#!/usr/bin/env python3
"""Microbenchmark instrumentado dos consumidores PADE, sem AMS/rede.

Cada operação usa o caminho local real ``ACLMessage -> react() -> OrderedInbox
-> deferToThread -> callback``. A métrica primária, porém, é delimitada dentro
da função computacional entregue ao worker: ela não incorpora espera de thread
ou entrega do callback pelo reactor.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import mas  # noqa: F401  # bootstrap local PADE vendorizado
from pade.acl.aid import AID
from pade.acl.messages import ACLMessage
from twisted.internet import reactor, task, threads
from twisted.internet.defer import inlineCallbacks

from component_benchmarks.imageset import build_selector_pool, build_suited_pool, cyclic_order
from domain.pipeline_events import FrameEvent, event_to_json
from domain.visual_events import VisualFrameEvent, visual_event_to_json
from mas.adapters.data_enhance_adapter import DataEnhanceAdapter
from mas.adapters.frame_selection_adapter import FrameSelectionAdapter
from mas.adapters.inference_adapter import InferenceAdapter
from mas.agents.data_enhance_agent import DataEnhanceAgent
from mas.agents.frame_selection import FrameSelectionAgent
from mas.agents.predict_weight_agent import PredictWeightAgent
from mas.agents.visual_event_agent import VisualEventAgent
from mas.infrastructure.frame_store import FrameStore
import mas.agents.data_enhance_agent as preprocessing_module
import mas.agents.frame_selection as selection_module
import mas.agents.predict_weight_agent as prediction_module
import mas.agents.visual_event_agent as visual_module


COMPONENTS = ("visual", "selection", "preprocessing", "prediction")
FIELDNAMES = [
    "iteration", "image_id", "service_time_ms", "thread_wait_ms",
    "callback_delay_ms", "total_latency_ms", "gc_occurred",
    "gc_generations", "gc_event_count",
]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--warmup", type=int, default=50)
    result.add_argument("--iterations", type=int, default=1000)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--pool-size", type=int, default=300)
    result.add_argument("--data-root", default="data/exp1")
    result.add_argument("--selector-model", default="infra/models/frame_selector.tflite")
    result.add_argument("--predictor-model", default="infra/models/sheep_weight_predictor.tflite")
    result.add_argument("--output-dir", default="benchmarks/runs")
    return result


def silence_agent_logs() -> None:
    """I/O de display não deve virar parte da latência de serviço."""
    for module in (selection_module, preprocessing_module, prediction_module, visual_module):
        module.display_message = lambda *args, **kwargs: None


class GcRecorder:
    """Registro global, thread-safe, dos callbacks normais do GC Python."""

    def __init__(self) -> None:
        self._events: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def callback(self, phase: str, info: dict[str, Any]) -> None:
        with self._lock:
            self._events.append({
                "time_ns": time.perf_counter_ns(),
                "phase": phase,
                "generation": int(info.get("generation", -1)),
                "collected": int(info.get("collected", 0)),
                "uncollectable": int(info.get("uncollectable", 0)),
            })

    def clear(self) -> None:
        with self._lock:
            self._events.clear()

    def summary_between(self, start_ns: int, end_ns: int) -> tuple[bool, str, int]:
        with self._lock:
            events = [event for event in self._events if start_ns <= event["time_ns"] <= end_ns]
        generations = sorted({event["generation"] for event in events})
        return bool(events), ";".join(map(str, generations)), len(events)


class Measurement:
    def __init__(self) -> None:
        self.dispatch_ns: int | None = None
        self.thread_start_ns: int | None = None
        self.thread_end_ns: int | None = None
        self.callback_delivery_ns: int | None = None


class TimedDeferExecutor:
    """Preserva deferToThread e coleta limites da infraestrutura assíncrona."""

    def __init__(self) -> None:
        self.current: Measurement | None = None

    def __call__(self, function, *args, **kwargs):
        measurement = self.current
        if measurement is None:
            raise RuntimeError("deferToThread requested without active measurement")
        measurement.dispatch_ns = time.perf_counter_ns()

        def wrapped():
            measurement.thread_start_ns = time.perf_counter_ns()
            try:
                return function(*args, **kwargs)
            finally:
                measurement.thread_end_ns = time.perf_counter_ns()

        deferred = threads.deferToThread(wrapped)

        def reactor_received(result):
            measurement.callback_delivery_ns = time.perf_counter_ns()
            return result

        # Registrado antes do callback do agente: marca a chegada de volta ao
        # reactor, sem misturar o callback de roteamento à métrica de serviço.
        deferred.addBoth(reactor_received)
        return deferred


def make_message(event, *, visual: bool = False) -> ACLMessage:
    message = ACLMessage(ACLMessage.INFORM)
    message.set_sender(AID(name="capture@localhost:5010" if visual else "upstream@localhost:5009"))
    message.set_ontology("visual-event" if visual else "pipeline-event")
    message.set_content(visual_event_to_json(event) if visual else event_to_json(event))
    return message


@inlineCallbacks
def wait_until_idle(agent):
    while agent._processing:
        yield task.deferLater(reactor, 0.0005, lambda: None)


def build_agent(name: str, args, store: FrameStore, executor: TimedDeferExecutor):
    sent: list[ACLMessage] = []
    if name == "selection":
        agent = FrameSelectionAgent(
            AID(name="selection@localhost:5012"),
            FrameSelectionAdapter(None, args.selector_model),
            "next@localhost:1", frame_store=store, defer_executor=executor,
        )
        agent.frame_selection_adapter.load_model()
        agent._record_selection = lambda *unused: None
    elif name == "preprocessing":
        agent = DataEnhanceAgent(
            AID(name="enhance@localhost:5011"), DataEnhanceAdapter(),
            "next@localhost:1", frame_store=store, defer_executor=executor,
        )
    elif name == "prediction":
        adapter = InferenceAdapter(args.predictor_model)
        adapter.load_model()
        agent = PredictWeightAgent(
            AID(name="predict@localhost:5013"), adapter, "single", "benchmark",
            frame_store=store, defer_executor=executor,
            call_later=lambda *unused: None, shutdown_callback=lambda: None,
        )
    else:
        agent = VisualEventAgent(
            AID(name="visual@localhost:5014"), "capture@localhost:1", "benchmark",
            frame_store=store, state_publisher=lambda unused: None,
            defer_executor=executor, reports_dir=args.output_dir,
        )
    agent.send = sent.append
    return agent


def event_for(name: str, sequence: int, frame_id: str, item: dict, store: FrameStore):
    if name == "visual":
        return VisualFrameEvent(
            sequence, store.retain(frame_id, owner="visual", passage_id="B"), "B", sequence,
            float(sequence), float(sequence), item["depth_filename"], item.get("label"), frame_id,
        )
    return FrameEvent(
        sequence, frame_id, "B", sequence, float(sequence), item["depth_filename"],
        item.get("label"), float(sequence),
    )


def finite_ms(value_ns: int | None, other_ns: int | None, field: str) -> float:
    if value_ns is None or other_ns is None:
        raise RuntimeError(f"missing {field} instrumentation timestamp")
    result = (value_ns - other_ns) / 1_000_000.0
    if not math.isfinite(result) or result < 0:
        raise RuntimeError(f"invalid {field}: {result!r}")
    return result


@inlineCallbacks
def benchmark_component(name: str, args, raw_pool, suited_pool, gc_recorder: GcRecorder):
    store = FrameStore()
    executor = TimedDeferExecutor()
    agent = build_agent(name, args, store, executor)
    pool = suited_pool if name in ("preprocessing", "prediction") else raw_pool
    order, _ = cyclic_order(len(pool), args.warmup + args.iterations, args.seed)

    # Prediction must receive the exact enhanced representation expected by the
    # runtime regressor; this preparation is outside the measured service call.
    if name == "prediction":
        enhance = DataEnhanceAdapter()
        pool = [{**item, "img": enhance.run(item["img"])} for item in pool]

    rows: list[dict[str, Any]] = []
    for global_index, pool_index in enumerate(order):
        item = pool[pool_index]
        frame_id = f"{name}-{global_index}"
        store.put(frame_id, item["img"])
        event = event_for(name, global_index, frame_id, item, store)
        measurement = Measurement()
        executor.current = measurement
        start_ns = time.perf_counter_ns()
        try:
            agent.react(make_message(event, visual=(name == "visual")))
            yield wait_until_idle(agent)
        finally:
            complete_ns = time.perf_counter_ns()
            executor.current = None
            store.discard(frame_id)

        if global_index == args.warmup - 1:
            # Warm-ups não compõem os CSVs. Uma coleta explícita antes da série
            # medida elimina lixo acumulado da preparação, mantendo GC normal.
            gc.collect()
            gc_recorder.clear()
        if global_index < args.warmup:
            continue

        iteration = global_index - args.warmup
        service = finite_ms(measurement.thread_end_ns, measurement.thread_start_ns, "service_time_ms")
        thread_wait = finite_ms(measurement.thread_start_ns, measurement.dispatch_ns, "thread_wait_ms")
        callback_delay = finite_ms(measurement.callback_delivery_ns, measurement.thread_end_ns, "callback_delay_ms")
        total = finite_ms(complete_ns, start_ns, "total_latency_ms")
        gc_occurred, generations, event_count = gc_recorder.summary_between(start_ns, complete_ns)
        rows.append({
            "iteration": iteration, "image_id": item["image_id"],
            "service_time_ms": f"{service:.9f}", "thread_wait_ms": f"{thread_wait:.9f}",
            "callback_delay_ms": f"{callback_delay:.9f}", "total_latency_ms": f"{total:.9f}",
            "gc_occurred": str(gc_occurred).lower(), "gc_generations": generations,
            "gc_event_count": event_count,
        })
    if len(rows) != args.iterations:
        raise RuntimeError(f"{name}: expected {args.iterations} rows, got {len(rows)}")
    return rows


def write_metadata(output: Path, args) -> None:
    def sha256(path: str) -> str:
        digest = hashlib.sha256()
        with Path(path).open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    metadata = {
        "protocol": "isolated PADE agents; one operation at a time; no AMS/network",
        "components": list(COMPONENTS), "warmup_excluded": args.warmup,
        "measured_iterations_per_component": args.iterations, "seed": args.seed,
        "data_root": args.data_root, "selector_model": args.selector_model,
        "selector_model_sha256": sha256(args.selector_model),
        "predictor_model": args.predictor_model,
        "predictor_model_sha256": sha256(args.predictor_model),
        "runtime_path": {
            "visual": "VisualEventAgent._observe_frame -> ActivityDetector.observe(raw depth)",
            "selection": "FrameSelectionAdapter.evaluate_with_score -> FrameSelection v3 ROI10",
            "preprocessing": "DataEnhanceAdapter.run -> DataEnhance",
            "prediction": "InferenceAdapter.predict -> PredictWeight TFLite",
        },
        "service_time_definition": "worker function execution only",
        "thread_wait_definition": "defer dispatch to worker start",
        "callback_delay_definition": "worker end to reactor delivery",
        "total_latency_definition": "react() entry to agent completion",
        "gc": "normal enabled; gc.collect() before measured series; callbacks recorded",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


@inlineCallbacks
def main(reactor, args):
    if args.warmup < 0 or args.iterations <= 1 or args.pool_size <= 0:
        raise ValueError("warmup >= 0, iterations > 1 and pool-size > 0 are required")
    if not Path(args.selector_model).is_file() or not Path(args.predictor_model).is_file():
        raise FileNotFoundError("final selector and predictor TFLite models are required")
    silence_agent_logs()
    output = Path(args.output_dir) / ("benchmark_pade_agents_instrumented_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    output.mkdir(parents=True, exist_ok=False)
    write_metadata(output, args)
    raw_pool, _ = build_selector_pool(args.data_root, args.seed, max(1, args.pool_size // 2))
    suited_pool, _ = build_suited_pool(args.data_root, args.seed + 1, args.pool_size)
    gc_recorder = GcRecorder()
    gc.callbacks.append(gc_recorder.callback)
    try:
        for component in COMPONENTS:
            rows = yield benchmark_component(component, args, raw_pool, suited_pool, gc_recorder)
            with (output / f"{component}_measurements.csv").open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
                writer.writeheader()
                writer.writerows(rows)
            print(f"{component}: {len(rows)} measured rows")
    finally:
        gc.callbacks.remove(gc_recorder.callback)
    print(f"__BENCHMARK_DIR__={output}")


if __name__ == "__main__":
    task.react(main, (parser().parse_args(),))
