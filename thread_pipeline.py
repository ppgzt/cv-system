"""Pipeline de threads data-driven por tag (paralelo a mas_pipeline.py:MASStrategy).

Mesmo pipeline, mesmos estágios, mesmos adapters do MASStrategy (PADE/Twisted/FIPA),
mas orquestrados como produtor-consumidor de threads em vez de agentes PADE:

    Capture (timer FPS) -> Q1 -> Select -> Q2 -> Enhance -> Q3 -> Predict

Cada estágio = 1 thread consumidora de uma queue.Queue ilimitada. O payload carrega
o np.ndarray da imagem diretamente (sem o FRAME_BUFFER global nem indireção por
frame_id do PADE). A terminação é determinística por sentinel:
    - ("END_ANIMAL", tag, total_frames, first, last) ao fim de cada animal
    - None após o último animal
como queue.Queue é FIFO, o END_ANIMAL de um animal chega DEPOIS de todos os seus
frames -> o Predict finaliza o animal com a média exata dos pesos (modo single) ou
inferência em batch (modo batch). Acabam o _in_flight/_maybe_finalize/lossy-FIPA.

Paridade científica com PADE/baseline por construção: os adapters
(DataEnhanceAdapter, FrameSelectionAdapter, InferenceAdapter), o ReportCollector e
os monitores (CPUMonitor/RAMMonitor/TempMonitor) são importados e chamados
textualmente idênticos — só a casca de orquestração muda.

Alvo: Raspberry Pi 5 (4 cores). TFLite/XNNPACK (num_threads=2 default) libera o GIL,
então select/enhance/predict se sobrepõem em cores distintas. 1 worker por estágio
deixa 2 cores livres enquanto predict roda -> sem starvation, sem throttle térmico.
"""

import os
import json
import time
import uuid
import queue
import threading
from datetime import datetime

import numpy as np

from domain.helpers.capture_schedule import (
    build_fixed_fps_schedule,
    nearest_index,
)


# Sentinel de fim de animal: (END_ANIMAL, tag, total_frames, first_capture, last_capture)
_END_ANIMAL = "END_ANIMAL"
# Sentinel de fim de pipeline: None


class ThreadPipeline:
    """Pipeline de threads data-driven por tag (engine padrão, --engine thread)."""

    ANOMALY_SPAN_SECONDS = 120.0

    def __init__(
        self,
        pid: str,
        mode: str,
        fps: float | None,
        num_animals: int | None = None,
        max_passage_seconds: float | None = None,
        data_root: str = "data/exp1",
        verbose: bool = False,
        native_timestamps: bool = False,
        capture_timing_enabled: bool = True,
    ):
        self.pid = pid
        self.mode = mode
        self.fps = fps
        self.num_animals = num_animals
        self.max_passage_seconds = max_passage_seconds
        self.data_root = data_root
        self.native_timestamps = native_timestamps
        self.verbose = verbose
        self.capture_timing_enabled = capture_timing_enabled

        if not self.native_timestamps and (self.fps is None or self.fps <= 0):
            raise ValueError("fps deve ser maior que zero no modo normal")

    # ------------------------------------------------------------------ #
    def _log(self, tag: str, msg: str):
        print(f"[{tag}] {msg}", flush=True)

    def _now(self) -> str:
        return datetime.now().isoformat()

    # ------------------------------------------------------------------ #
    # Capture (transplantado do DatasetCaptureBehaviour)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _nearest_index(times: np.ndarray, value: float) -> int:
        return nearest_index(times, value)

    def _capture_loop(
        self,
        dataset,
        animal_tags,
        q1,
        telemetry_context=None,
        capture_timing_recorder=None,
    ):
        # O caminho original permanece intacto. O modo nativo usa um produtor
        # alternativo, mas mantém as mesmas quatro threads do pipeline e as
        # mesmas filas/sentinelas downstream.
        if self.native_timestamps:
            self._capture_loop_native(
                dataset,
                animal_tags,
                q1,
                telemetry_context=telemetry_context,
                capture_timing_recorder=capture_timing_recorder,
            )
            return

        period = 1.0 / self.fps

        for tag_idx, tag in enumerate(animal_tags):
            if telemetry_context is not None:
                telemetry_context.set_capture_passage_id(tag)

            # Pré-carrega o simulation_index.json do animal
            index = dataset.load_index(tag)
            index.sort(key=lambda x: x["relative_time_ms"])
            times = np.array([x["relative_time_ms"] for x in index], dtype=float)
            frames = index

            end_ms = self._passage_end_ms(times)
            capture_schedule = build_fixed_fps_schedule(times, self.fps, end_ms)
            span_s = (times[-1] - times[0]) / 1000.0
            if span_s > self.ANOMALY_SPAN_SECONDS:
                self._log("capture_agent",
                          f"[WARN] Animal {tag} tem span anômalo: {span_s:.1f}s "
                          f"({len(index)} frames). Considere --max_passage_seconds.")
            self._log("capture_agent",
                      f"[START] Animal {tag} ({tag_idx + 1}/{len(animal_tags)}) "
                      f"- {len(index)} frames, span {span_s:.2f}s")

            captured_count = 0
            first_capture = None
            last_capture = None

            # Pacing de parede fiel ao TimedBehaviour (deadline-based).
            # FRAMEs seguem os ticks de captura; END_ANIMAL segue o limite
            # temporal real da passagem e não cria um tick adicional.
            passage_start = time.monotonic()
            passage_end_deadline = (
                passage_start + (end_ms - float(times[0])) / 1000.0
            )
            next_tick = passage_start
            for schedule_idx, capture_event in enumerate(capture_schedule):
                virtual_clock = capture_event.scheduled_capture_time_ms
                scheduled_monotonic_ns = round(next_tick * 1_000_000_000)
                j = capture_event.source_index
                frame = frames[j]
                img = dataset.load_depth(tag, frame["depth_filename"])
                if img is not None:
                    now_iso = self._now()
                    if captured_count == 0:
                        first_capture = now_iso
                    last_capture = now_iso
                    captured_count += 1

                    payload = {
                        "frame_id": str(uuid.uuid4())[:12],
                        "animal_id": tag,
                        "frame_index": captured_count,
                        "elapsed_time": round(virtual_clock, 2),
                        "label": frame.get("label"),
                        "depth_filename": frame.get("depth_filename"),
                        "img": img,
                    }
                    q1.put(payload)
                    if capture_timing_recorder is not None:
                        actual_enqueue_monotonic_ns = time.monotonic_ns()
                        capture_timing_recorder.record(
                            passage_id=tag,
                            capture_index=captured_count,
                            frame_id=payload["frame_id"],
                            source_filename=frame.get("depth_filename"),
                            source_relative_time_ms=float(frame["relative_time_ms"]),
                            scheduled_capture_time_ms=virtual_clock,
                            scheduled_monotonic_ns=scheduled_monotonic_ns,
                            actual_enqueue_monotonic_ns=(
                                actual_enqueue_monotonic_ns
                            ),
                        )

                    if self.verbose:
                        self._log("capture_agent",
                                  f"[CAPTURE] animal={tag} idx={captured_count} "
                                  f"t={virtual_clock:.1f}ms label={frame.get('label')}")

                next_tick += period
                next_deadline = (
                    next_tick
                    if schedule_idx + 1 < len(capture_schedule)
                    else passage_end_deadline
                )
                sleep_for = next_deadline - time.monotonic()
                if sleep_for > 0:
                    time.sleep(sleep_for)

            # Fim da passagem do animal
            q1.put((_END_ANIMAL, tag, captured_count, first_capture, last_capture))
            if telemetry_context is not None:
                telemetry_context.clear_capture_passage_id(tag)
            self._log("capture_agent",
                      f"[PASSAGE-COMPLETE] Animal {tag}: {captured_count} frames capturados.")

        q1.put(None)
        self._log("capture_agent",
                  f"[FINISH] Captura concluída para {len(animal_tags)} animais.")

    def _capture_loop_native(
        self,
        dataset,
        animal_tags,
        q1,
        telemetry_context=None,
        capture_timing_recorder=None,
    ):
        """Reproduz cada timestamp do dataset uma única vez.

        Existe somente um produtor, como no modo FPS. Os deadlines são
        relativos a um único relógio monotônico por animal, evitando que o
        tempo de leitura de PNG se acumule como drift. Se o pipeline estiver
        atrasado, não descartamos frames: a fila já é ilimitada no desenho
        atual e o produtor recupera o atraso naturalmente.
        """
        for tag_idx, tag in enumerate(animal_tags):
            if telemetry_context is not None:
                telemetry_context.set_capture_passage_id(tag)

            index = dataset.load_index(tag)
            index.sort(key=lambda x: x["relative_time_ms"])
            times = np.array([x["relative_time_ms"] for x in index], dtype=float)
            frames = index

            if not frames:
                self._log("capture_agent",
                          f"[WARN] Animal {tag} possui simulation_index vazio.")
                q1.put((_END_ANIMAL, tag, 0, None, None))
                if telemetry_context is not None:
                    telemetry_context.clear_capture_passage_id(tag)
                continue

            first_dataset_ms = float(times[0])
            end_ms = self._passage_end_ms(times)
            span_s = (times[-1] - times[0]) / 1000.0
            if span_s > self.ANOMALY_SPAN_SECONDS:
                self._log("capture_agent",
                          f"[WARN] Animal {tag} tem span anômalo: {span_s:.1f}s "
                          f"({len(index)} frames). Considere --max_passage_seconds.")
            self._log("capture_agent",
                      f"[START] Animal {tag} ({tag_idx + 1}/{len(animal_tags)}) "
                      f"- {len(index)} frames nativos, span {span_s:.2f}s")

            captured_count = 0
            first_capture = None
            last_capture = None
            replay_start = time.monotonic()

            for frame_time, frame in zip(times, frames):
                frame_time = float(frame_time)
                if frame_time > end_ms:
                    break

                deadline = replay_start + (frame_time - first_dataset_ms) / 1000.0
                scheduled_monotonic_ns = round(deadline * 1_000_000_000)
                sleep_for = deadline - time.monotonic()
                if sleep_for > 0:
                    time.sleep(sleep_for)

                img = dataset.load_depth(tag, frame["depth_filename"])
                if img is None:
                    self._log(
                        "capture_agent",
                        f"[WARN] load_depth retornou None para "
                        f"{tag}/{frame['depth_filename']}",
                    )
                    continue

                now_iso = self._now()
                if captured_count == 0:
                    first_capture = now_iso
                last_capture = now_iso
                captured_count += 1

                payload = {
                    "frame_id": str(uuid.uuid4())[:12],
                    "animal_id": tag,
                    "frame_index": captured_count,
                    # Mantém o timestamp original do índice, sem nearest-neighbor.
                    "elapsed_time": frame_time,
                    "dataset_timestamp_ms": frame_time,
                    "label": frame.get("label"),
                    "depth_filename": frame.get("depth_filename"),
                    "img": img,
                }
                q1.put(payload)
                if capture_timing_recorder is not None:
                    actual_enqueue_monotonic_ns = time.monotonic_ns()
                    capture_timing_recorder.record(
                        passage_id=tag,
                        capture_index=captured_count,
                        frame_id=payload["frame_id"],
                        source_filename=frame.get("depth_filename"),
                        source_relative_time_ms=frame_time,
                        scheduled_capture_time_ms=frame_time,
                        scheduled_monotonic_ns=scheduled_monotonic_ns,
                        actual_enqueue_monotonic_ns=actual_enqueue_monotonic_ns,
                    )

                if self.verbose:
                    self._log(
                        "capture_agent",
                        f"[CAPTURE] animal={tag} idx={captured_count} "
                        f"t={frame_time:.1f}ms label={frame.get('label')}",
                    )

            # Se max_passage_seconds terminar entre dois timestamps originais,
            # preserva o limite real da passagem sem inventar um frame/tick.
            passage_end_deadline = (
                replay_start + (end_ms - first_dataset_ms) / 1000.0
            )
            sleep_for = passage_end_deadline - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)

            q1.put((_END_ANIMAL, tag, captured_count, first_capture, last_capture))
            if telemetry_context is not None:
                telemetry_context.clear_capture_passage_id(tag)
            self._log(
                "capture_agent",
                f"[PASSAGE-COMPLETE] Animal {tag}: "
                f"{captured_count} frames capturados.",
            )

        q1.put(None)
        self._log(
            "capture_agent",
            f"[FINISH] Captura concluída para {len(animal_tags)} animais "
            "(timestamps nativos).",
        )

    def _passage_end_ms(self, times: np.ndarray) -> float:
        tmax = float(times[-1])
        if self.max_passage_seconds is not None:
            cap = float(times[0]) + self.max_passage_seconds * 1000.0
            return min(tmax, cap)
        return tmax

    # ------------------------------------------------------------------ #
    # Select worker
    # ------------------------------------------------------------------ #
    def _select_loop(self, selection_adapter, q1, q2):
        discarded = 0
        forwarded = 0
        # verbose: matriz label_real x decisão
        confusion: dict = {}

        while True:
            item = q1.get()
            if item is None:
                q2.put(None)
                break
            if isinstance(item, tuple) and item and item[0] == _END_ANIMAL:
                if self.verbose:
                    self._log("frame_selection_agent",
                              f"[SELECT-SUMMARY] animal={item[1]} total={item[2]} "
                              f"discarded={discarded} forwarded={forwarded}")
                q2.put(item)
                continue

            payload = item
            img = payload["img"]
            elapsed = payload.get("elapsed_time", 0.0)
            label = payload.get("label")
            animal_id = payload["animal_id"]

            suitable, prob = selection_adapter.evaluate_with_score(elapsed, img)

            try:
                from mas.utils.report_collector import ReportCollector
                ReportCollector().record_selection(
                    animal_id, payload.get("depth_filename"), label, suitable, prob)
            except Exception as e:
                self._log("frame_selection_agent", f"[REPORT-ERROR] record_selection failed: {e}")

            if self.verbose and label is not None:
                key = (label, bool(suitable))
                confusion.setdefault(animal_id, {})
                confusion[animal_id][key] = confusion[animal_id].get(key, 0) + 1

            if not suitable:
                discarded += 1
                payload.pop("img", None)  # libera a raw
                self._log("frame_selection_agent",
                          (f"[SELECT] frame_id={payload['frame_id']} animal={animal_id} "
                           f"label={label} -> DISCARDED (p={prob:.4f}). "
                           f"Discarded={discarded}, Forwarded={forwarded}")
                          if self.verbose else
                          (f"frame_id={payload['frame_id']} DISCARDED. "
                           f"Discarded={discarded}, Forwarded={forwarded}"))
            else:
                forwarded += 1
                self._log("frame_selection_agent",
                          (f"[SELECT] frame_id={payload['frame_id']} animal={animal_id} "
                           f"label={label} -> SUITABLE (p={prob:.4f}). "
                           f"Discarded={discarded}, Forwarded={forwarded}")
                          if self.verbose else
                          (f"frame_id={payload['frame_id']} SUITABLE. "
                           f"Discarded={discarded}, Forwarded={forwarded}"))
                q2.put(payload)

    # ------------------------------------------------------------------ #
    # Enhance worker
    # ------------------------------------------------------------------ #
    def _enhance_loop(self, enhance_adapter, q2, q3):
        while True:
            item = q2.get()
            if item is None:
                q3.put(None)
                break
            if isinstance(item, tuple) and item and item[0] == _END_ANIMAL:
                q3.put(item)
                continue

            payload = item
            try:
                payload["img"] = enhance_adapter.run(payload["img"])
            except Exception as e:
                self._log("data_enhance_agent", f"[ERROR] Enhancement failed: {e}")
                continue
            q3.put(payload)
            self._log("data_enhance_agent",
                      f"frame_id={payload['frame_id']} enhanced and forwarded.")

    # ------------------------------------------------------------------ #
    # Predict worker
    # ------------------------------------------------------------------ #
    def _predict_loop(self, inference_adapter, q3, herd_size, metrics):
        finalized = set()
        weights_by_animal: dict = {}
        labels_by_animal: dict = {}
        batch_imgs: dict = {}          # batch mode: acumula imgs por animal
        batch_payloads: dict = {}

        def finalize(animal_id):
            if animal_id in finalized:
                return
            weights = weights_by_animal.get(animal_id, [])
            predicted = float(np.mean(weights)) if weights else 0.0

            try:
                from mas.utils.report_collector import ReportCollector
                ReportCollector().record_final_prediction(animal_id, predicted)
            except Exception as e:
                self._log("predict_weight_agent", f"[REPORT-ERROR] record_final_prediction failed: {e}")

            if animal_id in metrics["animals"]:
                metrics["animals"][animal_id]["weight_prediction_final"] = self._now()
                if self.verbose:
                    from collections import Counter
                    lbl = dict(Counter(labels_by_animal.get(animal_id, [])))
                    self._log("predict_weight_agent",
                              f"[FINAL] Animal {animal_id}: n_suitable={len(weights)} "
                              f"| labels_dos_suitable={lbl} | peso_medio={predicted:.4f} kg")
                else:
                    self._log("predict_weight_agent",
                              f"[FINAL] Animal {animal_id} completed. Mean weight: {predicted:.4f} kg")

            finalized.add(animal_id)
            if len(finalized) >= herd_size:
                self._save_metrics(metrics)
                self._log("predict_weight_agent",
                          "[SHUTDOWN] All animals evaluated. Pipeline concluído.")

        while True:
            item = q3.get()
            if item is None:
                break
            if isinstance(item, tuple) and item and item[0] == _END_ANIMAL:
                _, animal_id, total_frames, first_capture, last_capture = item

                metrics["animals"].setdefault(animal_id, {"imgs": {}})
                entry = metrics["animals"][animal_id]
                entry["total_of_images"] = total_frames
                if first_capture:
                    entry["first_image_capture_time"] = first_capture
                if last_capture:
                    entry["last_image_capture_time"] = last_capture

                if self.mode == "batch":
                    imgs = batch_imgs.pop(animal_id, [])
                    payloads = batch_payloads.pop(animal_id, [])
                    if self.verbose:
                        for p in payloads:
                            if p.get("label") is not None:
                                labels_by_animal.setdefault(animal_id, []).append(p.get("label"))
                    entry["suitable_images"] = len(imgs)
                    if imgs:
                        idx_key = str(total_frames) if total_frames else "0"
                        start_ts = self._now()
                        result = inference_adapter.predict(imgs)
                        weights = [float(r[0]) for r in result] if result is not None else []
                        final_ts = self._now()
                        entry["imgs"][idx_key] = {
                            "weight_prediction_start": start_ts,
                            "weight_prediction_final": final_ts,
                        }
                        weights_by_animal[animal_id] = weights
                        self._log("predict_weight_agent",
                                  f"[BATCH INFERENCE] Ran on {len(imgs)} frames for animal {animal_id}")
                    else:
                        weights_by_animal[animal_id] = []
                else:
                    entry["suitable_images"] = len(weights_by_animal.get(animal_id, []))

                finalize(animal_id)
                continue

            # Frame
            payload = item
            animal_id = payload["animal_id"]
            frame_id = payload["frame_id"]
            img = payload.pop("img", None)
            if img is None:
                continue

            metrics["animals"].setdefault(animal_id, {"imgs": {}})

            if self.mode == "single":
                start_ts = self._now()
                result = inference_adapter.predict([img])
                weight = float(result[0][0]) if result is not None else None
                final_ts = self._now()

                if weight is not None:
                    weights_by_animal.setdefault(animal_id, []).append(weight)
                    if self.verbose and payload.get("label") is not None:
                        labels_by_animal.setdefault(animal_id, []).append(payload.get("label"))

                try:
                    from mas.utils.report_collector import ReportCollector
                    ReportCollector().record_prediction(
                        animal_id, payload.get("depth_filename"), weight)
                except Exception as e:
                    self._log("predict_weight_agent", f"[REPORT-ERROR] record_prediction failed: {e}")

                metrics["animals"][animal_id]["imgs"][str(payload["frame_index"])] = {
                    "weight_prediction_start": start_ts,
                    "weight_prediction_final": final_ts,
                }
                label_str = f" label={payload.get('label')}" if self.verbose else ""
                self._log("predict_weight_agent",
                          f"[PREDICTION] animal_id={animal_id} frame_id={frame_id}{label_str} "
                          f"weight={weight:.4f} kg" if weight is not None else
                          f"[PREDICTION] animal_id={animal_id} frame_id={frame_id} weight=None")
            else:
                # batch: acumula para inferência única ao fim do animal
                batch_imgs.setdefault(animal_id, []).append(img)
                batch_payloads.setdefault(animal_id, []).append(payload)

    # ------------------------------------------------------------------ #
    def _save_metrics(self, metrics):
        reports_dir = f"infra/reports/{self.pid}"
        os.makedirs(reports_dir, exist_ok=True)
        capture_mode = "native-timestamps" if self.native_timestamps else None
        if capture_mode is not None:
            metrics["capture_mode"] = capture_mode
        with open(os.path.join(reports_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)
        self._log("predict_weight_agent", f"[METRICS] Saved metrics.json to {reports_dir}")
        try:
            from mas.utils.report_collector import ReportCollector
            ReportCollector().generate_report(
                reports_dir, self.mode, self.fps, capture_mode=capture_mode
            )
            self._log("predict_weight_agent", f"[REPORT] Saved report.md to {reports_dir}")
        except Exception as e:
            self._log("predict_weight_agent", f"[REPORT-ERROR] generate_report failed: {e}")

    # ------------------------------------------------------------------ #
    def run(self):
        """Sobe o pipeline de threads e bloqueia até o rebanho ser finalizado."""
        run_monotonic_origin_ns = time.monotonic_ns()

        from dotenv import load_dotenv
        load_dotenv(override=True)

        import mas  # noqa: F401  (sys.path hack para mas/infra)
        from mas.utils.animal_dataset import AnimalDataset
        from mas.utils.report_collector import ReportCollector
        from mas.utils.cpu_monitor import CPUMonitor
        from mas.utils.ram_monitor import RAMMonitor
        from mas.utils.temp_monitor import TempMonitor
        from infra.profiling.telemetry import (
            CaptureTimingRecorder,
            HardwareTelemetryMonitor,
            QueueTelemetryMonitor,
            TelemetryContext,
        )
        from mas.adapters.data_enhance_adapter import DataEnhanceAdapter
        from mas.adapters.frame_selection_adapter import FrameSelectionAdapter
        from mas.adapters.inference_adapter import InferenceAdapter

        # 1. Dataset + ordem dos animais (alfabética por tag)
        dataset = AnimalDataset(self.data_root)
        animal_tags = dataset.list_tags(limit=self.num_animals)
        if not animal_tags:
            self._log("ThreadPipeline", f"[ERROR] nenhuma tag encontrada em {self.data_root}")
            return

        ReportCollector().reset()

        # 2. Adapters (lógica de domínio compartilhada, paridade com baseline/PADE)
        enhance_adapter = DataEnhanceAdapter()
        selection_adapter = FrameSelectionAdapter(suitable_window=None,
                                                  model_path="infra/models/frame_selector.tflite")
        inference_adapter = InferenceAdapter("infra/models/sheep_weight_predictor.tflite")

        self._log("ThreadPipeline", f"Iniciando pipeline de threads para PID: {self.pid}")
        if self.native_timestamps:
            self._log(
                "ThreadPipeline",
                f"Configuração: animais={len(animal_tags)}, "
                f"timestamps=nativos, mode={self.mode}",
            )
        else:
            self._log("ThreadPipeline",
                      f"Configuração: animais={len(animal_tags)}, fps={self.fps}, mode={self.mode}")

        # 3. Monitores (threads independentes de 1s — nunca bloqueadas pelo pipeline)
        cpu_monitor = CPUMonitor(pid=self.pid, reports_dir="infra/reports")
        ram_monitor = RAMMonitor(pid=self.pid, reports_dir="infra/reports")
        temp_monitor = TempMonitor(pid=self.pid, reports_dir="infra/reports")

        # 4. Carrega modelos sincronamente (remove a dança agent-ready do PADE)
        metrics = {
            "pid": self.pid,
            "load_model_start": self._now(),
            "load_model_final": None,
            "animals": {},
        }
        selection_adapter.load_model()
        inference_adapter.load_model()
        metrics["load_model_final"] = self._now()
        self._log("ThreadPipeline", "Modelos carregados.")

        # 5. Filas ilimitadas (captura a todo vapor, sem perda de frames)
        q1 = queue.Queue()  # capture -> select
        q2 = queue.Queue()  # select -> enhance
        q3 = queue.Queue()  # enhance -> predict

        # Metadados definidos pelo chamador; a camada de telemetria apenas os
        # registra e não interpreta políticas ou modos de captura.
        condition = "original_timing" if self.native_timestamps else "fixed_fps"
        capture_fps = None if self.native_timestamps else self.fps
        telemetry_context = TelemetryContext(
            run_id=self.pid,
            condition=condition,
            capture_fps=capture_fps,
            monotonic_origin_ns=run_monotonic_origin_ns,
        )
        queue_telemetry_monitor = QueueTelemetryMonitor(
            telemetry_context, q1, q2, q3, reports_dir="infra/reports"
        )
        hardware_telemetry_monitor = HardwareTelemetryMonitor(
            telemetry_context, reports_dir="infra/reports"
        )
        capture_timing_recorder = (
            CaptureTimingRecorder(telemetry_context, reports_dir="infra/reports")
            if self.capture_timing_enabled
            else None
        )
        telemetry_monitors = (
            queue_telemetry_monitor,
            hardware_telemetry_monitor,
        )

        capture_t = threading.Thread(
            target=self._capture_loop,
            args=(
                dataset,
                animal_tags,
                q1,
                telemetry_context,
                capture_timing_recorder,
            ),
            name="capture", daemon=True)
        select_t = threading.Thread(
            target=self._select_loop, args=(selection_adapter, q1, q2),
            name="select", daemon=True)
        enhance_t = threading.Thread(
            target=self._enhance_loop, args=(enhance_adapter, q2, q3),
            name="enhance", daemon=True)
        predict_t = threading.Thread(
            target=self._predict_loop, args=(inference_adapter, q3, len(animal_tags), metrics),
            name="predict", daemon=True)

        import psutil  # warmup do cpu_percent (igual baseline: primeira chamada = 0.0)
        psutil.cpu_percent(percpu=True)

        # Monitores ANTES dos workers; stop+join SEMPRE no finally
        cpu_monitor.start()
        ram_monitor.start()
        temp_monitor.start()
        for monitor in telemetry_monitors:
            monitor.start()

        capture_t.start()
        select_t.start()
        enhance_t.start()
        predict_t.start()

        try:
            predict_t.join()
        finally:
            # Interrompe primeiro os samplers novos, sem participar do lifecycle
            # operacional das passagens ou dos workers.
            for monitor in telemetry_monitors:
                monitor.stop()

            # Garante a escrita dos CSVs mesmo com exceção/Ctrl-C
            for m in (cpu_monitor, ram_monitor, temp_monitor):
                try:
                    m.stop()
                    m.join()
                except Exception as e:
                    self._log("ThreadPipeline", f"[WARN] monitor stop falhou: {e}")

            for monitor in telemetry_monitors:
                monitor.join(timeout=3.0)
                if monitor.is_alive():
                    self._log(
                        "ThreadPipeline",
                        f"[WARN] {monitor.name} não encerrou dentro do timeout",
                    )
                elif monitor.persist_error is not None:
                    self._log(
                        "ThreadPipeline",
                        f"[WARN] {monitor.name} não persistiu CSV: "
                        f"{monitor.persist_error}",
                    )

            if capture_timing_recorder is not None:
                if not capture_timing_recorder.persist():
                    self._log(
                        "ThreadPipeline",
                        "[WARN] CaptureTimingRecorder não persistiu CSV: "
                        f"{capture_timing_recorder.persist_error}",
                    )
                if capture_timing_recorder.dropped_events:
                    self._log(
                        "ThreadPipeline",
                        "[WARN] CaptureTimingRecorder descartou "
                        f"{capture_timing_recorder.dropped_events} eventos",
                    )
