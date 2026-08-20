"""Capture PADE data-driven com a semantica temporal do ThreadPipeline e suporte a Visual-Gated.

O scheduler usa planos compartilhados e deadlines monotonicos absolutos.
Suporta modos:
1. Fixed-FPS estático;
2. Original-Timing estático (native timestamps);
3. Visual-Adaptive (taxa dinâmica LOW/HIGH com envio irrestrito ao Selection);
4. Visual-Gated Adaptive (taxa dinâmica LOW/HIGH com gating no ramo pesado em LOW).
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import replace
from datetime import datetime
from typing import Callable

import numpy as np
from twisted.internet import reactor

import mas  # noqa: F401  (sys.path hack para pade/infra)

from pade.acl.aid import AID
from pade.acl.messages import ACLMessage
from pade.behaviours.protocols import Behaviour
from pade.core.agent import Agent
from pade.misc.utility import display_message

from domain.helpers.capture_schedule import (
    PassageCapturePlan,
    build_passage_capture_plan,
)
from domain.pipeline_events import (
    CaptureControlEvent,
    EndPassageEvent,
    EndPipelineEvent,
    FrameEvent,
    event_to_dict,
)
from domain.visual_activity import VisualState
from domain.visual_events import (
    VisualFrameEvent,
    VisualStateEvent,
    visual_event_to_dict,
)
from infra.profiling.telemetry import CaptureTimingRecorder, TelemetryContext
from mas.infrastructure.frame_store import FRAME_STORE, FrameStore
from mas.infrastructure.stream_sequence import StreamSequencer
from mas.utils.animal_dataset import AnimalDataset


PIPELINE_EVENT_ONTOLOGY = "pipeline-event"
VISUAL_EVENT_ONTOLOGY = "visual-event"
VISUAL_STATE_ONTOLOGY = "visual-state"
CAPTURE_CONTROL_ONTOLOGY = "capture-control"
CAPTURE_PASSAGE_STARTED_ONTOLOGY = "capture-passage-started"
VISUAL_LEASE_OWNER = "visual"


class DatasetCaptureBehaviour(Behaviour):
    """Agenda frames e controles por deadlines absolutos, com suporte adaptativo e gating."""

    ANOMALY_SPAN_SECONDS = 120.0

    def __init__(
        self,
        agent: Agent,
        dataset: AnimalDataset,
        next_agent_aid: str,
        selection_agent_aid: str,
        animal_tags: list[str],
        fps: float | None = None,
        low_fps: float | None = None,
        medium_fps: float | None = None,
        visual_gated: bool = False,
        orchestrator_agent_aid: str | None = None,
        max_passage_seconds: float | None = None,
        native_timestamps: bool = False,
        frame_store: FrameStore = FRAME_STORE,
        telemetry_context: TelemetryContext | None = None,
        capture_timing_recorder: CaptureTimingRecorder | None = None,
        visual_agent_aid: str | None = None,
        verbose: bool = False,
        *,
        call_later: Callable | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        iso_now: Callable[[], str] | None = None,
        frame_id_factory: Callable[[], str] | None = None,
    ):
        super().__init__(agent)
        self.dataset = dataset
        self.next_agent_aid = next_agent_aid
        self.selection_agent_aid = selection_agent_aid
        self.animal_tags = animal_tags
        self.fps = fps
        self.low_fps = low_fps
        self.medium_fps = medium_fps
        self.visual_gated = visual_gated
        self.orchestrator_agent_aid = orchestrator_agent_aid
        self.max_passage_seconds = max_passage_seconds
        self.native_timestamps = native_timestamps
        self.frame_store = frame_store
        self.telemetry_context = telemetry_context
        self.capture_timing_recorder = capture_timing_recorder
        self.visual_agent_aid = visual_agent_aid
        self.verbose = verbose

        self.adaptive_mode = self.low_fps is not None

        self._call_later = call_later or reactor.callLater
        self._monotonic = monotonic
        self._iso_now = iso_now or (lambda: datetime.now().isoformat())
        self._frame_id_factory = frame_id_factory or (
            lambda: str(uuid.uuid4())[:12]
        )
        self._sequencer = StreamSequencer()
        self._visual_sequencer = StreamSequencer()

        self.tag_idx = 0
        self._started = False
        self._finished = False
        self._current_tag: str | None = None
        self._current_frames: list[dict] = []
        self._current_plan = PassageCapturePlan(None, None, ())
        self._plan_index = 0
        self._source_cursor = 0
        self._passage_start = 0.0
        self.captured_count = 0
        self.first_capture: str | None = None
        self.last_capture: str | None = None

        self._current_rate = "LOW" if self.adaptive_mode else ("HIGH" if self.native_timestamps else "FIXED")
        self._next_low_scheduled_ms = 0.0
        self._scheduled_call = None
        self._pending_low_frames: dict[str, FrameEvent] = {}

    def on_start(self):
        super().on_start()
        if self.agent.simulation_started:
            self.start()

    def start(self) -> None:
        """Inicia uma unica vez, depois do handshake agent-ready."""
        if self._started or self._finished:
            return
        self._started = True
        self._call_later(0.0, self._begin_next_passage)

    def _cancel_scheduled_call(self) -> None:
        if self._scheduled_call is not None:
            try:
                if hasattr(self._scheduled_call, "active") and self._scheduled_call.active():
                    self._scheduled_call.cancel()
            except Exception:
                pass
            self._scheduled_call = None

    def _schedule_at(self, deadline: float, callback: Callable):
        self._cancel_scheduled_call()
        delay = max(0.0, deadline - self._monotonic())
        self._scheduled_call = self._call_later(delay, callback)
        return self._scheduled_call

    def _begin_next_passage(self) -> None:
        if self._finished:
            return
        if self.tag_idx >= len(self.animal_tags):
            self._finish_pipeline()
            return

        tag = self.animal_tags[self.tag_idx]
        index = list(self.dataset.load_index(tag))
        index.sort(key=lambda item: item["relative_time_ms"])
        times = np.array(
            [item["relative_time_ms"] for item in index], dtype=float
        )
        plan = build_passage_capture_plan(
            times,
            fps=self.fps if not self.adaptive_mode else self.low_fps,
            native_timestamps=self.native_timestamps,
            max_passage_seconds=self.max_passage_seconds,
        )

        self._current_tag = tag
        self._current_frames = index
        self._current_plan = plan
        self._plan_index = 0
        self._source_cursor = 0
        self._passage_start = self._monotonic()
        self.captured_count = 0
        self.first_capture = None
        self.last_capture = None
        self._pending_low_frames.clear()

        if self.adaptive_mode:
            self._current_rate = "LOW"
            self._next_low_scheduled_ms = float(times[0]) if len(times) > 0 else 0.0

        if self.telemetry_context is not None:
            self.telemetry_context.set_capture_passage_id(tag)

        if self.orchestrator_agent_aid is not None:
            self._send_message(
                receiver=self.orchestrator_agent_aid,
                ontology=CAPTURE_PASSAGE_STARTED_ONTOLOGY,
                payload={"passage_id": tag},
            )

        if not index:
            display_message(
                self.agent.aid.name,
                f"[WARN] Animal {tag} possui simulation_index vazio.",
            )
        else:
            span_s = (times[-1] - times[0]) / 1000.0
            if span_s > self.ANOMALY_SPAN_SECONDS:
                display_message(
                    self.agent.aid.name,
                    f"[WARN] Animal {tag} tem span anomalo: {span_s:.1f}s "
                    f"({len(index)} frames). Considere --max_passage_seconds.",
                )
            if self.adaptive_mode:
                gated_str = "Visual-Gated" if self.visual_gated else "Visual-Adaptive"
                timing = f"{gated_str} (LOW={self.low_fps}fps, HIGH=native)"
            else:
                timing = "nativos" if self.native_timestamps else f"{self.fps} fps"
            display_message(
                self.agent.aid.name,
                f"[START] Animal {tag} "
                f"({self.tag_idx + 1}/{len(self.animal_tags)}) - "
                f"{len(index)} frames, span {span_s:.2f}s, {timing}",
            )

        self._schedule_next_event()

    def _schedule_next_event(self) -> None:
        if not self.adaptive_mode:
            if self._plan_index < len(self._current_plan.events):
                capture_event = self._current_plan.events[self._plan_index]
                offset_s = (
                    capture_event.scheduled_capture_time_ms
                    - self._current_plan.first_timestamp_ms
                ) / 1000.0
                self._schedule_at(
                    self._passage_start + offset_s,
                    self._capture_planned_event,
                )
                return

            self._schedule_at(
                self._passage_start + self._current_plan.end_offset_s,
                self._finish_passage,
            )
            return

        # Modo Adaptativo Dinâmico
        if self._source_cursor >= len(self._current_frames):
            self._schedule_at(
                self._passage_start + self._current_plan.end_offset_s,
                self._finish_passage,
            )
            return

        if self._current_rate == "HIGH":
            target_idx = self._source_cursor
        else:
            # Modo LOW: encontra o primeiro frame com t >= next_low_scheduled_ms sem alterar _source_cursor
            target_idx = self._source_cursor
            while target_idx < len(self._current_frames):
                t = float(self._current_frames[target_idx]["relative_time_ms"])
                if t >= self._next_low_scheduled_ms - 1e-5:
                    break
                target_idx += 1

            if target_idx >= len(self._current_frames):
                self._schedule_at(
                    self._passage_start + self._current_plan.end_offset_s,
                    self._finish_passage,
                )
                return

        frame = self._current_frames[target_idx]
        target_t_ms = float(frame["relative_time_ms"])
        offset_s = (
            target_t_ms - self._current_plan.first_timestamp_ms
        ) / 1000.0
        self._schedule_at(
            self._passage_start + offset_s,
            lambda idx=target_idx: self._capture_adaptive_event(idx),
        )

    def _capture_planned_event(self) -> None:
        if self._finished or self._plan_index >= len(self._current_plan.events):
            return

        capture_event = self._current_plan.events[self._plan_index]
        self._plan_index += 1
        frame = self._current_frames[capture_event.source_index]
        self._emit_frame(frame, capture_event.scheduled_capture_time_ms)
        self._schedule_next_event()

    def _capture_adaptive_event(self, target_idx: int) -> None:
        if self._finished or target_idx >= len(self._current_frames):
            return

        self._source_cursor = target_idx + 1
        frame = self._current_frames[target_idx]
        t_current = float(frame["relative_time_ms"])

        if self._current_rate == "LOW" and self.low_fps is not None:
            self._next_low_scheduled_ms = t_current + (1000.0 / self.low_fps)

        self._emit_frame(frame, t_current)
        self._schedule_next_event()


    def _emit_frame(self, frame: dict, scheduled_capture_time_ms: float) -> None:
        tag = self._current_tag
        img = self.dataset.load_depth(tag, frame["depth_filename"])

        if img is None:
            display_message(
                self.agent.aid.name,
                f"[WARN] load_depth retornou None para "
                f"{tag}/{frame['depth_filename']}",
            )
            return

        frame_id = self._frame_id_factory()
        now_iso = self._iso_now()
        if self.captured_count == 0:
            self.first_capture = now_iso
        self.last_capture = now_iso
        self.captured_count += 1

        event = FrameEvent(
            stream_seq=0,
            frame_id=frame_id,
            passage_id=tag,
            capture_index=self.captured_count,
            elapsed_time=round(scheduled_capture_time_ms, 2),
            depth_filename=frame.get("depth_filename"),
            label=frame.get("label"),
            dataset_timestamp_ms=float(frame["relative_time_ms"]),
        )

        self.frame_store.put(frame_id, img)
        visual_event = None
        if self.visual_agent_aid is not None:
            lease_id = None
            try:
                lease_id = self.frame_store.retain(
                    frame_id,
                    owner=VISUAL_LEASE_OWNER,
                    passage_id=tag,
                )
                visual_event = VisualFrameEvent(
                    stream_seq=self._visual_sequencer.next_seq(),
                    lease_id=lease_id,
                    passage_id=tag,
                    capture_index=self.captured_count,
                    elapsed_time=event.elapsed_time,
                    dataset_timestamp_ms=event.dataset_timestamp_ms,
                    depth_filename=event.depth_filename,
                    label=event.label,
                    frame_id=frame_id,
                )
            except Exception as exc:
                if lease_id is not None:
                    self._release_visual_lease_safely(lease_id)
                self._log_visual_error(
                    f"[ERROR] Visual lease creation failed for "
                    f"frame_id={frame_id}: {exc}",
                )

        if self.capture_timing_recorder is not None:
            offset_s = (
                scheduled_capture_time_ms
                - self._current_plan.first_timestamp_ms
            ) / 1000.0
            self.capture_timing_recorder.register_scheduled_event(
                passage_id=tag,
                capture_index=self.captured_count,
                frame_id=frame_id,
                source_filename=frame.get("depth_filename") or "",
                source_relative_time_ms=float(frame["relative_time_ms"]),
                scheduled_capture_time_ms=scheduled_capture_time_ms,
                scheduled_monotonic_ns=int(
                    round((self._passage_start + offset_s) * 1_000_000_000)
                ),
            )

        # Roteamento Visual-Gated:
        # Se visual_gated está ativo e o frame foi adquirido em LOW:
        # armazena pendente e NÃO envia ao Selection até confirmação visual.
        if self.visual_gated and self._current_rate == "LOW":
            self._pending_low_frames[frame_id] = event
        else:
            try:
                self._send_pipeline_event(event)
            except Exception:
                if visual_event is not None:
                    self._release_visual_lease_safely(visual_event.lease_id)
                if self.capture_timing_recorder is not None:
                    self.capture_timing_recorder.discard_scheduled_event(frame_id)
                raise

        if visual_event is not None:
            try:
                self._send_visual_event(visual_event)
            except Exception as exc:
                self._release_visual_lease_safely(visual_event.lease_id)
                self._log_visual_error(
                    f"[ERROR] Visual event publication failed for "
                    f"frame_id={frame_id}: {exc}",
                )

        if self.verbose:
            display_message(
                self.agent.aid.name,
                f"[CAPTURE] animal={tag} idx={self.captured_count} "
                f"t={event.elapsed_time:.1f}ms label={event.label} "
                f"rate={self._current_rate} -> {self.next_agent_aid}",
            )

    def handle_visual_state(self, obs: VisualStateEvent) -> None:
        """Processa decisão visual por frame para roteamento Visual-Gated."""
        if obs.passage_id != self._current_tag:
            return

        frame_id = obs.frame_id
        if not self.visual_gated or not frame_id or frame_id not in self._pending_low_frames:
            return

        pending_event = self._pending_low_frames.pop(frame_id)

        if obs.is_trigger:
            # Trigger IDLE -> ACTIVE: encaminha o MESMO frame_id ao Selection
            if self.verbose:
                display_message(
                    self.agent.aid.name,
                    f"[TRIGGER FORWARD] frame_id={frame_id} (IDLE->ACTIVE). Encaminhando ao Selection.",
                )
            try:
                self._send_pipeline_event(pending_event)
            except Exception as exc:
                self.frame_store.discard(frame_id)
                display_message(
                    self.agent.aid.name,
                    f"[ERROR] Trigger frame forwarding failed for frame_id={frame_id}: {exc}",
                )
            # Upshift para capturas futuras
            self._current_rate = "HIGH"
            self._schedule_next_event()
        else:
            # Frame permaneceu IDLE: descarta entrada principal do FrameStore
            self.frame_store.discard(frame_id)

    def handle_capture_control(self, target_rate: str, passage_id: str) -> None:
        """Processa comando de taxa vindo do Orchestrator."""
        if passage_id != self._current_tag:
            return

        if target_rate != self._current_rate:
            previous = self._current_rate
            self._current_rate = target_rate
            if self.verbose:
                display_message(
                    self.agent.aid.name,
                    f"[CAPTURE RATE CHANGE] passage={passage_id}: {previous}->{target_rate}",
                )
            if target_rate == "LOW" and self.low_fps is not None and self._source_cursor < len(self._current_frames):
                current_t = float(self._current_frames[max(0, self._source_cursor - 1)]["relative_time_ms"])
                self._next_low_scheduled_ms = current_t + (1000.0 / self.low_fps)
            self._schedule_next_event()

    def _finish_passage(self) -> None:
        if self._finished or self._current_tag is None:
            return

        tag = self._current_tag

        # Limpeza defensiva de frames pendentes não admitidos
        for fid in self._pending_low_frames:
            self.frame_store.discard(fid)
        self._pending_low_frames.clear()

        event = EndPassageEvent(
            stream_seq=0,
            passage_id=tag,
            total_captured_frames=self.captured_count,
            first_capture_time=self.first_capture,
            last_capture_time=self.last_capture,
        )
        self._send_pipeline_event(event)
        if self.visual_agent_aid is not None:
            visual_end = EndPassageEvent(
                stream_seq=self._visual_sequencer.next_seq(),
                passage_id=tag,
                total_captured_frames=self.captured_count,
                first_capture_time=self.first_capture,
                last_capture_time=self.last_capture,
            )
            try:
                self._send_visual_event(visual_end)
            except Exception as exc:
                self._log_visual_error(
                    f"[ERROR] Visual EndPassageEvent publication failed: {exc}",
                )

        if self.verbose:
            display_message(
                self.agent.aid.name,
                f"[END_PASSAGE] animal={tag} frames={self.captured_count}",
            )

        self._current_tag = None
        self._current_frames = []
        self._current_plan = PassageCapturePlan(None, None, ())
        self._plan_index = 0
        self._source_cursor = 0
        self.tag_idx += 1
        self._call_later(0.0, self._begin_next_passage)

    def _finish_pipeline(self) -> None:
        if self._finished:
            return
        self._finished = True

        event = EndPipelineEvent(stream_seq=0)
        self._send_pipeline_event(event)
        if self.visual_agent_aid is not None:
            visual_end = EndPipelineEvent(
                stream_seq=self._visual_sequencer.next_seq()
            )
            try:
                self._send_visual_event(visual_end)
            except Exception as exc:
                self._log_visual_error(
                    f"[ERROR] Visual EndPipelineEvent publication failed: {exc}",
                )

        display_message(
            self.agent.aid.name,
            f"[CAPTURE SHUTDOWN] Concluidas {len(self.animal_tags)} passagens.",
        )

    def _send_pipeline_event(self, event) -> None:
        event = replace(event, stream_seq=self._sequencer.next_seq())
        self._send_message(
            receiver=self.next_agent_aid,
            ontology=PIPELINE_EVENT_ONTOLOGY,
            payload=event_to_dict(event),
        )

    def _send_visual_event(self, event) -> None:
        if self.visual_agent_aid is None:
            return
        self._send_message(
            receiver=self.visual_agent_aid,
            ontology=VISUAL_EVENT_ONTOLOGY,
            payload=visual_event_to_dict(event),
        )

    def _release_visual_lease_safely(self, lease_id: str) -> None:
        try:
            self.frame_store.release_lease(
                lease_id,
                owner=VISUAL_LEASE_OWNER,
            )
        except Exception as exc:
            self._log_visual_error(
                f"[ERROR] Visual lease cleanup failed for {lease_id}: {exc}"
            )

    def _log_visual_error(self, text: str) -> None:
        try:
            display_message(self.agent.aid.name, text)
        except Exception:
            pass

    def _send_message(self, *, receiver: str, ontology: str, payload: dict) -> None:
        msg = ACLMessage(ACLMessage.INFORM)
        msg.set_ontology(ontology)
        msg.add_receiver(AID(receiver))
        msg.set_content(json.dumps(payload, ensure_ascii=True))
        self.agent.send(msg)



class DatasetCaptureAgent(Agent):
    """Capture PADE para Fixed-FPS, Original-Timing ou Visual-Gated Adaptive."""

    def __init__(
        self,
        aid,
        dataset: AnimalDataset,
        next_agent_aid: str,
        selection_agent_aid: str,
        animal_tags: list[str],
        fps: float | None = None,
        low_fps: float | None = None,
        medium_fps: float | None = None,
        visual_gated: bool = False,
        orchestrator_agent_aid: str | None = None,
        max_passage_seconds: float | None = None,
        native_timestamps: bool = False,
        wait_for_aids: list[str] | None = None,
        frame_store: FrameStore = FRAME_STORE,
        telemetry_context: TelemetryContext | None = None,
        capture_timing_recorder: CaptureTimingRecorder | None = None,
        visual_agent_aid: str | None = None,
        debug: bool = False,
        verbose: bool = False,
    ):
        super().__init__(aid=aid, debug=debug)
        if not native_timestamps and low_fps is None and (fps is None or fps <= 0):
            raise ValueError("fps or low_fps must be specified")

        self.dataset = dataset
        self.next_agent_aid = next_agent_aid
        self.selection_agent_aid = selection_agent_aid
        self.animal_tags = animal_tags
        self.fps = fps
        self.low_fps = low_fps
        self.medium_fps = medium_fps
        self.visual_gated = visual_gated
        self.orchestrator_agent_aid = orchestrator_agent_aid
        self.max_passage_seconds = max_passage_seconds
        self.native_timestamps = native_timestamps
        self.frame_store = frame_store
        self.telemetry_context = telemetry_context
        self.capture_timing_recorder = capture_timing_recorder
        self.visual_agent_aid = visual_agent_aid
        self.verbose = verbose
        self.wait_for_aids = set(wait_for_aids) if wait_for_aids else set()
        self.ready_agents: set[str] = set()
        self.simulation_started = False

    def on_start(self):
        super().on_start()
        self.capture_behaviour = DatasetCaptureBehaviour(
            agent=self,
            dataset=self.dataset,
            next_agent_aid=self.next_agent_aid,
            selection_agent_aid=self.selection_agent_aid,
            animal_tags=self.animal_tags,
            fps=self.fps,
            low_fps=self.low_fps,
            medium_fps=self.medium_fps,
            visual_gated=self.visual_gated,
            orchestrator_agent_aid=self.orchestrator_agent_aid,
            max_passage_seconds=self.max_passage_seconds,
            native_timestamps=self.native_timestamps,
            frame_store=self.frame_store,
            telemetry_context=self.telemetry_context,
            capture_timing_recorder=self.capture_timing_recorder,
            visual_agent_aid=self.visual_agent_aid,
            verbose=self.verbose,
        )
        self.behaviours.append(self.capture_behaviour)

        if not self.wait_for_aids:
            self._start_simulation()
        else:
            display_message(
                self.aid.name,
                f"DatasetCaptureAgent aguardando agentes: {self.wait_for_aids}",
            )

    def _start_simulation(self):
        if not self.simulation_started:
            self.simulation_started = True
            if self.low_fps is not None:
                gated_str = "Visual-Gated" if self.visual_gated else "Visual-Adaptive"
                mode = f"{gated_str} (LOW={self.low_fps}fps)"
            else:
                mode = "Original-Timing" if self.native_timestamps else f"{self.fps} fps"
            display_message(
                self.aid.name,
                f"DatasetCaptureAgent IGNITED ({mode}). Iniciando captura.",
            )
        self.capture_behaviour.start()

    def react(self, message):
        super().react(message)
        ontology = message.ontology

        if ontology == "agent-ready":
            try:
                data = json.loads(message.content)
                agent_name = data.get("agent")
                self.ready_agents.add(agent_name)
                display_message(self.aid.name, f"Agent {agent_name} is READY.")

                if self.wait_for_aids.issubset(self.ready_agents):
                    self._start_simulation()
            except Exception as exc:
                display_message(
                    self.aid.name,
                    f"[ERROR] Processing agent-ready: {exc}",
                )

        elif ontology == CAPTURE_CONTROL_ONTOLOGY:
            try:
                data = json.loads(message.content)
                target_rate = data["target_rate"]
                passage_id = data["passage_id"]
                if hasattr(self, "capture_behaviour"):
                    self.capture_behaviour.handle_capture_control(target_rate, passage_id)
            except Exception as exc:
                display_message(
                    self.aid.name,
                    f"[ERROR] Processing capture-control: {exc}",
                )

        elif ontology == VISUAL_STATE_ONTOLOGY:
            try:
                data = json.loads(message.content)
                event = VisualStateEvent(
                    passage_id=data["passage_id"],
                    capture_index=data["capture_index"],
                    elapsed_time=data["elapsed_time"],
                    dataset_timestamp_ms=data.get("dataset_timestamp_ms"),
                    pdi_score=data.get("pdi_score", data.get("mad")),
                    moving=data.get("moving"),
                    visual_state=VisualState(data["visual_state"]),
                    transition=data.get("transition"),
                    processing_time_ms=data.get("processing_time_ms", 0.0),
                    depth_filename=data.get("depth_filename"),
                    frame_id=data.get("frame_id"),
                    is_trigger=data.get("is_trigger", False),
                    is_invalid=data.get("is_invalid", False),
                    p99_mm=data.get("p99_mm"),
                    fraction_ge_2500=data.get("fraction_ge_2500"),
                    mad=data.get("mad"),
                )
                if hasattr(self, "capture_behaviour"):
                    self.capture_behaviour.handle_visual_state(event)
            except Exception as exc:
                display_message(
                    self.aid.name,
                    f"[ERROR] Processing visual-state: {exc}",
                )
