"""Capture PADE data-driven com a semantica temporal do ThreadPipeline.

O scheduler usa planos compartilhados e deadlines monotonicos absolutos. A
fonte interna de verdade sao ``FrameEvent``, ``EndPassageEvent`` e
``EndPipelineEvent``. A aresta Capture -> Selection transporta esses contratos
diretamente; apenas a finalizacao in-process da Prediction permanece legada.
"""

from __future__ import annotations

import json
import time
import uuid
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
    EndPassageEvent,
    EndPipelineEvent,
    FrameEvent,
    event_to_dict,
)
from mas.infrastructure.frame_store import FRAME_STORE, FrameStore
from mas.infrastructure.stream_sequence import StreamSequencer
from mas.utils.animal_dataset import AnimalDataset


PIPELINE_EVENT_ONTOLOGY = "pipeline-event"


class DatasetCaptureBehaviour(Behaviour):
    """Agenda frames e controles por deadlines absolutos, sem bloquear reactor."""

    ANOMALY_SPAN_SECONDS = 120.0

    def __init__(
        self,
        agent: Agent,
        dataset: AnimalDataset,
        next_agent_aid: str,
        selection_agent_aid: str,
        animal_tags: list[str],
        fps: float | None,
        max_passage_seconds: float | None = None,
        native_timestamps: bool = False,
        predict_agent=None,
        frame_store: FrameStore = FRAME_STORE,
        verbose: bool = False,
        *,
        call_later: Callable | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        iso_now: Callable[[], str] | None = None,
        frame_id_factory: Callable[[], str] | None = None,
    ):
        super().__init__(agent)
        if not native_timestamps and (fps is None or fps <= 0):
            raise ValueError("fps must be greater than zero in fixed-fps mode")

        self.dataset = dataset
        self.next_agent_aid = next_agent_aid
        self.selection_agent_aid = selection_agent_aid
        self.animal_tags = animal_tags
        self.fps = fps
        self.max_passage_seconds = max_passage_seconds
        self.native_timestamps = native_timestamps
        self.predict_agent = predict_agent
        self.frame_store = frame_store
        self.verbose = verbose

        self._call_later = call_later or reactor.callLater
        self._monotonic = monotonic
        self._iso_now = iso_now or (lambda: datetime.now().isoformat())
        self._frame_id_factory = frame_id_factory or (
            lambda: str(uuid.uuid4())[:12]
        )
        self._sequencer = StreamSequencer()

        self.tag_idx = 0
        self._started = False
        self._finished = False
        self._current_tag: str | None = None
        self._current_frames: list[dict] = []
        self._current_plan = PassageCapturePlan(None, None, ())
        self._plan_index = 0
        self._passage_start = 0.0
        self.captured_count = 0
        self.first_capture: str | None = None
        self.last_capture: str | None = None

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

    def _schedule_at(self, deadline: float, callback: Callable) -> None:
        delay = max(0.0, deadline - self._monotonic())
        self._call_later(delay, callback)

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
            fps=self.fps,
            native_timestamps=self.native_timestamps,
            max_passage_seconds=self.max_passage_seconds,
        )

        self._current_tag = tag
        self._current_frames = index
        self._current_plan = plan
        self._plan_index = 0
        self._passage_start = self._monotonic()
        self.captured_count = 0
        self.first_capture = None
        self.last_capture = None

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
            timing = "nativos" if self.native_timestamps else f"{self.fps} fps"
            display_message(
                self.agent.aid.name,
                f"[START] Animal {tag} "
                f"({self.tag_idx + 1}/{len(self.animal_tags)}) - "
                f"{len(index)} frames, span {span_s:.2f}s, {timing}",
            )

        self._schedule_next_event()

    def _schedule_next_event(self) -> None:
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

    def _capture_planned_event(self) -> None:
        if self._finished or self._plan_index >= len(self._current_plan.events):
            return

        capture_event = self._current_plan.events[self._plan_index]
        self._plan_index += 1
        frame = self._current_frames[capture_event.source_index]
        tag = self._current_tag
        img = self.dataset.load_depth(tag, frame["depth_filename"])

        if img is None:
            display_message(
                self.agent.aid.name,
                f"[WARN] load_depth retornou None para "
                f"{tag}/{frame['depth_filename']}",
            )
            self._schedule_next_event()
            return

        frame_id = self._frame_id_factory()
        now_iso = self._iso_now()
        if self.captured_count == 0:
            self.first_capture = now_iso
        self.last_capture = now_iso
        self.captured_count += 1

        event = FrameEvent(
            stream_seq=self._sequencer.next_seq(),
            frame_id=frame_id,
            passage_id=tag,
            capture_index=self.captured_count,
            elapsed_time=round(capture_event.scheduled_capture_time_ms, 2),
            depth_filename=frame.get("depth_filename"),
            label=frame.get("label"),
            dataset_timestamp_ms=(
                float(frame["relative_time_ms"])
                if self.native_timestamps
                else None
            ),
        )

        # O array fica disponivel antes da publicacao dos metadados.
        self.frame_store.put(frame_id, img)
        # A futura CaptureTimingRecorder do PADE deve registrar a admissao
        # depois que Selection inserir este evento em sua OrderedInbox. O
        # send ACL abaixo, isoladamente, nao prova admissao na primeira borda.
        self._send_pipeline_event(event)

        if self.verbose:
            display_message(
                self.agent.aid.name,
                f"[CAPTURE] animal={tag} idx={self.captured_count} "
                f"t={event.elapsed_time:.1f}ms label={event.label} "
                f"seq={event.stream_seq} -> {self.next_agent_aid}",
            )

        self._schedule_next_event()

    def _finish_passage(self) -> None:
        if self._finished or self._current_tag is None:
            return

        tag = self._current_tag
        event = EndPassageEvent(
            stream_seq=self._sequencer.next_seq(),
            passage_id=tag,
            total_captured_frames=self.captured_count,
            first_capture_time=self.first_capture,
            last_capture_time=self.last_capture,
        )
        self._send_pipeline_event(event)

        # Ponte temporaria: Prediction ainda nao consome EndPassageEvent.
        if self.predict_agent is not None:
            self.predict_agent.notify_capture_done(
                tag,
                total_frames=event.total_captured_frames,
                first_capture=event.first_capture_time,
                last_capture=event.last_capture_time,
            )

        display_message(
            self.agent.aid.name,
            f"[PASSAGE-COMPLETE] Animal {tag}: "
            f"{self.captured_count} frames capturados, seq={event.stream_seq}.",
        )

        self.tag_idx += 1
        self._current_tag = None
        self._current_frames = []
        self._current_plan = PassageCapturePlan(None, None, ())

        # Nenhum ACK ou drain downstream: prepara N+1 imediatamente.
        self._begin_next_passage()

    def _finish_pipeline(self) -> None:
        event = EndPipelineEvent(stream_seq=self._sequencer.next_seq())
        self._send_pipeline_event(event)
        self._finished = True
        display_message(
            self.agent.aid.name,
            f"[FINISH] Captura concluida para {len(self.animal_tags)} animais; "
            f"EndPipeline seq={event.stream_seq}.",
        )

    def _send_pipeline_event(
        self,
        event: FrameEvent | EndPassageEvent | EndPipelineEvent,
    ) -> None:
        self._send_message(
            receiver=self.next_agent_aid,
            ontology=PIPELINE_EVENT_ONTOLOGY,
            payload=event_to_dict(event),
        )

    def _send_message(self, *, receiver: str, ontology: str, payload: dict) -> None:
        msg = ACLMessage(ACLMessage.INFORM)
        msg.set_ontology(ontology)
        msg.add_receiver(AID(receiver))
        msg.set_content(json.dumps(payload, ensure_ascii=True))
        self.agent.send(msg)


class DatasetCaptureAgent(Agent):
    """Capture PADE para Fixed-FPS ou Original-Timing data-driven."""

    def __init__(
        self,
        aid,
        dataset: AnimalDataset,
        next_agent_aid: str,
        selection_agent_aid: str,
        animal_tags: list[str],
        fps: float | None,
        max_passage_seconds: float | None = None,
        native_timestamps: bool = False,
        wait_for_aids: list[str] | None = None,
        predict_agent=None,
        frame_store: FrameStore = FRAME_STORE,
        debug: bool = False,
        verbose: bool = False,
    ):
        super().__init__(aid=aid, debug=debug)
        if not native_timestamps and (fps is None or fps <= 0):
            raise ValueError("fps must be greater than zero in fixed-fps mode")

        self.dataset = dataset
        self.next_agent_aid = next_agent_aid
        self.selection_agent_aid = selection_agent_aid
        self.animal_tags = animal_tags
        self.fps = fps
        self.max_passage_seconds = max_passage_seconds
        self.native_timestamps = native_timestamps
        self.predict_agent = predict_agent
        self.frame_store = frame_store
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
            max_passage_seconds=self.max_passage_seconds,
            native_timestamps=self.native_timestamps,
            predict_agent=self.predict_agent,
            frame_store=self.frame_store,
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
            mode = "Original-Timing" if self.native_timestamps else f"{self.fps} fps"
            display_message(
                self.aid.name,
                f"DatasetCaptureAgent IGNITED ({mode}). Iniciando captura.",
            )
        self.capture_behaviour.start()

    def react(self, message):
        super().react(message)
        if message.ontology != "agent-ready":
            return
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
