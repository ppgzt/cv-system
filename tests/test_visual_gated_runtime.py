#!/usr/bin/env python3
"""Testes de conformidade e integração para o runtime Visual-Gated e Control Plane Mínimo."""

from __future__ import annotations

import unittest
from typing import Any
import numpy as np

from domain.pipeline_events import (
    EndPassageEvent,
    EndPipelineEvent,
    FrameEvent,
    SelectionEvidenceEvent,
)
from domain.selection_hold import SelectionHold
from domain.visual_activity import VisualState
from domain.visual_events import VisualFrameEvent, VisualStateEvent
from mas.agents.dataset_capture_agent import (
    DatasetCaptureAgent,
    DatasetCaptureBehaviour,
)
from mas.agents.frame_selection import FrameSelectionAgent
from mas.agents.orchestrator_agent import OrchestratorAgent
from mas.infrastructure.frame_store import FrameStore
from mas.infrastructure.ordered_inbox import OrderedInbox
from pade.acl.aid import AID


class DummyDataset:
    """Mock do AnimalDataset para testes unitários rápidos."""

    def __init__(self, data: dict[str, list[dict[str, Any]]]):
        self._data = data
        self._frames = {
            (tag, item["depth_filename"]): np.full((240, 320), 1500, dtype=np.uint16)
            for tag, items in data.items()
            for item in items
        }

    def list_tags(self, limit: int | None = None) -> list[str]:
        tags = sorted(self._data.keys())
        return tags[:limit] if limit else tags

    def load_index(self, tag: str) -> list[dict]:
        return list(self._data.get(tag, []))

    def load_depth(self, tag: str, filename: str) -> np.ndarray | None:
        return self._frames.get((tag, filename))


class ManualClock:
    """Relógio manual para controlar a passagem do tempo em testes sem I/O."""

    def __init__(self, start_time: float = 1000.0):
        self._time = start_time
        self._callbacks: list[tuple[float, Any]] = []

    def monotonic(self) -> float:
        return self._time

    def monotonic_ns(self) -> int:
        return int(self._time * 1_000_000_000)

    def call_later(self, delay: float, callback: Any) -> Any:
        deadline = self._time + delay
        entry = (deadline, callback)
        self._callbacks.append(entry)
        self._callbacks.sort(key=lambda x: x[0])
        return entry

    def advance(self, delta_s: float) -> None:
        self._time += delta_s
        while True:
            ready = [cb for cb in self._callbacks if cb[0] <= self._time + 1e-9]
            if not ready:
                break
            # Remove os que serão executados agora
            ready_set = set(id(entry) for entry in ready)
            self._callbacks = [cb for cb in self._callbacks if id(cb) not in ready_set]
            for _, cb in ready:
                cb()



class TestVisualGatedRuntime(unittest.TestCase):
    def setUp(self):
        self.frame_store = FrameStore()
        self.clock = ManualClock()
        self.dataset_data = {
            "passage_1": [
                {"relative_time_ms": 0.0, "depth_filename": "f0.png", "label": "background"},
                {"relative_time_ms": 100.0, "depth_filename": "f1.png", "label": "parcial"},
                {"relative_time_ms": 200.0, "depth_filename": "f2.png", "label": "suited"},
                {"relative_time_ms": 300.0, "depth_filename": "f3.png", "label": "suited"},
                {"relative_time_ms": 400.0, "depth_filename": "f4.png", "label": "suited"},
            ],
            "passage_2": [
                {"relative_time_ms": 0.0, "depth_filename": "p2_f0.png", "label": "background"},
                {"relative_time_ms": 100.0, "depth_filename": "p2_f1.png", "label": "suited"},
            ],
        }
        self.dataset = DummyDataset(self.dataset_data)

    def test_idle_frame_does_not_reach_selection_in_visual_gated(self):
        """1. Frame em LOW com Visual IDLE não chega ao Selection e é removido do FrameStore."""
        selection_inbox = OrderedInbox()
        capture_agent = DatasetCaptureAgent(
            aid=AID("capture@localhost:5000"),
            dataset=self.dataset,
            next_agent_aid="selection@localhost:5002",
            selection_agent_aid="selection@localhost:5002",
            animal_tags=["passage_1"],
            low_fps=5.0,
            visual_gated=True,
            frame_store=self.frame_store,
            visual_agent_aid="visual@localhost:5004",
        )
        behaviour = DatasetCaptureBehaviour(
            agent=capture_agent,
            dataset=self.dataset,
            next_agent_aid="selection@localhost:5002",
            selection_agent_aid="selection@localhost:5002",
            animal_tags=["passage_1"],
            low_fps=5.0,
            visual_gated=True,
            frame_store=self.frame_store,
            visual_agent_aid="visual@localhost:5004",
            call_later=self.clock.call_later,
            monotonic=self.clock.monotonic,
        )
        # Intercepta envio para o Selection
        pipeline_sent = []
        behaviour._send_pipeline_event = lambda evt: pipeline_sent.append(evt)

        # Inicia a captura
        behaviour.start()
        self.clock.advance(0.0)  # Emite Frame 1 em t=0

        self.assertEqual(len(pipeline_sent), 0)  # Gated out!
        self.assertEqual(len(behaviour._pending_low_frames), 1)
        frame_id = list(behaviour._pending_low_frames.keys())[0]
        self.assertIsNotNone(self.frame_store.get(frame_id))

        # Visual observa o frame e confirma IDLE
        idle_obs = VisualStateEvent(
            passage_id="passage_1",
            capture_index=1,
            elapsed_time=0.0,
            dataset_timestamp_ms=0.0,
            mad=0.01,
            moving=False,
            visual_state=VisualState.IDLE,
            transition=None,
            processing_time_ms=1.0,
            frame_id=frame_id,
            is_trigger=False,
        )
        behaviour.handle_visual_state(idle_obs)

        # Confirmado: não foi enviado ao Selection e foi descartado do FrameStore
        self.assertEqual(len(pipeline_sent), 0)
        self.assertIsNone(self.frame_store.get(frame_id))
        self.assertEqual(len(behaviour._pending_low_frames), 0)

    def test_trigger_idle_to_active_reaches_selection_once_zero_copy(self):
        """2 e 3. Trigger IDLE->ACTIVE chega ao Selection uma vez e preserva entrada no FrameStore sem cópia."""
        capture_agent = DatasetCaptureAgent(
            aid=AID("capture@localhost:5000"),
            dataset=self.dataset,
            next_agent_aid="selection@localhost:5002",
            selection_agent_aid="selection@localhost:5002",
            animal_tags=["passage_1"],
            low_fps=5.0,
            visual_gated=True,
            frame_store=self.frame_store,
            visual_agent_aid="visual@localhost:5004",
        )
        behaviour = DatasetCaptureBehaviour(
            agent=capture_agent,
            dataset=self.dataset,
            next_agent_aid="selection@localhost:5002",
            selection_agent_aid="selection@localhost:5002",
            animal_tags=["passage_1"],
            low_fps=5.0,
            visual_gated=True,
            frame_store=self.frame_store,
            visual_agent_aid="visual@localhost:5004",
            call_later=self.clock.call_later,
            monotonic=self.clock.monotonic,
        )
        pipeline_sent = []
        behaviour._send_pipeline_event = lambda evt: pipeline_sent.append(evt)

        behaviour.start()
        self.clock.advance(0.0)

        frame_id = list(behaviour._pending_low_frames.keys())[0]
        original_array = self.frame_store.get(frame_id)

        # Visual simula liberação de sua read lease e dispara transição IDLE->ACTIVE
        trigger_obs = VisualStateEvent(
            passage_id="passage_1",
            capture_index=1,
            elapsed_time=0.0,
            dataset_timestamp_ms=0.0,
            mad=0.9,
            moving=True,
            visual_state=VisualState.ACTIVE,
            transition="IDLE->ACTIVE",
            processing_time_ms=1.0,
            frame_id=frame_id,
            is_trigger=True,
        )
        behaviour.handle_visual_state(trigger_obs)

        # 1. Chega ao Selection imediatamente
        self.assertEqual(len(pipeline_sent), 1)
        self.assertIsInstance(pipeline_sent[0], FrameEvent)
        self.assertEqual(pipeline_sent[0].frame_id, frame_id)

        # 2. O mesmo array permanece no FrameStore (Zero-Copy)
        stored_after_lease_release = self.frame_store.get(frame_id)
        self.assertIs(stored_after_lease_release, original_array)

        # 3. Trigger não altera FPS diretamente; somente o Orchestrator pode.
        self.assertEqual(behaviour._current_rate, "LOW")
        behaviour.handle_capture_control("HIGH", "passage_1", control_sequence=1)
        self.assertEqual(behaviour._current_rate, "HIGH")

    def test_visual_adaptive_vs_visual_gated_routing_distinction(self):
        """7. Visual-Adaptive envia frames LOW ao Selection, enquanto Visual-Gated filtra."""
        # A. Visual-Adaptive (visual_gated = False)
        agent_adaptive = DatasetCaptureAgent(
            aid=AID("capture@localhost:5000"),
            dataset=self.dataset,
            next_agent_aid="selection@localhost:5002",
            selection_agent_aid="selection@localhost:5002",
            animal_tags=["passage_1"],
            low_fps=5.0,
            visual_gated=False,
            frame_store=self.frame_store,
        )
        beh_adaptive = DatasetCaptureBehaviour(
            agent=agent_adaptive,
            dataset=self.dataset,
            next_agent_aid="selection@localhost:5002",
            selection_agent_aid="selection@localhost:5002",
            animal_tags=["passage_1"],
            low_fps=5.0,
            visual_gated=False,
            frame_store=self.frame_store,
            call_later=self.clock.call_later,
            monotonic=self.clock.monotonic,
        )
        sent_adaptive = []
        beh_adaptive._send_pipeline_event = lambda evt: sent_adaptive.append(evt)
        beh_adaptive.start()
        self.clock.advance(0.0)

        self.assertEqual(len(sent_adaptive), 1)  # Frame em LOW foi enviado imediatamente!

        # B. Visual-Gated (visual_gated = True)
        agent_gated = DatasetCaptureAgent(
            aid=AID("capture@localhost:5000"),
            dataset=self.dataset,
            next_agent_aid="selection@localhost:5002",
            selection_agent_aid="selection@localhost:5002",
            animal_tags=["passage_1"],
            low_fps=5.0,
            visual_gated=True,
            frame_store=self.frame_store,
        )
        beh_gated = DatasetCaptureBehaviour(
            agent=agent_gated,
            dataset=self.dataset,
            next_agent_aid="selection@localhost:5002",
            selection_agent_aid="selection@localhost:5002",
            animal_tags=["passage_1"],
            low_fps=5.0,
            visual_gated=True,
            frame_store=self.frame_store,
            visual_agent_aid="visual@localhost:5004",
            call_later=self.clock.call_later,
            monotonic=self.clock.monotonic,
        )
        sent_gated = []
        beh_gated._send_pipeline_event = lambda evt: sent_gated.append(evt)
        beh_gated.start()
        self.clock.advance(0.0)

        self.assertEqual(len(sent_gated), 0)  # Frame em LOW foi retido pelo gate!

    def test_end_passage_sent_even_with_zero_admitted_frames(self):
        """8. Passagem sem nenhum frame admitido ainda envia EndPassageEvent ao Selection."""
        capture_agent = DatasetCaptureAgent(
            aid=AID("capture@localhost:5000"),
            dataset=self.dataset,
            next_agent_aid="selection@localhost:5002",
            selection_agent_aid="selection@localhost:5002",
            animal_tags=["passage_2"],
            low_fps=5.0,
            visual_gated=True,
            frame_store=self.frame_store,
        )
        behaviour = DatasetCaptureBehaviour(
            agent=capture_agent,
            dataset=self.dataset,
            next_agent_aid="selection@localhost:5002",
            selection_agent_aid="selection@localhost:5002",
            animal_tags=["passage_2"],
            low_fps=5.0,
            visual_gated=True,
            frame_store=self.frame_store,
            visual_agent_aid="visual@localhost:5004",
            call_later=self.clock.call_later,
            monotonic=self.clock.monotonic,
        )
        pipeline_sent = []
        behaviour._send_pipeline_event = lambda evt: pipeline_sent.append(evt)

        behaviour.start()
        self.clock.advance(0.0)  # Frame 1 emitido em LOW (gated)
        frame_id = next(iter(behaviour._pending_low_frames))
        behaviour.handle_visual_state(VisualStateEvent(
            passage_id="passage_2", capture_index=1, elapsed_time=0.0,
            dataset_timestamp_ms=0.0, pdi_score=0.0, moving=False,
            visual_state=VisualState.IDLE, transition=None,
            processing_time_ms=1.0, frame_id=frame_id, is_trigger=False,
        ))

        # Avança o tempo até o fim da passagem sem nenhum trigger (todos IDLE)
        self.clock.advance(1.0)

        # Deve ter enviado EndPassageEvent e depois EndPipelineEvent
        end_passage_evts = [e for e in pipeline_sent if isinstance(e, EndPassageEvent)]
        self.assertEqual(len(end_passage_evts), 1)
        self.assertEqual(end_passage_evts[0].passage_id, "passage_2")

    def test_stale_events_from_prior_passage_do_not_alter_active_capture(self):
        """10. SelectionEvidenceEvent ou VisualStateEvent atrasado de passagem anterior é rejeitado."""
        orch = OrchestratorAgent(
            aid=AID("orchestrator@localhost:5008"),
            capture_agent_aid="capture@localhost:5000",
            n_hold=2,
        )
        orch.handle_passage_started("passage_B")
        self.assertEqual(orch.current_rate, "LOW")
        self.assertFalse(orch.selection_hold.hold_active)

        # Evidência antiga da passage_A chega atrasada
        stale_sel = SelectionEvidenceEvent(
            passage_id="passage_A",
            capture_index=15,
            frame_id="F15",
            stream_seq=15,
            accepted=True,
            probability=0.99,
        )
        self.assertFalse(orch.handle_selection_evidence(stale_sel))
        self.assertFalse(orch.selection_hold.hold_active)
        self.assertEqual(orch.current_rate, "LOW")

        # VisualStateEvent antigo da passage_A chega atrasado
        stale_vis = VisualStateEvent(
            passage_id="passage_A",
            capture_index=16,
            elapsed_time=999.0,
            dataset_timestamp_ms=999.0,
            mad=0.99,
            moving=True,
            visual_state=VisualState.ACTIVE,
            transition="IDLE->ACTIVE",
            processing_time_ms=1.0,
        )
        self.assertFalse(orch.handle_visual_state(stale_vis))
        self.assertEqual(orch.current_rate, "LOW")

    def test_subsequent_frames_in_high_reach_selection_and_no_lease_leak(self):
        """4 e 12. Frames seguintes em HIGH chegam ao Selection e leases não vazam."""
        capture_agent = DatasetCaptureAgent(
            aid=AID("capture@localhost:5000"),
            dataset=self.dataset,
            next_agent_aid="selection@localhost:5002",
            selection_agent_aid="selection@localhost:5002",
            animal_tags=["passage_1"],
            low_fps=5.0,
            visual_gated=True,
            frame_store=self.frame_store,
            visual_agent_aid="visual@localhost:5004",
        )
        behaviour = DatasetCaptureBehaviour(
            agent=capture_agent,
            dataset=self.dataset,
            next_agent_aid="selection@localhost:5002",
            selection_agent_aid="selection@localhost:5002",
            animal_tags=["passage_1"],
            low_fps=5.0,
            visual_gated=True,
            frame_store=self.frame_store,
            visual_agent_aid="visual@localhost:5004",
            call_later=self.clock.call_later,
            monotonic=self.clock.monotonic,
        )
        pipeline_sent = []
        visual_sent = []
        behaviour._send_pipeline_event = lambda evt: pipeline_sent.append(evt)
        def mock_send_visual(evt):
            visual_sent.append(evt)
            if hasattr(evt, "lease_id") and evt.lease_id:
                self.frame_store.release_lease(evt.lease_id, owner="visual")
        behaviour._send_visual_event = mock_send_visual




        behaviour.start()
        self.clock.advance(0.0)  # Frame 1 capturado em LOW (t=0ms)

        f1_id = list(behaviour._pending_low_frames.keys())[0]

        # Simula trigger no Frame 1
        trigger_obs = VisualStateEvent(
            passage_id="passage_1",
            capture_index=1,
            elapsed_time=0.0,
            dataset_timestamp_ms=0.0,
            mad=0.8,
            moving=True,
            visual_state=VisualState.ACTIVE,
            transition="IDLE->ACTIVE",
            processing_time_ms=1.0,
            frame_id=f1_id,
            is_trigger=True,
        )
        behaviour.handle_visual_state(trigger_obs)
        self.assertEqual(len(pipeline_sent), 1)
        self.assertEqual(behaviour._current_rate, "LOW")
        behaviour.handle_capture_control("HIGH", "passage_1", control_sequence=1)
        self.assertEqual(behaviour._current_rate, "HIGH")

        # Avança tempo para capturar Frame 2 (t=100ms em HIGH)
        self.clock.advance(0.15)
        self.assertEqual(len(pipeline_sent), 2)  # Frame 2 foi enviado diretamente em HIGH!
        self.assertEqual(pipeline_sent[1].capture_index, 2)

        # Avança até o fim da passagem
        self.clock.advance(1.0)

        # Leases no frame_store devem ser limpas ou liberadas (não vazam)
        self.assertEqual(self.frame_store.lease_count(), 0)

    def test_error_handling_trigger_forwarding_cleans_up_framestore(self):
        """11. Falha no envio do trigger limpa o FrameStore sem deixar entrada órfã."""
        capture_agent = DatasetCaptureAgent(
            aid=AID("capture@localhost:5000"),
            dataset=self.dataset,
            next_agent_aid="selection@localhost:5002",
            selection_agent_aid="selection@localhost:5002",
            animal_tags=["passage_1"],
            low_fps=5.0,
            visual_gated=True,
            frame_store=self.frame_store,
        )
        behaviour = DatasetCaptureBehaviour(
            agent=capture_agent,
            dataset=self.dataset,
            next_agent_aid="selection@localhost:5002",
            selection_agent_aid="selection@localhost:5002",
            animal_tags=["passage_1"],
            low_fps=5.0,
            visual_gated=True,
            frame_store=self.frame_store,
            call_later=self.clock.call_later,
            monotonic=self.clock.monotonic,
        )
        def failing_send(evt):
            raise RuntimeError("Synthetic transmission failure")
        behaviour._send_pipeline_event = failing_send

        behaviour.start()
        self.clock.advance(0.0)
        f1_id = list(behaviour._pending_low_frames.keys())[0]

        trigger_obs = VisualStateEvent(
            passage_id="passage_1",
            capture_index=1,
            elapsed_time=0.0,
            dataset_timestamp_ms=0.0,
            mad=0.8,
            moving=True,
            visual_state=VisualState.ACTIVE,
            transition="IDLE->ACTIVE",
            processing_time_ms=1.0,
            frame_id=f1_id,
            is_trigger=True,
        )
        behaviour.handle_visual_state(trigger_obs)

        # Confirmado: frame_id foi descartado defensivamente do FrameStore
        self.assertIsNone(self.frame_store.get(f1_id))

    def test_low_frame_suited_pending_trigger_forwarded_once_to_selection(self):
        """12. LOW frame suited pending vira trigger e o MESMO frame_id chega UMA vez ao Selection."""
        pipeline_sent = []
        capture_agent = DatasetCaptureAgent(
            aid=AID("capture@localhost:5000"),
            dataset=self.dataset,
            next_agent_aid="selection@localhost:5002",
            selection_agent_aid="selection@localhost:5002",
            animal_tags=["passage_1"],
            low_fps=5.0,
            visual_gated=True,
            frame_store=self.frame_store,
        )
        behaviour = DatasetCaptureBehaviour(
            agent=capture_agent,
            dataset=self.dataset,
            next_agent_aid="selection@localhost:5002",
            selection_agent_aid="selection@localhost:5002",
            animal_tags=["passage_1"],
            low_fps=5.0,
            visual_gated=True,
            frame_store=self.frame_store,
            call_later=self.clock.call_later,
            monotonic=self.clock.monotonic,
        )
        behaviour._send_pipeline_event = lambda evt: pipeline_sent.append(evt)
        behaviour._send_visual_event = lambda evt: None

        behaviour.start()
        self.clock.advance(0.0)  # Frame 1 capturado em LOW
        self.assertEqual(len(behaviour._pending_low_frames), 1)
        f1_id = list(behaviour._pending_low_frames.keys())[0]

        # Visual produz IDLE -> ACTIVE (is_trigger=True)
        trigger_obs = VisualStateEvent(
            passage_id="passage_1",
            capture_index=1,
            elapsed_time=0.0,
            dataset_timestamp_ms=0.0,
            pdi_score=0.85,
            moving=True,
            visual_state=VisualState.ACTIVE,
            transition="IDLE->ACTIVE",
            processing_time_ms=1.0,
            frame_id=f1_id,
            is_trigger=True,
        )
        behaviour.handle_visual_state(trigger_obs)

        # Valida: exatamente 1 evento encaminhado ao Selection com o MESMO frame_id
        self.assertEqual(len(pipeline_sent), 1)
        self.assertIsInstance(pipeline_sent[0], FrameEvent)
        self.assertEqual(pipeline_sent[0].frame_id, f1_id)
        self.assertEqual(behaviour._current_rate, "LOW")
        self.assertEqual(len(behaviour._pending_low_frames), 0)

    def test_stale_capture_control_cannot_overwrite_newer_rate(self):
        capture_agent = DatasetCaptureAgent(
            aid=AID("capture@localhost:5000"), dataset=self.dataset,
            next_agent_aid="selection@localhost:5002",
            selection_agent_aid="selection@localhost:5002",
            animal_tags=["passage_1"], low_fps=5.0, medium_fps=7.0,
            visual_gated=True, frame_store=self.frame_store,
        )
        behaviour = DatasetCaptureBehaviour(
            agent=capture_agent, dataset=self.dataset,
            next_agent_aid="selection@localhost:5002",
            selection_agent_aid="selection@localhost:5002",
            animal_tags=["passage_1"], low_fps=5.0, medium_fps=7.0,
            visual_gated=True, frame_store=self.frame_store,
            call_later=self.clock.call_later, monotonic=self.clock.monotonic,
        )
        behaviour._send_pipeline_event = lambda _evt: None
        behaviour._send_visual_event = lambda _evt: None
        behaviour.start()
        self.clock.advance(0.0)

        behaviour.handle_capture_control("HIGH", "passage_1", control_sequence=2)
        behaviour.handle_capture_control("LOW", "passage_1", control_sequence=1)
        self.assertEqual(behaviour._current_rate, "HIGH")

    def test_low_frame_idle_not_forwarded_and_framestore_cleaned(self):
        """13. LOW frame em IDLE não chega ao Selection e FrameStore é limpo."""
        pipeline_sent = []
        capture_agent = DatasetCaptureAgent(
            aid=AID("capture@localhost:5000"),
            dataset=self.dataset,
            next_agent_aid="selection@localhost:5002",
            selection_agent_aid="selection@localhost:5002",
            animal_tags=["passage_1"],
            low_fps=5.0,
            visual_gated=True,
            frame_store=self.frame_store,
        )
        behaviour = DatasetCaptureBehaviour(
            agent=capture_agent,
            dataset=self.dataset,
            next_agent_aid="selection@localhost:5002",
            selection_agent_aid="selection@localhost:5002",
            animal_tags=["passage_1"],
            low_fps=5.0,
            visual_gated=True,
            frame_store=self.frame_store,
            call_later=self.clock.call_later,
            monotonic=self.clock.monotonic,
        )
        behaviour._send_pipeline_event = lambda evt: pipeline_sent.append(evt)
        behaviour._send_visual_event = lambda evt: None

        behaviour.start()
        self.clock.advance(0.0)  # Frame 1 capturado em LOW
        f1_id = list(behaviour._pending_low_frames.keys())[0]

        # Frame permaneceu IDLE
        idle_obs = VisualStateEvent(
            passage_id="passage_1",
            capture_index=1,
            elapsed_time=0.0,
            dataset_timestamp_ms=0.0,
            pdi_score=0.02,
            moving=False,
            visual_state=VisualState.IDLE,
            transition="",
            processing_time_ms=1.0,
            frame_id=f1_id,
            is_trigger=False,
        )
        behaviour.handle_visual_state(idle_obs)

        # Valida: NENHUM frame enviado ao Selection e FrameStore descartou a entrada
        self.assertEqual(len(pipeline_sent), 0)
        self.assertIsNone(self.frame_store.get(f1_id))
        self.assertEqual(len(behaviour._pending_low_frames), 0)

    def test_last_low_frame_idle_defers_end_then_cleans_framestore(self):
        """O fim logico aguarda somente a decisao IDLE do ultimo frame LOW."""
        pipeline_sent = []
        capture_agent = DatasetCaptureAgent(
            aid=AID("capture@localhost:5000"),
            dataset=self.dataset,
            next_agent_aid="selection@localhost:5002",
            selection_agent_aid="selection@localhost:5002",
            animal_tags=["passage_1"],
            low_fps=5.0,
            visual_gated=True,
            frame_store=self.frame_store,
        )
        behaviour = DatasetCaptureBehaviour(
            agent=capture_agent,
            dataset=self.dataset,
            next_agent_aid="selection@localhost:5002",
            selection_agent_aid="selection@localhost:5002",
            animal_tags=["passage_1"],
            low_fps=5.0,
            visual_gated=True,
            frame_store=self.frame_store,
            visual_agent_aid="visual@localhost:5004",
            call_later=self.clock.call_later,
            monotonic=self.clock.monotonic,
        )
        behaviour._send_pipeline_event = lambda evt: pipeline_sent.append(evt)
        behaviour._send_visual_event = lambda evt: None

        behaviour.start()
        self.clock.advance(0.0)
        first_id = next(iter(behaviour._pending_low_frames))
        behaviour.handle_visual_state(VisualStateEvent(
            passage_id="passage_1", capture_index=1, elapsed_time=0.0,
            dataset_timestamp_ms=0.0, pdi_score=0.0, moving=False,
            visual_state=VisualState.IDLE, transition=None,
            processing_time_ms=1.0, frame_id=first_id, is_trigger=False,
        ))
        self.clock.advance(0.2)
        second_id = next(iter(behaviour._pending_low_frames))
        behaviour.handle_visual_state(VisualStateEvent(
            passage_id="passage_1", capture_index=2, elapsed_time=200.0,
            dataset_timestamp_ms=200.0, pdi_score=0.0, moving=False,
            visual_state=VisualState.IDLE, transition=None,
            processing_time_ms=1.0, frame_id=second_id, is_trigger=False,
        ))
        self.clock.advance(0.2)
        frame_id = next(iter(behaviour._pending_low_frames))
        self.clock.advance(1.5)

        self.assertEqual(behaviour._current_tag, "passage_1")
        self.assertTrue(behaviour._passage_finalization_pending)
        self.assertEqual(pipeline_sent, [])

        behaviour.handle_visual_state(VisualStateEvent(
            passage_id="passage_1", capture_index=1, elapsed_time=0.0,
            dataset_timestamp_ms=0.0, pdi_score=0.0, moving=False,
            visual_state=VisualState.IDLE, transition=None,
            processing_time_ms=1.0, frame_id=frame_id, is_trigger=False,
        ))

        self.clock.advance(0.0)
        self.assertEqual([type(evt) for evt in pipeline_sent], [EndPassageEvent, EndPipelineEvent])
        self.assertIsNone(self.frame_store.get(frame_id))
        self.assertEqual(len(self.frame_store), 0)

    def test_trigger_on_last_frame_is_forwarded_before_end_and_stale_duplicate_is_ignored(self):
        """O trigger tardio fecha a passagem em ordem e nao duplica na proxima."""
        pipeline_sent = []
        capture_agent = DatasetCaptureAgent(
            aid=AID("capture@localhost:5000"),
            dataset=self.dataset,
            next_agent_aid="selection@localhost:5002",
            selection_agent_aid="selection@localhost:5002",
            animal_tags=["passage_1", "passage_2"],
            low_fps=5.0,
            visual_gated=True,
            frame_store=self.frame_store,
        )
        behaviour = DatasetCaptureBehaviour(
            agent=capture_agent,
            dataset=self.dataset,
            next_agent_aid="selection@localhost:5002",
            selection_agent_aid="selection@localhost:5002",
            animal_tags=["passage_1", "passage_2"],
            low_fps=5.0,
            visual_gated=True,
            frame_store=self.frame_store,
            visual_agent_aid="visual@localhost:5004",
            call_later=self.clock.call_later,
            monotonic=self.clock.monotonic,
        )
        behaviour._send_pipeline_event = lambda evt: pipeline_sent.append(evt)
        behaviour._send_visual_event = lambda evt: None

        behaviour.start()
        self.clock.advance(0.0)  # Frame 1 da passage_1 capturado em LOW
        early_id = next(iter(behaviour._pending_low_frames))
        behaviour.handle_visual_state(VisualStateEvent(
            passage_id="passage_1", capture_index=1, elapsed_time=0.0,
            dataset_timestamp_ms=0.0, pdi_score=0.0, moving=False,
            visual_state=VisualState.IDLE, transition=None,
            processing_time_ms=1.0, frame_id=early_id, is_trigger=False,
        ))
        self.clock.advance(0.2)
        early_id = next(iter(behaviour._pending_low_frames))
        behaviour.handle_visual_state(VisualStateEvent(
            passage_id="passage_1", capture_index=2, elapsed_time=200.0,
            dataset_timestamp_ms=200.0, pdi_score=0.0, moving=False,
            visual_state=VisualState.IDLE, transition=None,
            processing_time_ms=1.0, frame_id=early_id, is_trigger=False,
        ))
        self.clock.advance(0.2)
        f1_id = next(iter(behaviour._pending_low_frames))

        # O fim temporal ocorre, mas a passagem ainda aguarda a resposta Visual.
        self.clock.advance(1.5)
        self.assertEqual(behaviour._current_tag, "passage_1")
        self.assertTrue(behaviour._passage_finalization_pending)

        trigger = VisualStateEvent(
            passage_id="passage_1",
            capture_index=3,
            elapsed_time=400.0,
            dataset_timestamp_ms=400.0,
            pdi_score=0.85,
            moving=True,
            visual_state=VisualState.ACTIVE,
            transition="IDLE->ACTIVE",
            processing_time_ms=1.0,
            frame_id=f1_id,
            is_trigger=True,
        )
        original_array = self.frame_store.get(f1_id)
        behaviour.handle_visual_state(trigger)

        # Mesmo frame, uma vez, seguido do EndPassage; nenhum downstream foi aguardado.
        forwarded = [evt for evt in pipeline_sent if isinstance(evt, FrameEvent)]
        self.assertEqual(len(forwarded), 1)
        self.assertEqual(forwarded[0].frame_id, f1_id)
        self.assertIs(self.frame_store.get(f1_id), original_array)
        first_end = next(i for i, evt in enumerate(pipeline_sent) if isinstance(evt, EndPassageEvent))
        self.assertLess(pipeline_sent.index(forwarded[0]), first_end)

        # A proxima passagem pode iniciar sem esperar Selection/Enhance/Prediction.
        self.clock.advance(0.0)
        self.assertEqual(behaviour._current_tag, "passage_2")

        # Duplicata tardia da passage_1 nao reencaminha nem altera a nova captura.
        behaviour.handle_visual_state(trigger)
        self.assertEqual(behaviour._current_rate, "LOW")
        self.assertEqual(len([evt for evt in pipeline_sent if isinstance(evt, FrameEvent)]), 1)


if __name__ == "__main__":
    unittest.main()
