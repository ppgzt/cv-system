#!/usr/bin/env python3
"""Testes unitários puros para SelectionHold e OrchestratorAgent."""

import unittest
import json
from domain.selection_hold import SelectionHold
from domain.visual_activity import VisualState
from domain.visual_events import VisualStateEvent
from domain.pipeline_events import SelectionEvidenceEvent
from mas.agents.orchestrator_agent import OrchestratorAgent
from pade.acl.aid import AID
from pade.acl.messages import ACLMessage


class TestSelectionHoldAndOrchestrator(unittest.TestCase):
    def test_acl_selection_evidence_accepts_canonical_event_type_field(self):
        orch = OrchestratorAgent(AID("o@localhost:1"), "c@localhost:2")
        orch.send = lambda _message: None
        orch.handle_passage_started("P")
        message = ACLMessage(ACLMessage.INFORM)
        message.set_sender(AID("selection@localhost:3"))
        message.set_ontology("selection-evidence")
        message.set_content(json.dumps({
            "event_type": "selection_evidence",
            "passage_id": "P",
            "capture_index": 1,
            "frame_id": "F",
            "stream_seq": 7,
            "accepted": True,
            "probability": 0.9,
        }))
        orch.react(message)
        self.assertTrue(orch.selection_hold.hold_active)

    def test_selection_hold_lifecycle(self):
        hold = SelectionHold(n_rejections_threshold=2)
        self.assertFalse(hold.hold_active)

        # 1. Rejeição com hold inativo não ativa hold
        self.assertFalse(hold.observe(accepted=False))

        # 2. Aceitação ativa hold
        self.assertTrue(hold.observe(accepted=True))
        self.assertEqual(hold.consecutive_rejections, 0)

        # 3. Primeira rejeição mantém hold ativo
        self.assertTrue(hold.observe(accepted=False))
        self.assertEqual(hold.consecutive_rejections, 1)

        # 4. Aceitação intermediária reseta o contador de rejeições
        self.assertTrue(hold.observe(accepted=True))
        self.assertEqual(hold.consecutive_rejections, 0)

        # 5. Primeira rejeição após reset mantém hold
        self.assertTrue(hold.observe(accepted=False))
        self.assertEqual(hold.consecutive_rejections, 1)

        # 6. Segunda rejeição consecutiva desativa o hold
        self.assertFalse(hold.observe(accepted=False))
        self.assertEqual(hold.consecutive_rejections, 2)

        # 7. Terceira rejeição mantém inativo
        self.assertFalse(hold.observe(accepted=False))

    def test_orchestrator_coordination_and_stale_event_protection(self):
        rate_changes = []

        def rate_callback(pid, rate):
            rate_changes.append((pid, rate))

        orch = OrchestratorAgent(aid=AID("orchestrator@localhost:5008"), capture_agent_aid="capture@localhost:5000")
        orch.on_rate_change = rate_callback

        # Inicia Passagem A
        orch.handle_passage_started("passage_A")
        self.assertEqual(orch.current_rate, "LOW")

        # 1. Visual ACTIVE na passagem A -> sobe para HIGH
        vis_active_a = VisualStateEvent(
            passage_id="passage_A",
            capture_index=1,
            elapsed_time=100.0,
            dataset_timestamp_ms=100.0,
            mad=0.5,
            moving=True,
            visual_state=VisualState.ACTIVE,
            transition="IDLE->ACTIVE",
            processing_time_ms=1.0,
            frame_id="F1",
            is_trigger=True,
        )
        orch.handle_visual_state(vis_active_a)
        self.assertEqual(orch.current_rate, "HIGH")
        self.assertEqual(rate_changes[-1], ("passage_A", "HIGH"))

        # 2. Selection aceita frame em passagem A -> ativa hold
        sel_acc_a = SelectionEvidenceEvent(
            passage_id="passage_A",
            capture_index=1,
            frame_id="F1",
            stream_seq=1,
            accepted=True,
            probability=0.9,
        )
        orch.handle_selection_evidence(sel_acc_a)
        self.assertTrue(orch.selection_hold.hold_active)

        # 3. Visual retorna a IDLE em passagem A -> hold ativo MANTÉM HIGH
        vis_idle_a = VisualStateEvent(
            passage_id="passage_A",
            capture_index=2,
            elapsed_time=200.0,
            dataset_timestamp_ms=200.0,
            mad=0.01,
            moving=False,
            visual_state=VisualState.IDLE,
            transition="ACTIVE->IDLE",
            processing_time_ms=1.0,
            frame_id="F2",
            is_trigger=False,
        )
        orch.handle_visual_state(vis_idle_a)
        self.assertEqual(orch.current_rate, "HIGH")  # Mantém HIGH!

        # 4. Selection rejeita 1x -> continua HIGH
        sel_rej_a1 = SelectionEvidenceEvent(
            passage_id="passage_A",
            capture_index=2,
            frame_id="F2",
            stream_seq=2,
            accepted=False,
            probability=0.2,
        )
        orch.handle_selection_evidence(sel_rej_a1)
        self.assertEqual(orch.current_rate, "HIGH")

        # 5. Selection rejeita 2x -> Hold expira -> downshift para LOW
        sel_rej_a2 = SelectionEvidenceEvent(
            passage_id="passage_A",
            capture_index=3,
            frame_id="F3",
            stream_seq=3,
            accepted=False,
            probability=0.1,
        )
        orch.handle_selection_evidence(sel_rej_a2)
        self.assertEqual(orch.current_rate, "LOW")
        self.assertEqual(rate_changes[-1], ("passage_A", "LOW"))

        # 6. PROTEÇÃO CONTRA EVENTOS STALE:
        # Inicia Passagem B
        orch.handle_passage_started("passage_B")
        self.assertEqual(orch.current_rate, "LOW")
        self.assertFalse(orch.selection_hold.hold_active)

        # Chega evidência ATRASADA da passagem A (accepted=True)
        stale_sel = SelectionEvidenceEvent(
            passage_id="passage_A",
            capture_index=10,
            frame_id="F10",
            stream_seq=10,
            accepted=True,
            probability=0.99,
        )
        accepted = orch.handle_selection_evidence(stale_sel)
        self.assertFalse(accepted)  # Rejeitado por stale!
        self.assertFalse(orch.selection_hold.hold_active)
        self.assertEqual(orch.current_rate, "LOW")  # Permanece LOW em B!

        # Chega evento Visual ATRASADO da passagem A (ACTIVE)
        stale_vis = VisualStateEvent(
            passage_id="passage_A",
            capture_index=11,
            elapsed_time=500.0,
            dataset_timestamp_ms=500.0,
            mad=0.8,
            moving=True,
            visual_state=VisualState.ACTIVE,
            transition="IDLE->ACTIVE",
            processing_time_ms=1.0,
        )
        vis_accepted = orch.handle_visual_state(stale_vis)
        self.assertFalse(vis_accepted)  # Rejeitado por stale!
        self.assertEqual(orch.current_rate, "LOW")  # Permanece LOW em B!


if __name__ == "__main__":
    unittest.main()
