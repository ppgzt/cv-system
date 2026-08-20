"""Agente Orquestrador do Control Plane Mínimo.

Combina:
1. Estado do Visual Event Agent (IDLE / ACTIVE);
2. Evidencia do Frame Selection Agent com Selection Hold (N=2);
3. Protecao contra eventos atrasados/stale de passagens anteriores;
4. Emissao de comandos de taxa (LOW / HIGH) para o DatasetCaptureAgent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

from domain.pipeline_events import CaptureControlEvent, SelectionEvidenceEvent
from domain.selection_hold import SelectionHold
from domain.visual_activity import VisualState
from domain.visual_events import VisualStateEvent
from mas.pade.misc.utility import display_message
from pade.core.agent import Agent
from pade.acl.messages import ACLMessage
from pade.acl.aid import AID

VISUAL_STATE_ONTOLOGY = "visual-state"
SELECTION_EVIDENCE_ONTOLOGY = "selection-evidence"
CAPTURE_CONTROL_ONTOLOGY = "capture-control"
CAPTURE_PASSAGE_STARTED_ONTOLOGY = "capture-passage-started"


class OrchestratorAgent(Agent):
    """Agente de controle reativo para coordenacao de taxa e retencao."""

    def __init__(
        self,
        aid: AID,
        capture_agent_aid: str,
        n_hold: int = 2,
        verbose: bool = False,
    ):
        super().__init__(aid=aid)
        self.capture_agent_aid = capture_agent_aid
        self.selection_hold = SelectionHold(n_rejections_threshold=n_hold)
        self.verbose = verbose

        self.active_capture_passage_id: str | None = None
        self.current_visual_state = VisualState.IDLE
        self.current_rate = "LOW"

        # Callbacks opcionais para desacoplamento e testes
        self.on_rate_change: Callable[[str, str], None] | None = None

    def on_start(self):
        super().on_start()
        # Notifica que o Orchestrator está pronto
        ready_msg = ACLMessage(ACLMessage.INFORM)
        ready_msg.set_ontology("agent-ready")
        ready_msg.add_receiver(AID(self.capture_agent_aid))
        ready_msg.set_content(json.dumps({"agent": self.aid.name}))
        self.send(ready_msg)
        if self.verbose:
            display_message(self.aid.name, "OrchestratorAgent pronto.")

    def react(self, message: ACLMessage):
        super().react(message)
        if message.performative != ACLMessage.INFORM:
            return

        ontology = message.ontology
        content = message.content

        if ontology == CAPTURE_PASSAGE_STARTED_ONTOLOGY:
            try:
                data = json.loads(content)
                passage_id = data["passage_id"]
                self.handle_passage_started(passage_id)
            except Exception as exc:
                display_message(
                    self.aid.name,
                    f"[ERROR] Failed to handle passage start: {exc}",
                )

        elif ontology == VISUAL_STATE_ONTOLOGY:
            try:
                data = json.loads(content)
                # Reconstruir evento
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
                self.handle_visual_state(event)
            except Exception as exc:
                display_message(
                    self.aid.name,
                    f"[ERROR] Failed to handle visual state event: {exc}",
                )

        elif ontology == SELECTION_EVIDENCE_ONTOLOGY:
            try:
                data = json.loads(content)
                event = SelectionEvidenceEvent(
                    passage_id=data["passage_id"],
                    capture_index=data["capture_index"],
                    frame_id=data["frame_id"],
                    stream_seq=data["stream_seq"],
                    accepted=data["accepted"],
                    probability=data["probability"],
                )
                self.handle_selection_evidence(event)
            except Exception as exc:
                display_message(
                    self.aid.name,
                    f"[ERROR] Failed to handle selection evidence: {exc}",
                )

    def handle_passage_started(self, passage_id: str) -> None:
        """Inicia o controle para uma nova passagem capturada."""
        self.active_capture_passage_id = passage_id
        self.selection_hold.reset()
        self.current_visual_state = VisualState.IDLE
        self.current_rate = "LOW"
        if self.verbose:
            display_message(
                self.aid.name,
                f"[ORCHESTRATOR] Início de controle da passagem={passage_id} (Taxa=LOW).",
            )

    def handle_visual_state(self, event: VisualStateEvent) -> bool:
        """Processa evento visual se pertencer a passagem ativa. Retorna se foi aceito."""
        if (
            self.active_capture_passage_id is None
            or event.passage_id != self.active_capture_passage_id
        ):
            if self.verbose:
                display_message(
                    self.aid.name,
                    f"[DEBUG] Descartando VisualStateEvent stale da passagem={event.passage_id} "
                    f"(ativa={self.active_capture_passage_id}).",
                )
            return False

        self.current_visual_state = event.visual_state
        self._evaluate_policy()
        return True

    def handle_selection_evidence(self, event: SelectionEvidenceEvent) -> bool:
        """Processa evidencia do Selection se pertencer a passagem ativa. Retorna se foi aceito."""
        if (
            self.active_capture_passage_id is None
            or event.passage_id != self.active_capture_passage_id
        ):
            if self.verbose:
                display_message(
                    self.aid.name,
                    f"[DEBUG] Descartando SelectionEvidenceEvent stale da passagem={event.passage_id} "
                    f"(ativa={self.active_capture_passage_id}).",
                )
            return False

        self.selection_hold.observe(event.accepted)
        self._evaluate_policy()
        return True

    def _evaluate_policy(self) -> None:
        """Avalia a politica de coordenacao reativa para a passagem ativa."""
        if self.active_capture_passage_id is None:
            return

        if self.current_visual_state == VisualState.ACTIVE:
            target_rate = "HIGH"
        else:
            # Visual IDLE: mantem HIGH somente se Selection Hold estiver ativo e ja estavamos em HIGH
            if self.current_rate == "HIGH" and self.selection_hold.hold_active:
                target_rate = "HIGH"
            else:
                target_rate = "LOW"
                if self.current_rate == "HIGH":
                    # Ao efetivar o downshift, o hold é resetado
                    self.selection_hold.reset()

        if target_rate != self.current_rate:
            previous_rate = self.current_rate
            self.current_rate = target_rate
            self._notify_rate_change(previous_rate, target_rate)

    def _notify_rate_change(self, previous_rate: str, target_rate: str) -> None:
        if self.active_capture_passage_id is None:
            return

        if self.verbose:
            display_message(
                self.aid.name,
                f"[ORCHESTRATOR] Transicao de taxa na passagem={self.active_capture_passage_id}: "
                f"{previous_rate}->{target_rate} (Visual={self.current_visual_state.value}, "
                f"Hold={self.selection_hold.hold_active})",
            )

        if self.on_rate_change is not None:
            self.on_rate_change(self.active_capture_passage_id, target_rate)

        # Envia comando ao DatasetCaptureAgent
        msg = ACLMessage(ACLMessage.INFORM)
        msg.set_ontology(CAPTURE_CONTROL_ONTOLOGY)
        msg.add_receiver(AID(self.capture_agent_aid))
        msg.set_content(
            json.dumps({
                "passage_id": self.active_capture_passage_id,
                "target_rate": target_rate,
                "reason": f"Visual={self.current_visual_state.value},Hold={self.selection_hold.hold_active}",
            })
        )
        self._safe_send(msg)

    def _safe_send(self, msg: ACLMessage) -> None:
        try:
            if hasattr(self, "agentInstance") and self.agentInstance is not None:
                self.send(msg)
        except Exception as exc:
            if self.verbose:
                display_message(self.aid.name, f"[WARN] send failed: {exc}")

