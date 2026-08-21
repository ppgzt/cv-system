"""Plano de controle: Visual + Selection Hold + Resource-Aware."""

from __future__ import annotations

import json
import time
from typing import Callable

from domain.pipeline_events import SelectionEvidenceEvent, event_from_dict
from domain.resource_events import ResourceState, ResourceStateEvent
from domain.selection_hold import SelectionHold
from domain.visual_activity import VisualState
from domain.visual_events import VisualStateEvent
from mas.pade.misc.utility import display_message
from mas.experiment_config import RESOURCE_STALE_AFTER_SECONDS
from pade.acl.aid import AID
from pade.acl.messages import ACLMessage
from pade.core.agent import Agent

VISUAL_STATE_ONTOLOGY = "visual-state"
SELECTION_EVIDENCE_ONTOLOGY = "selection-evidence"
RESOURCE_STATE_ONTOLOGY = "resource-state"
CAPTURE_CONTROL_ONTOLOGY = "capture-control"
CAPTURE_PASSAGE_STARTED_ONTOLOGY = "capture-passage-started"
_CONTROL_ONTOLOGIES = frozenset({
    CAPTURE_PASSAGE_STARTED_ONTOLOGY,
    VISUAL_STATE_ONTOLOGY,
    SELECTION_EVIDENCE_ONTOLOGY,
    RESOURCE_STATE_ONTOLOGY,
})

_RATE_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
_CAP_FOR_RESOURCE = {ResourceState.SAFE: "HIGH", ResourceState.WARNING: "MEDIUM", ResourceState.CRITICAL: "LOW"}


class OrchestratorAgent(Agent):
    """Calcula desired_rate (Visual/Hold) e aplica o cap de recursos."""

    def __init__(
        self,
        aid: AID,
        capture_agent_aid: str,
        n_hold: int = 2,
        verbose: bool = False,
        resource_control_enabled: bool = False,
        resource_stale_after_seconds: float = RESOURCE_STALE_AFTER_SECONDS,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ):
        super().__init__(aid=aid)
        self.capture_agent_aid = capture_agent_aid
        self.selection_hold = SelectionHold(n_rejections_threshold=n_hold)
        self.verbose = verbose
        self.resource_control_enabled = resource_control_enabled
        self.resource_stale_after_ns = int(resource_stale_after_seconds * 1_000_000_000)
        if self.resource_stale_after_ns <= 0:
            raise ValueError("resource_stale_after_seconds must be positive")
        self._monotonic_ns = monotonic_ns
        self.active_capture_passage_id: str | None = None
        self.current_visual_state = VisualState.IDLE
        self.resource_state = ResourceState.SAFE
        # Antes da primeira amostra não existe cap válido a preservar. O
        # bootstrap, portanto, não é uma condição LOW especial: Visual pode
        # elevar a primeira passagem normalmente. Depois da primeira amostra,
        # stale volta a congelar conservadoramente o cap efetivo conhecido.
        self.resource_cap = "HIGH"
        self.desired_rate = "LOW"
        self.current_rate = "LOW"
        self._last_visual_capture_index = -1
        self._last_selection_stream_seq = -1
        self._last_resource_sequence = -1
        self._last_resource_observed_ns: int | None = None
        self._resource_stale = resource_control_enabled
        self._control_sequence = 0
        self.on_rate_change: Callable[[str, str], None] | None = None
        self.on_control_state_change: Callable[[dict], None] | None = None

    def on_start(self):
        super().on_start()
        msg = ACLMessage(ACLMessage.INFORM)
        msg.set_ontology("agent-ready")
        msg.add_receiver(AID(self.capture_agent_aid))
        msg.set_content(json.dumps({"agent": self.aid.name}))
        self.send(msg)

    def react(self, message: ACLMessage):
        super().react(message)
        # PADE entrega ao agente tambem o INFORM de sistema do AMS que atualiza
        # sua tabela de agentes. Esse payload e pickle (bytes, prefixo 0x80),
        # enquanto o plano de controle abaixo possui contrato JSON textual.
        # Nunca tente decodificar mensagens de sistema/nem ontologias alheias
        # como se fossem eventos de controle.
        if message.system_message or message.ontology not in _CONTROL_ONTOLOGIES:
            return
        if message.performative != ACLMessage.INFORM:
            return
        try:
            if not isinstance(message.content, str):
                raise TypeError(
                    "control ACL content must be JSON text, got "
                    f"{type(message.content).__name__}"
                )
            data = json.loads(message.content)
            if message.ontology == CAPTURE_PASSAGE_STARTED_ONTOLOGY:
                self.handle_passage_started(data["passage_id"])
            elif message.ontology == VISUAL_STATE_ONTOLOGY:
                self.handle_visual_state(VisualStateEvent(
                    passage_id=data["passage_id"], capture_index=data["capture_index"], elapsed_time=data["elapsed_time"],
                    dataset_timestamp_ms=data.get("dataset_timestamp_ms"), pdi_score=data.get("pdi_score", data.get("mad")),
                    moving=data.get("moving"), visual_state=VisualState(data["visual_state"]), transition=data.get("transition"),
                    processing_time_ms=data.get("processing_time_ms", 0.0), depth_filename=data.get("depth_filename"),
                    frame_id=data.get("frame_id"), is_trigger=data.get("is_trigger", False), is_invalid=data.get("is_invalid", False),
                    p99_mm=data.get("p99_mm"), fraction_ge_2500=data.get("fraction_ge_2500"), mad=data.get("mad"),
                ))
            elif message.ontology == SELECTION_EVIDENCE_ONTOLOGY:
                evidence = event_from_dict(data)
                if not isinstance(evidence, SelectionEvidenceEvent):
                    raise ValueError("selection-evidence payload has wrong event type")
                self.handle_selection_evidence(evidence)
            elif message.ontology == RESOURCE_STATE_ONTOLOGY:
                self.handle_resource_state(ResourceStateEvent(
                    sequence=data["sequence"], observed_at_monotonic_ns=data["observed_at_monotonic_ns"],
                    state=ResourceState(data["state"]), metrics=data.get("metrics", {}), reasons=tuple(data.get("reasons", ())),
                ))
        except Exception as exc:
            display_message(self.aid.name, f"[ERROR] control event: {exc}")

    def handle_passage_started(self, passage_id: str) -> None:
        self.active_capture_passage_id = passage_id
        self.selection_hold.reset()
        self.current_visual_state = VisualState.IDLE
        self.desired_rate = "LOW"
        self.current_rate = "LOW"
        self._last_visual_capture_index = -1
        self._last_selection_stream_seq = -1
        self._emit_control_state()

    def handle_visual_state(self, event: VisualStateEvent) -> bool:
        if event.passage_id != self.active_capture_passage_id or event.capture_index <= self._last_visual_capture_index:
            return False
        self._last_visual_capture_index = event.capture_index
        self.current_visual_state = event.visual_state
        self._evaluate_policy()
        return True

    def handle_selection_evidence(self, event: SelectionEvidenceEvent) -> bool:
        if event.passage_id != self.active_capture_passage_id or event.stream_seq <= self._last_selection_stream_seq:
            return False
        self._last_selection_stream_seq = event.stream_seq
        was_holding = self.selection_hold.hold_active
        self.selection_hold.observe(event.accepted)
        if self.verbose and was_holding != self.selection_hold.hold_active:
            detail = "ON" if self.selection_hold.hold_active else "OFF after consecutive rejections"
            display_message(self.aid.name, f"[Orchestrator] passage={event.passage_id} Selection Hold {detail}")
        self._evaluate_policy()
        return True

    def handle_resource_state(self, event: ResourceStateEvent) -> bool:
        if not self.resource_control_enabled:
            return False
        if event.sequence <= self._last_resource_sequence:
            return False
        self._last_resource_sequence = event.sequence
        self.resource_state = event.state
        self.resource_cap = _CAP_FOR_RESOURCE[event.state]
        self._last_resource_observed_ns = event.observed_at_monotonic_ns
        self._resource_stale = self._resource_is_stale()
        self._evaluate_policy()
        return True

    def _resource_is_stale(self) -> bool:
        if not self.resource_control_enabled:
            return False
        if self._last_resource_observed_ns is None:
            return True
        return self._monotonic_ns() - self._last_resource_observed_ns > self.resource_stale_after_ns

    def _evaluate_policy(self) -> None:
        if self.active_capture_passage_id is None:
            return
        if self.current_visual_state is VisualState.ACTIVE:
            self.desired_rate = "HIGH"
        elif not self.selection_hold.hold_active:
            self.desired_rate = "LOW"
        stale_now = self._resource_is_stale()
        # Uma amostra stale não pode causar novo upshift. Congelamos o teto no
        # estado efetivo vigente; amostra fresca retoma o cap normal sem spam.
        if stale_now:
            # Sem amostra anterior não há um teto de recurso válido; aplicar
            # LOW aqui tornaria o primeiro animal diferente de todos os outros.
            stale_cap = "HIGH" if self._last_resource_observed_ns is None else self.current_rate
            self._resource_stale = True
        else:
            stale_cap = self.resource_cap
            self._resource_stale = False
        target = min((self.desired_rate, stale_cap), key=_RATE_ORDER.__getitem__)
        if target != self.current_rate:
            previous = self.current_rate
            self.current_rate = target
            self._notify_rate_change(previous, target)
        self._emit_control_state()

    def _emit_control_state(self) -> None:
        if self.on_control_state_change is not None:
            self.on_control_state_change({
                "passage_id": self.active_capture_passage_id, "visual_state": self.current_visual_state.value,
                "hold_active": self.selection_hold.hold_active, "desired_capture_state": self.desired_rate,
                "resource_state": self.resource_state.value, "resource_cap": self.resource_cap,
                "resource_fresh": not self._resource_stale,
                "effective_capture_state": self.current_rate,
            })

    def _notify_rate_change(self, previous_rate: str, target_rate: str) -> None:
        if self.active_capture_passage_id is None:
            return
        if self.verbose:
            display_message(
                self.aid.name,
                f"[Orchestrator] passage={self.active_capture_passage_id} "
                f"{previous_rate}->{target_rate} visual={self.current_visual_state.value} "
                f"hold={self.selection_hold.hold_active} resource={self.resource_state.value}",
            )
        if self.on_rate_change is not None:
            self.on_rate_change(self.active_capture_passage_id, target_rate)
        self._control_sequence += 1
        msg = ACLMessage(ACLMessage.INFORM)
        msg.set_ontology(CAPTURE_CONTROL_ONTOLOGY)
        msg.add_receiver(AID(self.capture_agent_aid))
        msg.set_content(json.dumps({
            "passage_id": self.active_capture_passage_id, "target_rate": target_rate,
            "control_sequence": self._control_sequence,
            "reason": f"desired={self.desired_rate},resource={self.resource_state.value},cap={self.resource_cap}",
        }))
        self._safe_send(msg)

    def _safe_send(self, msg: ACLMessage) -> None:
        try:
            if hasattr(self, "agentInstance") and self.agentInstance is not None:
                self.send(msg)
        except Exception as exc:
            if self.verbose:
                display_message(self.aid.name, f"[WARN] send failed: {exc}")
