"""Agente observacional de atividade visual no ramo Capture -> Visual."""

from __future__ import annotations

import csv
import json
import queue
import time
from pathlib import Path
from typing import Callable

from twisted.internet.threads import deferToThread

from pade.acl.aid import AID
from pade.acl.messages import ACLMessage
from pade.core.agent import Agent
from pade.misc.utility import display_message

from domain.pipeline_events import EndPassageEvent, EndPipelineEvent
from domain.visual_activity import (
    DEFAULT_IDLE_PATIENCE,
    DEFAULT_PDI_THRESHOLD,
    DEFAULT_PIXEL_THRESHOLD_MM,
    VisualActivityDetector,
    readonly_view,
)
from domain.visual_events import (
    VisualFrameEvent,
    VisualInputEvent,
    VisualStateEvent,
    visual_event_from_json,
)
from mas.infrastructure.frame_store import FRAME_STORE, FrameStore
from mas.infrastructure.ordered_inbox import OrderedInbox, OrderedInboxClosed


VISUAL_EVENT_ONTOLOGY = "visual-event"
VISUAL_STATE_ONTOLOGY = "visual-state"
VISUAL_LEASE_OWNER = "visual"


VISUAL_ACTIVITY_HEADER = (
    "passage_id",
    "capture_index",
    "elapsed_time",
    "dataset_timestamp_ms",
    "depth_filename",
    "label",
    "pdi_score",
    "moving",
    "visual_state",
    "transition",
    "is_invalid",
    "p99_mm",
    "fraction_ge_2500",
    "processing_time_ms",
    "frame_id",
    "is_trigger",
    "mad",
)



class VisualEventAgent(Agent):
    """Le leases RAW em ordem e produz estado IDLE/ACTIVE sem controlar FPS."""

    def __init__(
        self,
        aid,
        capture_agent_aid: str,
        pid: str,
        pdi_threshold: float = DEFAULT_PDI_THRESHOLD,
        idle_patience_frames: int = DEFAULT_IDLE_PATIENCE,
        pixel_threshold_mm: float = DEFAULT_PIXEL_THRESHOLD_MM,
        frame_store: FrameStore = FRAME_STORE,
        orchestrator_agent_aid: str | None = None,
        inbox: OrderedInbox[VisualInputEvent] | None = None,
        detector: VisualActivityDetector | None = None,
        state_publisher: Callable[[VisualStateEvent], None] | None = None,
        defer_executor: Callable = deferToThread,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        reports_dir: str = "infra/reports",
        debug: bool = False,
    ):
        super().__init__(aid=aid, debug=debug)
        self.capture_agent_aid = capture_agent_aid
        self.orchestrator_agent_aid = orchestrator_agent_aid
        self.pid = pid
        self.frame_store = frame_store
        self.inbox = inbox or OrderedInbox()
        self.detector = detector or VisualActivityDetector(
            pdi_threshold=pdi_threshold,
            idle_patience_frames=idle_patience_frames,
            pixel_threshold_mm=pixel_threshold_mm,
        )
        self.state_publisher = state_publisher
        self._defer_executor = defer_executor
        self._monotonic_ns = monotonic_ns
        self._output_path = Path(reports_dir) / pid / "visual_activity.csv"

        self.observations: list[VisualStateEvent] = []
        self._observation_records: list[dict] = []
        self.passage_final_states: dict[str, str] = {}
        self.latest_state_event: VisualStateEvent | None = None
        self._processing = False
        self._active_frame_seq: int | None = None
        self._persisted = False

    def on_start(self):
        super().on_start()
        message = ACLMessage(ACLMessage.INFORM)
        message.set_ontology("agent-ready")
        message.add_receiver(AID(self.capture_agent_aid))
        message.set_content(json.dumps({"agent": self.aid.name}))
        self.send(message)
        self._safe_log("VisualEventAgent ready (observational only).")

    def react(self, message):
        super().react(message)
        if message.performative != ACLMessage.INFORM:
            return
        if message.ontology != VISUAL_EVENT_ONTOLOGY:
            return
        try:
            event = visual_event_from_json(message.content)
            self.inbox.put(event)
        except (TypeError, ValueError, OrderedInboxClosed) as exc:
            self._safe_log(f"[ERROR] Invalid visual event: {exc}")
            return
        self._drain_inbox()

    def _drain_inbox(self) -> None:
        if self._processing:
            return
        while True:
            try:
                event = self.inbox.get(block=False)
            except (queue.Empty, OrderedInboxClosed):
                return

            if isinstance(event, VisualFrameEvent):
                self._processing = True
                self._active_frame_seq = event.stream_seq
                self._schedule_observation(event)
                return
            if isinstance(event, EndPassageEvent):
                self._handle_end_passage(event)
                continue
            if isinstance(event, EndPipelineEvent):
                self._handle_end_pipeline()
                return
            self._safe_log(
                f"[ERROR] Unsupported visual event: {type(event).__name__}"
            )

    def _schedule_observation(self, event: VisualFrameEvent) -> None:
        try:
            raw = self.frame_store.read_lease(
                event.lease_id,
                owner=VISUAL_LEASE_OWNER,
            )
            raw_readonly = readonly_view(raw)
            deferred = self._defer_executor(
                self._observe_frame,
                event,
                raw_readonly,
            )
            deferred.addCallbacks(
                self._observation_succeeded,
                self._observation_failed,
                callbackArgs=(event,),
                errbackArgs=(event,),
            )
        except Exception as exc:
            self._observation_failed(exc, event)

    def _observe_frame(
        self,
        event: VisualFrameEvent,
        raw_readonly,
    ) -> VisualStateEvent:
        started_ns = self._monotonic_ns()
        result = self.detector.observe(raw_readonly)
        elapsed_ms = (self._monotonic_ns() - started_ns) / 1_000_000.0
        is_trigger = (result.transition == "IDLE->ACTIVE")
        return VisualStateEvent(
            passage_id=event.passage_id,
            capture_index=event.capture_index,
            elapsed_time=event.elapsed_time,
            dataset_timestamp_ms=event.dataset_timestamp_ms,
            pdi_score=result.score,
            moving=result.moving,
            visual_state=result.visual_state,
            transition=result.transition,
            processing_time_ms=elapsed_ms,
            depth_filename=event.depth_filename,
            frame_id=event.frame_id,
            is_trigger=is_trigger,
            is_invalid=result.is_invalid,
            p99_mm=result.p99_mm,
            fraction_ge_2500=result.fraction_ge_2500,
            mad=result.mad,
        )

    def _observation_succeeded(
        self,
        observation: VisualStateEvent,
        event: VisualFrameEvent,
    ):
        try:
            self.observations.append(observation)
            self._observation_records.append(
                {
                    "passage_id": observation.passage_id,
                    "capture_index": observation.capture_index,
                    "elapsed_time": observation.elapsed_time,
                    "dataset_timestamp_ms": observation.dataset_timestamp_ms,
                    "depth_filename": observation.depth_filename,
                    "frame_id": observation.frame_id,
                    "is_trigger": observation.is_trigger,
                    # Ground truth e persistido apenas para analise; ele nao
                    # faz parte do VisualStateEvent publicado online.
                    "label": event.label,
                    "pdi_score": observation.pdi_score,
                    "moving": observation.moving,
                    "visual_state": observation.visual_state.value,
                    "transition": observation.transition,
                    "is_invalid": observation.is_invalid,
                    "p99_mm": observation.p99_mm,
                    "fraction_ge_2500": observation.fraction_ge_2500,
                    "processing_time_ms": observation.processing_time_ms,
                    "mad": observation.mad,
                }
            )
            self.latest_state_event = observation
            if self.state_publisher is not None:
                try:
                    self.state_publisher(observation)
                except Exception as exc:
                    self._safe_log(f"[ERROR] state_publisher failed: {exc}")

            if self.orchestrator_agent_aid is not None:
                try:
                    msg = ACLMessage(ACLMessage.INFORM)
                    msg.set_ontology(VISUAL_STATE_ONTOLOGY)
                    msg.add_receiver(AID(self.orchestrator_agent_aid))
                    msg.set_content(
                        json.dumps({
                            "passage_id": observation.passage_id,
                            "capture_index": observation.capture_index,
                            "elapsed_time": observation.elapsed_time,
                            "dataset_timestamp_ms": observation.dataset_timestamp_ms,
                            "pdi_score": observation.pdi_score,
                            "moving": observation.moving,
                            "visual_state": observation.visual_state.value,
                            "transition": observation.transition,
                            "processing_time_ms": observation.processing_time_ms,
                            "depth_filename": observation.depth_filename,
                            "frame_id": observation.frame_id,
                            "is_trigger": observation.is_trigger,
                            "is_invalid": observation.is_invalid,
                            "p99_mm": observation.p99_mm,
                            "fraction_ge_2500": observation.fraction_ge_2500,
                            "mad": observation.mad,
                        })
                    )
                    self._safe_send(msg)
                except Exception as exc:
                    self._safe_log(f"[ERROR] Failed to send visual state to orchestrator: {exc}")

            if observation.transition:
                score_str = f"{observation.pdi_score:.4f}" if observation.pdi_score is not None else "None"
                self._safe_log(
                    f"[VISUAL] passage={observation.passage_id} "
                    f"capture={observation.capture_index} "
                    f"pdi={score_str} "
                    f"transition={observation.transition}"
                )
        except Exception as exc:
            self._safe_log(
                f"[ERROR] Visual result handling failed for "
                f"lease={event.lease_id}: {exc}"
            )
        finally:
            self._release_and_finish(event)
        return observation

    def _safe_send(self, msg: ACLMessage) -> None:
        try:
            if hasattr(self, "agentInstance") and self.agentInstance is not None:
                self.send(msg)
        except Exception as exc:
            self._safe_log(f"[WARN] send failed: {exc}")


    def _observation_failed(self, failure, event: VisualFrameEvent):
        try:
            error = (
                failure.getErrorMessage()
                if hasattr(failure, "getErrorMessage")
                else str(failure)
            )
            self._safe_log(
                f"[ERROR] Visual observation failed for "
                f"lease={event.lease_id}: {error}"
            )
        finally:
            self._release_and_finish(event)
        return None

    def _release_and_finish(self, event: VisualFrameEvent) -> None:
        try:
            self.frame_store.release_lease(
                event.lease_id,
                owner=VISUAL_LEASE_OWNER,
            )
        except Exception as exc:
            self._safe_log(
                f"[ERROR] Visual lease release failed for "
                f"{event.lease_id}: {exc}"
            )
        finally:
            self._finish_current_frame(event)

    def _finish_current_frame(self, event: VisualFrameEvent) -> None:
        if self._active_frame_seq != event.stream_seq:
            return
        self._active_frame_seq = None
        self._processing = False
        self._drain_inbox()

    def _handle_end_passage(self, event: EndPassageEvent) -> None:
        try:
            final_state = self.detector.reset()
            self.passage_final_states[event.passage_id] = final_state.value
        except Exception as exc:
            self._safe_log(f"[ERROR] Visual passage reset failed: {exc}")
        try:
            released = self.frame_store.release_leases(
                owner=VISUAL_LEASE_OWNER,
                passage_id=event.passage_id,
            )
            if released:
                self._safe_log(
                    f"[WARN] Released {released} residual visual lease(s) for "
                    f"passage={event.passage_id}."
                )
        except Exception as exc:
            self._safe_log(f"[ERROR] Visual passage cleanup failed: {exc}")

    def _handle_end_pipeline(self) -> None:
        try:
            self.detector.reset()
        except Exception as exc:
            self._safe_log(f"[ERROR] Visual pipeline reset failed: {exc}")
        try:
            released = self.frame_store.release_leases(owner=VISUAL_LEASE_OWNER)
            if released:
                self._safe_log(
                    f"[WARN] Released {released} residual visual lease(s) at END."
                )
        except Exception as exc:
            self._safe_log(f"[ERROR] Visual pipeline cleanup failed: {exc}")
        try:
            self._persist_observations()
        finally:
            self.inbox.close()

    def stop_visual_monitoring(self) -> None:
        """Hook idempotente de cleanup; nunca encerra o reactor."""
        self.detector.reset()
        self.frame_store.release_leases(owner=VISUAL_LEASE_OWNER)
        self._persist_observations()

    def _persist_observations(self) -> None:
        if self._persisted:
            return
        try:
            self._output_path.parent.mkdir(parents=True, exist_ok=True)
            with self._output_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=VISUAL_ACTIVITY_HEADER)
                writer.writeheader()
                writer.writerows(self._observation_records)
            self._persisted = True
        except Exception as exc:
            self._safe_log(f"[ERROR] Visual CSV persistence failed: {exc}")

    def _safe_log(self, text: str) -> None:
        try:
            display_message(self.aid.name, text)
        except Exception:
            pass
