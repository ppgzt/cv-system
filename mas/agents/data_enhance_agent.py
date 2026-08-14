"""Preprocessing PADE ordenado, equivalente ao consumidor unico de q2.

Eventos recebidos por ACL sao primeiro admitidos numa ``OrderedInbox``. O
estagio executa no maximo uma transformacao por vez e so entao libera o evento
seguinte. ``qsize()`` mede eventos pendentes; o frame retirado e atualmente em
processamento nao faz parte da ocupacao, como em ``queue.Queue``.
"""

from __future__ import annotations

import json
import queue
from typing import Callable

from twisted.internet.threads import deferToThread

from pade.acl.aid import AID
from pade.acl.messages import ACLMessage
from pade.core.agent import Agent
from pade.misc.utility import display_message

from domain.pipeline_events import (
    EndPassageEvent,
    EndPipelineEvent,
    FrameEvent,
    PipelineEvent,
    event_from_json,
    event_to_dict,
    event_to_json,
)
from mas.adapters.data_enhance_adapter import DataEnhanceAdapter
from mas.infrastructure.frame_store import FRAME_STORE, FrameStore
from mas.infrastructure.ordered_inbox import OrderedInbox, OrderedInboxClosed
from mas.infrastructure.stream_sequence import StreamSequencer


PIPELINE_EVENT_ONTOLOGY = "pipeline-event"
LEGACY_FRAME_ONTOLOGY = "frame-enhanced"
LEGACY_END_PASSAGE_ONTOLOGY = "batch-ready"


class DataEnhanceAgent(Agent):
    """Consome Selection -> Preprocessing e sequencia sua propria saida."""

    def __init__(
        self,
        aid,
        data_enhance_adapter: DataEnhanceAdapter,
        next_agent_aid: str,
        frame_store: FrameStore = FRAME_STORE,
        inbox: OrderedInbox[PipelineEvent] | None = None,
        output_sequencer: StreamSequencer | None = None,
        defer_executor: Callable = deferToThread,
        debug: bool = False,
    ):
        super().__init__(aid=aid, debug=debug)
        self.data_enhance_adapter = data_enhance_adapter
        self.next_agent_aid = next_agent_aid
        self.frame_store = frame_store
        self.inbox = inbox or OrderedInbox()
        self.output_sequencer = output_sequencer or StreamSequencer()
        self._defer_executor = defer_executor

        self._processing = False
        self._active_frame_seq: int | None = None
        self._enhanced_by_passage: dict[str, int] = {}

    def react(self, message):
        super().react(message)
        if message.performative != ACLMessage.INFORM:
            return
        if message.ontology != PIPELINE_EVENT_ONTOLOGY:
            return

        try:
            event = event_from_json(message.content)
            self.inbox.put(event)
        except (TypeError, ValueError, OrderedInboxClosed) as exc:
            display_message(
                self.aid.name,
                f"[ERROR] Invalid ordered pipeline event: {exc}",
            )
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

            if isinstance(event, FrameEvent):
                self._processing = True
                self._active_frame_seq = event.stream_seq
                self._schedule_enhance(event)
                return

            if isinstance(event, EndPassageEvent):
                self._handle_end_passage(event)
                continue

            if isinstance(event, EndPipelineEvent):
                self._emit_pipeline_event(
                    EndPipelineEvent(
                        stream_seq=self.output_sequencer.next_seq(),
                    )
                )
                self.inbox.close()
                return

            display_message(
                self.aid.name,
                f"[ERROR] Unsupported event: {type(event).__name__}",
            )

    def _schedule_enhance(self, event: FrameEvent) -> None:
        try:
            raw = self.frame_store.get(event.frame_id)
        except Exception as exc:
            self._enhance_failed(exc, event)
            return

        if raw is None:
            try:
                display_message(
                    self.aid.name,
                    f"[WARN] frame_id={event.frame_id} not in FrameStore",
                )
            except Exception as exc:
                self._report_callback_exception("Enhance callback", event, exc)
            finally:
                self._finish_current_frame(event)
            return

        try:
            deferred = self._defer_executor(self.data_enhance_adapter.run, raw)
            deferred.addCallbacks(
                self._enhance_succeeded,
                self._enhance_failed,
                callbackArgs=(event,),
                errbackArgs=(event,),
            )
        except Exception as exc:
            self._enhance_failed(exc, event)
            return

    def _enhance_succeeded(self, enhanced, event: FrameEvent):
        try:
            self.frame_store.put(event.frame_id, enhanced)
            self._enhanced_by_passage[event.passage_id] = (
                self._enhanced_by_passage.get(event.passage_id, 0) + 1
            )
            output_event = self._resequence_frame(event)
            self._emit_pipeline_event(output_event)
            self._send_legacy_frame(output_event)
            display_message(
                self.aid.name,
                f"frame_id={event.frame_id} enhanced and forwarded.",
            )
        except Exception as exc:
            self._report_callback_exception("Enhance callback", event, exc)
        finally:
            self._finish_current_frame(event)
        return enhanced

    def _enhance_failed(self, failure, event: FrameEvent):
        try:
            self.frame_store.discard(event.frame_id)
            error = (
                failure.getErrorMessage()
                if hasattr(failure, "getErrorMessage")
                else str(failure)
            )
            display_message(
                self.aid.name,
                f"[ERROR] Enhancement failed for {event.frame_id}: {error}",
            )
        except Exception as exc:
            self._report_callback_exception("Enhance errback", event, exc)
        finally:
            self._finish_current_frame(event)
        return None

    def _report_callback_exception(
        self,
        callback_name: str,
        event: FrameEvent,
        error: Exception,
    ) -> None:
        try:
            display_message(
                self.aid.name,
                f"[ERROR] {callback_name} failed for {event.frame_id}: {error}",
            )
        except Exception:
            # Logging nao pode reter o lifecycle do consumidor ordenado.
            pass

    def _finish_current_frame(self, event: FrameEvent) -> None:
        if self._active_frame_seq != event.stream_seq:
            return
        self._active_frame_seq = None
        self._processing = False
        self._drain_inbox()

    def _handle_end_passage(self, event: EndPassageEvent) -> None:
        output_event = EndPassageEvent(
            stream_seq=self.output_sequencer.next_seq(),
            passage_id=event.passage_id,
            total_captured_frames=event.total_captured_frames,
            first_capture_time=event.first_capture_time,
            last_capture_time=event.last_capture_time,
        )
        self._emit_pipeline_event(output_event)

        # Bridge temporaria: Prediction ainda usa batch-ready. A contagem e
        # apenas dos frames efetivamente emitidos antes deste END ordenado.
        suitable_count = self._enhanced_by_passage.pop(event.passage_id, 0)
        self._send_legacy_end(output_event, suitable_count)

    def _resequence_frame(self, event: FrameEvent) -> FrameEvent:
        return FrameEvent(
            stream_seq=self.output_sequencer.next_seq(),
            frame_id=event.frame_id,
            passage_id=event.passage_id,
            capture_index=event.capture_index,
            elapsed_time=event.elapsed_time,
            depth_filename=event.depth_filename,
            label=event.label,
            dataset_timestamp_ms=event.dataset_timestamp_ms,
        )

    def _emit_pipeline_event(self, event: PipelineEvent) -> None:
        message = ACLMessage(ACLMessage.INFORM)
        message.set_ontology(PIPELINE_EVENT_ONTOLOGY)
        message.add_receiver(AID(self.next_agent_aid))
        message.set_content(event_to_json(event))
        self.send(message)

    def _send_legacy_frame(self, event: FrameEvent) -> None:
        payload = event_to_dict(event)
        payload.update({
            "animal_id": event.passage_id,
            "frame_index": event.capture_index,
        })
        self._send_legacy_message(LEGACY_FRAME_ONTOLOGY, payload)

    def _send_legacy_end(
        self,
        event: EndPassageEvent,
        suitable_count: int,
    ) -> None:
        payload = {
            "animal_id": event.passage_id,
            "suitable_count": suitable_count,
            "total_frames": event.total_captured_frames,
            "capture_metrics": {
                "first_image_capture_time": event.first_capture_time,
                "last_image_capture_time": event.last_capture_time,
            },
        }
        self._send_legacy_message(LEGACY_END_PASSAGE_ONTOLOGY, payload)

    def _send_legacy_message(self, ontology: str, payload: dict) -> None:
        message = ACLMessage(ACLMessage.INFORM)
        message.set_ontology(ontology)
        message.add_receiver(AID(self.next_agent_aid))
        message.set_content(json.dumps(payload, ensure_ascii=True))
        self.send(message)

    def on_start(self):
        super().on_start()
        display_message(self.aid.name, "DataEnhanceAgent started.")
