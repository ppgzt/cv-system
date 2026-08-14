"""Selection PADE ordenado, semanticamente equivalente ao consumidor de q1.

Mensagens ACL apenas admitem contratos na ``OrderedInbox``. Um unico fluxo
logico retira eventos em ``stream_seq`` e mantem no maximo uma inferencia de
Selection em andamento. Rejeicoes e erros descartam o frame, mas controles de
fim continuam atravessando a aresta.
"""

from __future__ import annotations

import json
import queue
import time
from typing import Callable

from twisted.internet.defer import DeferredSemaphore
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
    event_to_json,
)
from infra.profiling.telemetry import CaptureTimingRecorder, TelemetryContext
from mas.adapters.frame_selection_adapter import FrameSelectionAdapter
from mas.infrastructure.frame_store import FRAME_STORE, FrameStore
from mas.infrastructure.ordered_inbox import OrderedInbox, OrderedInboxClosed
from mas.infrastructure.stream_sequence import StreamSequencer


PIPELINE_EVENT_ONTOLOGY = "pipeline-event"


class FrameSelectionAgent(Agent):
    """Consome Capture -> Selection em ordem e sequencia sua propria saida."""

    def __init__(
        self,
        aid,
        frame_selection_adapter: FrameSelectionAdapter,
        next_agent_aid: str,
        predict_agent_aid: str | None = None,
        capture_agent_aid: str | None = None,
        frame_store: FrameStore = FRAME_STORE,
        inbox: OrderedInbox[PipelineEvent] | None = None,
        output_sequencer: StreamSequencer | None = None,
        telemetry_context: TelemetryContext | None = None,
        capture_timing_recorder: CaptureTimingRecorder | None = None,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        defer_executor: Callable = deferToThread,
        debug: bool = False,
        verbose: bool = False,
    ):
        super().__init__(aid=aid, debug=debug)
        self.frame_selection_adapter = frame_selection_adapter
        self.next_agent_aid = next_agent_aid
        # Compatibilidade de construcao do launcher historico. Selection nao
        # possui mais aresta nem protocolo de finalizacao direto com Prediction.
        self.predict_agent_aid = predict_agent_aid
        self.capture_agent_aid = capture_agent_aid
        self.frame_store = frame_store
        self.inbox = inbox or OrderedInbox()
        self.output_sequencer = output_sequencer or StreamSequencer()
        self.telemetry_context = telemetry_context
        self.capture_timing_recorder = capture_timing_recorder
        self._monotonic_ns = monotonic_ns
        self._defer_executor = defer_executor
        self.verbose = verbose

        self.discarded = 0
        self.forwarded = 0
        self.suitable_frames: dict[str, int] = {}
        self.confusion: dict[str, dict[tuple[str, bool], int]] = {}

        self._processing = False
        self._active_frame_seq: int | None = None
        self._inference_semaphore = DeferredSemaphore(1)

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

        # Admissao real na primeira borda: o timestamp é obtido imediatamente
        # depois do put, nunca no Agent.send(). Falha do relógio é observacional.
        try:
            actual_admission_ns = self._monotonic_ns()
        except Exception as exc:
            actual_admission_ns = None
            display_message(
                self.aid.name,
                f"[ERROR] Capture admission clock failed: {exc}",
            )

        try:
            if (
                isinstance(event, FrameEvent)
                and self.capture_timing_recorder is not None
                and actual_admission_ns is not None
            ):
                self.capture_timing_recorder.record_admission(
                    event.frame_id,
                    actual_admission_ns,
                )
            elif (
                isinstance(event, EndPassageEvent)
                and self.telemetry_context is not None
            ):
                # Limpeza condicional: N+1 pode ter iniciado antes da chegada
                # física deste END, sem apagar o contexto mais novo.
                self.telemetry_context.clear_capture_passage_id(
                    event.passage_id
                )
        except Exception as exc:
            # Telemetria nunca bloqueia o consumidor de domínio.
            display_message(
                self.aid.name,
                f"[ERROR] Capture telemetry admission failed: {exc}",
            )

        self._drain_inbox()

    def _drain_inbox(self) -> None:
        """Inicia somente o proximo evento logico disponivel."""
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
                self._schedule_evaluation(event)
                return

            if isinstance(event, EndPassageEvent):
                self._handle_end_passage(event)
                continue

            if isinstance(event, EndPipelineEvent):
                self._emit_event(
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

    def _schedule_evaluation(self, event: FrameEvent) -> None:
        try:
            img = self.frame_store.get(event.frame_id)
        except Exception as exc:
            self._selection_failed(exc, event)
            return

        if img is None:
            self._selection_succeeded((False, 0.0), event)
            return

        try:
            deferred = self._inference_semaphore.run(
                self._defer_executor,
                self.frame_selection_adapter.evaluate_with_score,
                event.elapsed_time,
                img,
            )
            deferred.addCallbacks(
                self._selection_succeeded,
                self._selection_failed,
                callbackArgs=(event,),
                errbackArgs=(event,),
            )
        except Exception as exc:
            self._selection_failed(exc, event)
            return

    def _selection_succeeded(self, result, event: FrameEvent):
        try:
            try:
                suitable, probability = result
            except Exception as exc:
                self._handle_selection_failure(exc, event)
            else:
                self._complete_selection(
                    event,
                    suitable=bool(suitable),
                    probability=float(probability),
                )
        except Exception as exc:
            self._report_callback_exception("Selection callback", event, exc)
        finally:
            self._finish_current_frame(event)
        return result

    def _complete_selection(
        self,
        event: FrameEvent,
        *,
        suitable: bool,
        probability: float,
    ) -> None:
        self._record_selection(event, suitable, probability)

        if self.verbose and event.label is not None:
            key = (event.label, suitable)
            passage_confusion = self.confusion.setdefault(event.passage_id, {})
            passage_confusion[key] = passage_confusion.get(key, 0) + 1

        if suitable:
            self.forwarded += 1
            self.suitable_frames[event.passage_id] = (
                self.suitable_frames.get(event.passage_id, 0) + 1
            )
            try:
                self._emit_event(self._resequence_frame(event))
            except Exception:
                # O downstream nao recebeu o evento; nao deixe raw orfao.
                self.frame_store.discard(event.frame_id)
                raise
            action = "SUITABLE"
        else:
            self.discarded += 1
            self.frame_store.discard(event.frame_id)
            action = "DISCARDED"

        display_message(
            self.aid.name,
            f"frame_id={event.frame_id} {action} (p={probability:.4f}). "
            f"Discarded={self.discarded}, Forwarded={self.forwarded}",
        )

    def _selection_failed(self, failure, event: FrameEvent):
        try:
            self._handle_selection_failure(failure, event)
        except Exception as exc:
            self._report_callback_exception("Selection errback", event, exc)
        finally:
            self._finish_current_frame(event)
        return None

    def _handle_selection_failure(self, failure, event: FrameEvent) -> None:
        self.discarded += 1
        self.frame_store.discard(event.frame_id)
        error = (
            failure.getErrorMessage()
            if hasattr(failure, "getErrorMessage")
            else str(failure)
        )
        display_message(
            self.aid.name,
            f"[ERROR] Selection failed for {event.frame_id}: {error}",
        )

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
        if self.verbose:
            self._log_confusion(event.passage_id, event.total_captured_frames)

        self._emit_event(
            EndPassageEvent(
                stream_seq=self.output_sequencer.next_seq(),
                passage_id=event.passage_id,
                total_captured_frames=event.total_captured_frames,
                first_capture_time=event.first_capture_time,
                last_capture_time=event.last_capture_time,
            )
        )
        self.suitable_frames.pop(event.passage_id, None)
        self.confusion.pop(event.passage_id, None)

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

    def _emit_event(self, event: PipelineEvent) -> None:
        message = ACLMessage(ACLMessage.INFORM)
        message.set_ontology(PIPELINE_EVENT_ONTOLOGY)
        message.add_receiver(AID(self.next_agent_aid))
        message.set_content(event_to_json(event))
        self.send(message)

    def _record_selection(
        self,
        event: FrameEvent,
        suitable: bool,
        probability: float,
    ) -> None:
        try:
            from mas.utils.report_collector import ReportCollector

            ReportCollector().record_selection(
                event.passage_id,
                event.depth_filename,
                event.label,
                suitable,
                probability,
            )
        except Exception as exc:
            display_message(
                self.aid.name,
                f"[REPORT-ERROR] record_selection failed: {exc}",
            )

    def _log_confusion(self, passage_id: str, total: int) -> None:
        confusion = self.confusion.get(passage_id, {})
        suited_ok = confusion.get(("suited", True), 0)
        suited_no = confusion.get(("suited", False), 0)
        false_positive = sum(
            value
            for (label, decision), value in confusion.items()
            if decision and label != "suited"
        )
        display_message(
            self.aid.name,
            f"[SELECT-SUMMARY] animal={passage_id} total={total} | "
            f"label 'suited' captados={suited_ok + suited_no} "
            f"(TP={suited_ok}, FN={suited_no}) | "
            f"nao-suited marcados suitable (FP)={false_positive}",
        )

    def on_start(self):
        super().on_start()
        display_message(
            self.aid.name,
            "FrameSelectionAgent started. Loading selection model...",
        )
        deferred = deferToThread(self.frame_selection_adapter.load_model)
        deferred.addCallback(self._on_model_loaded)
        deferred.addErrback(self._on_model_error)

    def _on_model_loaded(self, _):
        display_message(self.aid.name, "Selection Model loaded successfully.")
        if self.capture_agent_aid:
            message = ACLMessage(ACLMessage.INFORM)
            message.set_ontology("agent-ready")
            message.add_receiver(AID(self.capture_agent_aid))
            message.set_content(json.dumps({"agent": self.aid.name}))
            self.send(message)

    def _on_model_error(self, failure):
        error = (
            failure.getErrorMessage()
            if hasattr(failure, "getErrorMessage")
            else str(failure)
        )
        display_message(
            self.aid.name,
            f"[ERROR] Selection model load failed: {error}",
        )
