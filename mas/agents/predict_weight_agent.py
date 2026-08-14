"""Prediction/Aggregation PADE ordenada, equivalente ao consumidor de q3.

Mensagens ACL apenas admitem contratos na ``OrderedInbox``. Um unico fluxo
logico processa ``FrameEvent`` e somente o ``EndPassageEvent`` correspondente
finaliza a passagem. ``EndPipelineEvent`` persiste os resultados e aciona o
lifecycle global depois que todos os eventos anteriores foram processados.
"""

from __future__ import annotations

import json
import os
import queue
from collections import Counter
from datetime import datetime
from typing import Callable

import numpy as np
from twisted.internet import reactor
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
)
from mas.adapters.inference_adapter import InferenceAdapter
from mas.infrastructure.frame_store import FRAME_STORE, FrameStore
from mas.infrastructure.ordered_inbox import OrderedInbox, OrderedInboxClosed


PIPELINE_EVENT_ONTOLOGY = "pipeline-event"


class PredictWeightAgent(Agent):
    """Consome Preprocessing -> Prediction em ordem e agrega por passagem."""

    def __init__(
        self,
        aid,
        inference_adapter: InferenceAdapter,
        mode: str,
        pid: str,
        herd_size: int | None = None,
        capture_agent_aid: str | None = None,
        frame_store: FrameStore = FRAME_STORE,
        inbox: OrderedInbox[PipelineEvent] | None = None,
        defer_executor: Callable = deferToThread,
        call_later: Callable | None = None,
        shutdown_callback: Callable | None = None,
        now: Callable[[], str] | None = None,
        debug: bool = False,
        verbose: bool = False,
        fps: float | None = None,
        capture_mode: str | None = None,
    ):
        super().__init__(aid=aid, debug=debug)
        if mode not in {"single", "batch"}:
            raise ValueError("mode must be 'single' or 'batch'")

        self.inference_adapter = inference_adapter
        self.mode = mode
        self.pid = pid
        # Aceito somente para compatibilidade de construcao com o launcher
        # historico; nunca participa da finalizacao, que depende de END.
        self.herd_size = herd_size
        self.capture_agent_aid = capture_agent_aid
        self.frame_store = frame_store
        self.inbox = inbox or OrderedInbox()
        self.verbose = verbose
        self.fps = fps
        self.capture_mode = capture_mode

        self._defer_executor = defer_executor
        self._call_later = call_later or reactor.callLater
        self._shutdown_callback = shutdown_callback or reactor.stop
        self._now = now or (lambda: datetime.now().isoformat())
        self._inference_semaphore = DeferredSemaphore(1)

        self._processing = False
        self._active_event_seq: int | None = None
        self._global_finished = False
        self._metrics_saved = False

        self._predictions: dict[str, list[float]] = {}
        self._labels: dict[str, list[str]] = {}
        self._finalized: set[str] = set()
        self._total_inferences = 0
        self._batch_imgs: dict[str, list] = {}

        self.metrics = {
            "pid": self.pid,
            "load_model_start": None,
            "load_model_final": None,
            "animals": {},
        }
        if self.capture_mode is not None:
            self.metrics["capture_mode"] = self.capture_mode

    def react(self, message):
        """Admite apenas eventos canonicos; inferencia ocorre no consumidor."""
        super().react(message)
        if message.performative != ACLMessage.INFORM:
            return
        if message.ontology != PIPELINE_EVENT_ONTOLOGY:
            return

        try:
            event = event_from_json(message.content)
            self.inbox.put(event)
        except (TypeError, ValueError, OrderedInboxClosed) as exc:
            self._safe_log(f"[ERROR] Invalid ordered pipeline event: {exc}")
            return
        self._drain_inbox()

    def _drain_inbox(self) -> None:
        """Retira no maximo um evento que possa iniciar trabalho assincrono."""
        if self._processing or self._global_finished:
            return

        while True:
            try:
                event = self.inbox.get(block=False)
            except (queue.Empty, OrderedInboxClosed):
                return

            if isinstance(event, FrameEvent) and self.mode == "batch":
                # Acumulo sincrono, ainda sob o mesmo slot logico exclusivo.
                self._activate_event(event)
                try:
                    self._collect_batch_frame(event)
                finally:
                    self._release_current_event(event)
                continue

            self._activate_event(event)
            if isinstance(event, FrameEvent):
                self._schedule_single_inference(event)
                return
            if isinstance(event, EndPassageEvent):
                self._handle_end_passage(event)
                return
            if isinstance(event, EndPipelineEvent):
                self._handle_end_pipeline(event)
                return

            self._safe_log(f"[ERROR] Unsupported event: {type(event).__name__}")
            self._finish_current_event(event)
            return

    def _activate_event(self, event: PipelineEvent) -> None:
        self._processing = True
        self._active_event_seq = event.stream_seq

    def _finish_current_event(
        self,
        event: PipelineEvent,
        *,
        continue_consuming: bool = True,
    ) -> None:
        """Libera exatamente uma vez o evento ativo e retoma a inbox."""
        if not self._release_current_event(event):
            return
        if continue_consuming:
            try:
                self._drain_inbox()
            except Exception as exc:
                # Nao deixe uma falha auxiliar escapar para o callback
                # anterior e provocar uma segunda finalizacao/liberacao.
                self._safe_log(f"[ERROR] Could not resume prediction inbox: {exc}")

    def _release_current_event(self, event: PipelineEvent) -> bool:
        """Limpa o slot ativo uma unica vez, sem retirar outro evento."""
        if self._active_event_seq != event.stream_seq:
            return False
        self._active_event_seq = None
        self._processing = False
        return True

    def _schedule_single_inference(self, event: FrameEvent) -> None:
        try:
            img = self.frame_store.pop(event.frame_id)
        except Exception as exc:
            self._single_inference_failed(exc, event)
            return

        if img is None:
            try:
                self._safe_log(
                    f"[WARN] frame_id={event.frame_id} not in FrameStore"
                )
            finally:
                self._finish_current_event(event)
            return

        try:
            self.metrics["animals"].setdefault(event.passage_id, {"imgs": {}})
            start_ts = self._now()
            deferred = self._inference_semaphore.run(
                self._defer_executor,
                self.inference_adapter.predict,
                [img],
            )
            deferred.addCallbacks(
                self._single_inference_succeeded,
                self._single_inference_failed,
                callbackArgs=(event, start_ts),
                errbackArgs=(event,),
            )
        except Exception as exc:
            self._single_inference_failed(exc, event)

    def _single_inference_succeeded(
        self,
        result,
        event: FrameEvent,
        start_ts: str,
    ):
        try:
            self._total_inferences += 1
            weight = float(result[0][0]) if result is not None else None
            if weight is not None:
                self._predictions.setdefault(event.passage_id, []).append(weight)
                if self.verbose and event.label is not None:
                    self._labels.setdefault(event.passage_id, []).append(event.label)

            self._record_prediction(event, weight)
            self._record_single_metric(event, start_ts, self._now())
            weight_text = "None" if weight is None else f"{weight:.4f} kg"
            label_text = f" label={event.label}" if self.verbose else ""
            self._safe_log(
                f"[PREDICTION] animal_id={event.passage_id} "
                f"frame_id={event.frame_id}{label_text} weight={weight_text}"
            )
        except Exception as exc:
            self._report_callback_exception("Prediction callback", event, exc)
        finally:
            self._finish_current_event(event)
        return result

    def _single_inference_failed(self, failure, event: FrameEvent):
        try:
            self.frame_store.discard(event.frame_id)
            self._safe_log(
                f"[ERROR] Prediction failed for {event.frame_id}: "
                f"{self._failure_text(failure)}"
            )
        except Exception as exc:
            self._report_callback_exception("Prediction errback", event, exc)
        finally:
            self._finish_current_event(event)
        return None

    def _collect_batch_frame(self, event: FrameEvent) -> None:
        """Move um enhanced frame do store para o batch da sua passagem."""
        try:
            img = self.frame_store.pop(event.frame_id)
            if img is None:
                self._safe_log(
                    f"[WARN] frame_id={event.frame_id} not in FrameStore"
                )
                return
            self._batch_imgs.setdefault(event.passage_id, []).append(img)
            if self.verbose and event.label is not None:
                self._labels.setdefault(event.passage_id, []).append(event.label)
        except Exception as exc:
            try:
                self.frame_store.discard(event.frame_id)
            except Exception:
                pass
            self._report_callback_exception("Batch collection", event, exc)

    def _handle_end_passage(self, event: EndPassageEvent) -> None:
        """Finaliza exclusivamente ao consumir o END ordenado."""
        try:
            self._prepare_passage_metrics(event)
            if self.mode == "single":
                self._finalize_passage(event.passage_id)
                self._finish_current_event(event)
                return

            imgs = self._batch_imgs.pop(event.passage_id, [])
            self.metrics["animals"][event.passage_id]["suitable_images"] = len(imgs)
            if not imgs:
                self._predictions[event.passage_id] = []
                self._finalize_passage(event.passage_id)
                self._finish_current_event(event)
                return

            self._safe_log(
                f"[BATCH INFERENCE] Running full network on {len(imgs)} "
                f"frames for animal {event.passage_id}"
            )
            start_ts = self._now()
            deferred = self._inference_semaphore.run(
                self._defer_executor,
                self.inference_adapter.predict,
                imgs,
            )
            deferred.addCallbacks(
                self._batch_inference_succeeded,
                self._batch_inference_failed,
                callbackArgs=(event, start_ts),
                errbackArgs=(event,),
            )
        except Exception as exc:
            self._batch_inference_failed(exc, event)

    def _batch_inference_succeeded(
        self,
        result,
        event: EndPassageEvent,
        start_ts: str,
    ):
        try:
            self._total_inferences += 1
            weights = [float(item[0]) for item in result] if result is not None else []
            self._predictions[event.passage_id] = weights
            self._record_batch_metric(
                event.passage_id,
                event.total_captured_frames,
                start_ts,
                self._now(),
            )
            self._finalize_passage(event.passage_id)
        except Exception as exc:
            # O batch ja foi consumido: nao ha retry nem segunda finalizacao.
            self._report_callback_exception("Batch prediction callback", event, exc)
            if event.passage_id not in self._finalized:
                self._predictions[event.passage_id] = []
                self._finalize_passage_safely(event.passage_id)
        finally:
            self._finish_current_event(event)
        return result

    def _batch_inference_failed(self, failure, event: EndPassageEvent):
        try:
            self._predictions[event.passage_id] = []
            self._safe_log(
                f"[ERROR] Batch prediction failed for {event.passage_id}: "
                f"{self._failure_text(failure)}"
            )
            self._finalize_passage_safely(event.passage_id)
        finally:
            self._finish_current_event(event)
        return None

    def _prepare_passage_metrics(self, event: EndPassageEvent) -> None:
        entry = self.metrics["animals"].setdefault(event.passage_id, {"imgs": {}})
        entry["total_of_images"] = event.total_captured_frames
        if event.first_capture_time:
            entry["first_image_capture_time"] = event.first_capture_time
        if event.last_capture_time:
            entry["last_image_capture_time"] = event.last_capture_time
        if self.mode == "single":
            entry["suitable_images"] = len(
                self._predictions.get(event.passage_id, [])
            )

    def _finalize_passage(self, passage_id: str) -> float:
        if passage_id in self._finalized:
            weights = self._predictions.get(passage_id, [])
            return float(np.mean(weights)) if weights else 0.0

        weights = self._predictions.get(passage_id, [])
        predicted_weight = float(np.mean(weights)) if weights else 0.0
        self._record_final_prediction(passage_id, predicted_weight)

        entry = self.metrics["animals"].setdefault(passage_id, {"imgs": {}})
        entry["weight_prediction_final"] = self._now()
        if self.verbose:
            label_counts = dict(Counter(self._labels.get(passage_id, [])))
            self._safe_log(
                f"[FINAL] Animal {passage_id}: n_suitable={len(weights)} "
                f"| labels_dos_suitable={label_counts} "
                f"| peso_medio={predicted_weight:.4f} kg"
            )
        else:
            self._safe_log(
                f"[FINAL] Animal {passage_id} completed. "
                f"Mean weight: {predicted_weight:.4f} kg"
            )
        self._finalized.add(passage_id)
        return predicted_weight

    def _finalize_passage_safely(self, passage_id: str) -> None:
        try:
            self._finalize_passage(passage_id)
        except Exception as exc:
            self._safe_log(f"[ERROR] Finalization failed for {passage_id}: {exc}")

    def _record_single_metric(
        self,
        event: FrameEvent,
        start_ts: str,
        final_ts: str,
    ) -> None:
        entry = self.metrics["animals"].setdefault(event.passage_id, {"imgs": {}})
        metric = {
            "weight_prediction_start": start_ts,
            "weight_prediction_final": final_ts,
        }
        if event.label is not None:
            metric["label"] = event.label
        entry["imgs"][str(event.capture_index)] = metric

    def _record_batch_metric(
        self,
        passage_id: str,
        total_frames: int,
        start_ts: str,
        final_ts: str,
    ) -> None:
        entry = self.metrics["animals"].setdefault(passage_id, {"imgs": {}})
        key = str(total_frames) if total_frames else "0"
        entry["imgs"][key] = {
            "weight_prediction_start": start_ts,
            "weight_prediction_final": final_ts,
        }

    def _record_prediction(self, event: FrameEvent, weight: float | None) -> None:
        try:
            from mas.utils.report_collector import ReportCollector

            ReportCollector().record_prediction(
                event.passage_id,
                event.depth_filename,
                weight,
            )
        except Exception as exc:
            self._safe_log(f"[REPORT-ERROR] record_prediction failed: {exc}")

    def _record_final_prediction(self, passage_id: str, weight: float) -> None:
        try:
            from mas.utils.report_collector import ReportCollector

            ReportCollector().record_final_prediction(passage_id, weight)
        except Exception as exc:
            self._safe_log(
                f"[REPORT-ERROR] record_final_prediction failed: {exc}"
            )

    def _handle_end_pipeline(self, event: EndPipelineEvent) -> None:
        try:
            self._save_metrics()
            self._safe_log(
                "[SHUTDOWN] EndPipeline processed. Stopping reactor in 1s..."
            )
        except Exception as exc:
            self._safe_log(f"[ERROR] Global finalization failed: {exc}")
        finally:
            self._global_finished = True
            self.inbox.close()
            self._finish_current_event(event, continue_consuming=False)
            try:
                self._call_later(1.0, self._shutdown_callback)
            except Exception as exc:
                self._safe_log(f"[ERROR] Could not schedule shutdown: {exc}")

    def _save_metrics(self) -> None:
        if self._metrics_saved:
            return
        reports_dir = f"infra/reports/{self.pid}"
        os.makedirs(reports_dir, exist_ok=True)
        metrics_path = os.path.join(reports_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as stream:
            json.dump(self.metrics, stream, indent=4)
        self._metrics_saved = True
        self._safe_log(f"[METRICS] Saved metrics.json to {metrics_path}")

        try:
            from mas.utils.report_collector import ReportCollector

            ReportCollector().generate_report(
                reports_dir,
                self.mode,
                self.fps,
                capture_mode=self.capture_mode,
            )
            self._safe_log(f"[REPORT] Saved execution report to {reports_dir}/report.md")
        except Exception as exc:
            self._safe_log(f"[REPORT-ERROR] generate_report failed: {exc}")

    def on_start(self):
        super().on_start()
        self._safe_log("PredictWeightAgent started. Loading AI model in background...")
        self.metrics["load_model_start"] = self._now()
        deferred = self._defer_executor(self.inference_adapter.load_model)
        deferred.addCallback(self._on_model_loaded)
        deferred.addErrback(self._on_model_error)

    def _on_model_loaded(self, _):
        self.metrics["load_model_final"] = self._now()
        self._safe_log("AI Model loaded successfully.")
        if self.capture_agent_aid:
            message = ACLMessage(ACLMessage.INFORM)
            message.set_ontology("agent-ready")
            message.add_receiver(AID(self.capture_agent_aid))
            message.set_content(json.dumps({"agent": self.aid.name}))
            self.send(message)

    def _on_model_error(self, failure):
        self._safe_log(f"[ERROR] Model load failed: {self._failure_text(failure)}")
        return None

    def get_predictions_summary(self) -> dict:
        summary = {}
        for passage_id, weights in self._predictions.items():
            summary[passage_id] = {
                "n_predictions": len(weights),
                "mean_weight": round(float(np.mean(weights)), 4) if weights else 0.0,
                "weights": list(weights),
            }
        return summary

    def _report_callback_exception(
        self,
        callback_name: str,
        event: PipelineEvent,
        error: Exception,
    ) -> None:
        frame_id = getattr(event, "frame_id", None)
        suffix = f" for {frame_id}" if frame_id is not None else ""
        self._safe_log(f"[ERROR] {callback_name} failed{suffix}: {error}")

    def _safe_log(self, text: str) -> None:
        try:
            display_message(self.aid.name, text)
        except Exception:
            pass

    @staticmethod
    def _failure_text(failure) -> str:
        if hasattr(failure, "getErrorMessage"):
            return failure.getErrorMessage()
        return str(failure)
