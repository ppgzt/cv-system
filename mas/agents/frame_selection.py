"""Frame Selection Agent — gatekeeper that evaluates frame suitability.

Receives RAW frames from CaptureAgent (ontology "frame-capture"), runs the
trained TFLite selector on the raw depth (its own preprocessing), and either
forwards the suitable key to DataEnhanceAgent (ontology "frame-selected") or
deletes the raw from FRAME_BUFFER to free RAM immediately.

Control signals: passage-complete (from Capture) sets the expected frame
count per animal; batch-ready is sent straight to PredictWeightAgent
(predict_agent_aid) once all frames of an animal are processed.

The evaluation delegates to FrameSelectionAdapter via deferToThread.
"""

import json
import threading

from twisted.internet.threads import deferToThread
from twisted.internet.defer import DeferredSemaphore

from pade.acl.aid import AID
from pade.acl.messages import ACLMessage
from pade.core.agent import Agent
from pade.misc.utility import display_message

from mas.adapters.frame_selection_adapter import FrameSelectionAdapter
from mas.utils.globals import FRAME_BUFFER


class FrameSelectionAgent(Agent):
    """Frame selection PADE agent — suitability gatekeeper with GC."""

    def __init__(
        self,
        aid,
        frame_selection_adapter: FrameSelectionAdapter,
        next_agent_aid: str,
        predict_agent_aid: str = None,
        capture_agent_aid: str = None,
        debug: bool = False,
        verbose: bool = False,
    ):
        super().__init__(aid=aid, debug=debug)
        self.frame_selection_adapter = frame_selection_adapter
        self.next_agent_aid = next_agent_aid        # DataEnhanceAgent (suitable frames)
        self.predict_agent_aid = predict_agent_aid  # PredictWeightAgent (batch-ready)
        self.capture_agent_aid = capture_agent_aid
        self.verbose = verbose
        self.discarded = 0
        self.forwarded = 0
        self._lock = threading.Lock()
        self._inference_semaphore = DeferredSemaphore(1)

        self.expected_frames = {}
        self.processed_frames = {}
        self.suitable_frames = {}
        self.capture_metrics_buffer = {}
        # verbose: matriz de confusão label_real x decisão, por animal
        self.confusion: dict = {}

    def _parse_payload(self, message) -> dict | None:
        try:
            payload = json.loads(message.content)
        except (TypeError, json.JSONDecodeError):
            display_message(self.aid.name, "[WARN] invalid JSON payload")
            return None
        if not payload.get("frame_id"):
            display_message(self.aid.name, "[WARN] missing frame_id")
            return None
        return payload

    def _forward_frame(self, payload: dict):
        out = ACLMessage(ACLMessage.INFORM)
        out.set_ontology("frame-selected")
        out.add_receiver(AID(self.next_agent_aid))
        out.set_content(json.dumps(payload, ensure_ascii=True))
        self.send(out)

    def _check_batch_ready(self, animal_id: int):
        """Verify if all frames for the animal have been processed and fire batch-ready."""
        with self._lock:
            expected = self.expected_frames.get(animal_id)
            processed = self.processed_frames.get(animal_id, 0)

            if expected is not None and processed >= expected:
                if self.verbose:
                    self._log_confusion(animal_id, expected)

                msg = ACLMessage(ACLMessage.INFORM)
                msg.set_ontology("batch-ready")
                msg.add_receiver(AID(self.predict_agent_aid))
                msg.set_content(json.dumps({
                    "animal_id": animal_id,
                    "suitable_count": self.suitable_frames.get(animal_id, 0),
                    "total_frames": expected,
                    "capture_metrics": self.capture_metrics_buffer.get(animal_id, {})
                }, ensure_ascii=True))
                self.send(msg)

                # Cleanup state
                self.expected_frames.pop(animal_id, None)
                self.processed_frames.pop(animal_id, None)
                self.suitable_frames.pop(animal_id, None)
                self.capture_metrics_buffer.pop(animal_id, None)
                self.confusion.pop(animal_id, None)

                display_message(self.aid.name, f"[BATCH READY] Sent animal_id={animal_id} to Predict!")

    def _log_confusion(self, animal_id, total):
        """Resume label_real x decisão do seletor para o animal (modo verbose)."""
        cm = self.confusion.get(animal_id, {})
        suited_pred_ok = cm.get(("suited", True), 0)        # label=suited & suitable
        suited_pred_no = cm.get(("suited", False), 0)       # label=suited & discarded (FN)
        nonsuited_as_suited = sum(v for (lbl, dec), v in cm.items() if dec and lbl != "suited")  # FP
        display_message(
            self.aid.name,
            f"[SELECT-SUMMARY] animal={animal_id} total={total} | "
            f"label 'suited' captados={suited_pred_ok + suited_pred_no} "
            f"(TP={suited_pred_ok}, FN={suited_pred_no}) | "
            f"não-suited marcados suitable (FP)={nonsuited_as_suited}",
        )

    def _on_selection_complete(self, result, payload: dict):
        suitable, prob = result
        frame_id = payload["frame_id"]
        animal_id = payload["animal_id"]
        label = payload.get("label")
        depth_filename = payload.get("depth_filename")

        try:
            from mas.utils.report_collector import ReportCollector
            ReportCollector().record_selection(animal_id, depth_filename, label, suitable, prob)
        except Exception as e:
            display_message(self.aid.name, f"[REPORT-ERROR] record_selection failed: {e}")

        with self._lock:
            self.processed_frames[animal_id] = self.processed_frames.get(animal_id, 0) + 1
            if suitable:
                self.suitable_frames[animal_id] = self.suitable_frames.get(animal_id, 0) + 1
            if self.verbose and label is not None:
                key = (label, bool(suitable))
                self.confusion.setdefault(animal_id, {})[key] = \
                    self.confusion.get(animal_id, {}).get(key, 0) + 1

        if not suitable:
            self.discarded += 1
            with self._lock:
                FRAME_BUFFER.pop(frame_id, None)
            display_message(
                self.aid.name,
                (f"[SELECT] frame_id={frame_id} animal={animal_id} "
                 f"label={label} -> DISCARDED (p={prob:.4f}). "
                 f"Discarded={self.discarded}, Forwarded={self.forwarded}")
                if self.verbose else
                (f"frame_id={frame_id} DISCARDED (deleted from buffer). "
                 f"Discarded={self.discarded}, Forwarded={self.forwarded}"),
            )
        else:
            self.forwarded += 1
            display_message(
                self.aid.name,
                (f"[SELECT] frame_id={frame_id} animal={animal_id} "
                 f"label={label} -> SUITABLE (p={prob:.4f}). "
                 f"Discarded={self.discarded}, Forwarded={self.forwarded}")
                if self.verbose else
                (f"frame_id={frame_id} SUITABLE. "
                 f"Discarded={self.discarded}, Forwarded={self.forwarded}"),
            )
            self._forward_frame(payload)

        self._check_batch_ready(animal_id)

    def _on_selection_error(self, failure):
        display_message(
            self.aid.name,
            f"[ERROR] Evaluation failed: {failure.getErrorMessage()}",
        )

    def _schedule_evaluation(self, payload: dict):
        elapsed = payload.get("elapsed_time", 0.0)
        frame_id = payload.get("frame_id")
        
        with self._lock:
            # Selector runs on the RAW depth straight from Capture (DataEnhance
            # now runs downstream, only for suitable frames).
            img = FRAME_BUFFER.get(frame_id)
            
        if img is None:
            display_message(self.aid.name, f"[WARN] Image not found in buffer for {frame_id}")
            self._on_selection_complete((False, 0.0), payload)
            return
            
        d = self._inference_semaphore.run(deferToThread, self.frame_selection_adapter.evaluate_with_score, elapsed, img)
        d.addCallback(self._on_selection_complete, payload)
        d.addErrback(self._on_selection_error)

    def react(self, message):
        super().react(message)
        if message.performative != ACLMessage.INFORM:
            return
            
        if message.ontology == "passage-complete":
            try:
                data = json.loads(message.content)
                animal_id = data.get("animal_id")
                with self._lock:
                    self.expected_frames[animal_id] = data.get("total_frames")
                    self.capture_metrics_buffer[animal_id] = {
                        "first_image_capture_time": data.get("first_capture"),
                        "last_image_capture_time": data.get("last_capture")
                    }
                display_message(self.aid.name, f"[SYNC] Expected {self.expected_frames[animal_id]} frames for animal {animal_id}")
                self._check_batch_ready(animal_id)
            except Exception as e:
                display_message(self.aid.name, f"[ERROR] Parsing passage-complete: {e}")
            return
            
        if message.ontology != "frame-capture":
            return
            
        payload = self._parse_payload(message)
        if not payload:
            return

        self._schedule_evaluation(payload)

    def on_start(self):
        super().on_start()
        display_message(self.aid.name, "FrameSelectionAgent started. Loading selection model...")
        
        # Load model in background thread to avoid blocking the reactor
        d = deferToThread(self.frame_selection_adapter.load_model)
        d.addCallback(self._on_model_loaded)
        d.addErrback(self._on_selection_error)

    def _on_model_loaded(self, _):
        display_message(self.aid.name, "Selection Model loaded successfully.")
        
        # Notify CaptureAgent that we are ready
        if self.capture_agent_aid:
            msg = ACLMessage(ACLMessage.INFORM)
            msg.set_ontology("agent-ready")
            msg.add_receiver(AID(self.capture_agent_aid))
            msg.set_content(json.dumps({"agent": self.aid.name}))
            self.send(msg)
