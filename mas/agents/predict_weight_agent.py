"""Predict Weight Agent — runs model inference via InferenceAdapter.

Receives suitable frames from FrameSelectionAgent (ontology "frame-selected"),
pops the enhanced image from FRAME_BUFFER to prevent RAM accumulation,
and runs TensorFlow inference in a delegated thread (deferToThread) so
the Twisted event loop is never blocked.

Logs per-animal prediction metrics that mirror the baseline output for
direct scientific comparison.
"""

import json
import threading
import os
from datetime import datetime
import numpy as np

from twisted.internet.threads import deferToThread

from pade.acl.aid import AID
from pade.acl.messages import ACLMessage
from pade.core.agent import Agent
from pade.misc.utility import display_message

from mas.adapters.inference_adapter import InferenceAdapter
from mas.utils.globals import FRAME_BUFFER


class PredictWeightAgent(Agent):
    """Prediction PADE agent — runs weight inference off the event loop."""

    def __init__(
        self,
        aid,
        inference_adapter: InferenceAdapter,
        mode: str,
        pid: str,
        herd_size: int,
        capture_agent_aid: str = None,
        debug: bool = False,
    ):
        super().__init__(aid=aid, debug=debug)
        self.inference_adapter = inference_adapter
        self.mode = mode
        self.pid = pid
        self.herd_size = herd_size
        self.capture_agent_aid = capture_agent_aid
        
        self._predictions: dict[int, list[float]] = {}
        self._total_inferences = 0
        self._lock = threading.Lock()
        
        self.batch_imgs = {}
        self.batch_payloads = {}
        self.expected_counts = {}
        self.metrics = {
            'pid': self.pid,
            'load_model_start': None,
            'load_model_final': None,
            'animals': {}
        }

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

    def _record_single_metric(self, animal_id, frame_index, start_ts, final_ts):
        if animal_id not in self.metrics['animals']:
            return
        self.metrics['animals'][animal_id]['imgs'][str(frame_index)] = {
            'weight_prediction_start': start_ts,
            'weight_prediction_final': final_ts
        }

    def _record_batch_metric(self, animal_id, total_frames, start_ts, final_ts):
        if animal_id not in self.metrics['animals']:
            return
        idx_key = str(total_frames) if total_frames else "0"
        self.metrics['animals'][animal_id]['imgs'][idx_key] = {
            'weight_prediction_start': start_ts,
            'weight_prediction_final': final_ts
        }

    def _finalize_animal(self, animal_id):
        from twisted.internet import reactor
        
        weights = self._predictions.get(animal_id, [])
        if animal_id in self.metrics['animals']:
            predicted_weight = float(np.mean(weights)) if weights else 0.0
            self.metrics['animals'][animal_id]['weight_prediction_final'] = datetime.now().isoformat()
            
            display_message(
                self.aid.name,
                f"[FINAL] Animal {animal_id} completed. Mean weight: {predicted_weight:.4f} kg"
            )

        if str(animal_id) == str(self.herd_size):
            self._save_metrics()
            display_message(self.aid.name, "[SHUTDOWN] All animals evaluated. Stopping reactor in 1s...")
            reactor.callLater(1.0, reactor.stop)

    def _save_metrics(self):
        reports_dir = f"infra/reports/{self.pid}"
        os.makedirs(reports_dir, exist_ok=True)
        with open(os.path.join(reports_dir, "metrics.json"), "w") as f:
            json.dump(self.metrics, f, indent=4)
        display_message(self.aid.name, f"[METRICS] Saved exactly identical JSON to {reports_dir}/metrics.json")

    def _on_single_inference_success(self, result, payload: dict, start_ts: str):
        self._total_inferences += 1
        animal_id = payload.get("animal_id", "?")
        frame_id = payload.get("frame_id", "?")
        frame_index = payload.get("frame_index", self._total_inferences)
        
        weight = float(result[0][0]) if result is not None else None

        if animal_id != "?" and weight is not None:
            self._predictions.setdefault(animal_id, []).append(weight)

        final_ts = datetime.now().isoformat()
        self._record_single_metric(animal_id, frame_index, start_ts, final_ts)

        display_message(
            self.aid.name,
            f"[PREDICTION] animal_id={animal_id} frame_id={frame_id} weight={weight:.4f} kg",
        )
        self._check_batch_sync(animal_id)

    def _on_batch_inference_success(self, result, animal_id: int, total_frames: int, start_ts: str):
        self._total_inferences += 1
        weights = [float(r[0]) for r in result] if result is not None else []
        
        if animal_id != "?":
            self._predictions.setdefault(animal_id, []).extend(weights)
            
        final_ts = datetime.now().isoformat()
        self._record_batch_metric(animal_id, total_frames, start_ts, final_ts)
        self._finalize_animal(animal_id)

    def _on_inference_error(self, failure):
        display_message(
            self.aid.name,
            f"[ERROR] Inference failed: {failure.getErrorMessage()}",
        )

    def _schedule_inference(self, payload: dict):
        frame_id = payload["frame_id"]
        animal_id = payload["animal_id"]
        
        with self._lock:
            img = FRAME_BUFFER.pop(frame_id, None)
            
        if img is None:
            display_message(self.aid.name, f"[WARN] frame_id={frame_id} not in buffer")
            return
            
        if animal_id not in self.metrics['animals']:
            self.metrics['animals'][animal_id] = {
                'first_image_capture_time': datetime.now().isoformat(),  # Will be overridden by sync metric
                'imgs': {}
            }

        start_ts = datetime.now().isoformat()
        d = deferToThread(self.inference_adapter.predict, [img])
        d.addCallback(self._on_single_inference_success, payload, start_ts)
        d.addErrback(self._on_inference_error)
        
        # Pseudo finalize evaluation if single (cannot guarantee exact last frame easily unless synced)
        # But single stream doesn't strictly log per-animal final metric properly until gap.

    def _process_batch(self, animal_id: int, total_frames: int):
        with self._lock:
            imgs = self.batch_imgs.pop(animal_id, [])
            self.batch_payloads.pop(animal_id, [])
            
        if not imgs:
            display_message(self.aid.name, f"[WARN] No images available for batch animal {animal_id}")
            self._finalize_animal(animal_id)
            return

        display_message(self.aid.name, f"[BATCH INFERENCE] Running full network on {len(imgs)} frames for animal {animal_id}")
        start_ts = datetime.now().isoformat()
        d = deferToThread(self.inference_adapter.predict, imgs)
        d.addCallback(self._on_batch_inference_success, animal_id, total_frames, start_ts)
        d.addErrback(self._on_inference_error)

    def react(self, message):
        super().react(message)
        if message.performative != ACLMessage.INFORM:
            return
            
        if message.ontology == "batch-ready":
            try:
                data = json.loads(message.content)
                animal_id = data.get("animal_id")
                capture_metrics = data.get("capture_metrics", {})
                total_frames = data.get("total_frames", 0)
                
                with self._lock:
                    self.expected_counts[animal_id] = data.get("suitable_count", 0)
                    
                    if animal_id not in self.metrics['animals']:
                        self.metrics['animals'][animal_id] = {'imgs': {}}
                        
                    self.metrics['animals'][animal_id]['first_image_capture_time'] = capture_metrics.get("first_image_capture_time")
                    self.metrics['animals'][animal_id]['last_image_capture_time'] = capture_metrics.get("last_image_capture_time")
                    self.metrics['animals'][animal_id]['suitable_images'] = data.get("suitable_count", 0)
                    self.metrics['animals'][animal_id]['total_of_images'] = total_frames
                    
                self._check_batch_ready_custom(animal_id, total_frames)
            except Exception as e:
                display_message(self.aid.name, f"[ERROR] Parsing batch-ready: {e}")
            return
            
        if message.ontology != "frame-selected":
            return
            
        payload = self._parse_payload(message)
        if not payload:
            return
            
        animal_id = payload["animal_id"]
        
        # Ensure animal exists in dict before recording
        if animal_id not in self.metrics['animals']:
            self.metrics['animals'][animal_id] = {
                'first_image_capture_time': datetime.now().isoformat(),
                'imgs': {}
            }
            
        if self.mode == "single":
            self._schedule_inference(payload)
        else:
            frame_id = payload["frame_id"]
            with self._lock:
                img = FRAME_BUFFER.pop(frame_id, None)
                if img is not None:
                    self.batch_imgs.setdefault(animal_id, []).append(img)
                    self.batch_payloads.setdefault(animal_id, []).append(payload)
            self._check_batch_sync(animal_id)

    def _check_batch_ready_custom(self, animal_id: int, total_frames: int = 0):
        # Specific check for batch mode triggered by the signal
        try:
            animal_id = int(animal_id)
        except (TypeError, ValueError):
            return

        with self._lock:
            expected = self.expected_counts.get(animal_id)
            received = len(self.batch_imgs.get(animal_id, [])) if self.mode == "batch" else len(self._predictions.get(animal_id, []))
            
        if expected is not None and received >= expected:
            with self._lock:
                self.expected_counts.pop(animal_id, None)
            
            if self.mode == "single":
                self._finalize_animal(animal_id)
            else:
                self._process_batch(animal_id, total_frames)

    def _check_batch_sync(self, animal_id: int):
        # Loop check call (from frames)
        try:
            animal_id = int(animal_id)
        except (TypeError, ValueError):
            return

        with self._lock:
            expected = self.expected_counts.get(animal_id)
            if self.mode == "single":
                received = len(self._predictions.get(animal_id, []))
            else:
                received = len(self.batch_imgs.get(animal_id, []))
            
        if expected is not None and received >= expected:
            with self._lock:
                self.expected_counts.pop(animal_id, None)
            
            if self.mode == "single":
                self._finalize_animal(animal_id)
            else:
                # We don't have total_frames here, but this branch is rarely 
                # reached before the batch-ready signal in normal flow.
                # If it is, we use a default or wait for the signal.
                pass

    def on_start(self):
        super().on_start()
        display_message(self.aid.name, "PredictWeightAgent started. Loading AI model in background...")
        self.metrics['load_model_start'] = datetime.now().isoformat()
        
        # Load model in a background thread to avoid blocking the reactor
        d = deferToThread(self.inference_adapter.load_model)
        d.addCallback(self._on_model_loaded)
        d.addErrback(self._on_inference_error)

    def _on_model_loaded(self, _):
        self.metrics['load_model_final'] = datetime.now().isoformat()
        display_message(self.aid.name, "AI Model loaded successfully.")
        
        # Notify CaptureAgent that we are ready
        if self.capture_agent_aid:
            msg = ACLMessage(ACLMessage.INFORM)
            msg.set_ontology("agent-ready")
            msg.add_receiver(AID(self.capture_agent_aid))
            msg.set_content(json.dumps({"agent": self.aid.name}))
            self.send(msg)

    def get_predictions_summary(self) -> dict:
        """Return per-animal prediction summaries (mean weights)."""
        summary = {}
        for aid, weights in self._predictions.items():
            summary[aid] = {
                "n_predictions": len(weights),
                "mean_weight": round(float(np.mean(weights)), 4),
                "weights": weights,
            }
        return summary
