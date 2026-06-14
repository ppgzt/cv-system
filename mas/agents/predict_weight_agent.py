"""Predict Weight Agent — runs model inference via InferenceAdapter.

Receives suitable frames from FrameSelectionAgent (ontology "frame-selected"),
pops the enhanced image from FRAME_BUFFER to prevent RAM accumulation,
and runs TensorFlow inference in a delegated thread (deferToThread) so
the Twisted event loop is never blocked.

Logs per-animal prediction metrics that mirror the baseline output for
direct scientific comparison.

---
Deterministic termination & drain (see plan: delightful-sparking-sky.md)
-----------------------------------------------------------------------
PADE's `send()` is fire-and-forget (at-most-once, silently dropped under
load), so the terminal `pipeline-complete` signal can vanish — which is what
caused the pipeline to hang at 2 FPS x 100 animals. This agent therefore does
NOT rely on that signal to terminate. Instead a passive watchdog (LoopingCall)
guarantees shutdown using ground truth that lives in shared in-process memory
(`mas.utils.globals`), not on the lossy message bus:

  - `CAPTURE_MANIFEST`   {animal_id: frames_captured}  (written by CaptureAgent)
  - `CAPTURE_DONE_TS`    wall-clock of capture FINISH   (written by CaptureAgent)

The watchdog fires the existing normal `_shutdown()` path when all manifest
animals are finalized; if progress stalls (stragglers whose `batch-ready` was
lost), it force-drains them. A wall-clock hard deadline (`T_max`) is the
absolute backstop.

The watchdog is dormant during healthy runs (1 FPS): the normal
`pipeline-complete` -> flush -> shutdown path fires well before any grace
expires, so the `metrics.json` output is byte-identical to before.
"""

import json
import threading
import os
import time
from datetime import datetime
import numpy as np

from twisted.internet import reactor
from twisted.internet.task import LoopingCall
from twisted.internet.threads import deferToThread

from pade.acl.aid import AID
from pade.acl.messages import ACLMessage
from pade.core.agent import Agent
from pade.misc.utility import display_message

from mas.adapters.inference_adapter import InferenceAdapter
from mas.utils import globals as mas_globals
from mas.utils.globals import FRAME_BUFFER, CAPTURE_MANIFEST, update_queue_stat


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
        passage_time: int = 0,
        arrival_time: int = 0,
        debug: bool = False,
    ):
        super().__init__(aid=aid, debug=debug)
        self.inference_adapter = inference_adapter
        self.mode = mode
        self.pid = pid
        self.herd_size = herd_size
        self.capture_agent_aid = capture_agent_aid
        self.passage_time = passage_time
        self.arrival_time = arrival_time

        self._predictions: dict[int, list[float]] = {}
        self._total_inferences = 0
        self._lock = threading.Lock()

        # Idempotent finalization. `_in_progress` guards against two trigger
        # paths (_check_batch_ready_custom / _check_batch_sync) both scheduling
        # _process_batch for the same animal; `_finalized` marks animals whose
        # weight is already recorded. _finalize_animal is the single point that
        # promotes an animal from _in_progress -> _finalized.
        self._finalized: set[int] = set()
        self._in_progress: set[int] = set()

        self.batch_imgs = {}
        self.batch_payloads = {}
        self.expected_counts = {}
        self.metrics = {
            'pid': self.pid,
            'load_model_start': None,
            'load_model_final': None,
            'animals': {}
        }
        self.pending_inferences = 0
        self.pipeline_complete_received = False

        # --- Watchdog state -------------------------------------------------
        self._shutting_down = False
        self._flushing = False
        self.sim_start_ts = time.time()
        self.last_progress_ts = time.time()
        self._watchdog: LoopingCall | None = None

        # Generous by construction so healthy runs never trip them.
        env_deadline = os.getenv("MAS_HARD_DEADLINE")
        self._t_max = (
            float(env_deadline) if env_deadline
            else self.herd_size * (self.passage_time + self.arrival_time) * 2 + 300.0
        )
        env_grace = os.getenv("MAS_DRAIN_GRACE")
        self._long_grace = float(env_grace) if env_grace else 60.0
        self._short_grace = 5.0  # all-manifest-finalized: lost terminal signal
        self._watchdog_interval = 5.0

    # ------------------------------------------------------------------ #
    # Metrics recording (unchanged schema on the happy path)
    # ------------------------------------------------------------------ #
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
        """Record the per-animal prediction. Idempotent — single point that
        promotes animal_id into `_finalized`."""
        with self._lock:
            if animal_id in self._finalized:
                return
            self._in_progress.discard(animal_id)
            self._finalized.add(animal_id)
            finalized_count = len(self._finalized)

        weights = self._predictions.get(animal_id, [])
        if animal_id in self.metrics['animals']:
            predicted_weight = float(np.mean(weights)) if weights else 0.0
            self.metrics['animals'][animal_id]['weight_prediction_final'] = datetime.now().isoformat()

            display_message(
                self.aid.name,
                f"[FINAL] Animal {animal_id} completed. Mean weight: {predicted_weight:.4f} kg"
            )

        # Save metrics incrementally after each animal so data is never lost
        self._save_metrics()

        # Progress + observability
        self.last_progress_ts = time.time()
        update_queue_stat("finalized_animals", finalized_count)

    def _flush_and_shutdown(self):
        """Force-process remaining incomplete batches, save metrics, and stop.
        Idempotent via `_flushing`/`_shutting_down`. Skips animals already
        finalized or in progress."""
        if self._flushing or self._shutting_down:
            return
        self._flushing = True
        display_message(self.aid.name, "[FLUSH] Pipeline complete — processing remaining batches...")

        with self._lock:
            self.pipeline_complete_received = True

        # Process any batch that has images, even if incomplete
        for animal_id in list(self.batch_imgs.keys()):
            with self._lock:
                if animal_id in self._finalized or animal_id in self._in_progress:
                    continue
                self._in_progress.add(animal_id)

            imgs = self.batch_imgs.get(animal_id, [])
            if imgs:
                total = len(imgs)
                display_message(
                    self.aid.name,
                    f"[FLUSH-BATCH] Animal {animal_id}: running inference on {total} available frames"
                )
                start_ts = datetime.now().isoformat()
                with self._lock:
                    self.pending_inferences += 1
                update_queue_stat("pending_inference", self.pending_inferences)
                d = deferToThread(self.inference_adapter.predict, imgs)
                d.addCallback(self._on_batch_inference_success, animal_id, total, start_ts)
                d.addErrback(self._on_inference_error)
            else:
                with self._lock:
                    self._in_progress.discard(animal_id)

        self._check_shutdown_readiness()

    def _force_drain(self, reason: str):
        """Watchdog-triggered terminal drain. Guarantees every animal in
        CAPTURE_MANIFEST ends up with a recorded weight, then shuts down.
        Only runs on the failure path — never in a healthy run. Idempotent:
        a second invocation (e.g. another watchdog tick before shutdown lands)
        is a no-op once flushing has started."""
        if self._flushing or self._shutting_down:
            return
        display_message(
            self.aid.name,
            f"[WATCHDOG] Force-drain triggered ({reason}). "
            f"finalized={len(self._finalized)} manifest={len(CAPTURE_MANIFEST)}"
        )

        now_iso = datetime.now().isoformat()
        # 1. Placeholder animals: captured frames but zero suitable frames
        #    reached us (all discarded/lost). They still must get an entry.
        for animal_id, total_captured in CAPTURE_MANIFEST.items():
            with self._lock:
                already_done = (
                    animal_id in self._finalized
                    or (str(animal_id) in self.metrics['animals']
                        and self.metrics['animals'][str(animal_id)].get('weight_prediction_final'))
                )
            if already_done:
                continue
            if not self.batch_imgs.get(animal_id):
                self.metrics['animals'][str(animal_id)] = {
                    'imgs': {},
                    'total_of_images': total_captured,
                    'suitable_images': 0,
                    'weight_prediction_final': now_iso,
                    'incomplete': True,
                }
                with self._lock:
                    self._in_progress.discard(animal_id)
                    self._finalized.add(animal_id)
                    finalized_count = len(self._finalized)
                update_queue_stat("finalized_animals", finalized_count)
                display_message(
                    self.aid.name,
                    f"[WATCHDOG] Animal {animal_id}: placeholder (0 suitable frames)."
                )

        self._save_metrics()

        # 2. Animals WITH accumulated frames: reuse the flush path to run
        #    inference; it will call _shutdown when pending hits 0.
        self._flush_and_shutdown()

        # If nothing was left to infer, _flush_and_shutdown already took care
        # of shutdown via _check_shutdown_readiness.

    def _check_shutdown_readiness(self):
        with self._lock:
            is_done = (self.pending_inferences <= 0 and self.pipeline_complete_received)
        if is_done:
            self._shutdown()

    def _decrement_pending_inferences(self):
        with self._lock:
            self.pending_inferences -= 1
            current = self.pending_inferences
        update_queue_stat("pending_inference", current)
        self.last_progress_ts = time.time()
        self._check_shutdown_readiness()

    def _shutdown(self):
        """Save metrics and stop the reactor. Idempotent."""
        if self._shutting_down:
            return
        self._shutting_down = True
        self._save_metrics()
        if self._watchdog is not None and self._watchdog.running:
            self._watchdog.stop()
        display_message(
            self.aid.name,
            f"[SHUTDOWN] Animals finalized={len(self._finalized)}. Stopping reactor in 1s..."
        )
        reactor.callLater(1.0, reactor.stop)

    # ------------------------------------------------------------------ #
    # Watchdog
    # ------------------------------------------------------------------ #
    def _watchdog_tick(self):
        """Passive liveness check. Dormant during healthy runs."""
        if self._shutting_down:
            return

        now = time.time()

        # Hard deadline (wall clock) — absolute backstop against any hang.
        if now - self.sim_start_ts > self._t_max:
            self._force_drain(f"hard deadline {self._t_max:.0f}s exceeded")
            return

        if mas_globals.CAPTURE_DONE_TS is None:
            # Capture still running; nothing to drain yet.
            return

        # Refresh progress if any upstream stage still has pending work.
        # (QUEUE_STATS is a hint here, NOT a decision gate.)
        try:
            with mas_globals.QUEUE_STATS_LOCK:
                upstream_busy = (
                    mas_globals.QUEUE_STATS.get("pending_enhance", 0) > 0
                    or mas_globals.QUEUE_STATS.get("pending_eval", 0) > 0
                    or mas_globals.QUEUE_STATS.get("pending_inference", 0) > 0
                )
        except Exception:
            upstream_busy = False
        if upstream_busy:
            self.last_progress_ts = now

        manifest_size = len(CAPTURE_MANIFEST)
        with self._lock:
            finalized_count = len(self._finalized)
        idle = now - self.last_progress_ts

        if finalized_count >= manifest_size and manifest_size > 0:
            # All captured animals finalized. If the normal terminal signal was
            # lost, fire the normal shutdown after a short grace.
            if idle > self._short_grace:
                display_message(
                    self.aid.name,
                    "[WATCHDOG] All manifest animals finalized — completing."
                )
                self._shutdown()
            return

        # Stragglers remain. Force-drain only after a durable idle window
        # (no progress anywhere) so a legitimately slow drain is not cut short.
        if idle > self._long_grace:
            self._force_drain(f"no progress for {idle:.0f}s with {manifest_size - finalized_count} animals pending")

    # ------------------------------------------------------------------ #
    # Inference callbacks
    # ------------------------------------------------------------------ #
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
        self.last_progress_ts = time.time()
        self._try_finalize(animal_id)
        self._decrement_pending_inferences()

    def _on_batch_inference_success(self, result, animal_id: int, total_frames: int, start_ts: str):
        self._total_inferences += 1
        weights = [float(r[0]) for r in result] if result is not None else []

        if animal_id != "?":
            self._predictions.setdefault(animal_id, []).extend(weights)

        final_ts = datetime.now().isoformat()
        self._record_batch_metric(animal_id, total_frames, start_ts, final_ts)
        self._finalize_animal(animal_id)
        self._decrement_pending_inferences()

    def _on_inference_error(self, failure):
        display_message(
            self.aid.name,
            f"[ERROR] Inference failed: {failure.getErrorMessage()}",
        )
        self._decrement_pending_inferences()

    # ------------------------------------------------------------------ #
    # Inference scheduling
    # ------------------------------------------------------------------ #
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
        with self._lock:
            self.pending_inferences += 1
            current = self.pending_inferences
        update_queue_stat("pending_inference", current)
        d = deferToThread(self.inference_adapter.predict, [img])
        d.addCallback(self._on_single_inference_success, payload, start_ts)
        d.addErrback(self._on_inference_error)

    def _process_batch(self, animal_id: int, total_frames: int):
        with self._lock:
            imgs = self.batch_imgs.pop(animal_id, [])
            self.batch_payloads.pop(animal_id, [])
            # _try_finalize already added animal_id to _in_progress
            in_progress = animal_id in self._in_progress

        if not imgs:
            display_message(self.aid.name, f"[WARN] No images available for batch animal {animal_id}")
            if in_progress:
                self._finalize_animal(animal_id)
            return

        display_message(self.aid.name, f"[BATCH INFERENCE] Running full network on {len(imgs)} frames for animal {animal_id}")
        start_ts = datetime.now().isoformat()
        with self._lock:
            self.pending_inferences += 1
            current = self.pending_inferences
        update_queue_stat("pending_inference", current)
        d = deferToThread(self.inference_adapter.predict, imgs)
        d.addCallback(self._on_batch_inference_success, animal_id, total_frames, start_ts)
        d.addErrback(self._on_inference_error)

    # ------------------------------------------------------------------ #
    # Message handling
    # ------------------------------------------------------------------ #
    def react(self, message):
        super().react(message)
        if message.performative != ACLMessage.INFORM:
            return

        # Any inbound message is progress (the pipeline is alive).
        self.last_progress_ts = time.time()

        if message.ontology == "pipeline-complete":
            self._flush_and_shutdown()
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

                self._try_finalize(animal_id, total_frames)
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
            self._try_finalize(animal_id)

    # ------------------------------------------------------------------ #
    # Idempotent finalization gate (unifies the two trigger paths)
    # ------------------------------------------------------------------ #
    def _try_finalize(self, animal_id, total_frames: int = 0):
        """Single atomic gate for per-animal finalization. Prevents the two
        trigger paths (batch-ready and frame-selected) from both scheduling
        _process_batch for the same animal. At 1 FPS each animal finalizes
        exactly once, so this is a no-op there."""
        try:
            animal_id = int(animal_id)
        except (TypeError, ValueError):
            return

        with self._lock:
            if animal_id in self._finalized or animal_id in self._in_progress:
                return
            self._in_progress.add(animal_id)
            expected = self.expected_counts.get(animal_id)
            if self.mode == "single":
                received = len(self._predictions.get(animal_id, []))
            else:
                received = len(self.batch_imgs.get(animal_id, []))

        if expected is None:
            # expected count not yet known (batch-ready lost/delayed) — wait.
            with self._lock:
                self._in_progress.discard(animal_id)
            return

        if received >= expected:
            with self._lock:
                self.expected_counts.pop(animal_id, None)
            if self.mode == "single":
                self._finalize_animal(animal_id)
            else:
                # total_of_images for the batch metric key
                animal_entry = self.metrics['animals'].get(animal_id) or self.metrics['animals'].get(str(animal_id))
                tf = total_frames or (animal_entry.get('total_of_images', 0) if animal_entry else 0)
                self._process_batch(animal_id, tf)
        else:
            with self._lock:
                self._in_progress.discard(animal_id)

    def on_start(self):
        super().on_start()
        display_message(self.aid.name, "PredictWeightAgent started. Loading AI model in background...")
        self.metrics['load_model_start'] = datetime.now().isoformat()

        # Load model in a background thread to avoid blocking the reactor
        d = deferToThread(self.inference_adapter.load_model)
        d.addCallback(self._on_model_loaded)
        d.addErrback(self._on_model_load_error)

        # Start the passive termination watchdog. It runs regardless of model
        # load state and is dormant during healthy operation.
        self.sim_start_ts = time.time()
        self.last_progress_ts = time.time()
        self._watchdog = LoopingCall(self._watchdog_tick)
        self._watchdog.start(self._watchdog_interval, now=False)

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

    def _on_model_load_error(self, failure):
        display_message(
            self.aid.name,
            f"[ERROR] Failed to load model: {failure.getErrorMessage()}",
        )

    def _save_metrics(self):
        reports_dir = f"infra/reports/{self.pid}"
        os.makedirs(reports_dir, exist_ok=True)
        with open(os.path.join(reports_dir, "metrics.json"), "w") as f:
            json.dump(self.metrics, f, indent=4)
        display_message(self.aid.name, f"[METRICS] Saved exactly identical JSON to {reports_dir}/metrics.json")

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
