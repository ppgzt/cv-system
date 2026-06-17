"""Predict Weight Agent — runs model inference via InferenceAdapter.

Receives enhanced suitable frames from DataEnhanceAgent (ontology
"frame-enhanced"), pops the enhanced image from FRAME_BUFFER to prevent RAM
accumulation, and runs TFLite inference in a delegated thread (deferToThread)
so the Twisted event loop is never blocked.

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
from twisted.internet.defer import DeferredSemaphore
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
        verbose: bool = False,
        fps: float = None,
    ):
        super().__init__(aid=aid, debug=debug)
        self.inference_adapter = inference_adapter
        self.mode = mode
        self.pid = pid
        self.herd_size = herd_size
        self.capture_agent_aid = capture_agent_aid
        self.verbose = verbose
        self.fps = fps

        self._predictions: dict = {}
        self._total_inferences = 0
        self._finalized: set = set()
        self._labels: dict = {}
        self._lock = threading.Lock()
        self._inference_semaphore = DeferredSemaphore(1)

        # Terminação capture-driven (pipeline data-driven): o DatasetCaptureAgent
        # avisa IN-PROCESS quando a captura de cada animal acaba (ele sabe o
        # exato momento — último frame — pois tem o índice inteiro em memória).
        # Finalizamos deterministicamente quando a captura acabou E as inferências
        # em voo drenaram. Robusto a perdas no canal FIPA e a erros de inferência:
        # o peso final vira a média dos pesos que chegaram (0 se nenhum).
        self._capture_done: set = set()
        self._in_flight: dict = {}          # animal_id -> inferências pendentes
        self._batch_started: set = set()    # animal_id -> batch já disparado
        
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

    def _record_single_metric(self, animal_id, frame_index, start_ts, final_ts, label=None):
        if animal_id not in self.metrics['animals']:
            return
        entry = {
            'weight_prediction_start': start_ts,
            'weight_prediction_final': final_ts
        }
        if label is not None:
            entry['label'] = label
        self.metrics['animals'][animal_id]['imgs'][str(frame_index)] = entry

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

        # Idempotente: a finalização pode ser disparada pelo caminho
        # capture-driven (_maybe_finalize) e pelo caminho FIPA legado
        # (_check_batch_*). Garante uma única vez por animal.
        if animal_id in self._finalized:
            return

        weights = self._predictions.get(animal_id, [])
        predicted_weight = float(np.mean(weights)) if weights else 0.0

        try:
            from mas.utils.report_collector import ReportCollector
            ReportCollector().record_final_prediction(animal_id, predicted_weight)
        except Exception as e:
            display_message(self.aid.name, f"[REPORT-ERROR] record_final_prediction failed: {e}")

        if animal_id in self.metrics['animals']:
            self.metrics['animals'][animal_id]['weight_prediction_final'] = datetime.now().isoformat()

            if self.verbose:
                labels = self._labels.get(animal_id, [])
                from collections import Counter
                lbl_counts = dict(Counter(labels))
                display_message(
                    self.aid.name,
                    f"[FINAL] Animal {animal_id}: n_suitable={len(weights)} "
                    f"| labels_dos_suitable={lbl_counts} "
                    f"| peso_medio={predicted_weight:.4f} kg",
                )
            else:
                display_message(
                    self.aid.name,
                    f"[FINAL] Animal {animal_id} completed. Mean weight: {predicted_weight:.4f} kg"
                )

        # Tag-string ids (dataset capture) ou int ids (antigo): conta
        # finalizações até atingir o total de animais do rebanho.
        self._finalized.add(animal_id)
        if len(self._finalized) >= self.herd_size:
            self._save_metrics()
            display_message(self.aid.name, "[SHUTDOWN] All animals evaluated. Stopping reactor in 1s...")
            reactor.callLater(1.0, reactor.stop)

    def _save_metrics(self):
        reports_dir = f"infra/reports/{self.pid}"
        os.makedirs(reports_dir, exist_ok=True)
        with open(os.path.join(reports_dir, "metrics.json"), "w") as f:
            json.dump(self.metrics, f, indent=4)
        display_message(self.aid.name, f"[METRICS] Saved exactly identical JSON to {reports_dir}/metrics.json")
        
        try:
            from mas.utils.report_collector import ReportCollector
            ReportCollector().generate_report(reports_dir, self.mode, self.fps)
            display_message(self.aid.name, f"[REPORT] Saved execution report to {reports_dir}/report.md")
        except Exception as e:
            display_message(self.aid.name, f"[REPORT-ERROR] generate_report failed: {e}")

    # ------------------------------------------------------------------ #
    # Terminação capture-driven (pipeline data-driven por tag).
    # ------------------------------------------------------------------ #
    def notify_capture_done(self, animal_id, total_frames=None,
                            first_capture=None, last_capture=None):
        """Chamada IN-PROCESS pelo DatasetCaptureAgent quando a captura do
        animal acaba (último frame capturado). É o gatilho deterministicamente
        confiável para o cálculo do peso final — não depende do canal FIPA
        (que é fire-and-forget e perde mensações sob carga).

        Single: tenta finalizar assim que as inferências em voo drenarem.
        Batch : dispara o batch sobre os frames suitable acumulados.
        """
        self._capture_done.add(animal_id)

        with self._lock:
            entry = self.metrics['animals'].setdefault(animal_id, {'imgs': {}})
            if total_frames is not None:
                entry['total_of_images'] = total_frames
            if first_capture:
                entry['first_image_capture_time'] = first_capture
            if last_capture:
                entry['last_image_capture_time'] = last_capture
            # fallback; o batch-ready (se chegar via FIPA) sobrescreve com o
            # valor autoritativo vindo do FrameSelectionAgent.
            entry.setdefault('suitable_images', len(self._predictions.get(animal_id, [])))

        display_message(
            self.aid.name,
            f"[CAPTURE-DONE] Animal {animal_id}: captura concluída "
            f"(total={total_frames}). Modo={self.mode}.",
        )

        if self.mode == "batch":
            self._process_batch(animal_id)
        else:
            self._maybe_finalize(animal_id)

    def _maybe_finalize(self, animal_id):
        """Finaliza o animal se: captura concluída E nenhuma inferência em voo."""
        if animal_id in self._finalized:
            return
        if animal_id not in self._capture_done:
            return
        if self._in_flight.get(animal_id, 0) > 0:
            return
        self._finalize_animal(animal_id)

    def _on_single_inference_success(self, result, payload: dict, start_ts: str):
        self._total_inferences += 1
        animal_id = payload.get("animal_id", "?")
        frame_id = payload.get("frame_id", "?")
        frame_index = payload.get("frame_index", self._total_inferences)
        depth_filename = payload.get("depth_filename")
        
        weight = float(result[0][0]) if result is not None else None

        if animal_id != "?" and weight is not None:
            self._predictions.setdefault(animal_id, []).append(weight)
            if self.verbose and payload.get("label") is not None:
                self._labels.setdefault(animal_id, []).append(payload.get("label"))

        try:
            from mas.utils.report_collector import ReportCollector
            ReportCollector().record_prediction(animal_id, depth_filename, weight)
        except Exception as e:
            display_message(self.aid.name, f"[REPORT-ERROR] record_prediction failed: {e}")

        final_ts = datetime.now().isoformat()
        self._record_single_metric(animal_id, frame_index, start_ts, final_ts, payload.get("label"))

        label_str = f" label={payload.get('label')}" if self.verbose else ""
        display_message(
            self.aid.name,
            f"[PREDICTION] animal_id={animal_id} frame_id={frame_id}{label_str} weight={weight:.4f} kg",
        )

        if animal_id != "?":
            # drena exatamente uma unidade (dec no fim => se o callback raisar
            # antes, o errback é quem dec). Veja _on_inference_error.
            self._in_flight[animal_id] = max(0, self._in_flight.get(animal_id, 0) - 1)
        self._check_batch_sync(animal_id)        # caminho FIPA legado (backward-compat)
        if animal_id != "?":
            self._maybe_finalize(animal_id)      # caminho capture-driven (robusto)

    def _on_batch_inference_success(self, result, animal_id: int, total_frames: int, start_ts: str):
        self._total_inferences += 1
        weights = [float(r[0]) for r in result] if result is not None else []
        
        if animal_id != "?":
            self._predictions.setdefault(animal_id, []).extend(weights)
            
        final_ts = datetime.now().isoformat()
        self._record_batch_metric(animal_id, total_frames, start_ts, final_ts)
        self._finalize_animal(animal_id)

    def _on_inference_error(self, failure, animal_id=None):
        # Para inferência single: animal_id é passada para drenar o _in_flight e
        # tentar finalizar mesmo em caso de erro (não trava mais o animal).
        # Para load_model/batch: animal_id=None (apenas loga).
        if animal_id is not None and animal_id != "?":
            self._in_flight[animal_id] = max(0, self._in_flight.get(animal_id, 0) - 1)
        display_message(
            self.aid.name,
            f"[ERROR] Inference failed: {failure.getErrorMessage()}",
        )
        if animal_id is not None and animal_id != "?":
            self._maybe_finalize(animal_id)

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

        # capture-driven: conta a inferência como "em voo" para sabermos quando
        # drenar e finalizar o animal.
        self._in_flight[animal_id] = self._in_flight.get(animal_id, 0) + 1

        start_ts = datetime.now().isoformat()
        d = self._inference_semaphore.run(deferToThread, self.inference_adapter.predict, [img])
        d.addCallback(self._on_single_inference_success, payload, start_ts)
        d.addErrback(self._on_inference_error, animal_id)
        
        # Pseudo finalize evaluation if single (cannot guarantee exact last frame easily unless synced)
        # But single stream doesn't strictly log per-animal final metric properly until gap.

    def _process_batch(self, animal_id: int, total_frames: int = 0):
        # Guarda de idempotência: o batch pode ser disparado pelo caminho
        # capture-driven (notify_capture_done) e pelo FIPA batch-ready.
        if animal_id in self._batch_started:
            return
        self._batch_started.add(animal_id)

        with self._lock:
            imgs = self.batch_imgs.pop(animal_id, [])
            payloads = self.batch_payloads.pop(animal_id, [])
            if self.verbose:
                labels = [p.get("label") for p in payloads if p.get("label") is not None]
                if labels:
                    self._labels.setdefault(animal_id, []).extend(labels)

        if not imgs:
            display_message(self.aid.name, f"[WARN] No images available for batch animal {animal_id}")
            self._finalize_animal(animal_id)
            return

        display_message(self.aid.name, f"[BATCH INFERENCE] Running full network on {len(imgs)} frames for animal {animal_id}")
        start_ts = datetime.now().isoformat()
        d = self._inference_semaphore.run(deferToThread, self.inference_adapter.predict, imgs)
        d.addCallback(self._on_batch_inference_success, animal_id, total_frames, start_ts)
        d.addErrback(self._on_inference_error, animal_id)

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
            
        if message.ontology != "frame-enhanced":
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

    def _check_batch_ready_custom(self, animal_id, total_frames: int = 0):
        # Specific check for batch mode triggered by the signal.
        # animal_id pode ser tag-string (dataset capture) ou int (antigo);
        # usamos direto como chave (qualquer hashable serve).

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

    def _check_batch_sync(self, animal_id):
        # Loop check call (from frames). animal_id usado direto como chave.

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
