"""Benchmark do PREDITOR de peso (PredictWeight real, TFLite EfficientNet-B3).

A entrada do preditor é o tensor JÁ ENHANCEADO. Por isso, no setup(), cada
imagem suited do pool é pré-processada UMA VEZ pela DataEnhance real (fora de
qualquer cronômetro) — exatamente como chega ao estágio no pipeline.

Medições (§4.3):
- total_stage: do tensor enhanced até o peso escalar (asarray + set_tensor +
  invoke + get_tensor + copy + float), no interpretador real;
- tflite_total: set_tensor + invoke + get_tensor;
- invoke: somente interpreter.invoke().
Reuso integral do objeto PredictWeight (mesmo model_path, num_threads=2,
mesmo interpretador alocado uma vez). Equivalência com predict() verificada.
"""

from __future__ import annotations

import time
import numpy as np

from .base import BenchmarkBase

PREDICTOR_DEFAULTS = {"num_threads": 2}


class PredictorBenchmark(BenchmarkBase):
    name = "predictor"

    def __init__(self, model_path: str, pool, order, warmup, iterations, seed,
                 monitor=None, num_threads: int = 2):
        super().__init__(warmup, iterations, seed, monitor)
        self.model_path = model_path
        self.num_threads = num_threads
        self.pool = pool
        self.order = order
        self.pred = None

    def setup(self):
        from domain.modules.predict_weight import PredictWeight
        from domain.modules.data_enhance import DataEnhance

        self.pred = PredictWeight(model_path=self.model_path,
                                  num_threads=self.num_threads)
        in_det = self.pred._interpreter.get_input_details()
        out_det = self.pred._interpreter.get_output_details()
        self.input_shape = tuple(int(x) for x in in_det[0]["shape"])
        self.input_dtype = str(in_det[0]["dtype"])
        self.output_shape = tuple(int(x) for x in out_det[0]["shape"])
        self.output_dtype = str(out_det[0]["dtype"])
        if self.pred._cur_batch != 1:
            raise ValueError(
                "O benchmark do preditor mede exclusivamente o caminho de "
                f"imagem única; batch inicial detectado: {self.pred._cur_batch}")

        # Pré-enhancement real (fora do cronômetro): gera o tensor de entrada
        # do preditor a partir do depth cru suited, igual ao pipeline.
        enh = DataEnhance()
        for e in self.pool:
            e["enhanced_img"] = enh.run(e["img"])

    def _make_ctx(self, pool_idx: int, iteration: int) -> dict:
        e = self.pool[pool_idx]
        return {"iteration": iteration, "img": e["enhanced_img"],
                "image_id": e["image_id"], "animal_id": e["tag"]}

    # ------------------------------------------------------------------ #
    def _run_one(self, ctx: dict) -> dict:
        img = ctx["img"]
        pred = self.pred
        with pred._lock:
            t0 = time.perf_counter_ns()
            arr = np.asarray([img], dtype=np.float32)
            if arr.shape[0] != pred._cur_batch:
                raise RuntimeError(
                    "batch inesperado durante o benchmark: "
                    f"{arr.shape[0]} != {pred._cur_batch}")
            t_set0 = time.perf_counter_ns()
            pred._interpreter.set_tensor(pred._input_index, arr)
            t_inv0 = time.perf_counter_ns()
            pred._interpreter.invoke()
            t_inv1 = time.perf_counter_ns()
            result = pred._interpreter.get_tensor(pred._output_index)
            t_set1 = time.perf_counter_ns()
            out = result.copy()
            weight = float(out[0][0])
            t1 = time.perf_counter_ns()
        total = t1 - t0
        tflite = t_set1 - t_set0
        invoke = t_inv1 - t_inv0
        return {
            "iteration": ctx["iteration"],
            "image_id": ctx["image_id"],
            "animal_id": ctx["animal_id"],
            "prediction": weight,
            "total_stage_ns": total,
            "total_stage_ms": total / 1e6,
            "tflite_total_ns": tflite,
            "tflite_total_ms": tflite / 1e6,
            "invoke_ns": invoke,
            "invoke_ms": invoke / 1e6,
            "input_shape": str(self.input_shape),
            "input_dtype": str(self.input_dtype),
            "output_shape": str(self.output_shape),
            "output_dtype": str(self.output_dtype),
            "timestamp_monotonic_ns": ctx.get("timestamp_monotonic_ns"),
            "timestamp_utc": ctx.get("timestamp_utc"),
        }

    def _validate(self, result, ctx):
        w = result.get("prediction")
        if w is None or not np.isfinite(w):
            return False, f"predição não finita: {w}"
        return True, ""

    def _verify_correctness(self) -> dict:
        if not self.pool:
            return {"checked": False, "reason": "pool vazio"}
        try:
            enhanced = self.pool[0]["enhanced_img"]
            real = float(self.pred.predict([enhanced])[0][0])
            ctx = self._make_ctx(0, 0)
            instr = self._run_one(ctx)
            diff = abs(real - instr["prediction"])
            return {"checked": True, "predict_weight": real,
                    "instrumented_weight": instr["prediction"],
                    "abs_diff": diff,
                    "matches_within_1e-4": bool(diff <= 1e-4)}
        except Exception as e:  # noqa: BLE001
            return {"checked": False, "error": repr(e)}
