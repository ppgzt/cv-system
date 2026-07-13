"""Benchmark do SELETOR de frames (FrameSelection real, TFLite MobileNetV2).

Reuso integral do objeto real: mesmo model_path, num_threads=2, threshold=0.5,
mesma preprocess_fn (tf.function) e mesmo interpretador alocado uma vez.

Para decompor em total_stage / tflite_total / invoke (exigido em §4.1.A e
§4.1.B), executamos a SEQUÊNCIA EXATA do predict() — mesmos métodos reais
(_to_single_channel, _preprocess_fn, _input_tensor, set_tensor, invoke,
get_tensor), dentro do mesmo lock — apenas intercalando perf_counter_ns().
Nada é reimplementado. A equivalência com predict() é verificada em
_verify_correctness().
"""

from __future__ import annotations

import time
import numpy as np

from .base import BenchmarkBase

SELECTOR_DEFAULTS = {
    "threshold": 0.5,
    "num_threads": 2,
}


class SelectorBenchmark(BenchmarkBase):
    name = "selector"

    def __init__(self, model_path: str, pool, order, warmup, iterations, seed,
                 monitor=None, threshold: float = 0.5, num_threads: int = 2):
        super().__init__(warmup, iterations, seed, monitor)
        self.model_path = model_path
        self.threshold = threshold
        self.num_threads = num_threads
        self.pool = pool
        self.order = order
        self.sel = None

    # ------------------------------------------------------------------ #
    def setup(self):
        # Mesma construção do FrameSelectionAdapter.load_model() do pipeline.
        from domain.modules.frame_selection import FrameSelection
        self.sel = FrameSelection(
            model_path=self.model_path,
            threshold=self.threshold,
            num_threads=self.num_threads,
            suitable_window=None,
        )
        in_det = self.sel._interpreter.get_input_details()
        self.input_shape = tuple(int(x) for x in in_det[0]["shape"])
        self.input_dtype = str(in_det[0]["dtype"])

    def _make_ctx(self, pool_idx: int, iteration: int) -> dict:
        e = self.pool[pool_idx]
        return {
            "iteration": iteration,
            "img": e["img"],
            "image_id": e["image_id"],
            "animal_id": e["tag"],
            "true_class": e["true_class"],
        }

    # ------------------------------------------------------------------ #
    def _run_one(self, ctx: dict) -> dict:
        img = ctx["img"]
        sel = self.sel
        with sel._lock:
            t0 = time.perf_counter_ns()
            single = sel._to_single_channel(img)
            processed = sel._preprocess_fn(single)
            sel._input_tensor[0] = processed
            t_set0 = time.perf_counter_ns()
            sel._interpreter.set_tensor(sel._input_index, sel._input_tensor)
            t_inv0 = time.perf_counter_ns()
            sel._interpreter.invoke()
            t_inv1 = time.perf_counter_ns()
            probs = sel._interpreter.get_tensor(sel._output_index)  # [1,4]
            t_set1 = time.perf_counter_ns()
            prob0 = float(probs[0][0])
            suited = prob0 > sel.threshold
            t1 = time.perf_counter_ns()
        # pós-cálculo FORA da região cronometrada (validação/log)
        probs_list = [float(x) for x in probs[0]]

        total = t1 - t0
        tflite = t_set1 - t_set0
        invoke = t_inv1 - t_inv0
        return {
            "iteration": ctx["iteration"],
            "image_id": ctx["image_id"],
            "animal_id": ctx["animal_id"],
            "true_class": ctx["true_class"],
            "predicted_class": "suited" if suited else "not_suited",
            "predicted_argmax": int(np.argmax(probs_list)),
            "score": prob0,
            "total_stage_ns": total,
            "total_stage_ms": total / 1e6,
            "tflite_total_ns": tflite,
            "tflite_total_ms": tflite / 1e6,
            "invoke_ns": invoke,
            "invoke_ms": invoke / 1e6,
            "input_shape": str(self.input_shape),
            "input_dtype": str(self.input_dtype),
            "timestamp_monotonic_ns": ctx.get("timestamp_monotonic_ns"),
            "timestamp_utc": ctx.get("timestamp_utc"),
        }

    def _validate(self, result, ctx):
        score = result.get("score")
        if score is None or not np.isfinite(score):
            return False, "score não é finito"
        if not (0.0 <= score <= 1.0):
            return False, f"score fora de [0,1]: {score}"
        return True, ""

    # ------------------------------------------------------------------ #
    def _verify_correctness(self) -> dict:
        """Confirma que o caminho instrumentado == predict() real."""
        if not self.pool:
            return {"checked": False, "reason": "pool vazio"}
        try:
            img = self.pool[0]["img"]
            real = float(self.sel.predict(img))
            ctx = self._make_ctx(0, 0)
            instr = self._run_one(ctx)
            diff = abs(real - instr["score"])
            return {"checked": True, "predict_prob": real,
                    "instrumented_prob": instr["score"],
                    "abs_diff": diff,
                    "matches_within_1e-6": bool(diff <= 1e-6)}
        except Exception as e:  # noqa: BLE001
            return {"checked": False, "error": repr(e)}
