"""Benchmark do ENHANCEMENT (DataEnhance real).

medição principal (total_stage) = tempo da função REAL DataEnhance.run(img),
do recebimento da matriz depth até o tensor pronto para o preditor (§4.2).
Nenhuma reimplementação.

Decomposição opcional por sub-operação (--decompose-enhancer): quando ativada,
roda a MESMA sequência dos 4 transforms reais (enh.transfs) com timers aninhados.
É secundária e não substitui o total_stage real; por padrão fica desligada para
não inflar a medição principal com overhead de instrumentação.
"""

from __future__ import annotations

import time
import numpy as np

from .base import BenchmarkBase


class EnhancerBenchmark(BenchmarkBase):
    name = "enhancer"

    def __init__(self, pool, order, warmup, iterations, seed, monitor=None,
                 decompose: bool = False):
        super().__init__(warmup, iterations, seed, monitor)
        self.pool = pool
        self.order = order
        self.decompose = decompose
        self.enh = None

    def setup(self):
        from domain.modules.data_enhance import DataEnhance
        self.enh = DataEnhance()  # mesmos transforms do pipeline
        if self.pool:
            self.input_shape = tuple(np.asarray(self.pool[0]["img"]).shape)
            self.input_dtype = str(np.asarray(self.pool[0]["img"]).dtype)
        else:
            self.input_shape, self.input_dtype = None, None

    def _make_ctx(self, pool_idx: int, iteration: int) -> dict:
        e = self.pool[pool_idx]
        return {"iteration": iteration, "img": e["img"],
                "image_id": e["image_id"], "animal_id": e["tag"]}

    # ------------------------------------------------------------------ #
    def _run_one(self, ctx: dict) -> dict:
        img = ctx["img"]
        enh = self.enh

        if not self.decompose:
            # medição principal fiel: só um timer em volta do run() real.
            t0 = time.perf_counter_ns()
            out = enh.run(img)
            t1 = time.perf_counter_ns()
            total = t1 - t0
            per = {}
        else:
            # decomposição secundária: mesmos transforms reais, timers aninhados.
            cur = img
            t0 = time.perf_counter_ns()
            per = {}
            names = ["noise_removal", "adjust_scale", "replicate", "resize_pad"]
            for name, trf in zip(names, enh.transfs):
                ts = time.perf_counter_ns()
                cur = trf.transform(cur)
                te = time.perf_counter_ns()
                per[name] = te - ts
            out = cur
            t1 = time.perf_counter_ns()
            total = t1 - t0

        out_arr = np.asarray(out)
        row = {
            "iteration": ctx["iteration"],
            "image_id": ctx["image_id"],
            "animal_id": ctx["animal_id"],
            "total_stage_ns": total,
            "total_stage_ms": total / 1e6,
            "input_shape": str(self.input_shape),
            "input_dtype": str(self.input_dtype),
            "output_shape": str(out_arr.shape),
            "output_dtype": str(out_arr.dtype),
            "output_min": float(out_arr.min()),
            "output_max": float(out_arr.max()),
            "output_mean": float(out_arr.mean()),
            "output_std": float(out_arr.std()),
            "timestamp_monotonic_ns": ctx.get("timestamp_monotonic_ns"),
            "timestamp_utc": ctx.get("timestamp_utc"),
        }
        for n in ("noise_removal", "adjust_scale", "replicate", "resize_pad"):
            v = per.get(n)
            row[f"{n}_ns"] = ("" if v is None else v)
            row[f"{n}_ms"] = ("" if v is None else v / 1e6)
        return row

    def _validate(self, result, ctx):
        try:
            out_shape = eval(result["output_shape"])  # noqa: S307 (trusted tuple literal)
        except Exception:
            return False, "output_shape não parseável"
        if tuple(out_shape) != (300, 300, 3):
            return False, f"output_shape inesperado: {out_shape}"
        if result["output_dtype"] != "float32":
            return False, f"output_dtype inesperado: {result['output_dtype']}"
        for k in ("output_min", "output_max", "output_mean", "output_std"):
            if not np.isfinite(result[k]):
                return False, f"{k} não finito (NaN/inf)"
        return True, ""
