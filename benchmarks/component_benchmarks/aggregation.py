"""Benchmark da AGREGAÇÃO final (§4.4).

A agregação real do pipeline é `float(np.mean(weights))` — média aritmética das
predições individuais de um animal (ver ThreadPipeline._predict_loop.finalize).
Custo tipicamente baixo; medimos separadamente para tamanhos representativos do
conjunto de predições: 1, 5, 10, 20, 50.

Os vetores de pesos são sintéticos, gerados com seed fixa (faixa plausível de
kg). A operação cronometrada é EXATAMENTE a do pipeline (np.mean da lista ->
float). Esta métrica NÃO se mistura com as latências dos modelos.
"""

from __future__ import annotations

import time
import numpy as np

from .base import BenchmarkBase

DEFAULT_SIZES = [1, 5, 10, 20, 50]


class AggregationBenchmark(BenchmarkBase):
    name = "aggregation"

    def __init__(self, warmup, iterations, seed, monitor=None,
                 sizes=None, weight_low: float = 20.0,
                 weight_high: float = 80.0):
        super().__init__(warmup, iterations, seed, monitor,
                         record_warmup=False)
        self.sizes = list(sizes or DEFAULT_SIZES)
        self.weight_low = weight_low
        self.weight_high = weight_high

    def setup(self):
        # Gera os vetores de pesos UMA VEZ (seed fixa) — reprodutível.
        rng = np.random.default_rng(self.seed)
        self.weight_sets = {
            s: rng.uniform(self.weight_low, self.weight_high, size=s).tolist()
            for s in self.sizes
        }

    # Override total do loop: varre tamanhos em vez de pool de imagens.
    def run(self) -> dict:
        self.correctness = {}
        total_valid = 0

        # --- Warm-up por tamanho (descartável) -------------------------
        self._cur_component = f"{self.name}__warmup"
        for s in self.sizes:
            data = self.weight_sets[s]
            for _ in range(max(0, self.warmup)):
                try:
                    float(np.mean(data))
                except Exception:
                    pass

        # --- Medições válidas ------------------------------------------
        self._cur_component = self.name
        if self.monitor is not None:
            self.monitor.start()
        for s in self.sizes:
            data = self.weight_sets[s]
            for _ in range(self.iterations):
                ts_mono = time.monotonic_ns()
                ts_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                try:
                    t0 = time.perf_counter_ns()
                    result = float(np.mean(data))
                    t1 = time.perf_counter_ns()
                    if not np.isfinite(result):
                        self._record_failure(
                            {"iteration": total_valid + 1, "image_id": None,
                             "animal_id": None},
                            f"resultado não finito (size={s})", kind="validation")
                        continue
                    elapsed = t1 - t0
                    total_valid += 1
                    self.measurements.append({
                        "iteration": total_valid,
                        "num_predictions": s,
                        "aggregation_ns": elapsed,
                        "aggregation_ms": elapsed / 1e6,
                        "result": result,
                        "timestamp_monotonic_ns": ts_mono,
                        "timestamp_utc": ts_utc,
                    })
                except Exception as e:  # noqa: BLE001
                    self._record_failure(
                        {"iteration": total_valid + 1, "image_id": None,
                         "animal_id": None},
                        f"runtime exception: {e!r}", kind=type(e).__name__)
        if self.monitor is not None:
            self.monitor.stop()

        return {
            "component": self.name,
            "requested": self.iterations * len(self.sizes),
            "completed": total_valid,
            "valid": total_valid,
            "failures": len(self.failures),
            "warmup": self.warmup,
            "sizes": list(self.sizes),
        }

    def _record_failure(self, ctx, reason, kind="runtime"):
        self.failures.append({
            "component": self.name,
            "iteration": ctx.get("iteration"),
            "image_id": ctx.get("image_id"),
            "animal_id": ctx.get("animal_id"),
            "exception_type": kind,
            "message": reason,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                           time.gmtime()),
        })
