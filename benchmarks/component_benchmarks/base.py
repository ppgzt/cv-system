"""Classe base dos benchmarks de componentes.

Disciplina comum a todos (§5, §10, §11):
- warm-up SEPARADO das medições válidas (nunca misturado);
- loop de medição só conta uma medição VÁLIDA (falhas vão p/ failures.csv e
  não substituem silenciosamente);
- a região cronometrada fica inteiramente dentro de `_run_one`; nada de I/O,
  print, JSON, coleta de CPU/tempo dentro dela;
- timestamps monotônicos por iteração (fora da região cronometrada) p/ alinhar
  com o monitor de sistema nas correlações.
"""

from __future__ import annotations

import time


class BenchmarkBase:
    name: str = "base"

    def __init__(self, warmup: int, iterations: int, seed: int,
                 monitor=None, record_warmup: bool = True):
        self.warmup = warmup
        self.iterations = iterations
        self.seed = seed
        self.monitor = monitor
        self.record_warmup = record_warmup

        self.measurements: list[dict] = []
        self.warmup_rows: list[dict] = []
        self.failures: list[dict] = []
        self.correctness: dict = {}
        self.order: list[int] = []
        self.pool: list[dict] = []

        self._cur_component = self.name

    # ------------------------------------------------------------------ #
    # Componentes devem implementar:
    # ------------------------------------------------------------------ #
    def setup(self):
        """Carrega modelo real (mesmos args do pipeline) e monta pool/ordem."""
        raise NotImplementedError

    def _make_ctx(self, pool_idx: int, iteration: int) -> dict:
        raise NotImplementedError

    def _run_one(self, ctx: dict) -> dict:
        """Executa a região cronometrada real e devolve dict com durações."""
        raise NotImplementedError

    def _validate(self, result: dict, ctx: dict) -> tuple[bool, str]:
        return True, ""

    def _verify_correctness(self) -> dict:
        """Comparação opcional do caminho instrumentado vs função real."""
        return {}

    # ------------------------------------------------------------------ #
    @property
    def component_label(self) -> str:
        return self._cur_component

    def _record_failure(self, ctx: dict, reason: str, kind: str = "runtime"):
        self.failures.append({
            "component": self.name,
            "iteration": ctx.get("iteration"),
            "image_id": ctx.get("image_id"),
            "animal_id": ctx.get("animal_id"),
            "exception_type": kind,
            "message": reason,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })

    # ------------------------------------------------------------------ #
    def run(self) -> dict:
        """Executa warm-up + N medições válidas. Devolve um sumário da execução."""
        self.correctness = self._verify_correctness()

        # --- Warm-up (descartável, mesmo caminho das medições) ----------
        self._cur_component = f"{self.name}__warmup"
        for w in range(max(0, self.warmup)):
            idx = self.order[w % len(self.order)] if self.order else 0
            ctx = self._make_ctx(idx, w + 1)
            try:
                res = self._run_one(ctx)
                if self.record_warmup:
                    self.warmup_rows.append(res)
            except Exception as e:  # noqa: BLE001
                # Falha no warm-up é registrada mas não aborta.
                self._record_failure(ctx, f"warmup exception: {e!r}",
                                     kind="warmup")

        # --- Medições válidas -------------------------------------------
        self._cur_component = self.name
        if self.monitor is not None:
            self.monitor.start()

        max_attempts = self.iterations * 5  # safety: evita loop infinito
        attempt = 0
        while len(self.measurements) < self.iterations and attempt < max_attempts:
            idx = self.order[attempt % len(self.order)] if self.order else 0
            iteration = len(self.measurements) + 1
            ctx = self._make_ctx(idx, iteration)
            attempt += 1

            # timestamp monotônico FORA da região cronometrada (alinhamento)
            ctx["timestamp_monotonic_ns"] = time.monotonic_ns()
            ctx["timestamp_utc"] = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime())

            try:
                res = self._run_one(ctx)
                ok, reason = self._validate(res, ctx)
                if not ok:
                    self._record_failure(ctx, reason, kind="validation")
                    continue
            except Exception as e:  # noqa: BLE001
                self._record_failure(ctx, f"runtime exception: {e!r}",
                                     kind=type(e).__name__)
                continue
            self.measurements.append(res)

        if self.monitor is not None:
            self.monitor.stop()

        return {
            "component": self.name,
            "requested": self.iterations,
            "completed": len(self.measurements),
            "valid": len(self.measurements),
            "failures": len(self.failures),
            "warmup": self.warmup,
            "attempts": attempt,
        }
