"""Monitor de sistema (thread daemon, 1 Hz).

A coleta roda numa thread SEPARADA — nunca dentro da região cronometrada
(§9/§10). O impacto é constante entre componentes (mesmo intervalo, mesmas
leitura), então não enviesa a comparação relativa. A frequência default é 1 Hz,
configurável.

Cada amostra vira uma linha de system_monitor.csv com a coluna `component`
indicando qual benchmark estava em execução no momento.
"""

from __future__ import annotations

import threading
import time

import psutil

from . import environment


class SystemMonitor(threading.Thread):
    """Thread leve que amostra CPU/memória/tempo/freq/throttle a cada `interval`s."""

    def __init__(self, component_fn, interval: float = 1.0,
                 pid: int | None = None):
        super().__init__(name="SystemMonitor", daemon=True)
        self._component_fn = component_fn        # () -> str (componente corrente)
        self._interval = interval
        self._pid = pid
        # Não usar o nome ``_stop``: threading.Thread possui internamente um
        # método com esse nome, chamado durante join(). Sobrescrevê-lo com um
        # Event pode causar TypeError ao encerrar a thread.
        self._stop_event = threading.Event()
        self.rows: list[dict] = []
        # warmup do cpu_percent (primeira chamada retorna 0.0; igual ao pipeline)
        try:
            psutil.cpu_percent(percpu=True)
            if self._pid is not None:
                psutil.Process(self._pid).cpu_percent()
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    def _sample(self) -> dict:
        row = {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            "timestamp_monotonic_ns": time.monotonic_ns(),
            "component": self._component_fn(),
        }
        # CPU total (% agregado) e por-processo
        try:
            row["cpu_total_percent"] = round(psutil.cpu_percent(), 2)
        except Exception:
            row["cpu_total_percent"] = None
        try:
            row["cpu_per_core_percent"] = [round(x, 1)
                                           for x in psutil.cpu_percent(percpu=True)]
        except Exception:
            row["cpu_per_core_percent"] = None

        if self._pid is not None:
            try:
                proc = psutil.Process(self._pid)
                row["process_cpu_percent"] = round(proc.cpu_percent(), 2)
                row["process_rss_bytes"] = proc.memory_info().rss
            except Exception:
                row["process_cpu_percent"] = None
                row["process_rss_bytes"] = None
        else:
            row["process_cpu_percent"] = None
            row["process_rss_bytes"] = None

        # Memória disponível no sistema
        try:
            row["mem_available_bytes"] = psutil.virtual_memory().available
        except Exception:
            row["mem_available_bytes"] = None

        # Temperatura, frequência e throttle (Linux/RPi quando disponível)
        row["temperature_celsius"] = environment.read_temp_celsius()
        freq = environment.read_cpu_freq()
        row["cpu_freq_cur_hz"] = freq.get("scaling_cur_freq_hz")
        row["cpu_freq_min_hz"] = freq.get("scaling_min_freq_hz")
        row["cpu_freq_max_hz"] = freq.get("scaling_max_freq_hz")
        row["cpu_governor"] = freq.get("governor")
        thr = environment.read_throttled()
        row["throttled_raw"] = thr.get("raw")
        row["throttled_now"] = thr.get("throttled_now")
        row["undervoltage_now"] = thr.get("undervoltage_now")
        row["freq_capped_now"] = thr.get("freq_capped_now")
        return row

    # ------------------------------------------------------------------ #
    def run(self):
        # Primeira amostra imediata; demais a cada interval.
        while not self._stop_event.is_set():
            try:
                self.rows.append(self._sample())
            except Exception:
                pass
            self._stop_event.wait(self._interval)

    def stop(self, join: bool = True, timeout: float = 5.0):
        self._stop_event.set()
        if join:
            # amostra final p/ registrar o estado de fechamento
            try:
                self.rows.append(self._sample())
            except Exception:
                pass
            self.join(timeout)

    @staticmethod
    def empty_row(component: str) -> dict:
        """Linha placeholder quando o monitor está desligado."""
        return {"component": component, "timestamp_monotonic_ns": None}
