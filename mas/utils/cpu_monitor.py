"""CPU Monitor — cópia refatorada do baseline para uso no MAS.

Mesma biblioteca (psutil), mesma chamada, mesmo intervalo, mesmo CSV header.
Diferenças intencionais em relação ao baseline (infra/profiling/agents.py):

- Thread-safe explícito (threading.Lock) para acesso concorrente aos dados
- Método get_latest() para leitura segura da última métrica
- Suporte a reports_dir customizável (não hardcoded em "infra/reports")

Paridade garantida:
- psutil.cpu_percent(percpu=True, interval=1) — identico
- CSV header: ["timestamp", "cpu_core_0", ...] — identico
- Intervalo: 1s — identico
"""

import csv
import threading
import time
from datetime import datetime
from pathlib import Path

import psutil


class CPUMonitor(threading.Thread):
    """Monitor de CPU per-core.

    Coleta a utilização de cada core a cada 1 segundo e acumula
    os dados em memória. No stop(), escreve cpu.csv no mesmo
    formato do baseline para garantir comparabilidade.
    """

    def __init__(self, pid: str, reports_dir: str = "infra/reports"):
        super().__init__()
        self.pid = pid
        self.reports_dir = Path(reports_dir)
        self.running = True
        self._data: list[list] = []
        self._lock = threading.Lock()
        self.daemon = True

    def run(self):
        # The first call returns 0.0, so we ignore it
        psutil.cpu_percent(percpu=True)
        while self.running:
            cpu_percent = psutil.cpu_percent(percpu=True, interval=1)
            row = [datetime.now().isoformat()] + cpu_percent
            with self._lock:
                self._data.append(row)
            print(f"CPU: {cpu_percent}")

    def stop(self):
        self.running = False
        self._write_csv()

    def get_latest(self) -> list | None:
        """Retorna a última leitura de CPU [timestamp, core_0, ...] ou None."""
        with self._lock:
            if not self._data:
                return None
            return list(self._data[-1])

    def get_all_data(self) -> list[list]:
        """Retorna cópia de todos os dados acumulados."""
        with self._lock:
            return [list(row) for row in self._data]

    def _write_csv(self):
        with self._lock:
            data_copy = [list(row) for row in self._data]

        if not data_copy:
            return

        num_cores = len(data_copy[0]) - 1
        header = ["timestamp"] + [f"cpu_core_{i}" for i in range(num_cores)]

        output_dir = self.reports_dir / self.pid
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "cpu.csv", mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(data_copy)
