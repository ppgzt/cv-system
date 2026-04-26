"""RAM Monitor — cópia refatorada do baseline para uso no MAS.

Mesma biblioteca (psutil), mesma chamada, mesmo intervalo, mesmo CSV header.
Diferenças intencionais em relação ao baseline (infra/profiling/agents.py):

- Thread-safe explícito (threading.Lock) para acesso concorrente aos dados
- Método get_latest() para leitura segura da última métrica
- Suporte a reports_dir customizável (não hardcoded em "infra/reports")

Paridade garantida:
- psutil.virtual_memory() — identico
- CSV header: ["timestamp", "total", "available", "used", "percent", "free",
               "active", "inactive", "buffers", "cached"] — identico
- Intervalo: 1s — identico
"""

import csv
import threading
import time
import sys
from datetime import datetime
from pathlib import Path

import psutil


class RAMMonitor(threading.Thread):
    """Monitor de memória RAM.

    Coleta métricas detalhadas de memória a cada 1 segundo e acumula
    os dados em memória. No stop(), escreve mem.csv no mesmo
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
        while self.running:
            mem = psutil.virtual_memory()

            # Compatibilidade macOS vs Linux (Raspberry Pi)
            if sys.platform == 'darwin':
                line = [
                    datetime.now().isoformat(),
                    mem.total,
                    mem.available,
                    mem.used,
                    mem.percent,
                    mem.free,
                    getattr(mem, 'active', 0),
                    getattr(mem, 'inactive', 0),
                    0, # buffers (não disponível no mac)
                    0, # cached (não disponível no mac)
                ]
            else:
                # Código original do baseline para Raspberry Pi/Linux
                line = [
                    datetime.now().isoformat(),
                    mem.total,
                    mem.available,
                    mem.used,
                    mem.percent,
                    mem.free,
                    mem.active,
                    mem.inactive,
                    mem.buffers,
                    mem.cached,
                ]

            with self._lock:
                self._data.append(line)
            print(f"RAM: {mem}")

            time.sleep(1)

    def stop(self):
        self.running = False
        self._write_csv()

    def get_latest(self) -> list | None:
        """Retorna a última leitura de RAM
        [timestamp, total, available, used, percent, free, active, inactive, buffers, cached]
        ou None se nenhum dado foi coletado ainda.
        """
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

        header = [
            "timestamp",
            "total",
            "available",
            "used",
            "percent",
            "free",
            "active",
            "inactive",
            "buffers",
            "cached",
        ]

        output_dir = self.reports_dir / self.pid
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "mem.csv", mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(data_copy)
