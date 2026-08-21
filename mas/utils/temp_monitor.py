"""Temperature Monitor — collects CPU/SOC temperature metrics for MAS.

Similar to CPUMonitor and RAMMonitor, this runs in a dedicated thread,
collecting the system temperature every 1 second and saving it to temp.csv.
"""

import csv
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path


def read_temperature() -> float | None:
    # 1. Try psutil.sensors_temperatures() if available
    try:
        import psutil
        if hasattr(psutil, "sensors_temperatures"):
            temps = psutil.sensors_temperatures()
            # Try common keys for CPU/SOC thermal zones
            for key in ["cpu_thermal", "cpu-thermal", "soc_thermal", "cpu"]:
                if key in temps and temps[key]:
                    return float(temps[key][0].current)
            # Fallback to the first available sensor if the dict is not empty
            for key, sensors in temps.items():
                if sensors:
                    return float(sensors[0].current)
    except Exception:
        pass

    # 2. Try Linux sysfs /sys/class/thermal/thermal_zone0/temp (typical for Raspberry Pi)
    try:
        temp_path = "/sys/class/thermal/thermal_zone0/temp"
        if os.path.exists(temp_path):
            with open(temp_path, "r") as f:
                val = f.read().strip()
                return float(val) / 1000.0
    except Exception:
        pass

    return None


class TempMonitor(threading.Thread):
    """Monitor de Temperatura do CPU.

    Coleta a temperatura do processador a cada 1 segundo e acumula
    os dados em memória. No stop(), escreve temp.csv no mesmo
    formato de diretório dos outros monitores.
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
            temp = read_temperature()
            # Indisponibilidade é dado operacional, não temperatura zero.
            temp_val = float(temp) if temp is not None else None
            row = [datetime.now().isoformat(), temp_val]
            with self._lock:
                self._data.append(row)
            
            # Print matching the style of CPU/RAM monitors
            print("TEMP: unavailable" if temp_val is None else f"TEMP: {temp_val:.1f} C")
            time.sleep(1)

    def stop(self):
        self.running = False
        self._write_csv()

    def get_latest(self) -> list | None:
        """Retorna a última leitura de temperatura [timestamp, temp] ou None."""
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

        header = ["timestamp", "temperature"]

        output_dir = self.reports_dir / self.pid
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "temp.csv", mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(data_copy)
