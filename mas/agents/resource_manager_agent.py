"""Resource Manager Agent — monitoramento de CPU e RAM via threads dedicadas.

Este agente PADE usa cópias refatoradas dos monitores do baseline
(`mas.utils.cpu_monitor.CPUMonitor` e `mas.utils.ram_monitor.RAMMonitor`)
para garantir paridade total na coleta de métricas entre:

- Pipeline sequencial (baseline) → main.py + threads diretas
- Pipeline multiagente (MAS) → este agente

Diferenças intencionais em relação ao baseline:
- Thread-safe explícito com Lock
- Método get_latest() para leitura segura das últimas métricas
- Suporte a reports_dir customizável

Paridade garantida:
- Mesmas bibliotecas (psutil)
- Mesmas chamadas (cpu_percent, virtual_memory)
- Mesmos intervalos (1s)
- Mesmos CSV headers
- Mesmos paths de output (infra/reports/{pid}/cpu.csv + mem.csv)

O agente também grava as últimas métricas em um blackboard em memória
para uso futuro por outros agentes do MAS.
"""

import time
from datetime import datetime
from pathlib import Path

import mas  # noqa: F401 — side-effect: adiciona mas/ ao sys.path

from pade.core.agent import Agent
from pade.behaviours.protocols import TimedBehaviour
from pade.misc.utility import display_message

# Import monitors from mas.utils (Adapter approach)
from mas.utils.cpu_monitor import CPUMonitor
from mas.utils.ram_monitor import RAMMonitor

# Blackboard Adapter
from mas.adapters.blackboard_adapter import BlackboardAdapter, InMemoryBlackboardAdapter


class _PublishMetricsToBlackboard(TimedBehaviour):
    """TimedBehaviour que publica as últimas métricas no blackboard.

    Executa periodicamente e:
    1. Lê os dados via CPUMonitor.get_latest() e RAMMonitor.get_latest()
    2. Monta um snapshot no mesmo schema da POC mas-edge-vision
    3. Escreve no BlackboardAdapter para uso futuro por outros agentes
    """

    def __init__(
        self,
        agent: "ResourceManagerAgent",
        blackboard: BlackboardAdapter,
        interval_seconds: float,
    ):
        super().__init__(agent, interval_seconds)
        self.blackboard = blackboard
        self._version = 0

    def on_time(self):
        super().on_time()
        self._publish()

    def _publish(self):
        self._version += 1
        snapshot = self._build_snapshot()
        if snapshot is not None:
            self.blackboard.write_metrics(snapshot)

    def _build_snapshot(self) -> dict | None:
        cpu_mon = self.agent.cpu_monitor
        ram_mon = self.agent.ram_monitor

        if cpu_mon is None or ram_mon is None:
            return None

        cpu_latest = cpu_mon.get_latest()
        ram_latest = ram_mon.get_latest()

        if cpu_latest is None or ram_latest is None:
            return None

        now = time.time()

        # CPU: [timestamp, core_0, core_1, ...]
        cpu_cores = [float(v) for v in cpu_latest[1:]]
        cpu_percent = round(sum(cpu_cores) / len(cpu_cores), 2) if cpu_cores else 0.0

        # RAM: [timestamp, total, available, used, percent, free, active, inactive, buffers, cached]
        ram_percent = round(float(ram_latest[4]), 2)
        ram_total = int(ram_latest[1])
        ram_available = int(ram_latest[2])
        ram_used = int(ram_latest[3])
        ram_free = int(ram_latest[5])

        recorded_at_iso = datetime.fromtimestamp(now).isoformat(timespec="microseconds")

        metrics = {
            "timestamp": now,
            "cpu_percent": cpu_percent,
            "ram_percent": ram_percent,
            "cpu_cores": cpu_cores,
            "ram_total": ram_total,
            "ram_available": ram_available,
            "ram_used": ram_used,
            "ram_free": ram_free,
        }

        return {
            "version": self._version,
            "recorded_at": recorded_at_iso,
            "updated_at": recorded_at_iso,
            "metrics": metrics,
        }


class ResourceManagerAgent(Agent):
    """Agente responsável por monitorar recursos do sistema.

    Usa cópias refatoradas dos monitores do baseline com paridade total
    para a comparação baseline vs MAS.

    Output CSV idêntico ao baseline para comparação direta.
    Blackboard em memória para uso futuro por outros agentes.
    """

    def __init__(
        self,
        aid,
        pid: str,
        reports_dir: str = "infra/reports",
        blackboard: BlackboardAdapter | None = None,
        blackboard_interval_s: float = 5.0,
        debug: bool = False,
    ):
        super().__init__(aid=aid, debug=debug)
        self.pid = pid
        self.reports_dir = Path(reports_dir)
        self.cpu_monitor: CPUMonitor | None = None
        self.ram_monitor: RAMMonitor | None = None

        # Blackboard (implementado, não usado na comparação atual)
        self.blackboard = blackboard or InMemoryBlackboardAdapter()
        self._blackboard_publisher: _PublishMetricsToBlackboard | None = None
        self._blackboard_interval_s = blackboard_interval_s

    def on_start(self):
        super().on_start()
        display_message(self.aid.name, "ResourceManagerAgent iniciado.")

        # Criar diretório de reports (mesmo path do baseline)
        report_path = self.reports_dir / self.pid
        report_path.mkdir(parents=True, exist_ok=True)

        # Monitores refatorados do MAS — mesma lógica, mesma psutil, mesmo CSV
        self.cpu_monitor = CPUMonitor(pid=self.pid, reports_dir=str(self.reports_dir))
        self.ram_monitor = RAMMonitor(pid=self.pid, reports_dir=str(self.reports_dir))

        # Warmup do CPU (igual baseline: primeira chamada retorna 0.0)
        import psutil
        psutil.cpu_percent(percpu=True)

        self.cpu_monitor.start()
        self.ram_monitor.start()

        display_message(
            self.aid.name,
            f"Monitores iniciados — CPU e RAM a cada 1s (pid={self.pid})",
        )

        # Blackboard publisher (implementado, uso futuro)
        if self.blackboard is not None:
            self._blackboard_publisher = _PublishMetricsToBlackboard(
                agent=self,
                blackboard=self.blackboard,
                interval_seconds=self._blackboard_interval_s,
            )
            self.behaviours.append(self._blackboard_publisher)

    def stop_monitoring(self):
        """Para os monitores e escreve os CSVs.

        Método chamado explicitamente pelo launcher (MASStrategy)
        ou via signal handler, pois o PADE não chama on_shutdown automaticamente.
        """
        display_message(self.aid.name, "ResourceManagerAgent — stop_monitoring chamado.")

        if self.cpu_monitor:
            self.cpu_monitor.stop()
            self.cpu_monitor.join()

        if self.ram_monitor:
            self.ram_monitor.stop()
            self.ram_monitor.join()

        display_message(self.aid.name, "Monitores parados — CSVs gravados.")
