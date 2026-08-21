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
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import mas  # noqa: F401 — side-effect: adiciona mas/ ao sys.path

from pade.core.agent import Agent
from pade.behaviours.protocols import TimedBehaviour
from pade.misc.utility import display_message
from pade.acl.aid import AID
from pade.acl.messages import ACLMessage
from domain.resource_events import ResourceState, ResourceStateEvent
from mas.experiment_config import (
    RESOURCE_CRITICAL_TEMPERATURE_C,
    RESOURCE_STALE_AFTER_SECONDS,
    RESOURCE_WARNING_PREDICTION_BACKLOG,
    RESOURCE_WARNING_TEMPERATURE_C,
)

RESOURCE_STATE_ONTOLOGY = "resource-state"


@dataclass(frozen=True, slots=True)
class ResourceThresholds:
    """Thresholds congelados do controle experimental PIBIC.

    CPU e RAM continuam publicados para telemetria/blackboard, mas não fazem
    parte da classificação SAFE/WARNING/CRITICAL desta versão.
    """

    warning_temperature_c: float = RESOURCE_WARNING_TEMPERATURE_C
    critical_temperature_c: float = RESOURCE_CRITICAL_TEMPERATURE_C
    warning_prediction_backlog: int = RESOURCE_WARNING_PREDICTION_BACKLOG
    stale_after_seconds: float = RESOURCE_STALE_AFTER_SECONDS

# Import monitors from mas.utils (Adapter approach)
from mas.utils.cpu_monitor import CPUMonitor
from mas.utils.ram_monitor import RAMMonitor
from mas.utils.temp_monitor import TempMonitor

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
            self.agent.publish_resource_snapshot(snapshot)

    def _build_snapshot(self) -> dict | None:
        cpu_mon = self.agent.cpu_monitor
        ram_mon = self.agent.ram_monitor
        temp_mon = self.agent.temp_monitor

        cpu_latest = cpu_mon.get_latest() if cpu_mon else None
        ram_latest = ram_mon.get_latest() if ram_mon else None
        temp_latest = temp_mon.get_latest() if temp_mon else None

        now = time.time()

        # CPU: [timestamp, core_0, core_1, ...]
        cpu_cores = [float(v) for v in cpu_latest[1:]] if cpu_latest else []
        cpu_percent = round(sum(cpu_cores) / len(cpu_cores), 2) if cpu_cores else 0.0

        # RAM: [timestamp, total, available, used, percent, free, active, inactive, buffers, cached]
        ram_percent = round(float(ram_latest[4]), 2) if ram_latest else 0.0
        ram_total = int(ram_latest[1]) if ram_latest else 0
        ram_available = int(ram_latest[2]) if ram_latest else 0
        ram_used = int(ram_latest[3]) if ram_latest else 0
        ram_free = int(ram_latest[5]) if ram_latest else 0

        # Temperature: [timestamp, temp]; None significa sensor indisponível.
        temp_val = None if temp_latest is None or temp_latest[1] is None else float(temp_latest[1])

        prediction_backlog = self.agent.get_prediction_backlog()
        throttling = self.agent.get_throttling_reading()
        throttling_active = self.agent.throttling_active(throttling)

        recorded_at_iso = datetime.fromtimestamp(now).isoformat(timespec="microseconds")

        metrics = {
            "timestamp": now,
            "cpu_percent": cpu_percent,
            "ram_percent": ram_percent,
            "temperature": temp_val,
            "temperature_c": temp_val,
            "prediction_backlog": prediction_backlog,
            "throttled_raw": throttling.get("throttled_raw") if throttling else None,
            "throttled_mask": throttling.get("throttled_mask") if throttling else None,
            "throttling_command_available": (
                throttling.get("throttling_command_available") if throttling else None
            ),
            "throttling_active": throttling_active,
            "throttling_sample_monotonic_ns": (
                throttling.get("sampled_at_monotonic_ns") if throttling else None
            ),
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
            "sampled_at_monotonic_ns": time.monotonic_ns(),
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
        orchestrator_agent_aid: str | None = None,
        thresholds: ResourceThresholds | None = None,
        prediction_backlog_provider: Callable[[], int] | None = None,
        throttling_provider: Callable[[], dict | None] | None = None,
        control_enabled: bool = False,
        debug: bool = False,
    ):
        super().__init__(aid=aid, debug=debug)
        self.pid = pid
        self.reports_dir = Path(reports_dir)
        self.cpu_monitor: CPUMonitor | None = None
        self.ram_monitor: RAMMonitor | None = None
        self.temp_monitor: TempMonitor | None = None

        # Blackboard (implementado, não usado na comparação atual)
        self.blackboard = blackboard or InMemoryBlackboardAdapter()
        self._blackboard_publisher: _PublishMetricsToBlackboard | None = None
        self._blackboard_interval_s = blackboard_interval_s
        self.orchestrator_agent_aid = orchestrator_agent_aid
        self.thresholds = thresholds or ResourceThresholds()
        self.prediction_backlog_provider = prediction_backlog_provider
        self.throttling_provider = throttling_provider
        self.control_enabled = control_enabled
        self._resource_sequence = 0
        self._last_resource_state = ResourceState.SAFE

    def on_start(self):
        super().on_start()
        display_message(self.aid.name, "ResourceManagerAgent iniciado.")

        # Criar diretório de reports (mesmo path do baseline)
        report_path = self.reports_dir / self.pid
        report_path.mkdir(parents=True, exist_ok=True)

        # Monitores refatorados do MAS — mesma lógica, mesma psutil, mesmo CSV
        self.cpu_monitor = CPUMonitor(pid=self.pid, reports_dir=str(self.reports_dir))
        self.ram_monitor = RAMMonitor(pid=self.pid, reports_dir=str(self.reports_dir))
        self.temp_monitor = TempMonitor(pid=self.pid, reports_dir=str(self.reports_dir))

        # Warmup do CPU (igual baseline: primeira chamada retorna 0.0)
        import psutil
        psutil.cpu_percent(percpu=True)

        self.cpu_monitor.start()
        self.ram_monitor.start()
        self.temp_monitor.start()

        display_message(
            self.aid.name,
            f"Monitores iniciados — CPU, RAM e Temperatura a cada 1s (pid={self.pid})",
        )

        # Blackboard publisher (implementado, uso futuro)
        if self.blackboard is not None:
            self._blackboard_publisher = _PublishMetricsToBlackboard(
                agent=self,
                blackboard=self.blackboard,
                interval_seconds=self._blackboard_interval_s,
            )
            self.behaviours.append(self._blackboard_publisher)

    def get_prediction_backlog(self) -> int:
        """Retorna somente trabalho aceito aguardando Prediction."""
        if self.prediction_backlog_provider is None:
            return 0
        try:
            return max(0, int(self.prediction_backlog_provider()))
        except Exception:
            return 0

    def get_throttling_reading(self) -> dict | None:
        """Lê a última telemetria de hardware sem deixar falha escapar."""
        if self.throttling_provider is None:
            return None
        try:
            reading = self.throttling_provider()
            return dict(reading) if reading is not None else None
        except Exception:
            return None

    @staticmethod
    def throttling_active(reading: dict | None) -> bool | None:
        """True se algum bit de condição *atual* do Pi estiver ativo."""
        if reading is None or reading.get("throttling_command_available") is not True:
            return None
        return any(reading.get(name) is True for name in (
            "undervoltage_current",
            "arm_frequency_capped_current",
            "throttled_current",
            "soft_temperature_limit_current",
        ))

    def publish_resource_snapshot(self, snapshot: dict) -> ResourceStateEvent:
        """Classifica e, somente no modo resource-aware, notifica controle."""
        metrics = snapshot["metrics"]
        state, reasons = self._classify(metrics)
        self._resource_sequence += 1
        event = ResourceStateEvent(
            sequence=self._resource_sequence,
            observed_at_monotonic_ns=int(
                snapshot.get("sampled_at_monotonic_ns", time.monotonic_ns())
            ),
            state=state,
            metrics=dict(metrics),
            reasons=tuple(reasons),
        )
        if state is not self._last_resource_state:
            display_message(
                self.aid.name,
                f"[Resource] {self._last_resource_state.value}->{state.value} "
                f"reason={','.join(reasons) or 'recovered'}",
            )
            self._last_resource_state = state
        if self.control_enabled and self.orchestrator_agent_aid is not None:
            msg = ACLMessage(ACLMessage.INFORM)
            msg.set_ontology(RESOURCE_STATE_ONTOLOGY)
            msg.add_receiver(AID(self.orchestrator_agent_aid))
            msg.set_content(json.dumps({
                "sequence": event.sequence,
                "observed_at_monotonic_ns": event.observed_at_monotonic_ns,
                "state": event.state.value,
                "metrics": dict(event.metrics),
                "reasons": list(event.reasons),
            }))
            try:
                self.send(msg)
            except Exception:
                pass
        return event

    def _classify(self, metrics: dict) -> tuple[ResourceState, list[str]]:
        thresholds = self.thresholds
        throttling_active = metrics.get("throttling_active")
        temperature_c = metrics.get("temperature_c", metrics.get("temperature"))
        backlog = int(metrics.get("prediction_backlog", 0))

        if throttling_active is True:
            return ResourceState.CRITICAL, ["throttling_active"]
        if temperature_c is not None and float(temperature_c) >= thresholds.critical_temperature_c:
            return ResourceState.CRITICAL, [f"temperature_c>={thresholds.critical_temperature_c}"]
        if temperature_c is not None and float(temperature_c) >= thresholds.warning_temperature_c:
            return ResourceState.WARNING, [f"temperature_c>={thresholds.warning_temperature_c}"]
        if backlog >= thresholds.warning_prediction_backlog:
            return ResourceState.WARNING, [f"prediction_backlog>={thresholds.warning_prediction_backlog}"]
        return ResourceState.SAFE, []

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

        if self.temp_monitor:
            self.temp_monitor.stop()
            self.temp_monitor.join()

        display_message(self.aid.name, "Monitores parados — CSVs gravados.")
