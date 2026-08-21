"""Lifecycle de telemetria do engine PADE na camada de aplicacao.

O adaptador apenas injeta as tres inboxes logicas nos monitores genericos. Ele
nao importa agentes, blackboard, politicas ou internals de PADE/Twisted.
"""

from __future__ import annotations

from pathlib import Path
import csv
import time
from typing import Callable

from infra.profiling.telemetry import (
    HARDWARE_SAMPLE_INTERVAL,
    QUEUE_SAMPLE_INTERVAL,
    CaptureTimingRecorder,
    HardwareTelemetryMonitor,
    QueueTelemetryMonitor,
    TelemetryContext,
    read_arm_clock,
    read_throttled,
)


class PadeTelemetrySession:
    """Agrupa um único contexto e o lifecycle dos monitores de uma run."""

    def __init__(
        self,
        *,
        run_id: str,
        condition: str,
        capture_fps: float | None,
        monotonic_origin_ns: int,
        selection_inbox,
        enhance_inbox,
        prediction_inbox,
        reports_dir: str | Path = "infra/reports",
        capture_timing_enabled: bool = True,
        queue_interval: float = QUEUE_SAMPLE_INTERVAL,
        hardware_interval: float = HARDWARE_SAMPLE_INTERVAL,
        clock_reader: Callable[[float], dict] = read_arm_clock,
        throttling_reader: Callable[[float], dict] = read_throttled,
    ):
        self.context = TelemetryContext(
            run_id=run_id,
            condition=condition,
            capture_fps=capture_fps,
            monotonic_origin_ns=monotonic_origin_ns,
        )
        self.capture_timing_recorder = (
            CaptureTimingRecorder(self.context, reports_dir=reports_dir)
            if capture_timing_enabled
            else None
        )
        self.queue_monitor = QueueTelemetryMonitor(
            self.context,
            selection_inbox,
            enhance_inbox,
            prediction_inbox,
            interval=queue_interval,
            reports_dir=reports_dir,
        )
        self.hardware_monitor = HardwareTelemetryMonitor(
            self.context,
            interval=hardware_interval,
            reports_dir=reports_dir,
            clock_reader=clock_reader,
            throttling_reader=throttling_reader,
        )
        self._started = False
        self._stopped = False
        self.control_records: list[dict] = []
        self._control_path = Path(reports_dir) / run_id / "control_activity.csv"

    def record_control_state(self, state: dict) -> None:
        """Recebe somente transições/recomputações compactas do Orchestrator."""
        self.control_records.append({"monotonic_ns": time.monotonic_ns(), **state})

    def prediction_backlog(self) -> int:
        """Trabalho aceito e pré-processado aguardando Prediction.

        É a mesma borda observada como ``preprocessing_to_prediction_qsize``
        no CSV de filas. Não soma filas upstream e não inclui a inferência que
        já começou; portanto mede exatamente a pressão pendente no gargalo.
        """
        return self.queue_monitor.prediction_backlog()

    def latest_throttling(self) -> dict | None:
        """Última leitura já coletada pelo monitor de hardware, ou None."""
        row = self.hardware_monitor.get_latest()
        if row is None:
            return None
        return {
            "sampled_at_monotonic_ns": row.get("monotonic_ns"),
            "throttled_raw": row.get("throttled_raw"),
            "throttled_mask": row.get("throttled_mask"),
            "undervoltage_current": row.get("undervoltage_current"),
            "arm_frequency_capped_current": row.get("arm_frequency_capped_current"),
            "throttled_current": row.get("throttled_current"),
            "soft_temperature_limit_current": row.get("soft_temperature_limit_current"),
            "throttling_command_available": row.get("throttling_command_available"),
        }

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self.queue_monitor.start()
        self.hardware_monitor.start()

    def stop(self, join_timeout: float = 2.0) -> None:
        """Interrompe samplers, espera só suas threads e persiste timing."""
        if self._stopped:
            return
        self._stopped = True
        self.queue_monitor.stop()
        self.hardware_monitor.stop()
        if self._started:
            self.queue_monitor.join(timeout=join_timeout)
            self.hardware_monitor.join(timeout=join_timeout)
        if self.capture_timing_recorder is not None:
            self.capture_timing_recorder.persist()
        if self.control_records:
            self._control_path.parent.mkdir(parents=True, exist_ok=True)
            with self._control_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=self.control_records[0].keys())
                writer.writeheader()
                writer.writerows(self.control_records)
