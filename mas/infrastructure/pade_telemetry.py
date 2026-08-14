"""Lifecycle de telemetria do engine PADE na camada de aplicacao.

O adaptador apenas injeta as tres inboxes logicas nos monitores genericos. Ele
nao importa agentes, blackboard, politicas ou internals de PADE/Twisted.
"""

from __future__ import annotations

from pathlib import Path
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

