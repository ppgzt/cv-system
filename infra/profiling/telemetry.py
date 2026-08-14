"""Telemetria observacional de filas e hardware.

Este módulo não conhece engines, agentes, políticas ou decisões de controle.
O chamador fornece metadados opacos e, no caso das filas, referências públicas
com suporte a ``qsize()``. As amostras são mantidas em memória e persistidas em
CSV quando cada monitor é encerrado.
"""

from __future__ import annotations

import csv
import re
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable


QUEUE_SAMPLE_INTERVAL = 0.1
HARDWARE_SAMPLE_INTERVAL = 1.0
VCGENCMD_TIMEOUT = 0.5

QUEUE_TELEMETRY_HEADER = [
    "timestamp",
    "monotonic_ns",
    "elapsed_s",
    "run_id",
    "condition",
    "capture_fps",
    "capture_passage_id",
    "capture_to_selection_qsize",
    "selection_to_preprocessing_qsize",
    "preprocessing_to_prediction_qsize",
]

HARDWARE_TELEMETRY_HEADER = [
    "timestamp",
    "monotonic_ns",
    "elapsed_s",
    "run_id",
    "condition",
    "capture_fps",
    "capture_passage_id",
    "arm_clock_hz",
    "throttled_raw",
    "throttled_mask",
    "undervoltage_current",
    "arm_frequency_capped_current",
    "throttled_current",
    "soft_temperature_limit_current",
    "undervoltage_occurred",
    "arm_frequency_capping_occurred",
    "throttling_occurred",
    "soft_temperature_limit_occurred",
    "clock_command_available",
    "throttling_command_available",
]

CAPTURE_TIMING_HEADER = [
    "timestamp",
    "monotonic_ns",
    "elapsed_s",
    "run_id",
    "condition",
    "capture_fps",
    "passage_id",
    "capture_index",
    "frame_id",
    "source_filename",
    "source_relative_time_ms",
    "scheduled_capture_time_ms",
    "scheduled_monotonic_ns",
    "scheduled_elapsed_s",
    "actual_enqueue_monotonic_ns",
    "actual_enqueue_elapsed_s",
    "lateness_ms",
]

_THROTTLING_FLAG_NAMES = [
    "undervoltage_current",
    "arm_frequency_capped_current",
    "throttled_current",
    "soft_temperature_limit_current",
    "undervoltage_occurred",
    "arm_frequency_capping_occurred",
    "throttling_occurred",
    "soft_temperature_limit_occurred",
]

_ARM_CLOCK_PATTERN = re.compile(r"^\s*frequency\(\d+\)\s*=\s*(\d+)\s*$")
_THROTTLED_PATTERN = re.compile(
    r"^\s*throttled\s*=\s*(0[xX][0-9a-fA-F]+|\d+)\s*$"
)


class TelemetryContext:
    """Metadados opacos compartilhados pelos samplers de uma execução."""

    def __init__(
        self,
        run_id: str,
        condition: str,
        capture_fps: float | None,
        monotonic_origin_ns: int | None = None,
    ):
        self.run_id = run_id
        self.condition = condition
        self.capture_fps = capture_fps
        self.monotonic_origin_ns = (
            time.monotonic_ns()
            if monotonic_origin_ns is None
            else monotonic_origin_ns
        )
        self._capture_passage_id: str | None = None
        self._lock = threading.Lock()

    def set_capture_passage_id(self, passage_id: str) -> None:
        with self._lock:
            self._capture_passage_id = passage_id

    def clear_capture_passage_id(self, passage_id: str | None = None) -> None:
        """Limpa o contexto sem sincronizar ou aguardar nenhum estágio.

        Quando ``passage_id`` é fornecido, uma atualização mais nova não é
        apagada acidentalmente por uma limpeza atrasada.
        """
        with self._lock:
            if passage_id is None or self._capture_passage_id == passage_id:
                self._capture_passage_id = None

    def sample_metadata(self, monotonic_ns: int) -> dict:
        with self._lock:
            capture_passage_id = self._capture_passage_id
        return {
            "run_id": self.run_id,
            "condition": self.condition,
            "capture_fps": self.capture_fps,
            "capture_passage_id": capture_passage_id,
            "elapsed_s": (
                monotonic_ns - self.monotonic_origin_ns
            ) / 1_000_000_000.0,
        }


class CaptureTimingRecorder:
    """Recorder em memória do timing realizado de admissão de frames."""

    def __init__(
        self,
        context: TelemetryContext,
        reports_dir: str | Path = "infra/reports",
    ):
        self.context = context
        self.reports_dir = Path(reports_dir)
        self._rows: list[dict] = []
        self._pending_events: dict[str, dict] = {}
        self._lock = threading.Lock()
        self.persist_error: OSError | None = None
        self.dropped_events = 0

    def register_scheduled_event(
        self,
        *,
        passage_id: str,
        capture_index: int,
        frame_id: str,
        source_filename: str,
        source_relative_time_ms: float,
        scheduled_capture_time_ms: float,
        scheduled_monotonic_ns: int,
    ) -> bool:
        """Guarda o schedule até a admissão lógica ser observada.

        Este método não define onde ocorre a admissão. O produtor registra o
        deadline e o receptor da primeira borda chama ``record_admission``.
        Assim, runtimes assíncronos não confundem envio com enqueue lógico.
        """
        try:
            pending = {
                "passage_id": passage_id,
                "capture_index": capture_index,
                "frame_id": frame_id,
                "source_filename": source_filename,
                "source_relative_time_ms": source_relative_time_ms,
                "scheduled_capture_time_ms": scheduled_capture_time_ms,
                "scheduled_monotonic_ns": scheduled_monotonic_ns,
            }
            with self._lock:
                if frame_id in self._pending_events:
                    raise ValueError(f"duplicate pending frame_id: {frame_id}")
                self._pending_events[frame_id] = pending
            return True
        except Exception:
            with self._lock:
                self.dropped_events += 1
            return False

    def record_admission(
        self,
        frame_id: str,
        actual_admission_monotonic_ns: int,
    ) -> bool:
        """Completa um registro após a admissão na primeira borda lógica."""
        with self._lock:
            pending = self._pending_events.pop(frame_id, None)
        if pending is None:
            with self._lock:
                self.dropped_events += 1
            return False
        return self.record(
            **pending,
            actual_enqueue_monotonic_ns=actual_admission_monotonic_ns,
        )

    def discard_scheduled_event(self, frame_id: str) -> bool:
        """Descarta schedule cujo evento não pôde ser publicado."""
        with self._lock:
            return self._pending_events.pop(frame_id, None) is not None

    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending_events)

    def record(
        self,
        *,
        passage_id: str,
        capture_index: int,
        frame_id: str,
        source_filename: str,
        source_relative_time_ms: float,
        scheduled_capture_time_ms: float,
        scheduled_monotonic_ns: int,
        actual_enqueue_monotonic_ns: int,
    ) -> bool:
        """Registra um FRAME já enfileirado; nunca registra sentinelas."""
        try:
            metadata = self.context.sample_metadata(actual_enqueue_monotonic_ns)
            scheduled_elapsed_s = (
                scheduled_monotonic_ns - self.context.monotonic_origin_ns
            ) / 1_000_000_000.0
            actual_elapsed_s = metadata["elapsed_s"]
            row = {
                "timestamp": datetime.now().isoformat(),
                "monotonic_ns": actual_enqueue_monotonic_ns,
                "elapsed_s": actual_elapsed_s,
                "run_id": metadata["run_id"],
                "condition": metadata["condition"],
                "capture_fps": metadata["capture_fps"],
                "passage_id": passage_id,
                "capture_index": capture_index,
                "frame_id": frame_id,
                "source_filename": source_filename,
                "source_relative_time_ms": source_relative_time_ms,
                "scheduled_capture_time_ms": scheduled_capture_time_ms,
                "scheduled_monotonic_ns": scheduled_monotonic_ns,
                "scheduled_elapsed_s": scheduled_elapsed_s,
                "actual_enqueue_monotonic_ns": actual_enqueue_monotonic_ns,
                "actual_enqueue_elapsed_s": actual_elapsed_s,
                "lateness_ms": (
                    actual_enqueue_monotonic_ns - scheduled_monotonic_ns
                ) / 1_000_000.0,
            }
            with self._lock:
                self._rows.append(row)
            return True
        except Exception:
            with self._lock:
                self.dropped_events += 1
            return False

    def get_all_data(self) -> list[dict]:
        with self._lock:
            return [dict(row) for row in self._rows]

    def persist(self) -> bool:
        with self._lock:
            rows = [dict(row) for row in self._rows]
        try:
            output_dir = self.reports_dir / self.context.run_id
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / "capture_timing.csv", mode="w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=CAPTURE_TIMING_HEADER)
                writer.writeheader()
                writer.writerows(rows)
            return True
        except OSError as exc:
            self.persist_error = exc
            return False


def parse_arm_clock(raw: str | None) -> int | None:
    """Extrai Hz de ``frequency(48)=2400000000``."""
    if not raw:
        return None
    match = _ARM_CLOCK_PATTERN.fullmatch(raw)
    if match is None:
        return None
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return None


def parse_throttled_mask(raw: str | None) -> int | None:
    """Extrai a máscara numérica de ``throttled=0x...``."""
    if not raw:
        return None
    match = _THROTTLED_PATTERN.fullmatch(raw)
    if match is None:
        return None
    try:
        return int(match.group(1), 0)
    except (TypeError, ValueError):
        return None


def decode_throttled_mask(mask: int) -> dict[str, bool]:
    """Decodifica, separadamente, flags atuais e históricas do firmware."""
    return {
        "undervoltage_current": bool(mask & (1 << 0)),
        "arm_frequency_capped_current": bool(mask & (1 << 1)),
        "throttled_current": bool(mask & (1 << 2)),
        "soft_temperature_limit_current": bool(mask & (1 << 3)),
        "undervoltage_occurred": bool(mask & (1 << 16)),
        "arm_frequency_capping_occurred": bool(mask & (1 << 17)),
        "throttling_occurred": bool(mask & (1 << 18)),
        "soft_temperature_limit_occurred": bool(mask & (1 << 19)),
    }


def _run_vcgencmd(arguments: list[str], timeout: float) -> str | None:
    try:
        completed = subprocess.run(
            ["vcgencmd", *arguments],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def read_arm_clock(timeout: float = VCGENCMD_TIMEOUT) -> dict:
    raw = _run_vcgencmd(["measure_clock", "arm"], timeout)
    arm_clock_hz = parse_arm_clock(raw)
    return {
        "arm_clock_hz": arm_clock_hz,
        "clock_command_available": arm_clock_hz is not None,
    }


def _unavailable_throttling_reading(raw: str | None = None) -> dict:
    reading = {
        "throttled_raw": raw,
        "throttled_mask": None,
        "throttling_command_available": False,
    }
    reading.update({name: None for name in _THROTTLING_FLAG_NAMES})
    return reading


def read_throttled(timeout: float = VCGENCMD_TIMEOUT) -> dict:
    raw = _run_vcgencmd(["get_throttled"], timeout)
    if raw is None:
        return _unavailable_throttling_reading()
    mask = parse_throttled_mask(raw)
    if mask is None:
        return _unavailable_throttling_reading(raw)
    reading = {
        "throttled_raw": raw,
        "throttled_mask": mask,
        "throttling_command_available": True,
    }
    reading.update(decode_throttled_mask(mask))
    return reading


class _CsvTelemetryMonitor(threading.Thread):
    def __init__(
        self,
        *,
        name: str,
        context: TelemetryContext,
        interval: float,
        reports_dir: str | Path,
        filename: str,
        header: list[str],
    ):
        if interval <= 0:
            raise ValueError("telemetry interval must be greater than zero")
        super().__init__(name=name, daemon=True)
        self.context = context
        self.interval = interval
        self.reports_dir = Path(reports_dir)
        self.filename = filename
        self.header = list(header)
        self._stop_event = threading.Event()
        self._rows: list[dict] = []
        self._lock = threading.Lock()
        self.persist_error: OSError | None = None

    def stop(self) -> None:
        self._stop_event.set()

    def get_all_data(self) -> list[dict]:
        with self._lock:
            return [dict(row) for row in self._rows]

    def _append_sample(self) -> None:
        try:
            row = self._sample()
        except Exception:
            return
        with self._lock:
            self._rows.append(row)

    def _sample(self) -> dict:
        raise NotImplementedError

    def _write_csv(self) -> None:
        with self._lock:
            rows = [dict(row) for row in self._rows]
        try:
            output_dir = self.reports_dir / self.context.run_id
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / self.filename, mode="w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=self.header)
                writer.writeheader()
                writer.writerows(rows)
        except OSError as exc:
            self.persist_error = exc

    @staticmethod
    def _advance_deadline(deadline: float, interval: float) -> float:
        deadline += interval
        now = time.monotonic()
        if deadline <= now:
            missed = int((now - deadline) // interval) + 1
            deadline += missed * interval
        return deadline


class QueueTelemetryMonitor(_CsvTelemetryMonitor):
    """Amostra ocupação bruta de três filas públicas via ``qsize()``."""

    def __init__(
        self,
        context: TelemetryContext,
        capture_to_selection_queue,
        selection_to_preprocessing_queue,
        preprocessing_to_prediction_queue,
        *,
        interval: float = QUEUE_SAMPLE_INTERVAL,
        reports_dir: str | Path = "infra/reports",
    ):
        super().__init__(
            name="QueueTelemetryMonitor",
            context=context,
            interval=interval,
            reports_dir=reports_dir,
            filename="queue_telemetry.csv",
            header=QUEUE_TELEMETRY_HEADER,
        )
        self._capture_to_selection_queue = capture_to_selection_queue
        self._selection_to_preprocessing_queue = selection_to_preprocessing_queue
        self._preprocessing_to_prediction_queue = preprocessing_to_prediction_queue

    def _sample(self) -> dict:
        monotonic_ns = time.monotonic_ns()
        row = {
            "timestamp": datetime.now().isoformat(),
            "monotonic_ns": monotonic_ns,
        }
        row.update(self.context.sample_metadata(monotonic_ns))
        row.update({
            "capture_to_selection_qsize": self._capture_to_selection_queue.qsize(),
            "selection_to_preprocessing_qsize": (
                self._selection_to_preprocessing_queue.qsize()
            ),
            "preprocessing_to_prediction_qsize": (
                self._preprocessing_to_prediction_queue.qsize()
            ),
        })
        return row

    def run(self) -> None:
        deadline = time.monotonic()
        try:
            while not self._stop_event.is_set():
                self._append_sample()
                deadline = self._advance_deadline(deadline, self.interval)
                if self._stop_event.wait(max(0.0, deadline - time.monotonic())):
                    break
        finally:
            self._write_csv()


class HardwareTelemetryMonitor(_CsvTelemetryMonitor):
    """Amostra clock ARM e estado de throttling via ``vcgencmd``."""

    def __init__(
        self,
        context: TelemetryContext,
        *,
        interval: float = HARDWARE_SAMPLE_INTERVAL,
        reports_dir: str | Path = "infra/reports",
        command_timeout: float = VCGENCMD_TIMEOUT,
        clock_reader: Callable[[float], dict] = read_arm_clock,
        throttling_reader: Callable[[float], dict] = read_throttled,
    ):
        super().__init__(
            name="HardwareTelemetryMonitor",
            context=context,
            interval=interval,
            reports_dir=reports_dir,
            filename="hardware_telemetry.csv",
            header=HARDWARE_TELEMETRY_HEADER,
        )
        self.command_timeout = command_timeout
        self._clock_reader = clock_reader
        self._throttling_reader = throttling_reader

    def _sample(self) -> dict:
        monotonic_ns = time.monotonic_ns()
        row = {
            "timestamp": datetime.now().isoformat(),
            "monotonic_ns": monotonic_ns,
        }
        row.update(self.context.sample_metadata(monotonic_ns))
        try:
            clock = self._clock_reader(self.command_timeout)
        except Exception:
            clock = {
                "arm_clock_hz": None,
                "clock_command_available": False,
            }
        try:
            throttling = self._throttling_reader(self.command_timeout)
        except Exception:
            throttling = _unavailable_throttling_reading()
        row.update(clock)
        row.update(throttling)
        return row

    def run(self) -> None:
        # Evita lançar os dois subprocessos junto da inicialização dos workers.
        deadline = time.monotonic() + self.interval
        try:
            while not self._stop_event.wait(
                max(0.0, deadline - time.monotonic())
            ):
                self._append_sample()
                deadline = self._advance_deadline(deadline, self.interval)
        finally:
            self._write_csv()
