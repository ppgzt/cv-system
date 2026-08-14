"""Funções determinísticas compartilhadas pelo Capture e por análises offline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FixedFpsCaptureEvent:
    scheduled_capture_time_ms: float
    source_index: int


@dataclass(frozen=True)
class PassageCapturePlan:
    """Plano logico puro de uma passagem, independente do engine executor."""

    first_timestamp_ms: float | None
    end_timestamp_ms: float | None
    events: tuple[FixedFpsCaptureEvent, ...]

    @property
    def end_offset_s(self) -> float:
        if self.first_timestamp_ms is None or self.end_timestamp_ms is None:
            return 0.0
        return (self.end_timestamp_ms - self.first_timestamp_ms) / 1000.0


def nearest_index(times: np.ndarray, value: float) -> int:
    """Replica o nearest-neighbour do Capture, incluindo desempate anterior."""
    j = int(np.searchsorted(times, value))
    if j <= 0:
        return 0
    if j >= len(times):
        return len(times) - 1
    if abs(times[j - 1] - value) <= abs(times[j] - value):
        return j - 1
    return j


def build_fixed_fps_schedule(
    times: np.ndarray,
    fps: float,
    end_ms: float | None = None,
) -> list[FixedFpsCaptureEvent]:
    """Reconstrói os eventos virtuais usados pelo scheduler Fixed-FPS."""
    if fps <= 0:
        raise ValueError("fps must be greater than zero")
    if len(times) == 0:
        return []

    virtual_clock = float(times[0])
    passage_end_ms = float(times[-1]) if end_ms is None else float(end_ms)
    step_ms = 1000.0 / fps
    events = []
    while virtual_clock <= passage_end_ms:
        events.append(FixedFpsCaptureEvent(
            scheduled_capture_time_ms=virtual_clock,
            source_index=nearest_index(times, virtual_clock),
        ))
        virtual_clock += step_ms
    return events


def passage_end_ms(
    times: np.ndarray,
    max_passage_seconds: float | None = None,
) -> float | None:
    """Retorna o limite temporal real, inclusive para caps entre frames."""
    if len(times) == 0:
        return None
    tmax = float(times[-1])
    if max_passage_seconds is None:
        return tmax
    return min(tmax, float(times[0]) + max_passage_seconds * 1000.0)


def build_original_timing_schedule(
    times: np.ndarray,
    end_ms: float | None = None,
) -> list[FixedFpsCaptureEvent]:
    """Planeja uma admissao por timestamp original, sem nearest-neighbour."""
    if len(times) == 0:
        return []
    passage_end = float(times[-1]) if end_ms is None else float(end_ms)
    return [
        FixedFpsCaptureEvent(float(timestamp_ms), source_index)
        for source_index, timestamp_ms in enumerate(times)
        if float(timestamp_ms) <= passage_end
    ]


def build_passage_capture_plan(
    times: np.ndarray,
    *,
    fps: float | None,
    native_timestamps: bool,
    max_passage_seconds: float | None = None,
) -> PassageCapturePlan:
    """Compõe o plano Fixed-FPS ou Original-Timing usado pelos dois engines."""
    if len(times) == 0:
        return PassageCapturePlan(None, None, ())

    end_ms = passage_end_ms(times, max_passage_seconds)
    if native_timestamps:
        events = build_original_timing_schedule(times, end_ms)
    else:
        if fps is None:
            raise ValueError("fps is required for fixed-fps capture")
        events = build_fixed_fps_schedule(times, fps, end_ms)

    return PassageCapturePlan(
        first_timestamp_ms=float(times[0]),
        end_timestamp_ms=end_ms,
        events=tuple(events),
    )
