"""Funções determinísticas compartilhadas pelo Capture e por análises offline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FixedFpsCaptureEvent:
    scheduled_capture_time_ms: float
    source_index: int


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
