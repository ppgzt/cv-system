"""Configuração pequena e autoritativa das cinco condições PIBIC.

Os valores abaixo expõem o protocolo congelado; eles não alteram modelos nem
semântica dos estágios. ``mas-main.py`` é o único entrypoint experimental.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from domain.visual_activity import (
    DEFAULT_FRACTION_GE_2500_THRESHOLD,
    DEFAULT_IDLE_PATIENCE,
    DEFAULT_P99_THRESHOLD_MM,
    DEFAULT_PDI_THRESHOLD,
    DEFAULT_PIXEL_THRESHOLD_MM,
    DEFAULT_ROI_FRACTIONS,
)


DEFAULT_SELECTOR_THRESHOLD: Final[float] = 0.5
DEFAULT_LOW_FPS: Final[float] = 4.0
DEFAULT_MEDIUM_FPS: Final[float] = 7.0

RESOURCE_WARNING_TEMPERATURE_C: Final[float] = 75.0
RESOURCE_CRITICAL_TEMPERATURE_C: Final[float] = 80.0
RESOURCE_WARNING_PREDICTION_BACKLOG: Final[int] = 7
RESOURCE_STALE_AFTER_SECONDS: Final[float] = 10.0


@dataclass(frozen=True, slots=True)
class ExperimentMode:
    """Feature matrix of one official experimental condition."""

    name: str
    native_timestamps: bool
    requires_fixed_fps: bool
    visual_adaptive: bool
    visual_gated: bool
    resource_cap: bool


EXPERIMENT_MODES: Final[dict[str, ExperimentMode]] = {
    "original-timing": ExperimentMode(
        "original-timing", True, False, False, False, False,
    ),
    "fixed-fps": ExperimentMode(
        "fixed-fps", False, True, False, False, False,
    ),
    "visual-adaptive": ExperimentMode(
        "visual-adaptive", False, False, True, False, False,
    ),
    "visual-gated": ExperimentMode(
        "visual-gated", False, False, True, True, False,
    ),
    "resource-aware-visual-gated": ExperimentMode(
        "resource-aware-visual-gated", False, False, True, True, True,
    ),
}


def get_experiment_mode(name: str) -> ExperimentMode:
    try:
        return EXPERIMENT_MODES[name]
    except KeyError as exc:
        raise ValueError(f"unknown PIBIC experiment mode: {name!r}") from exc


__all__ = [
    "DEFAULT_FRACTION_GE_2500_THRESHOLD",
    "DEFAULT_IDLE_PATIENCE",
    "DEFAULT_LOW_FPS",
    "DEFAULT_MEDIUM_FPS",
    "DEFAULT_P99_THRESHOLD_MM",
    "DEFAULT_PDI_THRESHOLD",
    "DEFAULT_PIXEL_THRESHOLD_MM",
    "DEFAULT_ROI_FRACTIONS",
    "DEFAULT_SELECTOR_THRESHOLD",
    "EXPERIMENT_MODES",
    "ExperimentMode",
    "RESOURCE_CRITICAL_TEMPERATURE_C",
    "RESOURCE_STALE_AFTER_SECONDS",
    "RESOURCE_WARNING_PREDICTION_BACKLOG",
    "RESOURCE_WARNING_TEMPERATURE_C",
    "get_experiment_mode",
]
