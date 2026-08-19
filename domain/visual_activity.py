"""Deteccao visual leve e independente de PADE, modelos e controle de FPS."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class VisualState(str, Enum):
    IDLE = "IDLE"
    ACTIVE = "ACTIVE"


@dataclass(frozen=True, slots=True)
class VisualActivityResult:
    mad: float | None
    moving: bool | None
    visual_state: VisualState
    transition: str | None


def readonly_view(frame: np.ndarray) -> np.ndarray:
    """Cria view zero-copy somente leitura sem mudar o array original."""
    view = np.asarray(frame).view()
    view.flags.writeable = False
    return view


def mean_absolute_depth_difference(
    previous: np.ndarray,
    current: np.ndarray,
) -> float:
    """Calcula MAD em float32, evitando underflow de profundidade uint16."""
    previous_array = np.asarray(previous)
    current_array = np.asarray(current)
    if previous_array.shape != current_array.shape:
        raise ValueError(
            "depth frames must have the same shape: "
            f"{previous_array.shape!r} != {current_array.shape!r}"
        )
    previous_float = previous_array.astype(np.float32, copy=False)
    current_float = current_array.astype(np.float32, copy=False)
    return float(np.mean(np.abs(current_float - previous_float)))


class VisualActivityDetector:
    """Estado IDLE/ACTIVE por MAD e histerese de N observacoes."""

    def __init__(self, mad_threshold: float, idle_patience_frames: int):
        if mad_threshold < 0:
            raise ValueError("mad_threshold must be non-negative")
        if idle_patience_frames <= 0:
            raise ValueError("idle_patience_frames must be greater than zero")
        self.mad_threshold = float(mad_threshold)
        self.idle_patience_frames = int(idle_patience_frames)
        self.previous_raw: np.ndarray | None = None
        self.state = VisualState.IDLE
        self.no_motion_count = 0

    def observe(self, current_raw: np.ndarray) -> VisualActivityResult:
        current = np.asarray(current_raw)
        if self.previous_raw is None:
            self.previous_raw = current
            return VisualActivityResult(None, None, self.state, None)

        mad = mean_absolute_depth_difference(self.previous_raw, current)
        self.previous_raw = current
        return self.observe_mad(mad)

    def observe_mad(self, mad: float) -> VisualActivityResult:
        """Atualiza apenas a maquina de estados; util na analise offline."""
        mad = float(mad)
        moving = mad >= self.mad_threshold
        previous_state = self.state

        if moving:
            self.state = VisualState.ACTIVE
            self.no_motion_count = 0
        elif self.state is VisualState.ACTIVE:
            self.no_motion_count += 1
            if self.no_motion_count >= self.idle_patience_frames:
                self.state = VisualState.IDLE
                self.no_motion_count = 0

        transition = None
        if self.state is not previous_state:
            transition = f"{previous_state.value}->{self.state.value}"
        return VisualActivityResult(mad, moving, self.state, transition)

    def reset(self) -> VisualState:
        """Encerra a passagem visual e retorna seu estado antes do reset."""
        final_state = self.state
        self.previous_raw = None
        self.state = VisualState.IDLE
        self.no_motion_count = 0
        return final_state
