"""Deteccao visual leve e independente de PADE, modelos e controle de FPS."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy import ndimage


class VisualState(str, Enum):
    IDLE = "IDLE"
    ACTIVE = "ACTIVE"


# Configuração final congelada do Visual Event: faixa vertical full-width.
# Para 240×320, roi_slices produz y=72:162 e x=0:320.
DEFAULT_ROI_FRACTIONS = (0.30, 0.675, 0.00, 1.00)
DEFAULT_PIXEL_THRESHOLD_MM = 200.0
DEFAULT_PDI_THRESHOLD = 0.08747855917667238
DEFAULT_IDLE_PATIENCE = 3
DEFAULT_P99_THRESHOLD_MM = 2230.0
DEFAULT_FRACTION_GE_2500_THRESHOLD = 0.0027473958333333335


@dataclass(frozen=True, slots=True)
class VisualActivityResult:
    score: float | None
    moving: bool | None
    visual_state: VisualState
    transition: str | None
    is_invalid: bool = False
    p99_mm: float | None = None
    fraction_ge_2500: float | None = None
    mad: float | None = None  # alias de retrocompatibilidade opcional


def readonly_view(frame: np.ndarray) -> np.ndarray:
    """Cria view zero-copy somente leitura sem mudar o array original."""
    view = np.asarray(frame).view()
    view.flags.writeable = False
    return view


def mean_absolute_depth_difference(
    previous: np.ndarray,
    current: np.ndarray,
) -> float:
    """Calcula MAD em float32, mantido para scripts legados e análises offline."""
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


def check_quality_gate(
    frame: np.ndarray,
    p99_threshold_mm: float = DEFAULT_P99_THRESHOLD_MM,
    fraction_ge_2500_threshold: float = DEFAULT_FRACTION_GE_2500_THRESHOLD,
) -> tuple[bool, float, float]:
    """Quality gate conjuntivo: retorna (is_invalid, depth_p99_mm, fraction_ge_2500mm)."""
    flat = np.asarray(frame).ravel()
    p99_mm = float(np.quantile(flat, 0.99))
    fraction_ge_2500 = float(np.mean(flat >= 2500.0))
    is_invalid = bool(
        p99_mm >= p99_threshold_mm and fraction_ge_2500 >= fraction_ge_2500_threshold
    )
    return is_invalid, p99_mm, fraction_ge_2500


def roi_slices(
    shape: tuple[int, int],
    fractions: tuple[float, float, float, float] = DEFAULT_ROI_FRACTIONS,
) -> tuple[slice, slice]:
    """Retorna fatias (slice_y, slice_x) para a ROI configurada."""
    height, width = shape[:2]
    y0, y1, x0, x1 = fractions
    return (
        slice(int(round(y0 * height)), int(round(y1 * height))),
        slice(int(round(x0 * width)), int(round(x1 * width))),
    )


def compute_pdi_score(
    previous: np.ndarray,
    current: np.ndarray,
    roi_fractions: tuple[float, float, float, float] = DEFAULT_ROI_FRACTIONS,
    pixel_threshold_mm: float = DEFAULT_PIXEL_THRESHOLD_MM,
) -> float:
    """Calcula o score PDI (fração de pixels da maior componente conexa na ROI)."""
    previous_array = np.asarray(previous)
    current_array = np.asarray(current)
    if previous_array.shape != current_array.shape:
        raise ValueError(
            "depth frames must have the same shape: "
            f"{previous_array.shape!r} != {current_array.shape!r}"
        )
    region = roi_slices(current_array.shape, roi_fractions)
    prev_roi = previous_array[region].astype(np.float32, copy=False)
    curr_roi = current_array[region].astype(np.float32, copy=False)

    diff = np.abs(curr_roi - prev_roi)
    mask = diff >= pixel_threshold_mm

    labels, n_components = ndimage.label(
        mask, structure=np.ones((3, 3), dtype=np.uint8)
    )
    if not n_components:
        return 0.0

    sizes = np.bincount(labels.ravel())[1:]
    largest_area = int(np.max(sizes))
    changed_pixels = int(np.count_nonzero(mask))
    if changed_pixels == 0:
        return 0.0
    return float(largest_area / changed_pixels)


class VisualActivityDetector:
    """Detector Visual online com Quality Gate conjuntivo, ROI e PDI de componente conexa."""

    def __init__(
        self,
        pdi_threshold: float = DEFAULT_PDI_THRESHOLD,
        idle_patience_frames: int = DEFAULT_IDLE_PATIENCE,
        pixel_threshold_mm: float = DEFAULT_PIXEL_THRESHOLD_MM,
        roi_fractions: tuple[float, float, float, float] = DEFAULT_ROI_FRACTIONS,
        p99_threshold_mm: float = DEFAULT_P99_THRESHOLD_MM,
        fraction_ge_2500_threshold: float = DEFAULT_FRACTION_GE_2500_THRESHOLD,
    ):
        if pdi_threshold < 0:
            raise ValueError("pdi_threshold must be non-negative")
        if idle_patience_frames <= 0:
            raise ValueError("idle_patience_frames must be greater than zero")
        if pixel_threshold_mm <= 0:
            raise ValueError("pixel_threshold_mm must be greater than zero")

        self.pdi_threshold = float(pdi_threshold)
        self.idle_patience_frames = int(idle_patience_frames)
        self.pixel_threshold_mm = float(pixel_threshold_mm)
        self.roi_fractions = roi_fractions
        self.p99_threshold_mm = float(p99_threshold_mm)
        self.fraction_ge_2500_threshold = float(fraction_ge_2500_threshold)

        self.previous_raw: np.ndarray | None = None
        self.previous_valid: bool = False
        self.state = VisualState.IDLE
        self.no_motion_count = 0

    def observe(self, current_raw: np.ndarray) -> VisualActivityResult:
        """Processa frame depth cru online sob causalidade estrita."""
        current = np.asarray(current_raw)
        is_invalid, p99_mm, frac_ge_2500 = check_quality_gate(
            current,
            self.p99_threshold_mm,
            self.fraction_ge_2500_threshold,
        )

        previous_state = self.state

        if is_invalid:
            # INVALID: limpa histórico temporal, preserva estado atual
            self.previous_raw = None
            self.previous_valid = False
            self.no_motion_count = 0
            return VisualActivityResult(
                score=None,
                moving=None,
                visual_state=self.state,
                transition=None,
                is_invalid=True,
                p99_mm=p99_mm,
                fraction_ge_2500=frac_ge_2500,
                mad=None,
            )

        if not self.previous_valid or self.previous_raw is None:
            # Primeiro VALID após início ou após INVALID: vira baseline
            self.previous_raw = current
            self.previous_valid = True
            return VisualActivityResult(
                score=None,
                moving=None,
                visual_state=self.state,
                transition=None,
                is_invalid=False,
                p99_mm=p99_mm,
                fraction_ge_2500=frac_ge_2500,
                mad=None,
            )

        # Frame VALID com baseline existente: calcula PDI
        score = compute_pdi_score(
            self.previous_raw,
            current,
            self.roi_fractions,
            self.pixel_threshold_mm,
        )
        self.previous_raw = current

        moving = score >= self.pdi_threshold

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

        return VisualActivityResult(
            score=score,
            moving=moving,
            visual_state=self.state,
            transition=transition,
            is_invalid=False,
            p99_mm=p99_mm,
            fraction_ge_2500=frac_ge_2500,
            mad=score,
        )

    def observe_score(self, score: float) -> VisualActivityResult:
        """Atualiza a máquina de estados diretamente com score pré-computado (offline replay)."""
        score = float(score)
        moving = score >= self.pdi_threshold
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
        return VisualActivityResult(
            score=score,
            moving=moving,
            visual_state=self.state,
            transition=transition,
            is_invalid=False,
            mad=score,
        )

    def observe_mad(self, mad: float) -> VisualActivityResult:
        """Alias legado de observe_score para compatibilidade com análises offline."""
        return self.observe_score(mad)

    def reset(self) -> VisualState:
        """Encerra a passagem visual e retorna seu estado antes do reset."""
        final_state = self.state
        self.previous_raw = None
        self.previous_valid = False
        self.state = VisualState.IDLE
        self.no_motion_count = 0
        return final_state
