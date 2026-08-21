"""Eventos leves do plano de controle de recursos."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping


class ResourceState(str, Enum):
    SAFE = "SAFE"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True, slots=True)
class ResourceStateEvent:
    """Snapshot ordenado de recursos; nunca carrega frames."""

    sequence: int
    observed_at_monotonic_ns: int
    state: ResourceState
    # Métricas compactas: além de números, a telemetria preserva None para
    # sensores/comandos indisponíveis e o raw de vcgencmd para auditoria.
    metrics: Mapping[str, object]
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise ValueError("sequence must be non-negative")
        if self.observed_at_monotonic_ns < 0:
            raise ValueError("observed_at_monotonic_ns must be non-negative")
