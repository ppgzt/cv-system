"""Blackboard adapters para métricas de recursos.

Este módulo fornece o protocolo e a implementação de um blackboard
em memória para armazenar as últimas métricas coletadas pelos
monitores de profiling (CPU, RAM).

Mesmo schema da POC mas-edge-vision para manter paridade.
"""

import copy
import threading
from typing import Any, Protocol


class BlackboardAdapter(Protocol):
    """Protocolo para leitura e escrita de métricas no blackboard."""

    def write_metrics(self, snapshot: dict[str, Any]) -> None:
        ...

    def read_latest_metrics(self) -> dict[str, Any] | None:
        ...


class InMemoryBlackboardAdapter:
    """Thread-safe in-memory blackboard snapshot store.

    Armazena a última snapshot de métricas coletadas.
    Utilizado futuramente por outros agentes do MAS para
    consultar o estado atual dos recursos do sistema.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest_snapshot: dict[str, Any] | None = None

    def write_metrics(self, snapshot: dict[str, Any]) -> None:
        with self._lock:
            self._latest_snapshot = copy.deepcopy(snapshot)

    def read_latest_metrics(self) -> dict[str, Any] | None:
        with self._lock:
            if self._latest_snapshot is None:
                return None
            return copy.deepcopy(self._latest_snapshot)
