"""Frame store sincronizado para arrays compartilhados no mesmo processo.

PADE/ACL transporta eventos de controle e metadados; os dados volumosos de
imagem permanecem neste armazenamento in-process no mesmo dispositivo. Esta
fundacao nao oferece nem pretende oferecer transporte de imagens multi-host.
"""

from __future__ import annotations

import threading
from typing import Generic, TypeVar


FrameT = TypeVar("FrameT")
_MISSING = object()


class FrameStore(Generic[FrameT]):
    """Encapsula todos os acessos ao dicionario de frames sob um unico lock."""

    def __init__(self):
        self._frames: dict[str, FrameT] = {}
        self._lock = threading.RLock()

    def put(self, frame_id: str, value: FrameT) -> None:
        """Insere ou substitui um frame, permitindo raw -> enhanced."""
        with self._lock:
            self._frames[frame_id] = value

    def get(self, frame_id: str) -> FrameT | None:
        with self._lock:
            return self._frames.get(frame_id)

    def pop(self, frame_id: str) -> FrameT | None:
        with self._lock:
            return self._frames.pop(frame_id, None)

    def discard(self, frame_id: str) -> bool:
        """Remove um frame rejeitado e informa se ele existia."""
        with self._lock:
            return self._frames.pop(frame_id, _MISSING) is not _MISSING

    def clear(self) -> int:
        """Remove todos os frames e retorna quantos foram liberados."""
        with self._lock:
            removed = len(self._frames)
            self._frames.clear()
            return removed

    def __len__(self) -> int:
        with self._lock:
            return len(self._frames)

    def __contains__(self, frame_id: object) -> bool:
        with self._lock:
            return frame_id in self._frames


# Instancia destinada ao futuro caminho operacional PADE. Os agentes legados
# continuam usando FRAME_BUFFER ate serem migrados em fases posteriores.
FRAME_STORE: FrameStore = FrameStore()
