"""Frame store sincronizado para arrays compartilhados no mesmo processo.

PADE/ACL transporta eventos de controle e metadados; os dados volumosos de
imagem permanecem neste armazenamento in-process no mesmo dispositivo. Esta
fundacao nao oferece nem pretende oferecer transporte de imagens multi-host.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator, MutableMapping
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

    def keys_snapshot(self) -> tuple[str, ...]:
        """Retorna chaves imutaveis sem expor o dicionario interno."""
        with self._lock:
            return tuple(self._frames)


class FrameStoreMapping(MutableMapping[str, FrameT]):
    """Facade provisoria para consumidores que ainda esperam FRAME_BUFFER.

    Todos os acessos delegam ao mesmo ``FrameStore`` sincronizado; nao existe
    uma segunda copia dos arrays. A facade sera removida quando os consumidores
    PADE forem migrados para usar ``FrameStore`` diretamente.
    """

    def __init__(self, store: FrameStore[FrameT]):
        self._store = store

    def __getitem__(self, frame_id: str) -> FrameT:
        value = self._store.get(frame_id)
        if value is None:
            raise KeyError(frame_id)
        return value

    def __setitem__(self, frame_id: str, value: FrameT) -> None:
        self._store.put(frame_id, value)

    def __delitem__(self, frame_id: str) -> None:
        if not self._store.discard(frame_id):
            raise KeyError(frame_id)

    def __iter__(self) -> Iterator[str]:
        return iter(self._store.keys_snapshot())

    def __len__(self) -> int:
        return len(self._store)

    def get(self, frame_id: str, default=None):
        value = self._store.get(frame_id)
        return default if value is None else value

    def pop(self, frame_id: str, default=None):
        value = self._store.pop(frame_id)
        return default if value is None else value

    def clear(self) -> None:
        self._store.clear()


# Instancia autoritativa do caminho PADE. A facade legada FRAME_BUFFER delega
# a este mesmo objeto ate os consumidores serem migrados em fases posteriores.
FRAME_STORE: FrameStore = FrameStore()
