"""Sequenciamento monotônico por aresta logica do pipeline.

Cada sender logico deve possuir sua propria instancia: uma para
Capture->Selection, outra para Selection->Preprocessing e outra para
Preprocessing->Prediction. Nao existe contador global entre as arestas.
"""

from __future__ import annotations

import threading


class StreamSequencer:
    """Aloca numeros de sequencia crescentes de forma thread-safe."""

    def __init__(self, start: int = 0):
        if isinstance(start, bool) or not isinstance(start, int):
            raise TypeError("sequence start must be an integer")
        if start < 0:
            raise ValueError("sequence start must be non-negative")
        self._next_seq = start
        self._lock = threading.Lock()

    def next_seq(self) -> int:
        with self._lock:
            stream_seq = self._next_seq
            self._next_seq += 1
            return stream_seq

    @property
    def next_value(self) -> int:
        with self._lock:
            return self._next_seq
