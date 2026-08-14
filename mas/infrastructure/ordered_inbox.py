"""Inbox FIFO de aplicacao com restauracao explicita da ordem do stream."""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from typing import Generic, TypeVar


EventT = TypeVar("EventT")


class OrderedInboxClosed(RuntimeError):
    """Indica que uma inbox encerrada nao pode produzir novos eventos."""


class OrderedInbox(Generic[EventT]):
    """Ordena eventos por ``stream_seq`` antes de expo-los ao consumidor.

    ``qsize()`` representa a ocupacao pendente total da borda: eventos prontos
    mais eventos futuros retidos por um gap. ``ready_qsize()`` mede somente os
    eventos imediatamente consumiveis, e ``reorder_buffer_size()`` mede a
    parcela recebida fisicamente que ainda aguarda o fechamento de um gap.

    Ha um unico consumidor logico por instancia, como em cada estagio do
    ``ThreadPipeline``. O fechamento acorda consumidores bloqueados; eventos
    prontos ainda podem ser drenados, enquanto gaps nao resolvidos permanecem
    apenas para diagnostico em ``reorder_buffer_size()``.
    """

    def __init__(self, expected_seq: int = 0):
        if isinstance(expected_seq, bool) or not isinstance(expected_seq, int):
            raise TypeError("expected_seq must be an integer")
        if expected_seq < 0:
            raise ValueError("expected_seq must be non-negative")

        self._expected_seq = expected_seq
        self._ready: deque[EventT] = deque()
        self._reorder_buffer: dict[int, EventT] = {}
        self._closed = False
        self._condition = threading.Condition()

    def put(self, event: EventT) -> int:
        """Admite um evento e retorna quantos eventos ficaram prontos."""
        try:
            stream_seq = event.stream_seq  # type: ignore[attr-defined]
        except AttributeError as exc:
            raise TypeError("ordered inbox events must expose stream_seq") from exc
        if isinstance(stream_seq, bool) or not isinstance(stream_seq, int):
            raise TypeError("event stream_seq must be an integer")

        with self._condition:
            if self._closed:
                raise OrderedInboxClosed("cannot put into a closed ordered inbox")
            if stream_seq < self._expected_seq:
                raise ValueError(
                    f"stream_seq {stream_seq} was already released; "
                    f"expected {self._expected_seq}"
                )
            if stream_seq in self._reorder_buffer:
                raise ValueError(f"duplicate stream_seq {stream_seq}")

            self._reorder_buffer[stream_seq] = event
            released = 0
            while self._expected_seq in self._reorder_buffer:
                self._ready.append(
                    self._reorder_buffer.pop(self._expected_seq)
                )
                self._expected_seq += 1
                released += 1

            if released:
                self._condition.notify_all()
            return released

    def get(self, block: bool = True, timeout: float | None = None) -> EventT:
        """Retorna o proximo evento logico, em API semelhante a queue.Queue."""
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be non-negative")
        deadline = None if timeout is None else time.monotonic() + timeout

        with self._condition:
            while not self._ready:
                if self._closed:
                    raise OrderedInboxClosed("ordered inbox is closed and drained")
                if not block:
                    raise queue.Empty

                if deadline is None:
                    self._condition.wait()
                    continue

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise queue.Empty
                self._condition.wait(remaining)

            return self._ready.popleft()

    def close(self) -> None:
        """Impede novas admissoes e acorda consumidores bloqueados."""
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def qsize(self) -> int:
        """Ocupacao pendente total: eventos prontos mais reorder buffer."""
        with self._condition:
            return len(self._ready) + len(self._reorder_buffer)

    def ready_qsize(self) -> int:
        """Quantidade ordenada e imediatamente disponivel para consumo."""
        with self._condition:
            return len(self._ready)

    def reorder_buffer_size(self) -> int:
        """Quantidade recebida fisicamente, mas bloqueada por algum gap."""
        with self._condition:
            return len(self._reorder_buffer)

    def buffered_size(self) -> int:
        """Alias compativel de ``qsize()`` para a ocupacao pendente total."""
        return self.qsize()

    @property
    def expected_seq(self) -> int:
        with self._condition:
            return self._expected_seq

    @property
    def closed(self) -> bool:
        with self._condition:
            return self._closed
