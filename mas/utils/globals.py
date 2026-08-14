"""Compatibilidade temporaria do frame buffer compartilhado entre agentes.

O armazenamento autoritativo e ``FRAME_STORE``. ``FRAME_BUFFER`` preserva a
API de mapping usada pelos consumidores PADE ainda nao migrados, mas delega
ao mesmo store e nao mantem uma segunda copia dos arrays.

Keys are string frame identifiers; values are numpy ndarrays.
"""

from typing import Any

from mas.infrastructure.frame_store import FRAME_STORE, FrameStoreMapping

FRAME_BUFFER: FrameStoreMapping[Any] = FrameStoreMapping(FRAME_STORE)
