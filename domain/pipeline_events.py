"""Contratos de dados e controle compartilhados pelos engines do pipeline.

O modulo e independente de PADE, Twisted, modelos e telemetria. Imagens
volumosas nao fazem parte dos eventos: o engine PADE as referencia por
``frame_id`` em um frame store compartilhado no mesmo processo/dispositivo.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import ClassVar, Mapping, TypeAlias


def _validate_stream_seq(stream_seq: int) -> None:
    if isinstance(stream_seq, bool) or not isinstance(stream_seq, int):
        raise TypeError("stream_seq must be an integer")
    if stream_seq < 0:
        raise ValueError("stream_seq must be non-negative")


@dataclass(frozen=True, slots=True)
class FrameEvent:
    """Evento de admissao de um frame; nao contem o array da imagem."""

    EVENT_TYPE: ClassVar[str] = "frame"

    stream_seq: int
    frame_id: str
    passage_id: str
    capture_index: int
    elapsed_time: float
    depth_filename: str | None
    label: str | None
    dataset_timestamp_ms: float | None = None

    def __post_init__(self) -> None:
        _validate_stream_seq(self.stream_seq)


@dataclass(frozen=True, slots=True)
class EndPassageEvent:
    """Controle que delimita uma passagem sem representar uma imagem."""

    EVENT_TYPE: ClassVar[str] = "end_passage"

    stream_seq: int
    passage_id: str
    total_captured_frames: int
    first_capture_time: str | None
    last_capture_time: str | None

    def __post_init__(self) -> None:
        _validate_stream_seq(self.stream_seq)


@dataclass(frozen=True, slots=True)
class EndPipelineEvent:
    """Controle de encerramento global do stream em uma aresta logica."""

    EVENT_TYPE: ClassVar[str] = "end_pipeline"

    stream_seq: int

    def __post_init__(self) -> None:
        _validate_stream_seq(self.stream_seq)


PipelineEvent: TypeAlias = FrameEvent | EndPassageEvent | EndPipelineEvent

_EVENT_CLASSES = {
    FrameEvent.EVENT_TYPE: FrameEvent,
    EndPassageEvent.EVENT_TYPE: EndPassageEvent,
    EndPipelineEvent.EVENT_TYPE: EndPipelineEvent,
}


def event_to_dict(event: PipelineEvent) -> dict:
    """Converte um evento para payload serializavel, preservando seu tipo."""
    if not isinstance(event, (FrameEvent, EndPassageEvent, EndPipelineEvent)):
        raise TypeError(f"unsupported pipeline event: {type(event).__name__}")
    return {"event_type": event.EVENT_TYPE, **asdict(event)}


def event_from_dict(payload: Mapping) -> PipelineEvent:
    """Reconstrói um evento a partir de um payload de metadados/controle."""
    values = dict(payload)
    event_type = values.pop("event_type", None)
    event_class = _EVENT_CLASSES.get(event_type)
    if event_class is None:
        raise ValueError(f"unknown pipeline event type: {event_type!r}")
    try:
        return event_class(**values)
    except TypeError as exc:
        raise ValueError(f"invalid {event_type!r} event payload") from exc


def event_to_json(event: PipelineEvent) -> str:
    """Serializa somente metadados/controle; nunca serializa a imagem."""
    return json.dumps(event_to_dict(event), ensure_ascii=True)


def event_from_json(payload: str) -> PipelineEvent:
    """Reconstrói um evento de um conteudo JSON adequado a uma mensagem ACL."""
    decoded = json.loads(payload)
    if not isinstance(decoded, dict):
        raise ValueError("pipeline event JSON must contain an object")
    return event_from_dict(decoded)
