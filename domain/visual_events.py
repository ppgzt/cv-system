"""Contratos do ramo Capture -> Visual; nunca transportam arrays."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Mapping, TypeAlias

from domain.pipeline_events import EndPassageEvent, EndPipelineEvent
from domain.visual_activity import VisualState


@dataclass(frozen=True, slots=True)
class VisualFrameEvent:
    stream_seq: int
    lease_id: str
    passage_id: str
    capture_index: int
    elapsed_time: float
    dataset_timestamp_ms: float | None
    depth_filename: str | None
    label: str | None
    frame_id: str | None = None

    EVENT_TYPE = "visual_frame"

    def __post_init__(self) -> None:
        if isinstance(self.stream_seq, bool) or not isinstance(
            self.stream_seq, int
        ):
            raise TypeError("stream_seq must be an integer")
        if self.stream_seq < 0:
            raise ValueError("stream_seq must be non-negative")


VisualInputEvent: TypeAlias = (
    VisualFrameEvent | EndPassageEvent | EndPipelineEvent
)


@dataclass(frozen=True, slots=True)
class VisualStateEvent:
    """Resultado compacto que o Orchestrator e o Capturador consomem."""

    passage_id: str
    capture_index: int
    elapsed_time: float
    dataset_timestamp_ms: float | None
    moving: bool | None
    visual_state: VisualState
    transition: str | None
    processing_time_ms: float
    pdi_score: float | None = None
    depth_filename: str | None = None
    frame_id: str | None = None
    is_trigger: bool = False
    is_invalid: bool = False
    p99_mm: float | None = None
    fraction_ge_2500: float | None = None
    mad: float | None = None


_VISUAL_TYPES = {
    VisualFrameEvent.EVENT_TYPE: VisualFrameEvent,
    EndPassageEvent.EVENT_TYPE: EndPassageEvent,
    EndPipelineEvent.EVENT_TYPE: EndPipelineEvent,
}


def visual_event_to_dict(event: VisualInputEvent) -> dict:
    event_type = getattr(event, "EVENT_TYPE", None)
    if event_type not in _VISUAL_TYPES:
        raise TypeError(f"unsupported visual event: {type(event).__name__}")
    return {"event_type": event_type, **asdict(event)}


def visual_event_from_dict(payload: Mapping) -> VisualInputEvent:
    values = dict(payload)
    event_type = values.pop("event_type", None)
    event_class = _VISUAL_TYPES.get(event_type)
    if event_class is None:
        raise ValueError(f"unknown visual event type: {event_type!r}")
    try:
        return event_class(**values)
    except TypeError as exc:
        raise ValueError(f"invalid {event_type!r} visual event payload") from exc


def visual_event_to_json(event: VisualInputEvent) -> str:
    return json.dumps(visual_event_to_dict(event), ensure_ascii=True)


def visual_event_from_json(payload: str) -> VisualInputEvent:
    decoded = json.loads(payload)
    if not isinstance(decoded, dict):
        raise ValueError("visual event JSON must contain an object")
    return visual_event_from_dict(decoded)
