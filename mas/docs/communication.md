# MAS Communication Protocol

All inter-agent communication uses **FIPA-ACL INFORM** messages over TCP. Messages carry lightweight JSON metadata; numpy image arrays flow through the shared `FRAME_BUFFER` (see `mas/utils/globals.py`).

## Message Ontologies

| Ontology | Sender | Receiver | Purpose |
|---|---|---|---|
| `agent-ready` | FrameSelectionAgent, PredictWeightAgent | CaptureAgent | Signal that model is loaded and agent is ready |
| `frame-capture` | CaptureAgent | DataEnhanceAgent | Notify that a new frame was captured |
| `frame-enhanced` | DataEnhanceAgent | FrameSelectionAgent | Notify that a frame has been enhanced |
| `frame-selected` | FrameSelectionAgent | PredictWeightAgent | Forward a suitable frame for inference |
| `passage-complete` | CaptureAgent | FrameSelectionAgent | Signal that an animal left the camera |
| `batch-ready` | FrameSelectionAgent | PredictWeightAgent | Signal that all frames for an animal have been evaluated |

## Payload Schemas

### `agent-ready`

```json
{ "agent": "frame_selection_agent@localhost:5005" }
```

### `frame-capture`

```json
{
  "frame_id": "a1b2c3d4e5f6",
  "animal_id": 1,
  "frame_index": 5,
  "elapsed_time": 1.8234
}
```

- `frame_id`: 12-char UUID used as key in `FRAME_BUFFER`.
- `animal_id`: 1-based animal index.
- `frame_index`: sequential capture counter (1-based, incremented every tick regardless of suitability).
- `elapsed_time`: seconds since the current animal entered the frame, measured at capture time.

### `frame-enhanced`

Same payload as `frame-capture` — DataEnhanceAgent forwards the payload unchanged after enhancing the image in `FRAME_BUFFER`.

### `frame-selected`

Same payload as `frame-capture` — FrameSelectionAgent forwards the payload unchanged after confirming suitability.

### `passage-complete`

```json
{
  "animal_id": 1,
  "total_frames": 150,
  "first_capture": "2026-04-20T10:30:00.123456",
  "last_capture": "2026-04-20T10:30:30.456789"
}
```

- `total_frames`: total number of frames captured during the animal's passage.
- `first_capture` / `last_capture`: ISO timestamps of the first and last captured frames.

### `batch-ready`

```json
{
  "animal_id": 1,
  "suitable_count": 45,
  "total_frames": 150,
  "capture_metrics": {
    "first_image_capture_time": "2026-04-20T10:30:00.123456",
    "last_image_capture_time": "2026-04-20T10:30:30.456789"
  }
}
```

- `suitable_count`: number of frames that passed the suitability evaluation.
- `total_frames`: same as `passage-complete.total_frames`.
- `capture_metrics`: capture-side timing metrics forwarded from the `passage-complete` message.

## Sequence Diagrams

### Startup Coordination

```
FS-Agent                    PW-Agent                    CA-Agent
   │                           │                           │
   │── load model (thread) ──>│                           │
   │                           │── load model (thread) ──>│
   │                           │                           │
   │── agent-ready ─────────────────────────────────────>│
   │                           │── agent-ready ──────────>│
   │                           │                           │
   │                           │              simulation_started = True
   │                           │                           │
   │                           │              ─── PASSAGE cycle begins ───
```

### Single Mode — Per-Frame Flow

```
CA          DE          FS          PW
│            │           │           │
│── capture ─┤           │           │
│            │── enhance ┤           │
│            │           │── select ─┤
│            │           │           │── infer (thread)
│            │           │           │── record metric
│            │           │           │
│  ... more frames ...   │           │
│            │           │           │
│── passage-complete ───>│           │
│            │           │── batch-ready ──>│
│            │           │           │── finalize animal
│            │           │           │── compute mean weight
```

### Batch Mode — Per-Animal Flow

```
CA          DE          FS          PW
│            │           │           │
│── capture ─┤           │           │
│            │── enhance ┤           │
│            │           │── select ─┤── accumulate in batch_imgs
│            │           │           │
│  ... more frames ...   │           │
│            │           │           │
│── passage-complete ───>│           │
│            │           │── batch-ready ──>│
│            │           │           │── batch infer (thread)
│            │           │           │── finalize animal
```

## Frame Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRAME_BUFFER (dict)                       │
│                                                                  │
│  key: "a1b2c3d4e5f6"                                           │
│                                                                  │
│  [CaptureAgent]  ──SET──>  raw numpy array                      │
│  [DataEnhanceAgent] ─GET─> raw ──SET──> enhanced numpy array    │
│  [FrameSelectionAgent] ─GET─> enhanced                          │
│      if unsuitable: ──POP──>  (deleted, RAM freed)              │
│      if suitable:   (leave in place)                            │
│  [PredictWeightAgent] ──POP──>  enhanced (RAM freed)            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

Image data never appears in FIPA-ACL messages. Only the `frame_id` string is exchanged, and agents use it to look up or modify the image in `FRAME_BUFFER`. This design avoids base64 serialization overhead and keeps message sizes small.

## Threading Model

```
┌──────────────────────────────────────────┐
│            Twisted Reactor (main)         │
│                                           │
│  ┌─────────┐  ┌─────────┐  ┌──────────┐ │
│  │ Agent    │  │ Agent   │  │ Agent    │ │
│  │ react() │  │ react() │  │ react()  │ │
│  └─────────┘  └─────────┘  └──────────┘ │
│                                           │
│  deferToThread callbacks run here too     │
└───────────┬──────────────┬───────────────┘
            │              │
    ┌───────▼──────┐ ┌────▼──────────┐
    │ Thread pool  │ │ Thread pool   │
    │ (enhance,    │ │ (inference,   │
    │  select,     │ │  model load)  │
    │  capture)    │ │               │
    └──────────────┘ └───────────────┘
```

- All `react()` methods and `deferToThread` callbacks execute on the **reactor thread** (sequential, no concurrent access issues).
- CPU-intensive operations (enhancement, evaluation, inference, model loading) are delegated to background threads via `deferToThread`.
- `FRAME_BUFFER` accesses from background threads are protected by `threading.Lock`.
