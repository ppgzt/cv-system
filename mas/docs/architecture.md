# MAS Architecture

## Overview

The Multi-Agent System (MAS) implements the same computer-vision weight-prediction pipeline as the baseline, but replaces the synchronous, single-threaded loop with an asynchronous, distributed agent architecture built on **PADE** (Python Agent Development Environment) and the **Twisted** reactor.

The system processes a herd of animals one-by-one. For each animal, frames are captured for `passage_time` seconds, enhanced, filtered by a suitability window, and then fed to a Keras model for weight prediction. A gap of `arrival_time` seconds separates consecutive animals.

## Directory Structure

```
cv-system-multiagent/
├── main.py                                # Entry point — selects strategy via CLI arg
├── domain/
│   ├── pipelines.py                       # SingleStreamStrategy, BatchStreamStrategy, MASStrategy
│   ├── modules/
│   │   ├── image_capture.py               # ImageCapture.get_frame()
│   │   ├── data_enhance.py                # DataEnhance.run(img)
│   │   ├── frame_selection.py             # FrameSelection.evaluate(elapsed_time, img)
│   │   └── predict_weight.py              # PredictWeight.predict(imgs)
│   └── helpers/
│       └── transformations.py             # Image transformation primitives
│
├── mas/
│   ├── __init__.py                        # sys.path setup for PADE and project root
│   ├── agents/
│   │   ├── capture_agent.py               # CaptureAgent + CaptureBehaviour
│   │   ├── data_enhance_agent.py          # DataEnhanceAgent
│   │   ├── frame_selection.py             # FrameSelectionAgent
│   │   ├── predict_weight_agent.py        # PredictWeightAgent
│   │   └── resource_manager_agent.py      # ResourceManagerAgent (CPU/RAM monitoring)
│   ├── adapters/
│   │   ├── capture_adapter.py             # Wraps ImageCapture
│   │   ├── data_enhance_adapter.py        # Wraps DataEnhance
│   │   ├── frame_selection_adapter.py     # Wraps FrameSelection (lazy model load)
│   │   ├── inference_adapter.py           # Wraps PredictWeight (lazy model load)
│   │   ├── blackboard_adapter.py          # Thread-safe in-memory metrics blackboard
│   │   └── __init__.py
│   ├── utils/
│   │   ├── globals.py                     # FRAME_BUFFER — shared in-memory image store
│   │   ├── cpu_monitor.py                 # CPUMonitor thread
│   │   ├── ram_monitor.py                 # RAMMonitor thread
│   │   └── __init__.py
│   ├── pade/                              # Embedded PADE framework
│   └── docs/                              # This documentation
│
└── infra/
    ├── images/sample.png                  # Stub image used by ImageCapture
    ├── models/model_run1_epoch029.keras   # Trained Keras model
    ├── profiling/agents.py                # Baseline CPU/RAM monitors
    └── reports/{pid}/                     # Per-experiment output (metrics.json, cpu.csv, mem.csv)
```

## Agent Topology

```
                    ┌─────────────────┐
                    │ ResourceManager  │  (monitoring only, not in data path)
                    └─────────────────┘

 ┌────────────┐    ┌──────────────┐    ┌─────────────────┐    ┌────────────────┐
 │ Capture    │───>│ DataEnhance  │───>│ FrameSelection  │───>│ PredictWeight  │
 │ Agent      │    │ Agent        │    │ Agent           │    │ Agent          │
 └────────────┘    └──────────────┘    └─────────────────┘    └────────────────┘
       │                                        ^                      ^
       │         passage-complete               │                      │
       └────────────────────────────────────────┘                      │
       │         batch-ready                                           │
       └───────────────────────────────────────────────────────────────┘
              (via FrameSelectionAgent)
```

All agents run inside the **Twisted reactor** on the same host. Each agent listens on a dedicated TCP port. PADE's FIPA-ACL messaging carries only lightweight JSON metadata; numpy image arrays flow through the shared `FRAME_BUFFER` in-memory dict to avoid serialization overhead.

## Adapter Pattern

Each agent delegates its domain work to an adapter that wraps the corresponding `domain/modules/` class:

| Agent | Adapter | Domain Module |
|---|---|---|
| CaptureAgent | `CaptureAdapter` | `ImageCapture` |
| DataEnhanceAgent | `DataEnhanceAdapter` | `DataEnhance` |
| FrameSelectionAgent | `FrameSelectionAdapter` | `FrameSelection` |
| PredictWeightAgent | `InferenceAdapter` | `PredictWeight` |

Adapters are thin wrappers — they add no logic. `FrameSelectionAdapter` and `InferenceAdapter` support lazy model loading (via `load_model()`) so the Keras model is loaded in a background thread without blocking the reactor.

## FRAME_BUFFER

Defined in `mas/utils/globals.py`. A plain `dict[str, Any]` shared by all agents in the same process.

**Lifecycle of a frame key:**

1. **CaptureAgent** — stores raw image: `FRAME_BUFFER[frame_id] = img`
2. **DataEnhanceAgent** — reads, enhances, overwrites: `FRAME_BUFFER[frame_id] = enhanced`
3. **FrameSelectionAgent** — if unsuitable, deletes: `FRAME_BUFFER.pop(frame_id)`; if suitable, leaves in place
4. **PredictWeightAgent** — pops for inference: `FRAME_BUFFER.pop(frame_id)` (frees RAM immediately)

All accesses are guarded by `threading.Lock` because `deferToThread` callbacks run on background threads.

## Configuration

MASStrategy receives its parameters from `main.py` via CLI args:

```python
MASStrategy(
    pid=pid,                          # unique experiment ID
    mode='single' | 'batch',         # inference mode
    herd_size=int,                    # number of animals
    passage_time=int,                 # seconds each animal is visible
    arrival_time=int,                 # gap between animals (seconds)
    fselection_time=float,            # capture tick interval (seconds)
    fselection_window=float,          # suitable time window (seconds)
)
```

Network configuration is read from `.env`:

| Variable | Default | Purpose |
|---|---|---|
| `SMA_AMS_HOST` | `localhost` | AMS host |
| `SMA_AMS_PORT` | `8000` | AMS port |
| `SMA_AGENT_HOST` | `localhost` | Agent host |
| `SMA_AGENT_BASE_PORT` | `5003` | Base port (agents use base+0 through base+6) |

Port layout:
- `base+0`: CaptureAgent
- `base+1`: DataEnhanceAgent
- `base+2`: FrameSelectionAgent
- `base+3`: PredictWeightAgent
- `base+6`: ResourceManagerAgent

## Startup Sequence

1. `MASStrategy.run()` creates all adapters and agents.
2. Agents are registered with AMS and their TCP listeners are attached to the reactor.
3. `FrameSelectionAgent.on_start()` loads the selection model in a background thread, then sends `"agent-ready"` to CaptureAgent.
4. `PredictWeightAgent.on_start()` loads the inference model in a background thread, then sends `"agent-ready"` to CaptureAgent.
5. CaptureAgent waits until **both** ready signals are received, then sets `simulation_started = True` and begins the PASSAGE/ARRIVAL/RESTART cycle.

This ensures no frames are captured before models are loaded.

## Shutdown

When PredictWeightAgent finishes the last animal, it:
1. Saves `metrics.json` to `infra/reports/{pid}/`.
2. Schedules `reactor.stop()` after a 1-second grace period.

The reactor's `before shutdown` trigger calls `ResourceManagerAgent.stop_monitoring()`, which stops the CPU/RAM monitor threads, joins them, and writes `cpu.csv` and `mem.csv`.
