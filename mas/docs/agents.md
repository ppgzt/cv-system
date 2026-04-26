# MAS Agents — Detailed Reference

## CaptureAgent

**File:** `mas/agents/capture_agent.py`

Biological simulator that models cattle passing through a scale one-by-one. Uses a `TimedBehaviour` that pulses at a configurable FPS rate.

### Inner Class: CaptureBehaviour

Extends `pade.behaviours.protocols.TimedBehaviour`. Each tick computes the organic elapsed time since the current animal entered the frame and operates in one of three modes:

| Mode | Condition | Action |
|---|---|---|
| **PASSAGE** | `elapsed <= passage_time` | Capture a frame, store in FRAME_BUFFER, send FIPA-INFORM to DataEnhanceAgent |
| **ARRIVAL** | `passage_time < elapsed < passage_time + arrival_time` | Idle — animal is off-screen |
| **RESTART** | `elapsed >= passage_time + arrival_time` | Advance to next animal; if herd is complete, finish |

### Per-Tick Flow (PASSAGE mode)

1. Generate a 12-char UUID `frame_id`.
2. Call `capture_adapter.get_frame()` to get a numpy array.
3. Store in `FRAME_BUFFER[frame_id]` under a thread lock.
4. Send a `"frame-capture"` FIPA-INFORM to DataEnhanceAgent with payload:
   ```json
   {
     "frame_id": "a1b2c3d4e5f6",
     "animal_id": 1,
     "frame_index": 5,
     "elapsed_time": 1.8234
   }
   ```
5. Track `first_capture` / `last_capture` ISO timestamps and `captured_count`.

### Passage-Complete Signal

When `elapsed > passage_time` (animal leaves the frame), CaptureAgent sends a `"passage-complete"` message to **FrameSelectionAgent** (not directly to PredictWeightAgent):

```json
{
  "animal_id": 1,
  "total_frames": 150,
  "first_capture": "2026-04-20T10:30:00.123456",
  "last_capture": "2026-04-20T10:30:30.456789"
}
```

This signal carries the capture-side timestamps used later in metrics.

### Constructor Arguments

| Argument | Type | Description |
|---|---|---|
| `aid` | `AID` | PADE agent identifier |
| `capture_adapter` | `CaptureAdapter` | Wraps `ImageCapture` |
| `next_agent_aid` | `str` | AID name of DataEnhanceAgent |
| `selection_agent_aid` | `str` | AID name of FrameSelectionAgent (for passage-complete) |
| `interval_seconds` | `float` | Tick interval (default 0.2 = 5 FPS) |
| `herd_size` | `int` | Number of animals to process |
| `passage_time` | `int` | Seconds each animal is on camera |
| `arrival_time` | `int` | Seconds between consecutive animals |
| `wait_for_aids` | `list[str]` | AIDs that must send `"agent-ready"` before simulation starts |

---

## DataEnhanceAgent

**File:** `mas/agents/data_enhance_agent.py`

Applies image transformations (noise removal, scale adjustment, channel replication, resize to 300x300) via `DataEnhanceAdapter`.

### Message Handling

- **Receives:** `"frame-capture"` ontology from CaptureAgent
- **Sends:** `"frame-enhanced"` ontology to FrameSelectionAgent

### Per-Message Flow

1. Parse payload to extract `frame_id`.
2. Read raw image from `FRAME_BUFFER[frame_id]`.
3. Use `deferToThread` to run `data_enhance_adapter.run(img)` in a background thread.
4. Overwrite `FRAME_BUFFER[frame_id]` with the enhanced image.
5. Forward the **unchanged** payload to FrameSelectionAgent.

The `elapsed_time` in the payload is never modified — it always reflects the organic capture moment.

### Constructor Arguments

| Argument | Type | Description |
|---|---|---|
| `aid` | `AID` | PADE agent identifier |
| `data_enhance_adapter` | `DataEnhanceAdapter` | Wraps `DataEnhance` |
| `next_agent_aid` | `str` | AID name of FrameSelectionAgent |

---

## FrameSelectionAgent

**File:** `mas/agents/frame_selection.py`

Gatekeeper that evaluates frame suitability and acts as the synchronization point between capture-side timing and inference-side aggregation.

### Two Incoming Ontologies

| Ontology | Source | Action |
|---|---|---|
| `"frame-enhanced"` | DataEnhanceAgent | Evaluate suitability via adapter |
| `"passage-complete"` | CaptureAgent | Record expected frame count and capture timestamps |

### Frame Evaluation Flow

1. Parse payload to get `frame_id` and `elapsed_time`.
2. Read enhanced image from `FRAME_BUFFER[frame_id]`.
3. Use `deferToThread` to call `frame_selection_adapter.evaluate(elapsed_time, img)`.
4. **If unsuitable:** delete from `FRAME_BUFFER` immediately (free RAM), increment `discarded` counter.
5. **If suitable:** forward payload to PredictWeightAgent with ontology `"frame-selected"`, increment `forwarded` counter.
6. After each evaluation, check if all expected frames for this animal have been processed.
7. When `processed_frames >= expected_frames`, send `"batch-ready"` to PredictWeightAgent.

### Selection Logic

Delegated to `FrameSelection.evaluate()` in `domain/modules/frame_selection.py`:

```python
def evaluate(self, elapsed_time, img):
    suitable = elapsed_time <= self.suitable_window
    self.model(np.array([img]), training=False)  # model run (result unused)
    return suitable
```

Suitability is purely time-based: frames captured within `suitable_window` seconds from the animal's arrival are considered suitable. The model inference is executed for timing parity with the baseline but its output is ignored.

### Batch-Ready Signal

Sent to PredictWeightAgent when all frames for an animal have been evaluated:

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

### Startup

On `on_start()`, loads the selection model via `deferToThread(frame_selection_adapter.load_model)`. Once loaded, sends `"agent-ready"` to CaptureAgent.

### Constructor Arguments

| Argument | Type | Description |
|---|---|---|
| `aid` | `AID` | PADE agent identifier |
| `frame_selection_adapter` | `FrameSelectionAdapter` | Wraps `FrameSelection` |
| `next_agent_aid` | `str` | AID name of PredictWeightAgent |
| `capture_agent_aid` | `str` | AID name of CaptureAgent (for agent-ready signal) |

---

## PredictWeightAgent

**File:** `mas/agents/predict_weight_agent.py`

Runs Keras model inference. Supports two modes — **single** and **batch** — that mirror the baseline's `SingleStreamStrategy` and `BatchStreamStrategy`.

### Two Incoming Ontologies

| Ontology | Source | Action |
|---|---|---|
| `"frame-selected"` | FrameSelectionAgent | Process individual suitable frame |
| `"batch-ready"` | FrameSelectionAgent | Trigger aggregation / batch inference |

### Single Mode

For each `"frame-selected"` message:

1. Pop image from `FRAME_BUFFER[frame_id]`.
2. Record `weight_prediction_start` timestamp.
3. Use `deferToThread` to call `inference_adapter.predict([img])`.
4. On success, append weight to `_predictions[animal_id]`.
5. Record `weight_prediction_final` timestamp under `imgs[frame_index]`.
6. Check if all expected predictions have been received.

When `"batch-ready"` arrives and all predictions are complete:
- Compute `np.mean(weights)`.
- Set `weight_prediction_final` timestamp.
- If last animal: save `metrics.json` and stop reactor.

### Batch Mode

For each `"frame-selected"` message:

1. Pop image from `FRAME_BUFFER[frame_id]`.
2. Append to `batch_imgs[animal_id]` (accumulate, don't infer yet).

When `"batch-ready"` arrives and all suitable frames have been collected:
1. Call `inference_adapter.predict(all_imgs)` in a single batch.
2. Record per-animal metrics (start/final timestamps).
3. Compute `np.mean(weights)`.
4. If last animal: save `metrics.json` and stop reactor.

### Metrics Output

Saved to `infra/reports/{pid}/metrics.json` with the following structure:

```json
{
  "pid": "mas_single_2026-04-20T10:30:00",
  "load_model_start": "2026-04-20T10:29:55.000000",
  "load_model_final": "2026-04-20T10:29:58.000000",
  "animals": {
    "1": {
      "first_image_capture_time": "2026-04-20T10:30:00.123456",
      "imgs": {
        "2": { "weight_prediction_start": "...", "weight_prediction_final": "..." },
        "5": { "weight_prediction_start": "...", "weight_prediction_final": "..." }
      },
      "last_image_capture_time": "2026-04-20T10:30:30.456789",
      "weight_prediction_final": "2026-04-20T10:30:31.000000"
    }
  }
}
```

**Single mode** includes: `first_image_capture_time`, `imgs`, `last_image_capture_time`, `weight_prediction_final`.

**Batch mode** additionally includes: `suitable_images`, `total_of_images`.

### Startup

On `on_start()`, loads the inference model via `deferToThread(inference_adapter.load_model)`. Once loaded, sends `"agent-ready"` to CaptureAgent.

### Constructor Arguments

| Argument | Type | Description |
|---|---|---|
| `aid` | `AID` | PADE agent identifier |
| `inference_adapter` | `InferenceAdapter` | Wraps `PredictWeight` (lazy model load) |
| `mode` | `str` | `"single"` or `"batch"` |
| `pid` | `str` | Experiment ID (used for output path) |
| `herd_size` | `int` | Number of animals |
| `capture_agent_aid` | `str` | AID name of CaptureAgent (for agent-ready signal) |

---

## ResourceManagerAgent

**File:** `mas/agents/resource_manager_agent.py`

Monitors CPU and RAM usage via dedicated daemon threads. Produces CSV files in the same format as the baseline's `infra/profiling/agents.py` monitors, enabling direct scientific comparison.

### Monitors

| Monitor | File | Metric | Interval |
|---|---|---|---|
| CPUMonitor | `mas/utils/cpu_monitor.py` | Per-core CPU percentage | 1 second |
| RAMMonitor | `mas/utils/ram_monitor.py` | Total, available, used, percent, free, active, inactive, buffers, cached | 1 second |

### Output

- `infra/reports/{pid}/cpu.csv` — columns: `timestamp, cpu_core_0, cpu_core_1, ...`
- `infra/reports/{pid}/mem.csv` — columns: `timestamp, total, available, used, percent, free, active, inactive, buffers, cached`

The CSV headers and data formats are identical to the baseline monitors.

### Lifecycle

- `on_start()`: creates the reports directory, starts monitor threads and a `TimedBehaviour` that publishes snapshots to the in-memory blackboard every 5 seconds.
- `stop_monitoring()`: stops monitor threads, joins them, writes CSV files.

Registered as a `before shutdown` trigger on the Twisted reactor to ensure clean shutdown.
