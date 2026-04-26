# MAS Pipelines — Single and Batch Modes

The MAS supports two inference strategies, selected via the `mode` parameter in `MASStrategy`. Both mirror the baseline pipelines defined in `domain/pipelines.py`.

## Single Mode

Mirrors `SingleStreamStrategy` (file: `domain/pipelines.py`, line 20).

### Baseline Behavior

```
for each animal:
    start timer
    while elapsed < passage_time:
        capture frame
        enhance frame
        if frame is suitable:
            predict weight immediately
            record per-frame metric (start/final timestamps)
    compute mean of all predicted weights
    record weight_prediction_final timestamp
    wait arrival_time
```

Key characteristic: inference runs **per suitable frame**, and the mean weight is computed at the end of the passage.

### MAS Equivalent

| Step | Baseline | MAS Agent |
|---|---|---|
| Start timer | `start_at = datetime.now()` | `CaptureBehaviour.start_at = time.time()` |
| Capture frame | `ImageCapture.get_frame()` | `CaptureAgent` → `CaptureAdapter.get_frame()` |
| Enhance frame | `DataEnhance.run(img)` | `DataEnhanceAgent` → `DataEnhanceAdapter.run(img)` via `deferToThread` |
| Evaluate suitability | `FrameSelection.evaluate(elapsed_time, img)` | `FrameSelectionAgent` → `FrameSelectionAdapter.evaluate(elapsed_time, img)` via `deferToThread` |
| Predict weight | `PredictWeight.predict([img])` | `PredictWeightAgent` → `InferenceAdapter.predict([img])` via `deferToThread` |
| Compute mean | `np.mean(weights)` | `PredictWeightAgent._finalize_animal()` → `np.mean(weights)` |
| Wait for next animal | `time.sleep(arrival_time)` | `CaptureBehaviour` ARRIVAL mode (idle ticks) |

### Per-Frame Timing

In the baseline, the `elapsed_time` used for frame selection includes all prior processing time (capture + enhance + selection + prediction). In the MAS, `elapsed_time` is measured at the **capture moment only**, because downstream processing happens asynchronously. This means:

- MAS captures more frames per `passage_time` (processing doesn't block capture).
- The `elapsed_time` values in frame selection are smaller and more evenly spaced.
- This is an inherent difference in the async architecture and is part of what the experiment measures.

### Metrics JSON — Single Mode

Both baseline and MAS produce:

```json
{
  "1": {
    "first_image_capture_time": "...",
    "imgs": {
      "2": { "weight_prediction_start": "...", "weight_prediction_final": "..." },
      "5": { "weight_prediction_start": "...", "weight_prediction_final": "..." }
    },
    "last_image_capture_time": "...",
    "weight_prediction_final": "..."
  }
}
```

Keys in `imgs` are the sequential capture index (1-based). Only suitable frames have entries.

---

## Batch Mode

Mirrors `BatchStreamStrategy` (file: `domain/pipelines.py`, line 106).

### Baseline Behavior

```
for each animal:
    start timer
    imgs = []
    while elapsed < passage_time:
        capture frame
        enhance frame
        if frame is suitable:
            append to imgs (no prediction yet)
    predict weight on entire batch: PredictWeight.predict(imgs)
    compute mean of all predicted weights
    record weight_prediction_final timestamp
    wait arrival_time
```

Key characteristic: inference runs **once per animal** on the entire batch of suitable frames.

### MAS Equivalent

| Step | Baseline | MAS Agent |
|---|---|---|
| Collect suitable frames | `imgs.append(img)` | `PredictWeightAgent.batch_imgs[animal_id].append(img)` |
| Batch predict | `PredictWeight.predict(imgs)` | `PredictWeightAgent._process_batch()` → `InferenceAdapter.predict(all_imgs)` via `deferToThread` |
| Compute mean | `np.mean(weights)` | `PredictWeightAgent._finalize_animal()` → `np.mean(weights)` |

### Batch Coordination Flow

1. CaptureAgent captures frames and sends to DataEnhanceAgent.
2. DataEnhanceAgent enhances and forwards to FrameSelectionAgent.
3. FrameSelectionAgent evaluates suitability. Unsuitable frames are deleted from `FRAME_BUFFER`. Suitable frames are forwarded to PredictWeightAgent with `"frame-selected"` ontology.
4. PredictWeightAgent **accumulates** suitable images in `batch_imgs[animal_id]` (no inference yet).
5. CaptureAgent sends `"passage-complete"` to FrameSelectionAgent.
6. FrameSelectionAgent processes remaining frames. When all are done, sends `"batch-ready"` to PredictWeightAgent.
7. PredictWeightAgent runs a single batch inference on all accumulated images.
8. PredictWeightAgent finalizes the animal (mean weight, metrics).

### Metrics JSON — Batch Mode

Both baseline and MAS produce:

```json
{
  "1": {
    "first_image_capture_time": "...",
    "imgs": {
      "150": { "weight_prediction_start": "...", "weight_prediction_final": "..." }
    },
    "total_of_images": 150,
    "suitable_images": 45,
    "last_image_capture_time": "...",
    "weight_prediction_final": "..."
  }
}
```

- `imgs` contains a **single entry** keyed by `total_of_images` (the total capture count).
- `total_of_images` and `suitable_images` are recorded.

---

## Strategy Selection

The mode is determined in `main.py` (line 24) based on the CLI argument:

```
python main.py mas_single 5 30 5 0.2 10.0   → mode='single'
python main.py mas_batch  5 30 5 0.2 10.0   → mode='batch'
```

In `domain/pipelines.py` (line 26):
```python
mode = 'batch' if 'batch' in strategy else 'single'
```

---

## Comparison Summary

| Aspect | Baseline | MAS |
|---|---|---|
| Architecture | Synchronous, single-threaded | Asynchronous, multi-agent (PADE + Twisted) |
| Frame capture | Blocks on processing | Independent timer-based ticks |
| Image data flow | Variables in function scope | `FRAME_BUFFER` shared dict |
| Coordination | Sequential function calls | FIPA-ACL messages (ontologies) |
| Heavy computation | Runs on main thread | Delegated via `deferToThread` |
| Model instances | 1 (shared) | 2 (FrameSelection + PredictWeight) |
| Inference (single) | Per-frame, inline | Per-frame, async thread |
| Inference (batch) | End-of-passage, inline | End-of-passage, async thread |
| Monitoring | External threads (main.py) | ResourceManagerAgent (internal) |
| Metrics output | `metrics.json` | `metrics.json` (identical format) |
| Resource output | `cpu.csv`, `mem.csv` | `cpu.csv`, `mem.csv` (identical format) |
