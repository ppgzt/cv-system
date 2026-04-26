# Baseline Parity — Why MAS Metrics Are Scientifically Comparable

This document explains why the MAS implementation produces metrics that can be directly compared against the baseline synchronous pipelines (`SingleStreamStrategy` and `BatchStreamStrategy`).

---

## 1. Shared Domain Logic

The MAS never reimplements image processing, inference, or frame selection. Every agent delegates to an adapter that wraps the exact same `domain/modules/` classes used by the baseline:

| Domain Module | Baseline Usage | MAS Adapter | File |
|---|---|---|---|
| `ImageCapture` | `SingleStreamStrategy.__init__()` / `BatchStreamStrategy.__init__()` | `CaptureAdapter` | `mas/adapters/capture_adapter.py` |
| `DataEnhance` | `SingleStreamStrategy.__init__()` / `BatchStreamStrategy.__init__()` | `DataEnhanceAdapter` | `mas/adapters/data_enhance_adapter.py` |
| `FrameSelection` | `SingleStreamStrategy.__init__()` / `BatchStreamStrategy.__init__()` | `FrameSelectionAdapter` | `mas/adapters/frame_selection_adapter.py` |
| `PredictWeight` | `SingleStreamStrategy.__init__()` / `BatchStreamStrategy.__init__()` | `InferenceAdapter` | `mas/adapters/inference_adapter.py` |

Each adapter is a thin wrapper:

```python
# mas/adapters/capture_adapter.py
class CaptureAdapter:
    def __init__(self):
        self._capture = ImageCapture()       # same class as baseline

    def get_frame(self) -> np.ndarray:
        return self._capture.get_frame()      # same call as baseline
```

No additional logic, no preprocessing, no postprocessing. The same function is called with the same arguments.

---

## 2. Identical Frame Selection Logic

`FrameSelection.evaluate()` (file: `domain/modules/frame_selection.py`) performs two operations:

1. Time check: `suitable = elapsed_time <= self.suitable_window`
2. Model inference: `self.model(np.array([img]), training=False)` (result discarded)

Both baseline and MAS call this function with the same parameters (`elapsed_time`, `img`) through the same adapter. The only difference is **when** `elapsed_time` is measured:

- **Baseline**: after the previous frame's capture + enhance + (possibly) predict cycle.
- **MAS**: at the exact moment of capture (before any downstream processing).

This is an inherent architectural difference (sync vs async), not a logic difference. The frame selection function itself is identical.

---

## 3. Identical Inference Logic

`PredictWeight.predict()` (file: `domain/modules/predict_weight.py`):

```python
def predict(self, imgs: list):
    return self.model(np.array(imgs), training=False).numpy()
```

- **Single mode**: both baseline and MAS call `predict([img])` with a single image.
- **Batch mode**: both baseline and MAS call `predict(imgs)` with a list of all suitable images.

The inference adapter (`mas/adapters/inference_adapter.py`) calls the same `PredictWeight.predict()` method. Model weights are loaded from the same file (`infra/models/model_run1_epoch029.keras`).

---

## 4. Metrics JSON Structure Parity

### Single Mode

| Field | Baseline (`SingleStreamStrategy`) | MAS | Match? |
|---|---|---|---|
| `pid` | `{strategy}_{iso_timestamp}` | `{strategy}_{iso_timestamp}` | Yes |
| `load_model_start` | Before `keras.models.load_model()` | Before `deferToThread(load_model)` | Yes |
| `load_model_final` | After `keras.models.load_model()` | After model loaded in thread | Yes |
| `animals[id].first_image_capture_time` | ISO timestamp | ISO timestamp (from capture_metrics) | Yes |
| `animals[id].imgs[i].weight_prediction_start` | ISO timestamp before `predict()` | ISO timestamp before `deferToThread(predict)` | Yes |
| `animals[id].imgs[i].weight_prediction_final` | ISO timestamp after `predict()` | ISO timestamp after inference callback | Yes |
| `animals[id].last_image_capture_time` | ISO timestamp of last capture | ISO timestamp (from capture_metrics) | Yes |
| `animals[id].weight_prediction_final` | ISO timestamp after `np.mean()` | ISO timestamp after `np.mean()` in `_finalize_animal` | Yes |

Both have the same set of fields. JSON keys are strings in both cases (Python `json.dump` converts int keys to strings).

### Batch Mode

All fields from single mode, plus:

| Field | Baseline (`BatchStreamStrategy`) | MAS | Match? |
|---|---|---|---|
| `animals[id].total_of_images` | Loop counter `i` | `total_frames` from batch-ready | Yes |
| `animals[id].suitable_images` | Counter `s` | `suitable_count` from batch-ready | Yes |

### Key Naming

- `imgs` keys are the sequential capture index (1-based).
- In batch mode, there is a single entry keyed by `total_of_images`.
- Both baseline and MAS use the same key convention.

---

## 5. Resource Monitoring Parity

### Baseline (file: `infra/profiling/agents.py`)

Monitors started in `main.py` (line 40-41) for non-MAS strategies:

```python
cpu_monitor = CPUMonitor(pid=pid)
ram_monitor = RAMMonitor(pid=pid)
cpu_monitor.start()
ram_monitor.start()
```

### MAS (file: `mas/agents/resource_manager_agent.py`)

ResourceManagerAgent runs `CPUMonitor` and `RAMMonitor` from `mas/utils/`:

- Same `psutil` calls: `cpu_percent(percpu=True, interval=1)` and `virtual_memory()`.
- Same CSV format: identical column headers and data types.
- Same 1-second sampling interval.
- Same output files: `infra/reports/{pid}/cpu.csv` and `infra/reports/{pid}/mem.csv`.

The MAS monitors are refactored copies of the baseline monitors (the baseline monitors use inheritance from a base class that's coupled to the non-MAS code). The data collection logic is identical.

---

## 6. Known Intentional Differences

These differences are inherent to the async architecture and are part of what the experiment measures:

| Aspect | Baseline | MAS | Impact on comparison |
|---|---|---|---|
| Frame capture rate | Blocked by processing | Fixed interval (independent) | MAS captures more frames per passage_time |
| `elapsed_time` at selection | Includes prior processing time | Capture-time only | Slightly different suitability decisions |
| Model instances | 1 (shared) | 2 (one per loading agent) | Higher RAM in MAS |
| `load_model` timing | Single load | Two parallel loads | Different load_model metrics |
| Zero-suitable-frames case | `np.mean([])` → `nan` | Returns `0.0` | Edge case; unlikely in practice |

These are not bugs — they are the architectural differences the experiment is designed to measure.

---

## 7. Verification Checklist

To verify parity before running experiments:

- [ ] Both pipelines use the same model file: `infra/models/model_run1_epoch029.keras`
- [ ] Both pipelines use the same sample image: `infra/images/sample.png`
- [ ] Both pipelines produce `metrics.json` with matching field names
- [ ] Both pipelines produce `cpu.csv` and `mem.csv` with matching column headers
- [ ] `FrameSelection.evaluate()` is called with the same signature in both
- [ ] `PredictWeight.predict()` is called with the same image shape in both
- [ ] Adapters contain no additional logic beyond delegation to domain modules
