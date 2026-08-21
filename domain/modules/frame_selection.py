import numpy as np
import tensorflow as tf
from pathlib import Path


import threading

class FrameSelection:
    """Frame selection backed by the trained TFLite model (frame_selector.tflite).

    The model emits a 4-class softmax where only class 0 means "suited" (the
    animal is fully framed); classes 1-3 cover background / partial / noise.
    A frame is therefore suitable when prob(class 0) > threshold — a strictly
    model-driven decision.

    Inference runs through the TFLite interpreter, which uses the XNNPACK CPU
    delegate by default for low latency. The interpreter and its tensors are
    allocated once at construction and reused on every call (no per-inference
    reallocation).

    Preprocessing reproduces the training pipeline (v3 ROI10) EXACTLY:

        uint16 -> float32 -> ROI 10-90% crop (y0=0.10h, y1=0.90h, x0=0.10w, x1=0.90w)
        -> clip[0, 1950] mm -> resize 224x224 (bilinear, no padding)
        -> grayscale-to-RGB -> 2.0 * (d / 1950.0) - 1.0

    This deliberately differs from `DataEnhance` (clip 1950, /1950, 300x300
    with padding), so `evaluate()` MUST receive the RAW uint16 depth image,
    not the enhanced one.
    """

    # Preprocessing constants — must match the v3 ROI10 training pipeline.
    _IMG_SIZE = 224
    _CLIP_MAX = 1950.0
    _NORM_SCALE = 1950.0
    _ROI_Y_MIN = 0.10
    _ROI_Y_MAX = 0.90
    _ROI_X_MIN = 0.10
    _ROI_X_MAX = 0.90
    _EXPECTED_INPUT_SHAPE = (1, 224, 224, 3)
    _EXPECTED_OUTPUT_SHAPE = (1, 4)

    def __init__(self, model_path: str, threshold: float = 0.5,
                 num_threads: int = 2, suitable_window: float = None):
        """
        - model_path: path to the .tflite selection model.
        - threshold: a frame is suitable when prob(class 0) exceeds this.
        - num_threads: XNNPACK intra-op threads. Keep modest under MAS
          concurrency (the Twisted pool is min=2/max=4 on the Pi 5) to avoid
          CPU oversubscription.
        - suitable_window: kept for signature compatibility with existing
          callers; it is NOT used — the decision is model-only.
        """
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Frame Selection model not found: {path}")
        self.model_path = str(path)
        self.threshold = threshold
        self.suitable_window = suitable_window
        self._lock = threading.Lock()

        # Interpreter created once; XNNPACK is the default CPU delegate.
        self._interpreter = tf.lite.Interpreter(
            model_path=self.model_path, num_threads=num_threads)
        self._interpreter.allocate_tensors()

        in_details = self._interpreter.get_input_details()
        out_details = self._interpreter.get_output_details()
        if len(in_details) != 1 or len(out_details) != 1:
            raise ValueError("Frame Selection model must expose exactly one input and one output")
        input_detail, output_detail = in_details[0], out_details[0]
        input_shape = tuple(int(v) for v in input_detail["shape"])
        output_shape = tuple(int(v) for v in output_detail["shape"])
        if input_shape != self._EXPECTED_INPUT_SHAPE or input_detail["dtype"] != np.float32:
            raise ValueError(
                "Incompatible Frame Selection input: expected "
                f"shape={self._EXPECTED_INPUT_SHAPE}, dtype=float32; got "
                f"shape={input_shape}, dtype={input_detail['dtype']}"
            )
        if output_shape != self._EXPECTED_OUTPUT_SHAPE or output_detail["dtype"] != np.float32:
            raise ValueError(
                "Incompatible Frame Selection output: expected "
                f"shape={self._EXPECTED_OUTPUT_SHAPE}, dtype=float32; got "
                f"shape={output_shape}, dtype={output_detail['dtype']}"
            )
        self._input_index = input_detail['index']
        self._output_index = output_detail['index']

        # Reusable float32 input buffer [1, 224, 224, 3] (no realloc per call).
        self._input_tensor = np.zeros(
            (1, self._IMG_SIZE, self._IMG_SIZE, 3), dtype=np.float32)

        # Graph-compiled preprocessing (matches v3 ROI10 training process_path).
        # Input: single-channel [H, W, 1] (uint16 or float). Output: [224, 224, 3].
        roi_y_min = self._ROI_Y_MIN
        roi_y_max = self._ROI_Y_MAX
        roi_x_min = self._ROI_X_MIN
        roi_x_max = self._ROI_X_MAX
        clip_max = self._CLIP_MAX
        img_size = self._IMG_SIZE

        @tf.function
        def _preprocess_fn(img):
            img = tf.cast(img, tf.float32)
            h = tf.shape(img)[0]
            w = tf.shape(img)[1]
            y0 = tf.cast(tf.round(tf.cast(h, tf.float32) * roi_y_min), tf.int32)
            y1 = tf.cast(tf.round(tf.cast(h, tf.float32) * roi_y_max), tf.int32)
            x0 = tf.cast(tf.round(tf.cast(w, tf.float32) * roi_x_min), tf.int32)
            x1 = tf.cast(tf.round(tf.cast(w, tf.float32) * roi_x_max), tf.int32)
            img = img[y0:y1, x0:x1]
            img = tf.clip_by_value(img, 0.0, clip_max)
            img = tf.image.resize(img, [img_size, img_size], method="bilinear")
            img = tf.image.grayscale_to_rgb(img)
            img = 2.0 * (img / clip_max) - 1.0
            return img

        self._preprocess_fn = _preprocess_fn

    @staticmethod
    def _to_single_channel(raw) -> np.ndarray:
        """Reduce any raw depth image to a single-channel [H, W, 1] array."""
        img = np.asarray(raw)
        if img.ndim == 2:
            return img[:, :, np.newaxis]
        if img.ndim == 3:
            if img.shape[2] == 1:
                return img
            return img[:, :, :1]
        raise ValueError(f"Unsupported image shape: {img.shape}")

    def predict(self, raw) -> float:
        """Return prob(class 0) — the probability the frame is suited."""
        img = self._to_single_channel(raw)
        processed = self._preprocess_fn(img)
        
        with self._lock:
            self._input_tensor[0] = processed
            self._interpreter.set_tensor(self._input_index, self._input_tensor)
            self._interpreter.invoke()
            probs = self._interpreter.get_tensor(self._output_index)  # [1, 4]
            return float(probs[0][0])

    def evaluate(self, elapsed_time: float, img) -> bool:
        """Model-only decision: suitable when prob(class 0) > threshold.

        `elapsed_time` is kept in the signature for compatibility with existing
        callers but does not influence the decision.
        """
        return self.predict(img) > self.threshold
