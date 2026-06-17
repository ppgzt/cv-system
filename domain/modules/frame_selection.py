import numpy as np
import tensorflow as tf


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

    Preprocessing reproduces the training `process_path` EXACTLY:

        uint16 -> float32 -> resize 224x224 (bilinear, no padding)
        -> grayscale-to-RGB -> clip[0, 4000] -> (d / 2000) - 1.0

    This deliberately differs from `DataEnhance` (clip 1950, /1950, 300x300
    with padding), so `evaluate()` MUST receive the RAW uint16 depth image,
    not the enhanced one.
    """

    # Preprocessing constants — must match the training pipeline.
    _IMG_SIZE = 224
    _CLIP_MAX = 4000.0
    _NORM_SCALE = 2000.0

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
        self.model_path = model_path
        self.threshold = threshold
        self.suitable_window = suitable_window
        self._lock = threading.Lock()

        # Interpreter created once; XNNPACK is the default CPU delegate.
        self._interpreter = tf.lite.Interpreter(
            model_path=model_path, num_threads=num_threads)
        self._interpreter.allocate_tensors()

        in_details = self._interpreter.get_input_details()
        out_details = self._interpreter.get_output_details()
        self._input_index = in_details[0]['index']
        self._output_index = out_details[0]['index']

        # Reusable float32 input buffer [1, 224, 224, 3] (no realloc per call).
        self._input_tensor = np.zeros(
            (1, self._IMG_SIZE, self._IMG_SIZE, 3), dtype=np.float32)

        # Graph-compiled preprocessing (matches training process_path).
        # Input: single-channel [H, W, 1] (uint16 or float). Output: [224, 224, 3].
        @tf.function
        def _preprocess_fn(img):
            img = tf.cast(img, tf.float32)
            img = tf.image.resize(img, [self._IMG_SIZE, self._IMG_SIZE])
            img = tf.image.grayscale_to_rgb(img)
            img = tf.clip_by_value(img, 0.0, self._CLIP_MAX)
            img = (img / self._NORM_SCALE) - 1.0
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
