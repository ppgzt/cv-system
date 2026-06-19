"""Adapter for the FrameSelection domain module.

Bridges the MAS layer to the untouched domain frame selection logic.
"""

from domain.modules.frame_selection import FrameSelection


class FrameSelectionAdapter:
    """Thin wrapper around domain FrameSelection."""

    def __init__(self, suitable_window: float, passage_time: float, model_path: str) -> None:
        self.suitable_window = suitable_window
        self.passage_time = passage_time
        self.model_path = model_path
        self._selection = None

    def load_model(self):
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=self.model_path, num_threads=1)
        interpreter.allocate_tensors()
        self._selection = FrameSelection(self.suitable_window, self.passage_time, interpreter)

    def evaluate(self, elapsed_time: float, img) -> bool:
        if not self._selection:
            raise RuntimeError("Model not loaded yet. Call load_model() first.")
        return self._selection.evaluate(elapsed_time, img)
