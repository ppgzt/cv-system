"""Adapter for the FrameSelection domain module.

Bridges the MAS layer to the untouched domain frame selection logic.
"""

from domain.modules.frame_selection import FrameSelection


class FrameSelectionAdapter:
    """Thin wrapper around domain FrameSelection."""

    def __init__(self, suitable_window: float, model_path: str) -> None:
        self.suitable_window = suitable_window
        self.model_path = model_path
        self._selection = None

    def load_model(self):
        import keras
        model = keras.models.load_model(self.model_path)
        self._selection = FrameSelection(self.suitable_window, model)

    def evaluate(self, elapsed_time: float, img) -> bool:
        if not self._selection:
            raise RuntimeError("Model not loaded yet. Call load_model() first.")
        return self._selection.evaluate(elapsed_time, img)
