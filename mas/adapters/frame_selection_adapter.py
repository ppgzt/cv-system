"""Adapter for the FrameSelection domain module.

Bridges the MAS layer to the untouched domain frame selection logic.
"""

from domain.modules.frame_selection import FrameSelection


class FrameSelectionAdapter:
    """Thin wrapper around domain FrameSelection."""

    def __init__(self, suitable_window: float, model_path: str,
                 threshold: float = 0.5, num_threads: int = 2) -> None:
        self.suitable_window = suitable_window
        self.model_path = model_path
        self.threshold = threshold
        self.num_threads = num_threads
        self._selection = None

    def load_model(self):
        # Loads the trained TFLite selector (XNNPACK, single allocation).
        self._selection = FrameSelection(
            model_path=self.model_path,
            threshold=self.threshold,
            num_threads=self.num_threads,
            suitable_window=self.suitable_window,
        )

    def evaluate(self, elapsed_time: float, img) -> bool:
        if not self._selection:
            raise RuntimeError("Model not loaded yet. Call load_model() first.")
        return self._selection.evaluate(elapsed_time, img)

    def evaluate_with_score(self, elapsed_time: float, img) -> tuple[bool, float]:
        """Decisão + prob(class 0) numa única inferência.

        Retorna (suitable, prob). Útil p/ modo debug/logar a confiança do
        seletor sem rodar a inferência duas vezes.
        """
        if not self._selection:
            raise RuntimeError("Model not loaded yet. Call load_model() first.")
        prob = self._selection.predict(img)
        return (prob > self.threshold, float(prob))
