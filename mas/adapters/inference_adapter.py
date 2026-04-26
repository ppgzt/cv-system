"""Adapter for the PredictWeight domain module.

Bridges the MAS layer to the untouched domain inference logic.
"""

import numpy as np
from typing import Any

from domain.modules.predict_weight import PredictWeight


class InferenceAdapter:
    """Thin wrapper around domain PredictWeight with lazy loading support."""

    def __init__(self, model_or_path: Any) -> None:
        if isinstance(model_or_path, str):
            self.model_path = model_or_path
            self._predictor = None
        else:
            self._predictor = PredictWeight(model_or_path)

    def load_model(self) -> None:
        """Loads the Keras model. Should be called from a background thread."""
        if self._predictor is None:
            import keras
            model = keras.models.load_model(self.model_path)
            self._predictor = PredictWeight(model)

    def predict(self, imgs: list) -> np.ndarray:
        if self._predictor is None:
            self.load_model()
        return self._predictor.predict(imgs)
