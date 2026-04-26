"""Adapter for the ImageCapture domain module.

Bridges the MAS layer to the untouched domain capture logic.
"""

import numpy as np

from domain.modules.image_capture import ImageCapture


class CaptureAdapter:
    """Thin wrapper around domain ImageCapture."""

    def __init__(self) -> None:
        self._capture = ImageCapture()

    def get_frame(self) -> np.ndarray:
        return self._capture.get_frame()
