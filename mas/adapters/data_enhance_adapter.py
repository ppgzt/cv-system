"""Adapter for the DataEnhance domain module.

Bridges the MAS layer to the untouched domain enhancement logic.
"""

import numpy as np

from domain.modules.data_enhance import DataEnhance


class DataEnhanceAdapter:
    """Thin wrapper around domain DataEnhance."""

    def __init__(self) -> None:
        self._enhance = DataEnhance()

    def run(self, img: np.ndarray) -> np.ndarray:
        return self._enhance.run(img)
