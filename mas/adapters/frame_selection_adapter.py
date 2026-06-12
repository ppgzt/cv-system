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
        from keras import layers, models
        base_model = keras.applications.MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(300, 300, 3)
        )
        base_model.trainable = False
        inputs = keras.Input(shape=(300, 300, 3))
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(2, activation='softmax')(x)
        model = models.Model(inputs, outputs)
        
        self._selection = FrameSelection(self.suitable_window, model)

    def evaluate(self, elapsed_time: float, img) -> bool:
        if not self._selection:
            raise RuntimeError("Model not loaded yet. Call load_model() first.")
        return self._selection.evaluate(elapsed_time, img)
