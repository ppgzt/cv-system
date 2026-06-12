import numpy as np
import tensorflow as tf

class FrameSelection:

    '''
    Perform the frame selection task
    - suitable_window: the window of seconds that should be considered suitable
    - model: Keras/TF model object
    '''
    def __init__(self, suitable_window: float, model: object):
        self.suitable_window = suitable_window
        self.model = model

        @tf.function
        def _compiled_predict(img_tensor):
            return self.model(img_tensor, training=False)

        self._predict = _compiled_predict

    def evaluate(self, elapsed_time: float, img):
        self._predict(np.array([img]))
        return elapsed_time <= self.suitable_window