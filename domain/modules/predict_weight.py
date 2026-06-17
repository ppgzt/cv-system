import numpy as np
import tensorflow as tf
import threading


class PredictWeight:
    """Sheep weight regression backed by the TFLite model.

    Runs `sheep_weight_predictor.tflite` through the LiteRT interpreter, which
    uses the XNNPACK CPU delegate by default. The interpreter and its tensors
    are allocated once at construction; the batch dimension is resized only
    when it changes, so the common single-image path never reallocates.

    The model is a direct TFLite conversion of the former Keras model, so it
    consumes the SAME input the Keras model did: the DataEnhance-enhanced
    image, [N, 300, 300, 3] float32 in [0, 1]. Output is [N, 1] (weight in kg),
    matching the previous Keras output shape.
    """

    _INPUT_H = 300
    _INPUT_W = 300

    def __init__(self, model_path: str, num_threads: int = 2):
        self.model_path = model_path
        self._lock = threading.Lock()

        self._interpreter = tf.lite.Interpreter(
            model_path=model_path, num_threads=num_threads)
        self._interpreter.allocate_tensors()

        in_details = self._interpreter.get_input_details()
        out_details = self._interpreter.get_output_details()
        self._input_index = in_details[0]['index']
        self._output_index = out_details[0]['index']
        # Default batch size after allocate_tensors() is 1.
        self._cur_batch = int(in_details[0]['shape'][0])

    def predict(self, imgs: list) -> np.ndarray:
        """Run weight regression on a list of enhanced images -> [N, 1]."""
        arr = np.asarray(imgs, dtype=np.float32)  # [N, 300, 300, 3]
        n = arr.shape[0]
        
        with self._lock:
            if n != self._cur_batch:
                self._interpreter.resize_tensor_input(
                    self._input_index, [n, self._INPUT_H, self._INPUT_W, 3])
                self._interpreter.allocate_tensors()
                self._cur_batch = n
            self._interpreter.set_tensor(self._input_index, arr)
            self._interpreter.invoke()
            return self._interpreter.get_tensor(self._output_index).copy()  # [N, 1]
