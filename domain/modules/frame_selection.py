import numpy as np
import time
import threading
import tensorflow as tf

class FrameSelection:

    '''
    Perform the frame selection task
    - suitable_window: the window of seconds that should be considered suitable
    - model: the tflite Interpreter
    '''
    def __init__(self, suitable_window: float,  model: object):
        self.suitable_window = suitable_window
        self.interpreter = model
        if self.interpreter:
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
        self.lock = threading.Lock()

    def evaluate(self, elapsed_time: float, img):
        suite = False

        if elapsed_time <= self.suitable_window:
            suite = True
        
        if self.interpreter:
            # Resize from 300x300 to 224x224 as required by the tflite model
            img_resized = tf.image.resize(img, (224, 224)).numpy()
            
            with self.lock:
                input_data = np.array([img_resized], dtype=np.float32)
                self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
                self.interpreter.invoke()
                output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
                _ = output_data.copy()

        return suite