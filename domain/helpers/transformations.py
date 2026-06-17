import tensorflow as tf
import numpy as np

class Replicate1DtoNDimChannel:

    def __init__(self, dim: int):
        self.dim = dim

    def transform(self, data: np.array):
        return np.repeat(data[:, :, np.newaxis], self.dim, axis=2)
        
class NoiseRemovalSetMaxValue:

    def __init__(self, max_value: int):
        self.max_value = max_value

    def transform(self, data: np.array):
        return np.minimum(data, self.max_value)

class AdjustScaleWithFixedMaxValue:

    def __init__(self, max_value: int):
        self.max_value = max_value

    def transform(self, data: np.array):
        data = data.astype('float32')
        data /= self.max_value
        
        return data

class ResizeImageWithPadding:

    def __init__(self, shape: tuple):
        self.shape = shape

    def transform(self, data: np.array):    
        img = tf.image.convert_image_dtype(data, tf.float32)
    
        # Redimensiona mantendo proporção (lado menor ajustado)
        resized_img = tf.image.resize_with_pad(img, self.shape[0], self.shape[1])
        
        return resized_img