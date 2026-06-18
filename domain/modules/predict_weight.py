import numpy as np

class PredictWeight:

    def __init__(self, model: object):
        self.model = model

    def predict(self, imgs: list):
        return self.model.predict(np.array(imgs), batch_size=4, verbose=0)