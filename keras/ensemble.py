#!/usr/bin/env python

import numpy as np


class MeanEnsemble:
    def __init__(self, models):
        self.models = models

    def predict(self, X, batch_size=32):
        predictions = np.array([model.predict(X, batch_size) for model in self.models])

        return np.mean(predictions, axis=0)
