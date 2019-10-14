import operator

import numpy as np
from numpy import ndarray

from classifiers.classifier import Classifier


class PNNClassifier(Classifier):
    def __init__(self, g=0.1):
        super().__init__()

        self.g = g
        self._W = None
        self._y = None

    def fit(self, X: ndarray, y: ndarray):
        if X.shape[0] != y.shape[0]:
            raise Exception()
        self._W = np.apply_along_axis(self._normalize, 1, X)
        self._y = y

    def predict(self, sample: ndarray) -> int:
        sample = self._normalize(sample)
        estimations = np.apply_along_axis(
            self._estimate, 1,
            self._W, sample)

        prob = dict.fromkeys(set(self._y), 0)

        for p, y in zip(estimations, self._y):
            prob[y] += p
        return int(max(prob.items(), key=operator.itemgetter(1))[0])

    def _estimate(self, x: ndarray, y: ndarray):
        return np.e ** (-1 * (np.linalg.norm(x - y) ** 2) / (2 * self.g ** 2))

