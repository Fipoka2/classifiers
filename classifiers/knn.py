from collections import Counter

import numpy as np
from numpy import ndarray

from classifiers.classifier import Classifier


class KNNClassifier(Classifier):
    def __init__(self, n_neighbors=5):
        super().__init__()

        self._n = n_neighbors
        self._data = None

    def fit(self, X: ndarray, y: ndarray):
        if X.shape[0] != y.shape[0]:
            raise Exception()
        norm = np.apply_along_axis(self._normalize, 1, X)
        self._data = (norm, y)

    def predict(self, x) -> int:
        x = self._normalize(x)
        dist = np.apply_along_axis(
            lambda v: np.linalg.norm(v - x), 1,
            self._data[0])
        arr = np.array(list(zip(dist, self._data[1])))
        partial_order = np.argpartition(arr[:, 0], self._n - 1)
        smallest = arr[partial_order][:self._n]
        return int(Counter(smallest[:, 1]).most_common(1)[0][0])
