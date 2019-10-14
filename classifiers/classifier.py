from abc import ABC, abstractmethod
import numpy as np
from numpy import ndarray


class Classifier(ABC):

    @abstractmethod
    def fit(self, X: ndarray, y: ndarray):
        pass

    @abstractmethod
    def predict(self, x: ndarray) -> int:
        pass

    @classmethod
    def _normalize(cls, x):
        return x / np.linalg.norm(x)

    def score(self, X: ndarray, y: ndarray) -> float:
        if y.size == 0:
            raise Exception()

        n_correct = 0
        for x, ans in zip(X, y):
            predicted = self.predict(x)
            if predicted == ans:
                n_correct += 1
        return n_correct / y.size

    def detail_score(self, X: ndarray, y: ndarray) -> tuple:
        if y.size == 0:
            raise Exception()
        predictions = []
        n_correct = 0
        for x, ans in zip(X, y):
            predicted = self.predict(x)
            predictions.append(predicted)
            if predicted == ans:
                n_correct += 1
        return n_correct / y.size, predictions
