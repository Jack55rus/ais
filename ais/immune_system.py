from abc import ABCMeta, abstractmethod
from detector import Detector
import numpy as np


class AIS(metaclass=ABCMeta):

    @abstractmethod
    def fit(self, X):
        pass

    @abstractmethod
    def predict(self, X):
        pass


class ImmuneMemory:
    def __init__(self, init_memory=None):
        self.memory = init_memory
        self.initialize_memory()

    def initialize_memory(self):
        if self.memory is None:
            self.memory = []

    def expand(self, det: Detector):
        self.memory.append(det)

    def clear(self):
        self.memory = []


class NegativeSelection(AIS, ImmuneMemory):

    def __init__(self, num_detectors=10, criterion='euclidean', init_memory=None):
        AIS.__init__(self)
        ImmuneMemory.__init__(self, init_memory)
        assert criterion in ['euclidean']
        self.num_detectors = num_detectors
        self.criterion = criterion
        self.immune_memory = []

    def fit(self, X):
        dim = X.shape[1]
        for det_num in range(self.num_detectors):
            det = Detector(dim=dim)
            det.change_radius(X)
            if det.get_params()[0] > 0:
                self.expand(det)

    def predict(self, X):
        preds = np.zeros(shape=X.shape[0], dtype=np.uint8)
        for i, ag in enumerate(X):
            for detector in self.memory:
                if detector.is_ag_inside(ag):  # if inside detector
                    preds[i] = 1  # fraud
                    continue
        return preds



