import pickle
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Any, Union

import numpy as np
from tqdm import tqdm

from ais.immune_tools.detector import (
    Detector,
    DistanceCalculator,
    Point,
    RandomPointGenerator,
)
from ais.utils.input_check import array_like_check, nan_check


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

    def expand_memory(self, det: Detector):
        self.memory.append(det)

    def clear_memory(self):
        self.memory = []

    def is_memory_empty(self) -> bool:
        return True if len(self.memory) == 0 else False

    def get_current_memory_size(self) -> int:
        return len(self.memory)


class NegativeSelection(AIS, ImmuneMemory):
    def __init__(self, num_detectors=10, criterion="euclidean", init_memory=None, eps=None):
        AIS.__init__(self)
        ImmuneMemory.__init__(self, init_memory)
        assert criterion in ["euclidean"]
        self.num_detectors = num_detectors
        self.criterion = criterion
        self.eps = eps

    # noinspection PyPep8Naming
    def fit(self, X: Any):
        array_like_check(X)
        X = nan_check(X)
        dim = X.shape[1]  # dim of the data
        lower_boundary = X.min(axis=0, keepdims=True)
        upper_boundary = X.max(axis=0, keepdims=True)
        pbar = tqdm(total=self.num_detectors, desc="Training progress")
        while self.get_current_memory_size() < self.num_detectors:
            # first, generate a random point
            rand_point = RandomPointGenerator(dim=dim, low=lower_boundary, high=upper_boundary)
            candidate_point = rand_point()
            inside_detector = False
            # second, check if this point is inside the existing Detectors
            if not self.is_memory_empty():  # if there are detectors in memory
                for det in self.memory:
                    if det.is_point_inside(point=candidate_point):
                        inside_detector = True  # if inside -> go back to step 1
            # otherwise calculate all distances between the candidate and all self's ags
            # then, find min value among them
            if not inside_detector or self.is_memory_empty():
                min_dist_to_ag = DistanceCalculator.calc_min_dist_between_ags_and_point(ags=X, point=candidate_point)
                detector = Detector(center=candidate_point, radius=min_dist_to_ag, eps=self.eps)
                if self.eps is not None and min_dist_to_ag - self.eps > 0:
                    self.expand_memory(detector)
                    pbar.update(1)
                elif self.eps is None:
                    self.expand_memory(detector)
                    pbar.update(1)
        pbar.close()

    def save_model(self, filename: Union[str, Path]):
        with open(str(filename), "wb") as fout:
            pickle.dump(self.memory, fout)

    def load_model(self, filename: Union[str, Path]):
        with open(str(filename), "rb") as fin:
            self.memory = pickle.load(fin)

    def predict(self, X: Any) -> np.array:
        # 0 - normal, 1 = anomaly
        # for each point in X, go through each detector in memory
        # if a point falls inside a detector - this is an anomaly
        # otherwise it is a normal point
        array_like_check(X)
        X = nan_check(X)
        num_samples = X.shape[0]
        answers = [0] * num_samples
        for sample_id in tqdm(range(num_samples), desc="Test progress"):
            sample_coords = X[sample_id, :]
            sample_point = Point(sample_coords)
            for det in self.memory:
                if det.is_point_inside(point=sample_point):
                    answers[sample_id] = 1
                    break
        return np.array(answers)
