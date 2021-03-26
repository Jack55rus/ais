from abc import ABCMeta, abstractmethod
from detector import Detector, RandomPointGenerator, RandomHyperSphereGenerator, Point, HyperSphere
from ais.detector import DistanceCalculator
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

    def expand_memory(self, det: Detector):
        self.memory.append(det)

    def clear_memory(self):
        self.memory = []

    def is_memory_empty(self) -> bool:
        return True if len(self.memory) == 0 else False

    def get_current_memory_size(self) -> int:
        return len(self.memory)


class NegativeSelection(AIS, ImmuneMemory):

    def __init__(self, num_detectors=10, criterion='euclidean', init_memory=None):
        AIS.__init__(self)
        ImmuneMemory.__init__(self, init_memory)
        assert criterion in ['euclidean']
        self.num_detectors = num_detectors
        self.criterion = criterion

    def fit(self, X: np.ndarray):
        dim = X.shape[1]  # dim of the data
        while self.get_current_memory_size() < self.num_detectors:
            # first, generate a random point
            rand_point = RandomPointGenerator(dim=dim)
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
                detector = Detector(center=candidate_point, radius=min_dist_to_ag, eps=None)
                self.expand_memory(detector)

    def predict(self, X: np.ndarray) -> np.array:
        # 0 - normal, 1 = anomaly
        # for each point in X, go through each detector in memory
        # if a point falls inside a detector - this is an anomaly
        # otherwise it is a normal point
        num_samples = X.shape[0]
        answers = [0] * num_samples
        for sample_id in range(num_samples):
            sample_coords = X[sample_id, :]
            sample_point = Point(sample_coords)
            for det in self.memory:
                if det.is_point_inside(point=sample_point):
                    answers[sample_id] = 1
                    break
        return np.array(answers)





# class NegativeSelection(AIS, ImmuneMemory):
#
#     def __init__(self, num_detectors=10, criterion='euclidean', init_memory=None):
#         AIS.__init__(self)
#         ImmuneMemory.__init__(self, init_memory)
#         assert criterion in ['euclidean']
#         self.num_detectors = num_detectors
#         self.criterion = criterion
#
#     def fit(self, X):
#         dim = X.shape[1]
#         for det_num in range(self.num_detectors):
#             det = Detector(dim=dim)
#             det.change_radius(X)
#             if det.get_params()[0] > 0:
#                 self.expand(det)
#
#     def predict(self, X):
#         preds = np.zeros(shape=X.shape[0], dtype=np.uint8)
#         for i, ag in enumerate(X):
#             for detector in self.memory:
#                 if detector.is_ag_inside(ag):  # if inside detector
#                     preds[i] = 1  # fraud
#                     continue
#         return preds
