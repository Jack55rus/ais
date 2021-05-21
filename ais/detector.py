from abc import ABCMeta, abstractmethod
from typing import Optional

import numpy as np


class Point:
    def __init__(self, coords: np.array):
        self.coords = coords
        self.dim = len(self.coords)

    def get_dim(self) -> int:
        return self.dim

    def get_coords(self) -> np.array:
        return self.coords


class HyperSphere:
    def __init__(self, center: Point, radius: float):
        self.center = center
        self.radius = radius

    def get_radius(self) -> float:
        return self.radius

    def get_coords(self) -> Point:
        return self.center

    def get_params(self) -> tuple[Point, float]:
        return self.center, self.radius

    def is_point_inside(self, point: Point):
        # distance = self.coords.calc_dist_to_other_point(point)  # distance between center and some point
        distance = DistanceCalculator.calc_dist_between_points(
            self.center, point
        )  # distance between center and some point
        return True if distance < self.radius else False


class DistanceCalculator:
    @staticmethod
    def calc_dist_between_points(point_1: Point, point_2: Point) -> float:
        assert point_1.get_dim() == point_2.get_dim(), "dimensionality must be the same"
        return np.linalg.norm(point_1.get_coords() - point_2.get_coords())

    @classmethod
    def calc_dist_between_ags_and_point(cls, ags: np.ndarray, point: Point) -> np.ndarray:
        assert ags.shape[1] == point.get_dim(), "dimensionality must be the same"
        point_coord = point.get_coords()
        point_coord = np.expand_dims(point_coord, axis=0)
        point_coord_rep = np.repeat(point_coord, ags.shape[0], axis=0)
        return np.linalg.norm(ags - point_coord_rep, axis=1)

    @classmethod
    def calc_min_dist_between_ags_and_point(cls, ags: np.ndarray, point: Point) -> float:
        return np.min(cls.calc_dist_between_ags_and_point(ags, point))


class RandomParamsGenerator(metaclass=ABCMeta):
    """ This is a base class for generating random parts of the algorithm """

    @abstractmethod
    def __init__(self, *args):
        pass

    @abstractmethod
    def __call__(self, *args):
        pass


class RandomPointGenerator(RandomParamsGenerator):
    def __init__(self, dim, low, high):
        RandomParamsGenerator.__init__(self)
        self.dim = dim
        self.low = low
        self.high = high

    def __call__(self) -> Point:
        # return Point(np.random.rand(self.dim).flatten())
        return Point(np.random.uniform(low=self.low, high=self.high).flatten())


class RandomHyperSphereGenerator(RandomParamsGenerator):
    def __init__(self, dim):
        RandomParamsGenerator.__init__(self)
        self.point = RandomPointGenerator(dim)
        self.dim = dim

    def __call__(self) -> HyperSphere:
        center = self.point()
        radius = np.random.random()
        return HyperSphere(center, radius)


class Detector(HyperSphere):
    def __init__(self, center: Point, radius: float, eps: Optional[float] = None):
        HyperSphere.__init__(self, center, radius)
        self.eps = eps
        self.init_radius()

    def change_radius(self, value: float):
        # not sure if this method can be applied explicitly, but I'll keep for now
        self.radius += value

    def init_radius(self):
        if self.eps is not None:
            self.change_radius(self.eps)
