import numpy as np
from math import sqrt
from abc import ABCMeta, abstractmethod


class Point:
    def __init__(self, args: np.array):
        self.coords = args
        self.dim = len(self.coords)

    def get_dim(self) -> int:
        return self.dim

    def get_coords(self) -> np.array:
        return self.coords

    def calc_dist_to_other_point(self, other_point: 'Point') -> float:
        return np.linalg.norm(self.coords - other_point)

    # todo: method to calc dist to multiple points (think over some class for handling this stuff)
    # def calc_dist_to_multiple_points(self, points: 'Point'):


class HyperSphere:
    def __init__(self, coords: Point, radius: float):
        self.coords = coords
        self.radius = radius

    def get_radius(self) -> float:
        return self.radius

    def get_coords(self) -> Point:
        return self.coords

    def get_params(self) -> tuple[Point, float]:
        return self.coords, self.radius

    def is_point_inside(self, point: Point):
        distance = self.coords.calc_dist_to_other_point(point)  # distance between center and some point
        return True if sqrt(distance) < self.radius else False


class RandomParamsGenerator(metaclass=ABCMeta):
    ''' This is a base class for generating random parts of the algorithm '''
    def __init__(self, *args):
        pass

    def __call__(self, *args):
        pass


class RandomPointGenerator(RandomParamsGenerator):
    def __init__(self, dim):
        RandomParamsGenerator.__init__(self)
        self.dim = dim

    def __call__(self) -> Point:
        return Point(np.random.rand(self.dim).flatten())


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
    def __init__(self, center: Point, radius: float, eps: float = 1e-8):
        HyperSphere.__init__(self, center, radius)
        self.eps = eps

    def change_radius(self, value: float):
        self.radius += value




# class Detector:
#     def __init__(self, dim):
#         self.dim = dim  # dimensionality
#         self.coords = np.random.rand(self.dim)  # define random coords for the Detector
#         self.radius = np.random.random()  # do the same for its radius
#         self.eps = 1e-8  # how much detector radius is less than the minimum dist to ag
#         # add range for params later
#
#     def get_params(self):
#         return self.radius, self.coords
#
#     def is_ag_inside(self, ag):
#         '''
#         Checks if an antigen is inside the detector
#         :param ag: np.array (dim,)
#         :return: True if point inside, False otherwise
#         '''
#         distance = np.linalg.norm(self.coords - ag)
#         return True if sqrt(distance) < self.radius else False
#
#     def ag_distance(self, ag: np.array) -> float:
#         '''
#         Computes distance between an antigen and a center of the detector
#         :param ag:
#         :return:
#         '''
#         return np.linalg.norm(self.coords - ag)
#
#     def min_ag_dist(self, ags):
#         unsqueezed_coords = np.expand_dims(self.coords, axis=0)
#         repeated_coords = np.repeat(unsqueezed_coords, ags.shape[0], axis=0)
#         min_dist = min(np.linalg.norm(ags - repeated_coords, axis=1))
#         return min_dist
#
#     def change_radius(self, ags):
#         self.radius = self.min_ag_dist(ags) - self.eps
