import numpy as np
from math import sqrt


class Detector:
    # params: radius, coords
    # methods: get params, create
    def __init__(self, dim):
        self.dim = dim  # dimensionality
        self.coords = np.random.rand(self.dim)  # define random coords for the Detector
        self.radius = np.random.random()  # do the same for its radius
        self.eps = 1e-8  # how much detector radius is less than the minimum dist to ag
        # add range for params later

    def get_params(self):
        return self.radius, self.coords

    def is_ag_inside(self, ag):
        '''
        Checks if an antigen is inside the detector
        :param ag: np.array (dim,)
        :return: True if point inside, False otherwise
        '''
        distance = np.linalg.norm(self.coords - ag)
        return True if sqrt(distance) < self.radius else False

    def ag_distance(self, ag: np.array) -> float:
        '''
        Computes distance between an antigen and a center of the detector
        :param ag:
        :return:
        '''
        return np.linalg.norm(self.coords - ag)

    def min_ag_dist(self, ags):
        unsqueezed_coords = np.expand_dims(self.coords, axis=0)
        repeated_coords = np.repeat(unsqueezed_coords, ags.shape[0], axis=0)
        min_dist = min(np.linalg.norm(ags - repeated_coords, axis=1))
        return min_dist

    def change_radius(self, ags):
        self.radius = self.min_ag_dist(ags) - self.eps






