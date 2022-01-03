import numpy as np

from ais.detector import DistanceCalculator, Point


def test_dist_between_points():
    p1 = Point(coords=np.array([0.0, 1.0]))
    p2 = Point(coords=np.array([1.0, 2.0]))
    assert DistanceCalculator.calc_dist_between_points(p1, p2) == 2 ** 0.5


def test_dist_between_ags_and_point():
    ags = np.array([[1.0, 1.0], [2.0, 2.0]])
    p = Point(coords=np.array([1.0, 2.0]))
    assert np.all(DistanceCalculator.calc_dist_between_ags_and_point(ags=ags, point=p) == np.array([1.0, 1.0]))


def test_min_dist_between_ags_and_point():
    ags = np.array([[1.0, 1.0, 0.0], [2.0, 0.0, 1.0], [0.5, 0.5, 0.5]])
    p = Point(coords=np.array([1.0, 2.0, 0.5]))
    # d1 = 1.118; d2 = 2.291; d3 = 1.581
    assert round(DistanceCalculator.calc_min_dist_between_ags_and_point(ags=ags, point=p), 2) == 1.12


# todo: then random params test
