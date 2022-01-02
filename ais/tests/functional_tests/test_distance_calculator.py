import numpy as np

from ais.detector import DistanceCalculator, Point


def test_dist_between_points():
    p1 = Point(coords=np.array([0.0, 1.0]))
    p2 = Point(coords=np.array([1.0, 2.0]))
    assert DistanceCalculator.calc_dist_between_points(p1, p2) == 2 ** 0.5


def test_dist_between_ags_and_point():
    ags = np.array([[1.0, 1.0], [2, 2.0]])
    p = Point(coords=np.array([1.0, 2.0]))
    print(DistanceCalculator.calc_dist_between_ags_and_point(ags=ags, point=p))
    assert np.all(DistanceCalculator.calc_dist_between_ags_and_point(ags=ags, point=p) == np.array([1.0, 1.0]))


# todo: calc_min_dist_between_ags_and_point test; then random params test
