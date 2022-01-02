import numpy as np

from ais.detector import HyperSphere, Point


def test_point():
    points = [
        Point(coords=np.array([0.1, 3, 0.5])),
        Point(coords=np.array([3, 0.5])),
        Point(coords=np.array([0.1])),
        Point(coords=np.array([0.1, 3, 0.5, 1.0, 0.87])),
    ]
    ans = [3, 2, 1, 5]
    for p, a in zip(points, ans):
        assert p.get_dim() == a


def test_hyperspheres():
    sphere = HyperSphere(center=Point(coords=np.array([0.0, 0.0])), radius=1)
    points = [Point(coords=np.array([0.1, 0.5])), Point(coords=np.array([1.1, 0.5]))]
    ans = [True, False]
    for p, a in zip(points, ans):
        assert sphere.is_point_inside(p) == a
