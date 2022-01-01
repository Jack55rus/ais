import numpy as np

from ais.detector import Point


def test_point():
    p = Point(coords=np.array([0.1, 3, 0.5]))
    assert p.get_dim() == 3
