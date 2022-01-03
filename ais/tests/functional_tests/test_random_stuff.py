import numpy as np

from ais.detector import RandomPointGenerator


def test_random_point():
    rand_p = RandomPointGenerator(
        dim=4, low=np.array([[-1.0], [-1.0], [-1.0], [-1.0]]), high=np.array([[1.0], [1.0], [1.0], [1.0]])
    )
    assert np.min(rand_p().get_coords()) > -1.0
    assert np.max(rand_p().get_coords()) < 1.0
