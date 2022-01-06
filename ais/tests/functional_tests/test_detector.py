import numpy as np

from ais.detector import Detector, Point


def test_detector():
    det = Detector(center=Point(coords=np.array([1.2, 1.6])), radius=0.3)
    det.change_radius(value=0.4)
    assert det.radius == 0.7
