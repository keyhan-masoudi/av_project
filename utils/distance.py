import numpy as np


def get_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
