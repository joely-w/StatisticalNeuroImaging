from typing import Tuple
from random import randint
import numpy as np


def affinity(a, b, mock=False) -> float:
    """
    Calculates affinity of two adjacent voxels.
    :param a:
    :param b:
    :param mock: If no image is loaded, set mock=True to continue with random numbers.
    :return float:
    """
    alpha = 15
    return np.exp(-alpha*abs(intensity(a, mock) - intensity(b, mock)))


def intensity(coord: Tuple[int, int, int], mock=False):
    """
    Return intensity of voxel at co-ord.
    If no image is loaded, set mock=True to continue with random numbers.
    :param coord: Coordinate of voxel in 3D image.
    :param mock: Return random number instead of image.
    :return int:
    """
    if mock:
        return randint(1, 100)
    else:
        # TODO implement intensity fetching once 3D image loading has been implemented.
        return
