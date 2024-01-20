from typing import Tuple
from random import randint
import numpy as np

from PIL import Image

img = Image.open('elephant.jpg').convert('L')
px = img.load()


def affinity(a, b, mock=False) -> float:
    """
    Calculates affinity of two adjacent voxels.
    :param a:
    :param b:
    :param mock: If no image is loaded, set mock=True to continue with random numbers.
    :return float:
    """
    alpha = 15
    return np.exp(-alpha * abs(intensity(a, mock) - intensity(b, mock)))


def intensity(coord: Tuple[int, int, int], mock=False):
    """
    Return intensity of voxel at co-ord.
    If no image is loaded, set mock=True to continue with random numbers.
    :param coord: Coordinate of voxel in 3D image.
    :param mock: Return random number instead of image.
    :return int:
    """
    if mock:
        return int(round(0.7 * randint(1, 20) + 0.3 * randint(20, 255), 0))
    else:
        # TODO implement intensity fetching once 3D image loading has been implemented.
        return img.getpixel((coord[0], coord[1]))
