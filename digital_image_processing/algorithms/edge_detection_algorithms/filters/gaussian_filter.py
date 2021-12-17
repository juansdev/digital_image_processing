import numpy as np
from .mask_filter import mask_filter
from digital_image_processing.tools.logger_base import log as log_message


def gaussian_filter(img: np.array, size=5, sigma=1.4) -> np.array:
    """Runs the Gaussian Filter algorithm

    Reference:
    Gedraite, Estevao & Hadad, M.. (2011). Investigation on the effect of a Gaussian Blur in image
    filtering and segmentation. 393-396.

    :param img: The input image. Must be a gray scale image
    :type img: ndarray
    :param size: The size of kernel gaussian
    :type size: int
    :param sigma: The sigma
    :type sigma: int

    :return: The estimated local filter for each pixel
    :rtype: ndarray
    """

    log_message.info('Applying gaussian filter.')
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g = np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    ret = mask_filter(img, g)
    return ret
