import numpy as np
import cv2.cv2 as cv2

from tools.logger_base import log as log_message


def threshold_value_gaussian(img_to_threshold: np.ndarray) -> np.ndarray:
    """ Runs the threshold using as value a gaussian-weighted.

    Reference:
    Dnyandeo, S. V., & Nipanikar, R. S., Mrs. (2016). A Review of Adaptive Thresholding Techniques for
    Vehicle Number Plate Recognition. IJARCCE, 5(4), 944–946. https://doi.org/10.17148/IJARCCE.2016.54232

    :param img_to_threshold: The input image. Must be a gray scale image
    :type img_to_threshold: ndarray

    :return: The estimated local threshold for each pixel
    :rtype: ndarray
    """

    log_message.info('========Gaussian\'s Thresholding==========')
    value_gaussian = cv2.adaptiveThreshold(img_to_threshold, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 199, 5)
    assert not np.logical_and(value_gaussian > 0, value_gaussian < 255).any(), \
        'Image used with threshold value gaussian isn\'t monochrome'
    return ~value_gaussian
