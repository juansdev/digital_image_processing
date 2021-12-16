import numpy as np
import cv2.cv2 as cv2

from tools.logger_base import log as log_message


def threshold_value_mean(img_to_threshold: np.ndarray) -> np.ndarray:
    """ Runs the threshold using mean value.

    Reference:
    Dnyandeo, S. V., & Nipanikar, R. S., Mrs. (2016). A Review of Adaptive Thresholding Techniques for
    Vehicle Number Plate Recognition. IJARCCE, 5(4), 944–946. https://doi.org/10.17148/IJARCCE.2016.54232

    :param img_to_threshold: The input image. Must be a gray scale image
    :type img_to_threshold: ndarray

    :return: The estimated local threshold for each pixel
    :rtype: ndarray
    """

    log_message.info('========Mean\'s Thresholding==========')
    blur = cv2.medianBlur(img_to_threshold, 5)
    value_mean = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
    assert not np.logical_and(value_mean > 0, value_mean < 255).any(), \
        'Image used with threshold value mean isn\'t monochrome'
    return ~value_mean
