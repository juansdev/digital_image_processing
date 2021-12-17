import numpy as np
import cv2.cv2 as cv2

from digital_image_processing.algorithms.edge_detection_algorithms.threshold.threshold_based_edge_detection import (
    roberts_threshold
)
from digital_image_processing.tools.logger_base import log as log_message


def roberts_operator(img_to_roberts: np.ndarray) -> np.ndarray:
    """Runs the Roberts Operator algorithm

    Reference:
    Jain, R., Kasturi, R., & Schunck, B. G. (1995). Machine Vision (1.a ed.). Mcgraw-Hill College.

    :param img_to_roberts: The input image. Must be a gray scale image
    :type img_to_roberts: ndarray

    :return: The estimated local operator for each pixel
    :rtype: ndarray
    """

    log_message.info('========Roberts Operator==========')
    #  Roberts operator
    kernelx = np.array([[1, 0],
                        [0, -1]], dtype=int)
    kernely = np.array([[0, -1],
                        [1, 0]], dtype=int)
    x = cv2.filter2D(img_to_roberts, cv2.CV_16S, kernelx)
    y = cv2.filter2D(img_to_roberts, cv2.CV_16S, kernely)
    #  Convert to uint8
    abs_x = cv2.convertScaleAbs(x)
    abs_y = cv2.convertScaleAbs(y)
    ret = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)
    roberts = roberts_threshold(ret)
    assert not np.logical_and(roberts > 0, roberts < 255).any(), 'Image roberts operator isn\'t monochrome'
    return roberts
