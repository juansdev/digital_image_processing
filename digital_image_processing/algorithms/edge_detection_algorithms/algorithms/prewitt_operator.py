import numpy as np
import cv2.cv2 as cv2

from digital_image_processing.algorithms.edge_detection_algorithms.threshold.threshold_based_edge_detection import (
    lpk_threshold,
)
from digital_image_processing.tools.logger_base import log as log_message


def prewitt_operator(img_to_prewitt: np.ndarray) -> np.ndarray:
    """Runs the Prewitt Operator algorithm

    Reference:
    Comparison of Edge Detection Algorithms for Automated Radiographic
    Measurement of the Carrying Angle. Journal of Biomedical Engineering and Medical Imaging,
    2(6). https://doi.org/10.14738/jbemi.26.1753.

    :param img_to_prewitt: The input image. Must be a gray scale image
    :type img_to_prewitt: ndarray

    :return: The estimated local operator for each pixel
    :rtype: ndarray
    """

    log_message.info('========Prewitt Operador==========')
    prewitt_x = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]], dtype='int')
    prewitt_y = np.array([
        [1, 1, 1],
        [0, 0, 0],
        [-1, -1, -1]], dtype='int')
    gaussianBlur = cv2.GaussianBlur(img_to_prewitt, (3, 3), 0)
    # Threshold 127 because the brightness range is 0 ~ 255
    ret, binary = lpk_threshold(gaussianBlur)
    dx = cv2.filter2D(binary, cv2.CV_16S, prewitt_x)
    dy = cv2.filter2D(binary, cv2.CV_16S, prewitt_y)
    absX = cv2.convertScaleAbs(dx)
    absY = cv2.convertScaleAbs(dy)
    prewitt = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    ret, prewitt = lpk_threshold(prewitt)
    assert not np.logical_and(prewitt > 0, prewitt < 255).any(), 'Image prewitt operator isn\'t monochrome'
    return prewitt
