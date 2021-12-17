import numpy as np
import cv2.cv2 as cv2

from digital_image_processing.algorithms.edge_detection_algorithms.threshold.threshold_based_edge_detection import (
    cs_threshold,
)
from digital_image_processing.tools.logger_base import log as log_message
from digital_image_processing.algorithms.edge_detection_algorithms.filters import (
    mask_filter
)


def sobel_operator(img_to_sobel: np.ndarray) -> np.ndarray:
    """Runs the Sobel Operator algorithm

    Reference:
    Comparison of Edge Detection Algorithms for Automated Radiographic Measurement of the Carrying Angle.
    Journal of Biomedical Engineering and Medical Imaging, 2(6). https://doi.org/10.14738/jbemi.26.1753. Nasution,
    T. Y., Zarlis, M., & Nasution, M. K. (2017).
    Sobel, Irwin. (2014). An Isotropic 3x3 Image Gradient Operator. Presentation at Stanford A.I. Project 1968.

    :param img_to_sobel: The input image. Must be a gray scale image
    :type img_to_sobel: ndarray

    :return: The estimated local operator for each pixel
    :rtype: ndarray
    """

    log_message.info('========Sobel Operator==========')
    gaussianBlur = cv2.GaussianBlur(img_to_sobel, (3, 3), 0)
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]], dtype='float') / 8
    sobel_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]], dtype='float') / 8
    dx = mask_filter(gaussianBlur, sobel_x)
    dy = mask_filter(gaussianBlur, sobel_y)
    ret: np.ndarray = np.uint8(np.round(np.sqrt(dx ** 2 + dy ** 2)))
    ret, weak, strong = cs_threshold(ret)
    sobel: np.ndarray = (ret > weak) * 255
    assert not np.logical_and(sobel > 0, sobel < 255).any(), 'Image sobel operator isn\'t monochrome'
    return sobel
