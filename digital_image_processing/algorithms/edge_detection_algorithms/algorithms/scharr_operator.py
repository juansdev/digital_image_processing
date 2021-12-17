import numpy as np
import cv2.cv2 as cv2

from digital_image_processing.algorithms.edge_detection_algorithms.threshold.threshold_based_edge_detection import (
    scharr_threshold,
)
from digital_image_processing.tools.logger_base import log as log_message


def scharr_operator(img_to_scharr: np.ndarray) -> np.ndarray:
    """Runs the Scharr Operator algorithm

    Reference:
    Comparison of Edge Detection Algorithms for Automated Radiographic Measurement of the Carrying Angle.
    Journal of Biomedical Engineering and Medical Imaging, 2(6). https://doi.org/10.14738/jbemi.26.1753. Nasution,
    T. Y., Zarlis, M., & Nasution, M. K. (2017).

    :param img_to_scharr: The input image. Must be a gray scale image
    :type img_to_scharr: ndarray

    :return: The estimated local operator for each pixel
    :rtype: ndarray
    """

    log_message.info('========Scharr Operator==========')
    gaussianBlur = cv2.GaussianBlur(img_to_scharr, (3, 3), 0)
    x = cv2.Scharr(gaussianBlur, cv2.CV_16S, 1, 0)
    y = cv2.Scharr(gaussianBlur, cv2.CV_16S, 0, 1)
    abs_x = cv2.convertScaleAbs(x)
    abs_y = cv2.convertScaleAbs(y)
    scharr = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)
    _, scharr = scharr_threshold(scharr)
    assert not np.logical_and(scharr > 0, scharr < 255).any(), 'Image scharr operator isn\'t monochrome'
    return scharr
