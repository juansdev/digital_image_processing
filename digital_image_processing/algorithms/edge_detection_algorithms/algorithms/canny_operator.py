import numpy as np

from digital_image_processing.algorithms.edge_detection_algorithms.threshold.threshold_based_edge_detection import (
    cs_threshold,
)
from digital_image_processing.tools.logger_base import log as log_message
from digital_image_processing.algorithms.edge_detection_algorithms.filters import (
    sobel_filter,
    gaussian_filter,
)
from digital_image_processing.algorithms.edge_detection_algorithms.others_algorithms import (
    non_max_suppression,
    hysteresis,
)


def canny_operator(img_to_canny: np.ndarray) -> np.ndarray:
    """Runs the Canny Operator algorithm

    Reference: Canny, J. (1986). A Computational Approach to Edge Detection. IEEE Transactions on Pattern Analysis
    and Machine Intelligence, PAMI-8(6), 679–698. https://doi.org/10.1109/tpami.1986.4767851

    :param img_to_canny: The input image. Must be a gray scale image
    :type img_to_canny: ndarray

    :return: The estimated local operator for each pixel
    :rtype: ndarray
    """

    log_message.info('========Canny Operator==========')
    smooth_ret = gaussian_filter(img_to_canny)
    img_gradient_intensity, theta = sobel_filter(smooth_ret)
    img_non_max_suppression = non_max_suppression(img_gradient_intensity, theta)
    img_threshold, weak, strong = cs_threshold(img_non_max_suppression)
    canny = hysteresis(img_threshold, weak, strong)
    assert not np.logical_and(canny > 0, canny < 255).any(), 'Image canny operator isn\'t monochrome'
    return canny
