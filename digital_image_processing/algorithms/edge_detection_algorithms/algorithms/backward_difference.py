import math
import numpy as np

from digital_image_processing.algorithms.edge_detection_algorithms.threshold.threshold_based_edge_detection import (
    fbc_threshold,
)
from digital_image_processing.tools.logger_base import log as log_message


def backward_difference(img_to_backward: np.ndarray) -> np.ndarray:
    """Runs the Backward Difference algorithm

    Reference:
    Bustacara-Medina, C., & Flórez-Valencia, L. (2016). Comparison and Evaluation of First Derivatives
    Estimation. Computer Vision and Graphics, 121–133. https://doi.org/10.1007/978-3-319-46418-3_11

    :param img_to_backward: The input image. Must be a gray scale image
    :type img_to_backward: ndarray

    :return: The estimated local operator for each pixel
    :rtype: ndarray
    """

    log_message.info('========Backward Difference==========')
    img_h, img_w = img_to_backward.shape
    ret = np.copy(img_to_backward)
    for i in range(img_h):
        for j in range(img_w):
            if i <= 0 or j <= 0:
                ret[i][j] = 0
            else:
                dx = float(img_to_backward[i][j]) - float(img_to_backward[i - 1][j])
                dy = float(img_to_backward[i][j]) - float(img_to_backward[i][j - 1])
                ret[i][j] = np.uint8(np.round(math.sqrt(dx ** 2 + dy ** 2)))
    _, backward = fbc_threshold(ret)
    assert not np.logical_and(backward > 0, backward < 255).any(), 'Image backward operator isn\'t monochrome'
    return backward
