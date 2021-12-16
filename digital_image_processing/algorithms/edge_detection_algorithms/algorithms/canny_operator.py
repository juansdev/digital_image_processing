import numpy as np

from algorithms.edge_detection_algorithms.threshold.threshold_based_edge_detection import (
    cs_threshold,
)
from tools.logger_base import log as log_message
from algorithms.edge_detection_algorithms.filters import (
    sobel_filter,
    gaussian_filter,
)
from algorithms.edge_detection_algorithms.others_algorithms import (
    non_max_suppression,
    hysteresis,
)


def canny_operator(img_to_canny: np.ndarray) -> np.ndarray:
    log_message.info('========Canny Operator==========')
    smooth_ret = gaussian_filter(img_to_canny)
    img_gradient_intensity, theta = sobel_filter(smooth_ret)
    img_non_max_suppression = non_max_suppression(img_gradient_intensity, theta)
    img_threshold, weak, strong = cs_threshold(img_non_max_suppression)
    canny = hysteresis(img_threshold, weak, strong)
    assert not np.logical_and(canny > 0, canny < 255).any(), 'Image canny operator isn\'t monochrome'
    return canny
