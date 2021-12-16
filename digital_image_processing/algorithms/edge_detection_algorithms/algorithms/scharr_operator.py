import numpy as np
import cv2.cv2 as cv2

from algorithms.edge_detection_algorithms.threshold.threshold_based_edge_detection import (
    scharr_threshold,
)
from tools.logger_base import log as log_message


def scharr_operator(img_to_scharr: np.ndarray) -> np.ndarray:
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
