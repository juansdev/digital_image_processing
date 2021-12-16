import numpy as np
import cv2.cv2 as cv2

from algorithms.edge_detection_algorithms.threshold.threshold_based_edge_detection import (
    lpk_threshold,
)
from tools.logger_base import log as log_message


def laplacian_operator(img_to_laplacian: np.ndarray) -> np.ndarray:
    log_message.info('========Laplacian Operator==========')
    gaussianBlur = cv2.GaussianBlur(img_to_laplacian, (3, 3), 0)
    ret, binary = lpk_threshold(gaussianBlur)
    dst = cv2.Laplacian(binary, cv2.CV_16S, ksize=3)
    laplacian = cv2.convertScaleAbs(dst)
    assert not np.logical_and(laplacian > 0, laplacian < 255).any(), 'Image laplacian operator isn\'t monochrome'
    return laplacian
