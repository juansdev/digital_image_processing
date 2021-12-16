import math
import numpy as np

from algorithms.edge_detection_algorithms.threshold.threshold_based_edge_detection import (
    fbc_threshold,
)
from tools.logger_base import log as log_message


def central_difference(img_to_central: np.ndarray) -> np.ndarray:
    log_message.info('========Central Difference==========')
    img_h, img_w = img_to_central.shape
    ret = np.copy(img_to_central)
    for i in range(img_h):
        for j in range(img_w):
            if i <= 0 or j <= 0 or i >= img_h - 1 or j >= img_w - 1:
                ret[i][j] = 0
            else:
                dx = float(img_to_central[i + 1][j]) - float(img_to_central[i - 1][j])
                dy = float(img_to_central[i][j + 1]) - float(img_to_central[i][j - 1])
                ret[i][j] = np.uint8(np.round(math.sqrt(dx ** 2 + dy ** 2) / 2))
    _, central = fbc_threshold(ret)
    assert not np.logical_and(central > 0, central < 255).any(), 'Image central operator isn\'t monochrome'
    return central
