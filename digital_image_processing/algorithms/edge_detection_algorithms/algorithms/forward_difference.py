import math
import numpy as np

from algorithms.edge_detection_algorithms.threshold.threshold_based_edge_detection import (
    fbc_threshold,
)
from tools.logger_base import log as log_message


def forward_difference(img_to_forward: np.ndarray) -> np.ndarray:
    log_message.info('========Forward Difference==========')
    img_h, img_w = img_to_forward.shape
    ret = np.copy(img_to_forward)
    for i in range(img_h):
        for j in range(img_w):
            if i >= img_h - 1 or j >= img_w - 1:
                ret[i][j] = 0
            else:
                dx = float(img_to_forward[i + 1][j]) - float(img_to_forward[i][j])
                dy = float(img_to_forward[i][j + 1]) - float(img_to_forward[i][j])
                ret[i][j] = np.uint8(np.round(math.sqrt(dx ** 2 + dy ** 2)))
    _, forward = fbc_threshold(ret)
    assert not np.logical_and(forward > 0, forward < 255).any(), 'Image forward operator isn\'t monochrome'
    return forward
