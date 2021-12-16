import numpy as np
from .mask_filter import mask_filter
from tools.logger_base import log as log_message


def gaussian_filter(img: np.array, size=5, sigma=1.4) -> np.array:
    log_message.info('Applying gaussian filter.')
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g = np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    ret = mask_filter(img, g)
    return ret
