import numpy as np

from tools.logger_base import log as log_message
from algorithms.edge_detection_algorithms.filters import (
    mask_filter
)


# Implementation of Difference of Gaussian (DoG)
def dog(img_to_dog: np.ndarray) -> np.ndarray:
    log_message.info('========Difference of Gaussian==========')
    m, n = [(ss - 1.) / 2. for ss in (5, 5)]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    f1 = np.exp(-(x * x + y * y) / (2. * 0.5 * 0.5))
    f1 = f1 / f1.sum()
    f2 = np.exp(-(x * x + y * y) / (2. * 1 * 1))
    f2 = f2 / f2.sum()
    f = f1 - f2
    dog_img = mask_filter(img_to_dog, f - f.mean())
    # FIXME
    assert not np.logical_and(dog_img > 0, dog_img < 255).any(), 'Image DoG operator isn\'t monochrome'
    return np.uint8(np.round(np.absolute(dog_img)))
