import numpy as np
import cv2.cv2 as cv2
from tools.logger_base import log as log_message
from typing import Tuple


def lpk_threshold(gaussian_blur: np.array) -> Tuple[float, np.array]:
    log_message.info('Applying threshold.')
    return cv2.threshold(gaussian_blur, 127, 255, cv2.THRESH_BINARY)
