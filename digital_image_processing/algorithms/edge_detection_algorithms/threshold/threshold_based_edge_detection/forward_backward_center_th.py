import numpy as np
import cv2.cv2 as cv2
from tools.logger_base import log as log_message
from typing import Tuple


def fbc_threshold(img: np.array) -> Tuple[float, np.array]:
    log_message.info('Applying threshold.')
    return cv2.threshold(img, 255*.15, 255, cv2.THRESH_BINARY)