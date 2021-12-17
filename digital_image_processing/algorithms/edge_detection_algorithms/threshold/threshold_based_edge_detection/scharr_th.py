import numpy as np
import cv2.cv2 as cv2
from digital_image_processing.tools.logger_base import log as log_message
from typing import Tuple


def scharr_threshold(img: np.array) -> Tuple[float, np.array]:
    """Runs the Threshold algorithm for Scharr operator

    Reference:
    Topal, C., & Akinlar, C. (2012). Edge Drawing: A combined real-time edge and segment detector.
    Journal of Visual Communication and Image Representation, 23(6), 862–872.
    https://doi.org/10.1016/j.jvcir.2012.05.004

    :param img: The input image. Must be a gray scale image
    :type img: ndarray

    :return: The estimated local threshold for each pixel
    :rtype: ndarray
    """

    log_message.info('Applying threshold.')
    return cv2.threshold(img, 144, 255, cv2.THRESH_BINARY)
