import numpy as np
import cv2.cv2 as cv2
from digital_image_processing.tools.logger_base import log as log_message
from typing import Tuple


def fbc_threshold(img: np.array) -> Tuple[float, np.array]:
    """Runs the Threshold algorithm for Forward Backward or Center difference

    Reference:
    M. Adnan Al-Alaoui, "Direct approach to image edge detection using differentiators," 2010 17th IEEE International
    Conference on Electronics, Circuits and Systems, 2010, pp. 154-157, doi: 10.1109/ICECS.2010.5724477.

    :param img: The input image. Must be a gray scale image
    :type img: ndarray

    :return: The estimated local threshold for each pixel
    :rtype: ndarray
    """

    log_message.info('Applying threshold.')
    return cv2.threshold(img, 255*.15, 255, cv2.THRESH_BINARY)
