import numpy as np
import cv2.cv2 as cv2
from digital_image_processing.tools.logger_base import log as log_message
from typing import Tuple


def lpk_threshold(gaussian_blur: np.array) -> Tuple[float, np.array]:
    """Runs the Threshold algorithm for Laplacian, Prewitt or Kirsch operator

    Reference:
    E. (2021, 19 julio). [Python image processing] 42. Detailed explanation of Python image sharpening and edge
    detection (Roberts, Prewitt, Sobel, Laplacian, canny, log). Pythonmana.
    Recuperado 16 de diciembre de 2021, de https://pythonmana.com/2021/07/20210730134158901A.html

    :param gaussian_blur: The input image. Must be a gray scale image
    :type gaussian_blur: ndarray

    :return: The estimated local threshold for each pixel
    :rtype: ndarray
    """

    log_message.info('Applying threshold.')
    return cv2.threshold(gaussian_blur, 127, 255, cv2.THRESH_BINARY)
