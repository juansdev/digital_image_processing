import numpy as np
import cv2.cv2 as cv2

from digital_image_processing.tools.logger_base import log as log_message


def hough_transform(img_to_hough: np.array, img_original: np.array) -> np.array:
    """Runs the Hough Transform algorithm

    Reference:
    Coste, Arthur. (2012). Image Processing : Hough Transform. 10.13140/RG.2.2.30430.48969.

    :param img_to_hough: The input image. Must be a gray scale image
    :type img_to_hough: ndarray
    :param img_original: The input original image.
    :type img_original: ndarray

    :return: The estimated local for each pixel
    :rtype: ndarray
    """

    log_message.info('Finds the edges using Hough Transform.')
    edges = cv2.Canny(img_to_hough, 50, 150, apertureSize=3)
    minLineLength = 100
    maxLineGap = 10
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength, maxLineGap)
    for x1, y1, x2, y2 in lines[0]:
        cv2.line(img_original, (x1, y1), (x2, y2), (0, 255, 0), 2)
    ret = img_original
    return ret
