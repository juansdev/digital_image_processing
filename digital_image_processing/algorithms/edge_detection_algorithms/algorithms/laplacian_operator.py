import numpy as np
import cv2.cv2 as cv2

from digital_image_processing.algorithms.edge_detection_algorithms.threshold.threshold_based_edge_detection import (
    lpk_threshold,
)
from digital_image_processing.tools.logger_base import log as log_message


def laplacian_operator(img_to_laplacian: np.ndarray) -> np.ndarray:
    """Runs the Laplacian Operator algorithm

    Reference:
    Detection of Intensity Changes with Subpixel
    Accuracy Using Laplacian-Gaussian Masks. IEEE Transactions on Pattern Analysis and Machine Intelligence,
    PAMI-8(5), 651–664. https://doi.org/10.1109/tpami.1986.4767838.

    :param img_to_laplacian: The input image. Must be a gray scale image
    :type img_to_laplacian: ndarray

    :return: The estimated local operator for each pixel
    :rtype: ndarray
    """

    log_message.info('========Laplacian Operator==========')
    gaussianBlur = cv2.GaussianBlur(img_to_laplacian, (3, 3), 0)
    ret, binary = lpk_threshold(gaussianBlur)
    dst = cv2.Laplacian(binary, cv2.CV_16S, ksize=3)
    laplacian = cv2.convertScaleAbs(dst)
    assert not np.logical_and(laplacian > 0, laplacian < 255).any(), 'Image laplacian operator isn\'t monochrome'
    return laplacian
