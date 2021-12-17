import numpy as np
import cv2.cv2 as cv2

from scipy import signal
from digital_image_processing.tools.logger_base import log as log_message
from digital_image_processing.algorithms.edge_detection_algorithms.others_algorithms import (
    handle_img_padding,
    zero_cross_detection
)


def log(img_to_log: np.ndarray) -> np.ndarray:
    """Runs the Laplacian of Gaussian algorithm

    Reference:
    S. R. Gunn, "Edge detection error in the discrete Laplacian of Gaussian," Proceedings 1998
    International Conference on Image Processing. ICIP98 (Cat. No.98CB36269), 1998, pp. 515-519 vol.2,
    doi: 10.1109/ICIP.1998.723491. Huertas, A., & Medioni, G. (1986).
    Comparison of Edge Detection Algorithms for Automated Radiographic
    Measurement of the Carrying Angle. Journal of Biomedical Engineering and Medical Imaging,
    2(6). https://doi.org/10.14738/jbemi.26.1753.

    :param img_to_log: The input image. Must be a gray scale image
    :type img_to_log: ndarray

    :return: The estimated local operator for each pixel
    :rtype: ndarray
    """

    log_message.info('========Laplacian of Gaussian==========')
    log_kernel = np.array([
        [0, 0, 1, 0, 0],
        [0, 1, 2, 1, 0],
        [1, 2, -16, 2, 1],
        [0, 1, 2, 1, 0],
        [0, 0, 1, 0, 0]
    ])
    sobelx = cv2.Sobel(img_to_log, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_to_log, cv2.CV_64F, 0, 1, ksize=3)
    sobel_first_derivative = cv2.magnitude(sobelx, sobely)
    sobel_test = np.empty_like(sobel_first_derivative)
    sobel_test[:] = sobel_first_derivative
    sobel_test[sobel_test > 200] = 255
    sobel_test[sobel_test < 200] = 0
    ######################
    log_img = signal.convolve2d(img_to_log, log_kernel)
    zero_crossing_log = zero_cross_detection(log_img)
    zero_crossing_log = handle_img_padding(img_to_log, zero_crossing_log)
    log_img = cv2.bitwise_and(zero_crossing_log, sobel_test)
    log_img = (log_img > 0) * 255
    assert not np.logical_and(log_img > 0, log_img < 255).any(), 'Image LoG operator isn\'t monochrome'
    return log_img
