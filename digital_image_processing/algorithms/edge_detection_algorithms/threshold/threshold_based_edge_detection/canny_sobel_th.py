import numpy as np
from digital_image_processing.tools.logger_base import log as log_message


def cs_threshold(img: np.array, low_threshold_ratio=0.05, high_threshold_ratio=0.09) -> tuple:
    """Runs the Threshold algorithm for Canny or Sobel operator

    Reference:
    Goel, K., Sehrawat, M., & Agarwal, A. (2017). Finding the optimal threshold values for edge detection
    of digital images & comparing among Bacterial Foraging Algorithm, canny and Sobel Edge Detector. 2017
    International Conference on Computing, Communication and Automation (ICCCA), 1076-1080.

    :param img: The input image. Must be a gray scale image
    :type img: ndarray
    :param low_threshold_ratio: Low threshold ratio
    :type low_threshold_ratio: float
    :param high_threshold_ratio: High threshold ratio
    :type high_threshold_ratio: float

    :return: The estimated local threshold for each pixel
    :rtype: ndarray
    """

    log_message.info('Applying threshold.')
    high_threshold = img.max() * high_threshold_ratio
    low_threshold = high_threshold * low_threshold_ratio

    M, N = img.shape
    res = np.zeros((M, N), dtype=np.int32)

    weak = np.int32(25)
    strong = np.int32(255)

    strong_i, strong_j = np.where(img >= high_threshold)

    weak_i, weak_j = np.where((img <= high_threshold) & (img >= low_threshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return res, weak, strong
