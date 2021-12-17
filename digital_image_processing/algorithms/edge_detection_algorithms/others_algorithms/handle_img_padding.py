import numpy as np


def handle_img_padding(img: np.ndarray, img_zero_crossing: np.ndarray) -> np.ndarray:
    """Runs the Handle Img Padding algorithm

    :param img: The input image. Must be a gray scale image
    :type img: ndarray
    :param img_zero_crossing: The image with zero crossing
    :type img_zero_crossing: ndarray

    :return: Image with padding applied
    :rtype: ndarray
    """

    M1, N1 = img.shape[:2]
    M2, N2 = img_zero_crossing.shape[:2]
    padding_x = int(np.abs(M2 - M1)/2)
    padding_y = int(np.abs(N2 - N1)/2)
    res = img_zero_crossing[padding_x:M1+padding_x, padding_y: N1+padding_y]
    return res
