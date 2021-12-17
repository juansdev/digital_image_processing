import numpy as np

from digital_image_processing.tools.logger_base import log as log_message


def hysteresis(img: np.ndarray, weak: int, strong: int = 255):
    """Runs the Hysteresis algorithm

    Reference:
    Sornam, M., Kavitha, M. S., & Nivetha, M. (2016). Hysteresis thresholding based edge detectors for
    inscriptional image enhancement. 2016 IEEE International Conference on Computational Intelligence and Computing
    Research (ICCIC). Published. https://doi.org/10.1109/iccic.2016.7919568

    :param img: The input image. Must be a gray scale image
    :type img: ndarray
    :param weak: Weak
    :type weak: int
    :param strong: Strong for default 255
    :type strong: int

    :return: Image with hysteresis applied
    :rtype: ndarray
    """

    log_message.info('Applying hysteresis.')
    M, N = img.shape
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if img[i, j] == weak:
                try:
                    if ((img[i + 1, j - 1] == strong) or (img[i + 1, j] == strong) or (img[i + 1, j + 1] == strong)
                            or (img[i, j - 1] == strong) or (img[i, j + 1] == strong)
                            or (img[i - 1, j - 1] == strong) or (img[i - 1, j] == strong) or (
                                    img[i - 1, j + 1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img
