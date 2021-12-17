import numpy as np
from digital_image_processing.tools.logger_base import log as log_message


def roberts_threshold(img: np.ndarray) -> np.ndarray:
    """Runs the Threshold algorithm for Roberts operator

    Reference:
    Fisher, R., Perkins, S., Walker, A., & Wolfart, E. (s. f.). Feature Detectors - Roberts Cross Edge Detector.
    homepages. Recuperado 16 de diciembre de 2021, de https://homepages.inf.ed.ac.uk/rbf/HIPR2/roberts.htm

    :param img: The input image. Must be a gray scale image
    :type img: ndarray

    :return: The estimated local threshold for each pixel
    :rtype: ndarray
    """

    log_message.info('Applying threshold.')
    return ((img * 5) > 80) * 255
