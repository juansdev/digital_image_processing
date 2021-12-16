import numpy as np

from pythreshold.local_th.niblack import niblack_threshold
from tools.logger_base import log as log_message


def niblack_thresholding_method(img_to_niblack: np.ndarray) -> np.ndarray:
    """ Runs the niblack's thresholding algorithm.

    Reference:
    Niblack, W.: ‘An introduction to digital image
    processing’ (Prentice- Hall, Englewood Cliffs, NJ, 1986), pp. 115–116

    Modifications: Using integral images to compute the local mean and
    standard deviation

    :param img_to_niblack: The input image
    :type img_to_niblack: ndarray

    :return: The estimated local threshold for each pixel
    :rtype: ndarray
    """

    log_message.info('========Niblack\'s Thresholding==========')
    niblack_th = niblack_threshold(img_to_niblack)
    niblack_th = ((img_to_niblack >= niblack_th) * 255).astype(np.uint8)
    assert not np.logical_and(niblack_th > 0, niblack_th < 255).any(), \
        'Image adaptive threshold using johannsen\'s method isn\'t monochrome'
    return niblack_th
