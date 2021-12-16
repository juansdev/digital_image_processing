import numpy as np

from pythreshold.local_th.bradley_roth import bradley_roth_threshold
from tools.logger_base import log as log_message


def bradley_thresholding_method(img_to_bradley: np.ndarray) -> np.ndarray:
    """ Runs the Bradley-Roth thresholding algorithm.

    Reference:
    Bradley, D., & Roth, G. (2007). Adaptive thresholding
    using the integral image. Journal of Graphics Tools, 12(2), 13-21.

    :param img_to_bradley: The input image
    :type img_to_bradley: ndarray

    :return: The estimated local threshold for each pixel
    :rtype: ndarray
    """

    log_message.info('========Bradley\'s Thresholding==========')
    bradley_th = bradley_roth_threshold(img_to_bradley)
    bradley_th = ((img_to_bradley >= bradley_th) * 255).astype(np.uint8)
    assert not np.logical_and(bradley_th > 0, bradley_th < 255).any(), \
        'Image adaptive threshold using bradley\'s method isn\'t monochrome'
    return bradley_th
