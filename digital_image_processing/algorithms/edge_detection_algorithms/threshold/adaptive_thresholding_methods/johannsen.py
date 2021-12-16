import numpy as np

from pythreshold.global_th.entropy.johannsen import johannsen_threshold
from tools.logger_base import log as log_message


def johannsen_thresholding_method(img_to_feng: np.ndarray) -> np.ndarray:
    """ Runs the Johannsen's threshold algorithm.

    Reference:
    Johannsen, G., and J. Bille ‘‘A Threshold Selection Method Using
    Information Measures,’’ Proceedings of the Sixth International Conference
    on Pattern Recognition, Munich, Germany (1982): 140–143.

    :param img_to_feng: The input image
    :type img_to_feng: ndarray

    :return: The estimated threshold
    :rtype: int
    """

    log_message.info('========Johannsen\'s Thresholding==========')
    johannsen_th = johannsen_threshold(img_to_feng)
    johannsen_th = ((img_to_feng >= johannsen_th) * 255).astype(np.uint8)
    assert not np.logical_and(johannsen_th > 0, johannsen_th < 255).any(), \
        'Image adaptive threshold using johannsen\'s method isn\'t monochrome'
    return johannsen_th
