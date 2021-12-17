import numpy as np

from pythreshold.global_th.entropy.pun import pun_threshold
from digital_image_processing.tools.logger_base import log as log_message


def pun_thresholding_method(img_to_pun: np.ndarray) -> np.ndarray:
    """ Runs the Pun's threshold algorithm.

    Reference:
    Pun, T. ‘‘A New Method for Grey-Level Picture Thresholding Using the
    Entropy of the Histogram,’’ Signal Processing 2, no. 3 (1980): 223–237.

    :param img_to_pun: The input image
    :type img_to_pun: ndarray

    :return: The estimated threshold
    :rtype: int
    """

    log_message.info('========Pun\'s Thresholding==========')
    pun_th = pun_threshold(img_to_pun)
    pun_th = ((img_to_pun >= pun_th) * 255).astype(np.uint8)
    assert not np.logical_and(pun_th > 0, pun_th < 255).any(), \
        'Image adaptive threshold using p-tile method isn\'t monochrome'
    return pun_th
