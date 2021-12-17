import numpy as np

from pythreshold.local_th.nick import nick_threshold
from digital_image_processing.tools.logger_base import log as log_message


def nick_thresholding_method(img_to_nick: np.ndarray) -> np.ndarray:
    """ Runs the NICK thresholding algorithm.

    Reference:
    Khurshid, K., Siddiqi, I., Faure, C., & Vincent, N.
    (2009, January). Comparison of Niblack inspired Binarization methods for
    ancient documents. In IS&T/SPIE Electronic Imaging (pp. 72470U-72470U).
    International Society for Optics and Photonics.

    Modifications: Using integral images to compute the local mean and the
    NICK variation of the Niblack standard deviation term

    :param img_to_nick: The input image
    :type img_to_nick: ndarray

    :return: The estimated local threshold for each pixel
    :rtype: ndarray
    """

    log_message.info('========Nick\'s Thresholding==========')
    nick_th = nick_threshold(img_to_nick)
    nick_th = ((img_to_nick >= nick_th) * 255).astype(np.uint8)
    assert not np.logical_and(nick_th > 0, nick_th < 255).any(), \
        'Image adaptive threshold using nick\'s method isn\'t monochrome'
    return nick_th
