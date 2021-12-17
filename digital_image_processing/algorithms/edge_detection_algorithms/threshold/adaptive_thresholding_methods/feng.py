import numpy as np

from pythreshold.local_th.feng import feng_threshold
from digital_image_processing.tools.logger_base import log as log_message


def feng_thresholding_method(img_to_feng: np.ndarray) -> np.ndarray:
    """ Runs the Feng's thresholding algorithm.

    Reference:
    Algorithm proposed in: Meng-Ling Feng and Yap-Peng Tan, “Contrast adaptive
    thresholding of low quality document images”, IEICE Electron. Express,
    Vol. 1, No. 16, pp.501-506, (2004).

    Modifications: Using integral images to compute the local mean and the
    standard deviation

    :param img_to_feng: The input image. Must be a gray scale image
    :type img_to_feng: ndarray

    :return: The estimated local threshold for each pixel
    :rtype: ndarray
    """

    log_message.info('========Feng\'s Thresholding==========')
    feng_th = feng_threshold(img_to_feng)
    feng_th = ((img_to_feng >= feng_th) * 255).astype(np.uint8)
    assert not np.logical_and(feng_th > 0, feng_th < 255).any(), \
        'Image adaptive threshold using feng\'s method isn\'t monochrome'
    return feng_th
