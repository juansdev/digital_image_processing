import numpy as np

from pythreshold.local_th.bernsen import bernsen_threshold
from digital_image_processing.tools.logger_base import log as log_message


def bernsen_thresholding_method(img_to_kapur: np.ndarray) -> np.ndarray:
    """Runs the Bernsen thresholding algorithm

    Reference:
    Eyupoglu, Can. (2016). Implementation of Bernsen’s Locally Adaptive Binarization Method for Gray Scale Images.

    :param img_to_kapur: The input image. Must be a gray scale image
    :type img_to_kapur: ndarray

    :return: The estimated local threshold for each pixel
    :rtype: ndarray
    """

    log_message.info('========Bernsen\'s Thresholding==========')
    bernsen_th = bernsen_threshold(img_to_kapur)
    bernsen_th = ((img_to_kapur >= bernsen_th) * 255).astype(np.uint8)
    assert not np.logical_and(bernsen_th > 0, bernsen_th < 255).any(), \
        'Image adaptive threshold using bernsen\'s method isn\'t monochrome'
    return bernsen_th
