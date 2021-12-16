import numpy as np

from pythreshold.local_th.contrast import contrast_threshold
from tools.logger_base import log as log_message


def contrast_thresholding_method(img_to_contrast: np.ndarray) -> np.ndarray:
    """Runs the contrast thresholding algorithm

    Reference:
    Parker, J. R. (2010). Algorithms for image processing and
    computer vision. John Wiley & Sons.

    :param img_to_contrast: The input image. Must be a gray scale image
    :type img_to_contrast: ndarray

    :return: The estimated local threshold for each pixel
    :rtype: ndarray
    """

    log_message.info('========Contrast\'s Thresholding==========')
    contrast_th = contrast_threshold(img_to_contrast)
    contrast_th = ((img_to_contrast >= contrast_th) * 255).astype(np.uint8)
    assert not np.logical_and(contrast_th > 0, contrast_th < 255).any(), \
        'Image adaptive threshold using bradley\'s method isn\'t monochrome'
    return contrast_th
