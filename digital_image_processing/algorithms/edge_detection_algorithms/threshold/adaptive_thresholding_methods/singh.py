import numpy as np

from pythreshold.local_th.singh import singh_threshold
from digital_image_processing.tools.logger_base import log as log_message


def singh_thresholding_method(img_to_singh: np.ndarray) -> np.ndarray:
    """ Runs the Singh thresholding algorithm

    Reference:
    Singh, O. I., Sinam, T., James, O., & Singh, T. R. (2012). Local contrast
    and mean based thresholding technique in image binarization. International
    Journal of Computer Applications, 51, 5-10.

    Modifications: Using integral images to compute local mean
        and standard deviation

    :param img_to_singh: The input image
    :type img_to_singh: ndarray

    :return: The estimated local threshold for each pixel
    :rtype: ndarray
    """

    log_message.info('========Singh\'s Thresholding==========')
    singh_th = singh_threshold(img_to_singh)
    singh_th = ((img_to_singh >= singh_th) * 255).astype(np.uint8)
    assert not np.logical_and(singh_th > 0, singh_th < 255).any(), \
        'Image adaptive threshold using singh\'s method isn\'t monochrome'
    return singh_th
