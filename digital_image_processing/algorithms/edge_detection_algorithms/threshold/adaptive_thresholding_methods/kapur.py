import numpy as np

from pythreshold.global_th.entropy.kapur import kapur_threshold
from digital_image_processing.tools.logger_base import log as log_message


def kapur_thresholding_method(img_to_kapur: np.ndarray) -> np.ndarray:
    """ Runs the Kapur's thresholding method.

    Reference:
    Roy, P., Dutta, S., Dey, N., Dey, G., Chakraborty, S., & Ray, R. (2014). Adaptive thresholding: A
    comparative study. 2014 International Conference on Control, Instrumentation, Communication and Computational
    Technologies (ICCICCT). Published. https://doi.org/10.1109/iccicct.2014.6993140

    :param img_to_kapur: The input image
    :type img_to_kapur: ndarray

    :return: The image with the estimated threshold
    :rtype: ndarray
    """

    log_message.info('========Kapur\'s Thresholding==========')
    kapur_th = kapur_threshold(img_to_kapur)
    kapur_th = ~((img_to_kapur >= kapur_th) * 255).astype(np.uint8)
    assert not np.logical_and(kapur_th > 0, kapur_th < 255).any(), \
        'Image adaptive threshold using bernsen\'s method isn\'t monochrome'
    return kapur_th
