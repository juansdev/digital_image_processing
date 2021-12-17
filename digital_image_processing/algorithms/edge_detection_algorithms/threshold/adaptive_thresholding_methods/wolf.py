import numpy as np

from pythreshold.local_th.wolf import wolf_threshold
from digital_image_processing.tools.logger_base import log as log_message


def wolf_thresholding_method(img_to_wolf: np.ndarray) -> np.ndarray:
    """ Runs the Wolf's thresholding algorithm.

    Reference:
    C. Wolf, J-M. Jolion, “Extraction and Recognition
    of Artificial Text in Multimedia Documents”, Pattern Analysis and
    Applications, 6(4):309-326, (2003).

    Modifications: Using integral images to compute the local mean and the
    standard deviation

    :param img_to_wolf: The input image
    :type img_to_wolf: ndarray

    :return: The estimated local threshold for each pixel
    :rtype: ndarray
    """

    log_message.info('========Wolf\'s Thresholding==========')
    wolf_th = wolf_threshold(img_to_wolf)
    wolf_th = ((img_to_wolf >= wolf_th) * 255).astype(np.uint8)
    assert not np.logical_and(wolf_th > 0, wolf_th < 255).any(), \
        'Image adaptive threshold using wolf\'s method isn\'t monochrome'
    return wolf_th
