import numpy as np

from pythreshold.local_th.sauvola import sauvola_threshold
from digital_image_processing.tools.logger_base import log as log_message


def sauvola_thresholding_method(img_to_sauvola: np.ndarray) -> np.ndarray:
    """ Runs the sauvola's thresholding algorithm.

    Reference:
    Sauvola, J., Seppanen, T., Haapakoski, S., and Pietikainen, M.:
    ‘Adaptive document thresholding’. Proc. 4th Int. Conf. on Document
    Analysis and Recognition, Ulm Germany, 1997, pp. 147–152.

    Modifications: Using integral images to compute local mean
        and standard deviation

    :param img_to_sauvola: The input image
    :type img_to_sauvola: ndarray

    :return: The estimated local threshold for each pixel
    :rtype: ndarray
    """

    log_message.info('========Sauvola\'s Thresholding==========')
    sauvola_th = sauvola_threshold(img_to_sauvola)
    sauvola_th = ((img_to_sauvola >= sauvola_th) * 255).astype(np.uint8)
    assert not np.logical_and(sauvola_th > 0, sauvola_th < 255).any(), \
        'Image adaptive threshold using sauvola\'s method isn\'t monochrome'
    return sauvola_th
