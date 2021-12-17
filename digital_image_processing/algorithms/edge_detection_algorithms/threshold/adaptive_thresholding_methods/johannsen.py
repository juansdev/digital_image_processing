import numpy as np

from pythreshold.global_th.entropy.johannsen import johannsen_threshold
from digital_image_processing.tools.logger_base import log as log_message


def johannsen_thresholding_method(img_to_feng: np.ndarray) -> np.ndarray:
    """ Runs the Johannsen's threshold algorithm.

    Reference:
    Sahoo, Prasanna & Soltani, Sasan & Wong, Andrew. (1988). A Survey of Thresholding Techniques. Computer Vision,
    Graphics, and Image Processing. 41. 233-260. 10.1016/0734-189X(88)90022-9.

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
