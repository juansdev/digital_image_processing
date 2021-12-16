import numpy as np

from pythreshold.global_th.p_tile import p_tile_threshold
from tools.logger_base import log as log_message


def p_tile_thresholding_method(img_to_p_tile: np.ndarray) -> np.ndarray:
    """Runs the p-tile threshold algorithm.

    Reference:
    Parker, J. R. (2010). Algorithms for image processing and
    computer vision. John Wiley & Sons.

    :param img_to_p_tile: The input image
    :type img_to_p_tile: ndarray

    :return: The p-tile global threshold
    :rtype int
    """

    log_message.info('========P Tile\'s Thresholding==========')
    p_tile_th = p_tile_threshold(img_to_p_tile, .5)
    p_tile_th = ((img_to_p_tile >= p_tile_th) * 255).astype(np.uint8)
    assert not np.logical_and(p_tile_th > 0, p_tile_th < 255).any(), \
        'Image adaptive threshold using p-tile method isn\'t monochrome'
    return p_tile_th
