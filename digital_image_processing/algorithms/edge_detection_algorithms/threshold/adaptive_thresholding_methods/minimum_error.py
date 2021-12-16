import numpy as np

from pythreshold.global_th.min_err import min_err_threshold
from tools.logger_base import log as log_message


def minimum_err_thresholding_method(img_to_minimum_err: np.ndarray) -> np.ndarray:
    """Runs the minimum error thresholding algorithm.

    Reference:
    Kittler, J. and J. Illingworth. ‘‘On Threshold Selection Using Clustering
    Criteria,’’ IEEE Transactions on Systems, Man, and Cybernetics 15, no. 5
    (1985): 652–655.

    :param img_to_minimum_err: The input image
    :type img_to_minimum_err: ndarray

    :return: The threshold that minimize the error
    :rtype: ndarray
    """

    log_message.info('========Minimum error Thresholding==========')
    minimum_err_th = min_err_threshold(img_to_minimum_err)
    minimum_err_th = ((img_to_minimum_err >= minimum_err_th) * 255).astype(np.uint8)
    assert not np.logical_and(minimum_err_th > 0, minimum_err_th < 255).any(), \
        'Image adaptive threshold using minimum error method isn\'t monochrome'
    return minimum_err_th
