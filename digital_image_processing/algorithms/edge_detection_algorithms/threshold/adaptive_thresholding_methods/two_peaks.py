import numpy as np

from pythreshold.global_th.two_peaks import two_peaks_threshold
from tools.logger_base import log as log_message


def two_peaks_thresholding_method(img_to_two_peaks: np.ndarray) -> np.ndarray:
    """Runs the two peaks threshold algorithm. It selects two peaks
    from the histogram and return the index of the minimum value
    between them.

    The first peak is deemed to be the maximum value fo the histogram,
    while the algorithm will look for the second peak by multiplying the
    histogram values by the square of the distance from the first peak.
    This gives preference to peaks that are not close to the maximum.

    Reference:
    Parker, J. R. (2010). Algorithms for image processing and
    computer vision. John Wiley & Sons.

    :param img_to_two_peaks: The input image
    :type img_to_two_peaks: ndarray

    :return: The threshold between the two founded peaks with the
        minimum histogram value
    :rtype: int
    """

    log_message.info('========Two peaks\'s Thresholding==========')
    two_peaks_th = two_peaks_threshold(img_to_two_peaks)
    two_peaks_th = ((img_to_two_peaks >= two_peaks_th) * 255).astype(np.uint8)
    assert not np.logical_and(two_peaks_th > 0, two_peaks_th < 255).any(), \
        'Image adaptive threshold using two peaks method isn\'t monochrome'
    return two_peaks_th
