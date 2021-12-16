import numpy as np
import cv2.cv2 as cv2

from tools.logger_base import log as log_message


def otsu_thresholding_method(img_to_otsu: np.ndarray) -> dict:
    """ Runs the otsu thresholding algorithm.

    Reference:
    Roy, P., Dutta, S., Dey, N., Dey, G., Chakraborty, S., & Ray, R. (2014). Adaptive thresholding: A
    comparative study. 2014 International Conference on Control, Instrumentation, Communication and Computational
    Technologies (ICCICCT). Published. https://doi.org/10.1109/iccicct.2014.6993140

    Modifications: Using integral images to compute the local mean and the
    NICK variation of the Niblack standard deviation term

    :param img_to_otsu: The input image
    :type img_to_otsu: ndarray

    :return: The estimated local threshold for each pixel
    :rtype: ndarray
    """

    log_message.info('========Otsu\'s Thresholding==========')
    _, otsu = cv2.threshold(img_to_otsu, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blur = cv2.GaussianBlur(img_to_otsu, (5, 5), 0)
    _, otsu_with_gaussian_filter = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    assert not np.logical_and(otsu > 0, otsu < 255).any(), 'Image adaptive threshold using ' \
                                                           'otsu\'s method isn\'t monochrome'
    assert not np.logical_and(otsu_with_gaussian_filter > 0, otsu_with_gaussian_filter < 255).any(), \
        'Image adaptive threshold using otsu\'s method with gaussian filter isn\'t monochrome'
    return {'img': [otsu, otsu_with_gaussian_filter],
            'title': ['otsu', '*otsu_with_gaussian_filter']}
