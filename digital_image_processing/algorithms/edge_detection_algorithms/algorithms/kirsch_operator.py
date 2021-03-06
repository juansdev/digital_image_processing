import numpy as np

from digital_image_processing.algorithms.edge_detection_algorithms.threshold.threshold_based_edge_detection import (
    lpk_threshold,
)
from digital_image_processing.tools.logger_base import log as log_message


def kirsch_operator(img_to_kirsch: np.ndarray) -> np.ndarray:
    """Runs the Kirsch Operator algorithm

    Reference:
    AlNouri, M., al Saei, J., Younis, M., Bouri, F., al Habash, M. A., Shah, M. H., & al Dosari,
    M. (2015). Comparison of Edge Detection Algorithms for Automated Radiographic Measurement of the Carrying Angle.
    Journal of Biomedical Engineering and Medical Imaging, 2(6). https://doi.org/10.14738/jbemi.26.1753

    :param img_to_kirsch: The input image. Must be a gray scale image
    :type img_to_kirsch: ndarray

    :return: The estimated local operator for each pixel
    :rtype: ndarray
    """

    log_message.info('========Kirsch Operator==========')
    m, n = img_to_kirsch.shape
    mask_kirsch = np.zeros((m, n))
    for i in range(2, m - 1):
        for j in range(2, n - 1):
            d1 = np.square(5 * img_to_kirsch[i - 1, j - 1] + 5 * img_to_kirsch[i - 1, j] + 5 *
                           img_to_kirsch[i - 1, j + 1] - 3 * img_to_kirsch[i, j - 1] - 3 * img_to_kirsch[i, j + 1] -
                           3 * img_to_kirsch[i + 1, j - 1] - 3 * img_to_kirsch[i + 1, j] - 3 *
                           img_to_kirsch[i + 1, j + 1])
            d2 = np.square((-3) * img_to_kirsch[i - 1, j - 1] + 5 * img_to_kirsch[i - 1, j] + 5 *
                           img_to_kirsch[i - 1, j + 1] - 3 * img_to_kirsch[i, j - 1] + 5 * img_to_kirsch[i, j + 1] - 3
                           * img_to_kirsch[i + 1, j - 1] - 3 * img_to_kirsch[i + 1, j] - 3 *
                           img_to_kirsch[i + 1, j + 1])
            d3 = np.square((-3) * img_to_kirsch[i - 1, j - 1] - 3 * img_to_kirsch[i - 1, j] + 5
                           * img_to_kirsch[i - 1, j + 1] - 3 * img_to_kirsch[i, j - 1] + 5 * img_to_kirsch[i, j + 1] -
                           3 * img_to_kirsch[i + 1, j - 1] - 3 * img_to_kirsch[i + 1, j] + 5 *
                           img_to_kirsch[i + 1, j + 1])
            d4 = np.square(
                (-3) * img_to_kirsch[i - 1, j - 1] - 3 * img_to_kirsch[i - 1, j] - 3 * img_to_kirsch[i - 1, j + 1] -
                3 * img_to_kirsch[i, j - 1] + 5 * img_to_kirsch[i, j + 1] - 3 * img_to_kirsch[i + 1, j - 1] +
                5 * img_to_kirsch[i + 1, j] + 5 * img_to_kirsch[i + 1, j + 1])
            d5 = np.square(
                (-3) * img_to_kirsch[i - 1, j - 1] - 3 * img_to_kirsch[i - 1, j] - 3 * img_to_kirsch[i - 1, j + 1] - 3
                * img_to_kirsch[i, j - 1] - 3 * img_to_kirsch[i, j + 1] + 5 * img_to_kirsch[i + 1, j - 1] +
                5 * img_to_kirsch[i + 1, j] + 5 * img_to_kirsch[i + 1, j + 1])
            d6 = np.square(
                (-3) * img_to_kirsch[i - 1, j - 1] - 3 * img_to_kirsch[i - 1, j] - 3 * img_to_kirsch[i - 1, j + 1] +
                5 * img_to_kirsch[i, j - 1] - 3 * img_to_kirsch[i, j + 1] + 5 * img_to_kirsch[i + 1, j - 1] +
                5 * img_to_kirsch[i + 1, j] - 3 * img_to_kirsch[i + 1, j + 1])
            d7 = np.square(
                5 * img_to_kirsch[i - 1, j - 1] - 3 * img_to_kirsch[i - 1, j] - 3 * img_to_kirsch[i - 1, j + 1] +
                5 * img_to_kirsch[i, j - 1] - 3 * img_to_kirsch[i, j + 1] + 5 * img_to_kirsch[i + 1, j - 1] -
                3 * img_to_kirsch[i + 1, j] - 3 * img_to_kirsch[i + 1, j + 1])
            d8 = np.square(
                5 * img_to_kirsch[i - 1, j - 1] + 5 * img_to_kirsch[i - 1, j] - 3 * img_to_kirsch[i - 1, j + 1] +
                5 * img_to_kirsch[i, j - 1] - 3 * img_to_kirsch[i, j + 1] - 3 * img_to_kirsch[i + 1, j - 1] -
                3 * img_to_kirsch[i + 1, j] - 3 * img_to_kirsch[i + 1, j + 1])
            # The first method: take the maximum value in all directions, the effect is not good, use another method
            list_ret = [d1, d2, d3, d4, d5, d6, d7, d8]
            mask_kirsch[i, j] = int(np.sqrt(max(list_ret)))
    # Apply lpk_threshold
    ret, binary = lpk_threshold(mask_kirsch)
    kirsch = binary
    # FIXME, Generate imagen with extension .tif
    assert not np.logical_and(kirsch > 0, kirsch < 255).any(), 'Image kirsch operator isn\'t monochrome'
    return kirsch

