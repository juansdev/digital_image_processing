import numpy as np

from digital_image_processing.tools.logger_base import log as log_message
from digital_image_processing.algorithms.edge_detection_algorithms.filters import (
    mask_filter
)


def dog(img_to_dog: np.ndarray) -> np.ndarray:
    """Runs the Difference of Gaussian algorithm

    Reference:
    Abd El-Fattah El-Sennary, H., Eid Hussien, M., & El-Mgeid Amin Ali, A. (2019). Edge Detection of an
    Image Based on Extended Difference of Gaussian. American Journal of Computer Science and Technology, 2(3),
    35. https://doi.org/10.11648/j.ajcst.20190203.11. AlNouri, M., al Saei, J., Younis, M., Bouri, F., al Habash,
    M. A., Shah, M. H., & al Dosari, M. (2015).
    Comparison of Edge Detection Algorithms for Automated Radiographic
    Measurement of the Carrying Angle. Journal of Biomedical Engineering and Medical Imaging,
    2(6). https://doi.org/10.14738/jbemi.26.1753.

    :param img_to_dog: The input image. Must be a gray scale image
    :type img_to_dog: ndarray

    :return: The estimated local operator for each pixel
    :rtype: ndarray
    """

    log_message.info('========Difference of Gaussian==========')
    m, n = [(ss - 1.) / 2. for ss in (5, 5)]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    f1 = np.exp(-(x * x + y * y) / (2. * 0.5 * 0.5))
    f1 = f1 / f1.sum()
    f2 = np.exp(-(x * x + y * y) / (2. * 1 * 1))
    f2 = f2 / f2.sum()
    f = f1 - f2
    dog_img = mask_filter(img_to_dog, f - f.mean())
    # FIXME
    assert not np.logical_and(dog_img > 0, dog_img < 255).any(), 'Image DoG operator isn\'t monochrome'
    return np.uint8(np.round(np.absolute(dog_img)))
