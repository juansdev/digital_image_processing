from scipy import ndimage
import numpy as np
from digital_image_processing.tools.logger_base import log as log_message


def sobel_filter(img):
    """Runs the Sobel Filter algorithm

    Reference:
    Comparison of Edge Detection Algorithms for Automated Radiographic Measurement of the Carrying Angle.
    Journal of Biomedical Engineering and Medical Imaging, 2(6). https://doi.org/10.14738/jbemi.26.1753. Nasution,
    T. Y., Zarlis, M., & Nasution, M. K. (2017).
    Sobel, Irwin. (2014). An Isotropic 3x3 Image Gradient Operator. Presentation at Stanford A.I. Project 1968.

    :param img: The input image. Must be a gray scale image
    :type img: ndarray
    :param f: Kernel
    :type f: ndarray

    :return: The estimated local filter for each pixel
    :rtype: ndarray
    """

    log_message.info('Applying sobel filter.')
    Kx = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]], np.float32)
    Ky = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]], np.float32)

    Ix = ndimage.filters.convolve(img, Kx)
    Iy = ndimage.filters.convolve(img, Ky)

    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)

    return G, theta
