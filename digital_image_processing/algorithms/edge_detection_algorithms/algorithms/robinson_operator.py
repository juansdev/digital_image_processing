import math
import numpy as np
import cv2.cv2 as cv2
from skimage import util
from scipy import ndimage
from tools.logger_base import log as log_message


def robinson_operator(img_to_robinson: np.ndarray) -> dict:
    log_message.info('========Robinson Operator==========')
    template_north = np.array([[1, 2, 1],
                               [0, 0, 0],
                               [-1, -2, -1]])
    north = cv2.filter2D(img_to_robinson, -1, template_north)
    template_sobel_horizontal = np.array([[-1, -2, -1],
                                          [0, 0, 0],
                                          [1, 2, 1]])
    sobel_horizontal = cv2.filter2D(img_to_robinson, -1, template_sobel_horizontal)
    template_sobel_vertical = np.array([[-1, 0, 1],
                                        [-2, 0, 2],
                                        [-1, 0, 1]])
    sobel_vertical = cv2.filter2D(img_to_robinson, -1, template_sobel_vertical)
    template_west = np.array([[1, 0, -1],
                              [2, 0, -2],
                              [1, 0, -1]])
    west = cv2.filter2D(img_to_robinson, -1, template_west)
    template_north_west = np.array([[2, 1, 0],
                                    [1, 0, -1],
                                    [0, -1, -2]])
    north_west = cv2.filter2D(img_to_robinson, -1, template_north_west)
    template_south_east = np.array([[-2, -1, 0],
                                    [-1, 0, 1],
                                    [0, 1, 2]])
    south_east = cv2.filter2D(img_to_robinson, -1, template_south_east)
    template_north_east = np.array([[0, 1, 2],
                                    [-1, 0, 1],
                                    [-2, -1, 0]])
    north_east = cv2.filter2D(img_to_robinson, -1, template_north_east)
    template_south_west = np.array([[0, -1, -2],
                                    [1, 0, -1],
                                    [2, 1, 0]])
    south_west = cv2.filter2D(img_to_robinson, -1, template_south_west)

    four_templates = np.append([template_sobel_vertical], [template_south_east], axis=0)
    four_templates = np.append(four_templates, [template_sobel_horizontal], axis=0)
    four_templates = np.append(four_templates, [template_south_west], axis=0)
    h4_7 = np.negative(four_templates)
    e_k = np.zeros(img_to_robinson.shape)

    h0_7 = np.concatenate((four_templates, h4_7), axis=0)

    for filter_template in h0_7:
        e_k = np.maximum(ndimage.filters.convolve(img_to_robinson, filter_template), e_k)

    k_k = img_to_robinson
    for v in range(0, img_to_robinson.shape[1]):
        for u in range(0, img_to_robinson.shape[0]):
            k_k[u][v] = math.pi / 4 * e_k[u][v]
    four_templates = util.invert(k_k)
    # FIXME
    assert not np.logical_and(north > 0, north < 255).any(), 'Image north robinson operator isn\'t monochrome'
    assert not np.logical_and(sobel_horizontal > 0, sobel_horizontal < 255).any(), \
        'Image sobel horizontal robinson operator isn\'t monochrome'
    assert not np.logical_and(sobel_vertical > 0, sobel_vertical < 255).any(), \
        'Image sobel vertical robinson operator isn\'t monochrome'
    assert not np.logical_and(west > 0, west < 255).any(), \
        'Image west robinson operator isn\'t monochrome'
    assert not np.logical_and(north_west > 0, north_west < 255).any(), \
        'Image north west robinson operator isn\'t monochrome'
    assert not np.logical_and(south_east > 0, south_east < 255).any(), \
        'Image south east robinson operator isn\'t monochrome'
    assert not np.logical_and(north_east > 0, north_east < 255).any(), \
        'Image north east robinson operator isn\'t monochrome'
    assert not np.logical_and(south_west > 0, south_west < 255).any(), \
        'Image south west robinson operator isn\'t monochrome'
    assert not np.logical_and(four_templates > 0, four_templates < 255).any(), \
        'Image four templates robinson operator isn\'t monochrome'
    return {'img': [north, sobel_horizontal, sobel_vertical, west, north_west, south_east, north_east, south_west,
                    four_templates],
            'title': ['north', 'sobel_horizontal', 'sobel_vertical', 'west', 'north_west', 'south_east', 'north_east',
                      'south_west', '*four_templates']}
