import numpy as np

from digital_image_processing.tools.logger_base import log as log_message
from copy import deepcopy


def lbp(img_to_lbp: np.array) -> np.array:
    """Runs the LBP algorithm

    Reference:
    Wang, Xinyu & Cao, Jie & Hao, Qun & Zhang, Kaiyu & Zihan, Wang & Rizvi, Saad. (2018).
    LBP Based Edge Detection Method for Depth Images with Low Resolutions. IEEE Photonics Journal.
    PP. 1-1. 10.1109/JPHOT.2018.2884772.

    :param img_to_lbp: The input image. Must be a gray scale image
    :type img_to_lbp: ndarray

    :return: The estimated local for each pixel
    :rtype: ndarray
    """

    log_message.info('Finds the edges using LBP.')
    basic_array = np.zeros(img_to_lbp.shape, np.uint8)
    for i in range(basic_array.shape[0] - 1):
        for j in range(basic_array.shape[1] - 1):
            sum_list = []
            y = deepcopy(i)
            x = deepcopy(j)
            if img_to_lbp[y - 1, x] > img_to_lbp[y, x]:
                sum_list.append(1)
            else:
                sum_list.append(0)
            if img_to_lbp[y - 1, x + 1] > img_to_lbp[y, x]:
                sum_list.append(1)
            else:
                sum_list.append(0)
            if img_to_lbp[y, x + 1] > img_to_lbp[y, x]:
                sum_list.append(1)
            else:
                sum_list.append(0)
            if img_to_lbp[y + 1, x + 1] > img_to_lbp[y, x]:
                sum_list.append(1)
            else:
                sum_list.append(0)
            if img_to_lbp[y + 1, x] > img_to_lbp[y, x]:
                sum_list.append(1)
            else:
                sum_list.append(0)
            if img_to_lbp[y + 1, x - 1] > img_to_lbp[y, x]:
                sum_list.append(1)
            else:
                sum_list.append(0)
            if img_to_lbp[y, x - 1] > img_to_lbp[y, x]:
                sum_list.append(1)
            else:
                sum_list.append(0)
            if img_to_lbp[y - 1, x - 1] > img_to_lbp[y, x]:
                sum_list.append(1)
            else:
                sum_list.append(0)
            res = 0
            bit_num = 0  # Shift left
            for y in sum_list[::-1]:
                res += y << bit_num  # Shifting n bits to the left is equal to multiplying by 2 to the nth power
                bit_num += 1
            basic_array[i, j] = res
    ret = basic_array
    return ret
