import numpy as np
from tools.logger_base import log as log_message


def cs_threshold(img: np.array, low_threshold_ratio=0.05, high_threshold_ratio=0.09) -> tuple:
    log_message.info('Applying threshold.')
    high_threshold = img.max() * high_threshold_ratio
    low_threshold = high_threshold * low_threshold_ratio

    M, N = img.shape
    res = np.zeros((M, N), dtype=np.int32)

    weak = np.int32(25)
    strong = np.int32(255)

    strong_i, strong_j = np.where(img >= high_threshold)

    weak_i, weak_j = np.where((img <= high_threshold) & (img >= low_threshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return res, weak, strong
