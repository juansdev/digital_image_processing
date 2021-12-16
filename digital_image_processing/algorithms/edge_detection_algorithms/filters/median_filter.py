import numpy as np
from tools.logger_base import log as log_message


def median_filter(img: np.array, f: np.array) -> np.array:
    log_message.info('Applying medium filter.')
    img_h, img_w = img.shape
    ret = np.array(img, dtype='float')
    img = np.array(img, dtype='float')
    f_h, f_w = f.shape
    assert f_h % 2 == 1, 'assume filter size is odd'
    f_size = np.int32((f_h - 1) / 2)

    for i in range(img_h):
        for j in range(img_w):
            if i - f_size < 0 or j - f_size < 0 or i + f_size >= img_h or j + f_size >= img_w:
                ret[i][j] = 0
                continue
            v = np.zeros(9)
            count = 0
            for di in range(-f_size, f_size + 1):
                for dj in range(-f_size, f_size + 1):
                    ci = i + di
                    cj = j + dj
                    v[count] = img[ci, cj]
                    count = count+1
            ret[i][j] = np.median(v)
    return ret
