import numpy as np
from tools.logger_base import log as log_message


def mask_filter(img_to_filter: np.array, f: np.array) -> np.array:
    log_message.info('Applying mask filter.')
    img_h, img_w = img_to_filter.shape
    ret = np.array(img_to_filter, dtype='float')
    img_to_filter = np.array(img_to_filter, dtype='float')
    f_h, f_w = f.shape
    assert f_h % 2 == 1, 'assume filter size is odd'
    f_size = np.int32((f_h - 1) / 2)
    for i in range(img_h):
        for j in range(img_w):
            # Si excede limite, cambiar tonalidad a 0
            if (i - f_size < 0 or j - f_size < 0
                    or i + f_size >= img_h or j + f_size >= img_w):
                ret[i][j] = 0
                continue
            v = 0
            # Si esta dentro del limite, cambiar tonalidad al valor multiplicado por la mascara.
            for di in range(-f_size, f_size + 1):
                for dj in range(-f_size, f_size + 1):
                    ci = i + di
                    cj = j + dj
                    fi = di + f_size
                    fj = dj + f_size
                    v += f[fi, fj] * img_to_filter[ci, cj]
            ret[i][j] = v
    return ret
