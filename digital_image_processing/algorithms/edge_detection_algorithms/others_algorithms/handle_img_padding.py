import numpy as np


def handle_img_padding(img1, img2):
    M1, N1 = img1.shape[:2]
    M2, N2 = img2.shape[:2]
    padding_x = int(np.abs(M2 - M1)/2)
    padding_y = int(np.abs(N2 - N1)/2)
    img2 = img2[padding_x:M1+padding_x, padding_y: N1+padding_y]
    return img2
