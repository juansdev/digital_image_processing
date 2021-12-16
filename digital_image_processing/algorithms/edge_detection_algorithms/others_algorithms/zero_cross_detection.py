import numpy as np

def zero_cross_detection(image):
    z_c_image = np.zeros(image.shape)

    for i in range(0,image.shape[0]-1):
        for j in range(0,image.shape[1]-1):
            if image[i][j]>0:
                if image[i+1][j] < 0 or image[i+1][j+1] < 0 or image[i][j+1] < 0:
                    z_c_image[i,j] = 1
            elif image[i][j] < 0:
                if image[i+1][j] > 0 or image[i+1][j+1] > 0 or image[i][j+1] > 0:
                    z_c_image[i,j] = 1
    return z_c_image