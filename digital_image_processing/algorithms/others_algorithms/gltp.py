import numpy as np

from matplotlib import pyplot as plt
from digital_image_processing.tools.logger_base import log as log_message
from digital_image_processing.tools.tools_img import use_img_with_plt
from digital_image_processing.algorithms.edge_detection_algorithms.filters import median_filter


def gltp(img_to_gltp: np.array, use_median_filter: bool = True) -> np.array:
    """Runs the GLTP algorithm

    :param img_to_gltp: The input image. Must be a gray scale image
    :type img_to_gltp: ndarray
    :param use_median_filter: Use median filter?
    :type use_median_filter: bool

    :return: The estimated local for each pixel
    :rtype: ndarray
    """

    log_message.info('')

    # Apply median filter
    g = img_to_gltp
    h = median_filter(g * 255, np.zeros((3, 3))) / 255
    image_gray = []
    for img in (h, g):
        image_gray.append(use_img_with_plt(img))
    image_with_filter, image_withoutfilter = image_gray
    if use_median_filter:
        title_to_img = 'with_filter'
        g = h
    else:
        title_to_img = 'without_filter'

    # Residuos
    filas, columnas = g.shape
    vecinos = np.zeros(filas * columnas * 9).reshape(filas, columnas, 9)
    vecinos[0:filas - 1, 0:columnas - 0, 0] = g[1:filas - 0, 0:columnas - 0]
    vecinos[0:filas - 1, 0:columnas - 1, 1] = g[1:filas - 0, 1:columnas - 0]
    vecinos[0:filas - 0, 0:columnas - 1, 2] = g[0:filas - 0, 1:columnas - 0]
    vecinos[1:filas - 0, 0:columnas - 1, 3] = g[0:filas - 1, 1:columnas - 0]
    vecinos[1:filas - 0, 0:columnas - 0, 4] = g[0:filas - 1, 0:columnas - 0]
    vecinos[1:filas - 0, 1:columnas - 0, 5] = g[0:filas - 1, 0:columnas - 1]
    vecinos[0:filas - 0, 1:columnas - 0, 6] = g[0:filas - 0, 0:columnas - 1]
    vecinos[0:filas - 1, 1:columnas - 0, 7] = g[1:filas - 0, 0:columnas - 1]

    for v in range(8):
        vecinos[:, :, 8] = vecinos[:, :, 8] + (vecinos[:, :, v] - g) / 8

    image_waste = use_img_with_plt(vecinos[:, :, 8])

    # Histograma de residuos
    vals = vecinos[:, :, 8].flatten()
    b, bins, patches = plt.hist(vals, 255)
    x = (bins[0:len(bins) - 1] + bins[1:len(bins)]) / 2
    d = b / sum(b)
    c = 0
    i = 0

    for _ in range(len(d)):
        c = c + d[i]
        if c > 0.5:
            break
        i += 1

    y = (x[i] + x[i - 1]) / 2
    k = np.sum(np.abs((x - y) * d))

    fig = plt.figure()
    ax2 = fig.gca()
    ax2.plot(x, d, color='black')
    ax2.set_xlabel('Intensity')
    ax2.set_ylabel('Probability')
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    image_histogram = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # Bordes por filtro
    image_edges = use_img_with_plt(np.abs(vecinos[:, :, 8]) > y + 2 * k)

    return {'img': [image_with_filter, image_withoutfilter, image_waste, image_histogram, image_edges],
            'title': ['image_with_filter', 'image_without_filter', f'image_{title_to_img}_waste',
                      f'image_{title_to_img}_histogram', f'*image_{title_to_img}_edges']}
