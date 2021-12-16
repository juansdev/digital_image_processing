import numpy as np
import tkinter
from matplotlib import pyplot as plt
from PIL import Image


def use_img_with_plt(img_original: np.array) -> np.array:
    root = tkinter.Tk()
    dpi = root.winfo_fpixels('1i')
    img = Image.fromarray(img_original)
    fig, ax = plt.subplots(figsize=[pixel / dpi for pixel in img.size], dpi=dpi)
    ax.imshow(img_original, cmap='gray')
    ax.axis('off')
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    res = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return res
