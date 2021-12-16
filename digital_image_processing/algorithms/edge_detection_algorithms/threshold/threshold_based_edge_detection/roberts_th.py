import numpy as np
from tools.logger_base import log as log_message


def roberts_threshold(img: np.ndarray) -> np.ndarray:
    log_message.info('Applying threshold.')
    return ((img * 5) > 80) * 255
