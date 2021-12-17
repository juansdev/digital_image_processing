import os

from .tools.logger_base import log as log_message
from timeit import default_timer
from .algorithms.edge_detection_algorithms.threshold.adaptive_thresholding_methods import (
    threshold_value_mean,
    threshold_value_gaussian,
    otsu_thresholding_method,
    kapur_thresholding_method,
    p_tile_thresholding_method,
    two_peaks_thresholding_method,
    minimum_err_thresholding_method,
    pun_thresholding_method,
    johannsen_thresholding_method,
    sauvola_thresholding_method,
    niblack_thresholding_method,
    wolf_thresholding_method,
    nick_thresholding_method,
    bradley_thresholding_method,
    bernsen_thresholding_method,
    contrast_thresholding_method,
    singh_thresholding_method,
    feng_thresholding_method
)
from .algorithms.edge_detection_algorithms.algorithms import (
    backward_difference,
    forward_difference,
    central_difference,
    log,
    dog,
    canny_operator,
    laplacian_operator,
    prewitt_operator,
    sobel_operator,
    roberts_operator,
    scharr_operator,
    kirsch_operator,
    robinson_operator
)
from .algorithms.main import ApplyAlgorithms


__copyright__ = 'Copyright 2021'
__author__ = u'Juan Guillermo Serrano Ramírez, Sergio Orjuela'


def test_edge_detection_algorithms(path_input='input_test', path_output='output_test', apply_consensus_ground=False):
    """Runs all the package algorithms of edge detection algorithms on the input image and save the results in the root
    folder where the script was run.

    :param path_input: Path of the folder input
    :type path_input: str
    :param path_output: Path of the folder output
    :type path_output: str
    """

    log_message.info('========Applying the package algorithms of edge detection algorithms========')
    list_algorithms = [backward_difference,
                       forward_difference,
                       central_difference,
                       log,
                       # dog,
                       canny_operator,
                       laplacian_operator,
                       prewitt_operator,
                       sobel_operator,
                       roberts_operator,
                       scharr_operator,
                       kirsch_operator
                       # robinson_operator
                       ]
    path_output = os.path.join(path_output, 'edge_detection')
    start = default_timer()
    apply_edge_detection_algorithms = ApplyAlgorithms(list_algorithms,
                                                      path_input=path_input,
                                                      path_output=path_output,
                                                      apply_consensus_ground=apply_consensus_ground)
    apply_edge_detection_algorithms.apply_algorithms()
    stop = default_timer()
    log_message.info('Execution time of all edge detection algorithms: {0}'.format(stop - start))
    log_message.info('========Successfully applied algorithms of edge detection algorithms========')


def test_adaptive_thresholding_algorithms(path_input='input_test', path_output='output_test'):
    """Runs all the package adaptive thresholding methods on the input image and save the results in the root
    folder where the script was run.

    :param path_input: Path of the folder input
    :type path_input: str
    :param path_output: Path of the folder output
    :type path_output: str
    """

    log_message.info('========Applying the package adaptive thresholding methods========')
    list_algorithms = [threshold_value_mean,
                       threshold_value_gaussian,
                       otsu_thresholding_method,
                       kapur_thresholding_method,
                       p_tile_thresholding_method,
                       two_peaks_thresholding_method,
                       minimum_err_thresholding_method,
                       pun_thresholding_method,
                       johannsen_thresholding_method,
                       sauvola_thresholding_method,
                       niblack_thresholding_method,
                       wolf_thresholding_method,
                       nick_thresholding_method,
                       bradley_thresholding_method,
                       bernsen_thresholding_method,
                       contrast_thresholding_method,
                       singh_thresholding_method,
                       feng_thresholding_method]
    path_output = os.path.join(path_output, 'adaptive_thresholding')
    start = default_timer()
    apply_edge_detection_algorithms = ApplyAlgorithms(list_algorithms,
                                                      path_input=path_input,
                                                      path_output=path_output)
    apply_edge_detection_algorithms.apply_algorithms()
    stop = default_timer()
    log_message.info('Execution time of all adaptive thresholding methods: {0}'.format(stop - start))
    log_message.info('========Successfully applied adaptive thresholding methods========')
