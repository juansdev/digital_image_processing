import unittest

from algorithms.edge_detection_algorithms.threshold.adaptive_thresholding_methods import (
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
from algorithms.edge_detection_algorithms.algorithms import (
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
from algorithms.main import ApplyAlgorithms


class TestApplyAlgorithms(unittest.TestCase):
    """Runs all the package algorithms on the input image with default parameters and save the results in the root
    folder where the script was run.
    """

    PATH_INPUT: str = 'input_test'
    PATH_OUTPUT: str = 'output_test'

    def test_one_edge_detection_algorithms(self):
        list_algorithms = [threshold_value_mean]
        apply_edge_detection_algorithms = ApplyAlgorithms(list_algorithms,
                                                          path_input=self.PATH_INPUT,
                                                          path_output=self.PATH_OUTPUT)
        apply_edge_detection_algorithms.apply_algorithms()

    def test_several_edge_detection_algorithms(self):
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
        apply_edge_detection_algorithms = ApplyAlgorithms(list_algorithms,
                                                          path_input=self.PATH_INPUT,
                                                          path_output=self.PATH_OUTPUT)
        apply_edge_detection_algorithms.apply_algorithms()

    def test_all_edge_detection_algorithms(self):
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
        apply_edge_detection_algorithms = ApplyAlgorithms(list_algorithms,
                                                          path_input=self.PATH_INPUT,
                                                          path_output=self.PATH_OUTPUT,
                                                          apply_consensus_ground=True)
        apply_edge_detection_algorithms.apply_algorithms()


if __name__ == '__main__':
    unittest.main()
