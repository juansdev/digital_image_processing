import argparse
import os

from .utils import test_adaptive_thresholding_algorithms, test_edge_detection_algorithms
from .tools.logger_base import log as log_message


def test_algorithms_main():
    """Main entry point for the test algorithms script"""

    # Parsing arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input-dir', help='Input directory')
    ap.add_argument('-o', '--output-dir', help='Output directory')
    ap.add_argument('-all', '--all-algorithms', dest='all_algorithms', action='store_true',
                    help="Test all algorithms")
    ap.add_argument('-ed', '--edge-detection-algorithms', dest='edge_detection_algorithms', action='store_true',
                    help="Test edge detection algorithms")
    ap.add_argument('-at', '--adaptive-thresholding', dest='adaptive_thresholding', action='store_true',
                    help="Test adaptive thresholding algorithms")
    ap.add_argument('-cg', '--consensus-ground', dest='consensus_ground', action='store_true',
                    help="Use consensus ground for edge detection algorithms")
    ap.set_defaults(all_algorithms=False)
    ap.set_defaults(edge_detection_algorithms=False)
    ap.set_defaults(adaptive_thresholding=False)
    ap.set_defaults(consensus_ground=False)
    args = ap.parse_args()

    if not args.input_dir:
        args.input_dir = os.path.join(os.getcwd(), 'input_test')
        log_message.warning(f'Input directory not established, the default path will be used {args.input_dir!r}')
    if not args.output_dir:
        args.output_dir = os.path.join(os.getcwd(), 'output_test')
        log_message.warning(f'Output directory not established, the default path will be used {args.output_dir!r}')
    if args.all_algorithms:
        log_message.info('Algorithms to use: All')
        test_edge_detection_algorithms(args.input_dir, args.output_dir, args.consensus_ground)
        test_adaptive_thresholding_algorithms(args.input_dir, args.output_dir)
    else:
        if args.edge_detection_algorithms:
            test_edge_detection_algorithms(args.input_dir, args.output_dir, args.consensus_ground)
        if args.adaptive_thresholding:
            test_adaptive_thresholding_algorithms(args.input_dir, args.output_dir)
        if not args.adaptive_thresholding and not args.edge_detection_algorithms:
            log_message.error('''You must use at least one of the following arguments:
                                --all-algorithms | -all
                                --edge-detection-algorithms | -ed
                                --adaptive-thresholding | -at
                                
                                Ex: python main.py --all-algorithms
                                Use for more help: python main.py --help''')
