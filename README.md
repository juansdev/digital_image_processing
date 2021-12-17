# Digital Image Processing

**Digital Image Processing** is a python package featuring Numpy/Scipy/OpenCV implementations of image edge detection and adaptive thresholding algorithms.

## Installing

**Digital Image Processing** can be easily installed by typing the following command

    pip install digital_image_processing.zip

## Usage

It's recommended to run the Script as administrator and only use relative paths

    from digital_image_processing.utils import test_edge_detection_algorithms

    # Testing all the included edge detection algorithms using a default input/output directory path with default images
    test_edge_detection_algorithms()

    # Testing all the included edge detection algorithms using a custom input/output directory path with custom images
    path_input = 'input_directory'
    path_output = 'output_directory'
    test_edge_detection_algorithms(path_input, path_output)

Or just type in a terminal:

    digital_image_processing -i input -o output -ed

## Included Algorithms

* Edge Detection Algorithms
    * Bustacara-Medina, C., & Flórez-Valencia, L. (2016). Comparison and Evaluation of First Derivatives
    Estimation. Computer Vision and Graphics, 121–133. 
    https://doi.org/10.1007/978-3-319-46418-3_11 (**Backward Difference**, **Forward Difference**, **Central Difference**)
    * Comparison of Edge Detection Algorithms for Automated Radiographic Measurement of the Carrying Angle.
    Journal of Biomedical Engineering and Medical Imaging, 2(6). https://doi.org/10.14738/jbemi.26.1753. Nasution,
    T. Y., Zarlis, M., & Nasution, M. K. (2017). (**Sobel Operator**, **Scharr Operator**, **Prewitt Operator**, **Frei-Chen Operator**, **Robinson Operator**, **DoG**, **LoG**, **Hough Transform**)
    * Sobel, Irwin. (2014). An Isotropic 3x3 Image Gradient Operator. Presentation at Stanford A.I. Project 1968. (**Sobel Operator**)
    * Jain, R., Kasturi, R., & Schunck, B. G. (1995). Machine Vision (1.a ed.). Mcgraw-Hill College. (**roberts operator**)
    * S. R. Gunn, "Edge detection error in the discrete Laplacian of Gaussian," Proceedings 1998
    International Conference on Image Processing. ICIP98 (Cat. No.98CB36269), 1998, pp. 515-519 vol.2,
    doi: 10.1109/ICIP.1998.723491. Huertas, A., & Medioni, G. (1986). (**LoG**)
    * Detection of Intensity Changes with Subpixel
    Accuracy Using Laplacian-Gaussian Masks. IEEE Transactions on Pattern Analysis and Machine Intelligence,
    PAMI-8(5), 651–664. https://doi.org/10.1109/tpami.1986.4767838. (**Laplacian Operator**)
    * AlNouri, M., al Saei, J., Younis, M., Bouri, F., al Habash, M. A., Shah, M. H., & al Dosari,
    M. (2015). Comparison of Edge Detection Algorithms for Automated Radiographic Measurement of the Carrying Angle.
    Journal of Biomedical Engineering and Medical Imaging, 2(6). https://doi.org/10.14738/jbemi.26.1753 (**Kirsch Operator**)
    * Abd El-Fattah El-Sennary, H., Eid Hussien, M., & El-Mgeid Amin Ali, A. (2019). Edge Detection of an
    Image Based on Extended Difference of Gaussian. American Journal of Computer Science and Technology, 2(3), 35. 
    https://doi.org/10.11648/j.ajcst.20190203.11. AlNouri, M., al Saei, J., Younis, M., Bouri, F., al Habash,
M. A., Shah, M. H., & al Dosari, M. (2015). (**DoG**)
* Filters used for Edge Detection
    * Gedraite, Estevao & Hadad, M.. (2011). Investigation on the effect of a Gaussian Blur in image
    filtering and segmentation. 393-396.(**Gaussian Filter**)
    * Comparison of Edge Detection Algorithms for Automated Radiographic Measurement of the Carrying Angle.
    Journal of Biomedical Engineering and Medical Imaging, 2(6). https://doi.org/10.14738/jbemi.26.1753. Nasution,
    T. Y., Zarlis, M., & Nasution, M. K. (2017).
    Sobel, Irwin. (2014). An Isotropic 3x3 Image Gradient Operator. Presentation at Stanford A.I. Project 1968. (**Sobel Filter**)
    * Micek, J. & Kapitulík, Ján. (2003). Median filter. Journal of Information, Control and Management Systems. 1. (**Median Filter**)
* Other algorithms used for edge detection
    * Sornam, M., Kavitha, M. S., & Nivetha, M. (2016). Hysteresis thresholding based edge detectors for
    inscriptional image enhancement. 2016 IEEE International Conference on Computational Intelligence and Computing
    Research (ICCIC). Published. https://doi.org/10.1109/iccic.2016.7919568 (**hysteresis**)
    * Neubeck, Alexander & Van Gool, Luc. (2006). Efficient Non-Maximum Suppression. Proceedings of
    International Conference on Pattern Recognition. 3. 850-855. 10.1109/ICPR.2006.479. (**Non Max Supression**)
    * R. Gonzalez and R. Woods Digital Image Processing, Addison-Wesley Publishing Company, 1992, p 442. (**Zero Cross Detection**)
* Threshold used for edge detection
    * Goel, K., Sehrawat, M., & Agarwal, A. (2017). Finding the optimal threshold values for edge detection
    of digital images & comparing among Bacterial Foraging Algorithm, canny and Sobel Edge Detector. 2017
    International Conference on Computing, Communication and Automation (ICCCA), 1076-1080. (**Threshold for Canny and Sobel**)
    * M. Adnan Al-Alaoui, "Direct approach to image edge detection using differentiators," 2010 17th IEEE International
    Conference on Electronics, Circuits and Systems, 2010, pp. 154-157, doi: 10.1109/ICECS.2010.5724477. (**Threshold for Forward, Backward and Center**)
    * E. (2021, 19 julio). [Python image processing] 42. Detailed explanation of Python image sharpening and edge
    detection (Roberts, Prewitt, Sobel, Laplacian, canny, log). Pythonmana.
    Recuperado 16 de diciembre de 2021, de https://pythonmana.com/2021/07/20210730134158901A.html (**Threshold for Laplacian, Prewitt and Kirsch**)
    * Fisher, R., Perkins, S., Walker, A., & Wolfart, E. (s. f.). Feature Detectors - Roberts Cross Edge Detector.
    homepages. Recuperado 16 de diciembre de 2021, de https://homepages.inf.ed.ac.uk/rbf/HIPR2/roberts.htm (**Threshold for Roberts**)
    * Topal, C., & Akinlar, C. (2012). Edge Drawing: A combined real-time edge and segment detector.
    Journal of Visual Communication and Image Representation, 23(6), 862–872.
    https://doi.org/10.1016/j.jvcir.2012.05.004 (**Threshold for Scharr**)
* Adaptive Thresholding
    * Eyupoglu, Can. (2016). Implementation of Bernsen’s Locally Adaptive Binarization Method for Gray Scale Images. (**Bernsen**)
    * C. Wolf, J-M. Jolion, “Extraction and Recognition
    of Artificial Text in Multimedia Documents”, Pattern Analysis and
    Applications, 6(4):309-326, (2003). (**Wolf**)
    * Parker, J. R. (2010). Algorithms for image processing and
    computer vision. John Wiley & Sons. (**Two Peaks**)
    * Singh, O. I., Sinam, T., James, O., & Singh, T. R. (2012). Local contrast
    and mean based thresholding technique in image binarization. International
    Journal of Computer Applications, 51, 5-10. (**Singh**)
    * Sauvola, J., Seppanen, T., Haapakoski, S., and Pietikainen, M.:
    ‘Adaptive document thresholding’. Proc. 4th Int. Conf. on Document
    Analysis and Recognition, Ulm Germany, 1997, pp. 147–152.(**Sauvola**)
    * Pun, T. ‘‘A New Method for Grey-Level Picture Thresholding Using the
    Entropy of the Histogram,’’ Signal Processing 2, no. 3 (1980): 223–237. (**Pun**)
    * Parker, J. R. (2010). Algorithms for image processing and
    computer vision. John Wiley & Sons. (**P tile)
    * Roy, P., Dutta, S., Dey, N., Dey, G., Chakraborty, S., & Ray, R. (2014). Adaptive thresholding: A
    comparative study. 2014 International Conference on Control, Instrumentation, Communication and Computational
    Technologies (ICCICCT). Published. https://doi.org/10.1109/iccicct.2014.6993140 (**Otsu**)
    * Khurshid, K., Siddiqi, I., Faure, C., & Vincent, N.
    (2009, January). Comparison of Niblack inspired Binarization methods for
    ancient documents. In IS&T/SPIE Electronic Imaging (pp. 72470U-72470U).
    International Society for Optics and Photonics. (**Nick**)
    * Niblack, W.: ‘An introduction to digital image
    processing’ (Prentice- Hall, Englewood Cliffs, NJ, 1986), pp. 115–116 (**Niblack**)
    * Kittler, J. and J. Illingworth. ‘‘On Threshold Selection Using Clustering
    Criteria,’’ IEEE Transactions on Systems, Man, and Cybernetics 15, no. 5
    (1985): 652–655. (**Minimum error**)