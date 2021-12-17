# -*- coding: utf-8 -*-

"""
digital_image_processing.algorithms.main
~~~~~~~~~~~~~~~~~
This module contains the application of algorithms on digital images.
"""
import PIL

import digital_image_processing
import inspect
import itertools
import numpy as np
import matplotlib.image as mpimg
import os
import string
import shutil
import cv2.cv2 as cv2
from typing import Union, List
from pathlib import Path
from digital_image_processing.tools.logger_base import log as log_message
from PIL import Image
from timeit import default_timer

__author__ = "Juan Guillermo Serrano Ramírez & Sergio Orjuela"
__date__ = "12/08/2021 02:42:41 PM$"


class BaseApplyAlgorithms:
    r"""A class used as a parent class for classes that apply algorithms to digital images

    :param path_input: Path of the folder input
    :type path_input: str
    :param path_output: Path of the folder output
    :type path_output: str
    """

    def __init__(self, path_input: str, path_output: str):
        self.__path_input: str = path_input
        self.__path_output: str = path_output

    def create_folders(self, name_algorithms: List[str], apply_consensus_ground: bool) -> None:
        log_message.info(f'Removing old output.')
        shutil.rmtree(self.__path_output, ignore_errors=True)
        log_message.info(f'Creating new output folder and if not exist the input folder.')
        try:
            Path(self.__path_input).mkdir(parents=True)
        except OSError:
            log_message.debug(f'Input folder already exists.')
        else:
            input('Input folder created, please add images in the folder before continue or ignore this message '
                  'to add test images, press \'Enter\' to continue.')
        name_imgs: List[str] = [name_img for name_img in os.listdir(self.path_input)
                                if os.path.isfile(os.path.join(self.path_input, name_img))]
        if not name_imgs:
            input(f'There are no images in the input folder, test images will be saved in the folder. '
                  f'Ctrl + Z and enter for cancel operation.')
            path_test_img: str = os.path.join(os.path.dirname(digital_image_processing.__file__), 'test_imgs')
            path_test_imgs: List[str] = [os.path.join(path_test_img, name_img)
                                         for name_img in os.listdir(path_test_img)]
            start = default_timer()
            [shutil.copy(path_test_img, self.path_input) for path_test_img in path_test_imgs]
            stop = default_timer()
            log_message.info(f'Test images saved successfully'
                             'Execution time: {0}'.format(stop - start))
        Path(self.__path_output).mkdir(parents=True, exist_ok=True)
        [Path(os.path.join(self.__path_output, path)).mkdir(parents=True, exist_ok=True)
         for path in name_algorithms]
        if apply_consensus_ground:
            Path(os.path.join(self.__path_output, 'consensus_ground')).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def read_img(path_filename: str, apply_gray_scale=False) -> np.ndarray:
        img_to_read: np.ndarray = cv2.imread(path_filename)
        if apply_gray_scale:
            log_message.info(f'Converting image {path_filename!r} to grayscale.')
            # convert to gray image
            gimg: np.ndarray = cv2.cvtColor(img_to_read, cv2.COLOR_BGR2GRAY)
            if np.array_equal(gimg, gimg >= 1):
                # binary image, 1-255
                return gimg
            if np.array_equal(gimg, np.uint8(gimg)):
                # gray image, 0-255
                return gimg
            # 0-1
            assert gimg.min() >= 0 and gimg.max() <= 1, gimg
            gimg: np.ndarray = gimg * 255
            # 0-255
            return np.array(gimg, dtype='uint8')
        else:
            return img_to_read

    @staticmethod
    def save_img(filename: np.ndarray, path_to_save: str) -> None:
        try:
            mpimg.imsave(path_to_save, filename, cmap='gray')
        except KeyError:
            cv2.imwrite(path_to_save, filename)
        log_message.info(f'Result of the algorithm saved in {path_to_save}.')

    @staticmethod
    def get_name_and_type_of_args(algorithm_method: callable):
        name_args: List[str] = [param for param in inspect.getfullargspec(algorithm_method)
                                if param][0]
        dict_types_args = inspect.getfullargspec(algorithm_method).annotations
        try:
            del dict_types_args['return']
        except KeyError:
            pass
        type_args: list = list(dict_types_args.values())
        return name_args, type_args

    @staticmethod
    def verify_and_add_np_array_args(name_args: list, type_args: list, *args) -> tuple[list, list]:
        value_params: list = []
        np_array_args = [[i, types] for i, types in enumerate(type_args)
                         if np.ndarray == types]
        if len(np_array_args):
            np_array_args_used = []
            for index_extra_np_array, list_extra_np_array in enumerate(np_array_args):
                index_name_arg, type_np_array = list_extra_np_array
                # First parameter value is expected to be the gray scale image
                if index_name_arg == 0:
                    value_params.append(args[0])
                    np_array_args_used.append(np_array_args[index_extra_np_array])
                # Second parameter value is expected to be the original image
                elif index_name_arg == 1:
                    value_params.append(args[1])
                    np_array_args_used.append(np_array_args[index_extra_np_array])
            for np_array_arg_used in np_array_args_used:
                name_args[np_array_arg_used[0]] = ''
                del np_array_args[np_array_args.index(np_array_arg_used)]
            name_args = list(filter(None, name_args))
        else:
            np_array_args = []
        assert not len(np_array_args), 'Numpy array values must be clarified in source code'
        return name_args, value_params

    @staticmethod
    def verify_and_add_params(algorithm_name: string, name_args: string, type_args: any) -> list:
        log_message.debug(f'Params of the function: {name_args}.')
        value_params = []
        extra_args_params = [types for types in type_args
                             if np.ndarray != types]
        for index_method_param, param in enumerate(name_args):
            while True:
                type_param: any = extra_args_params[index_method_param]
                value: str = input(f'Ingrese un valor para el {param!r} de la funcion {algorithm_name!r}'
                                   f'\nEl valor debe de ser de tipo {type_param}: ')
                if type_param in (int, float, complex):
                    try:
                        value: float = float(value)
                    except ValueError:
                        log_message.error(f'El valor ingresado no corresponde a un tipo de valor valido, '
                                          f'debe ser de tipo {type_param}')
                        continue
                elif type_param == bool:
                    try:
                        value: str = value.lower()
                        if value == 'true':
                            value: bool = True
                        elif value == 'false':
                            value: bool = False
                        else:
                            value: int = int(value)
                    except ValueError:
                        log_message.error(f'El valor ingresado no corresponde a un tipo de valor valido, '
                                          f'debe ser de tipo {type_param}')
                        continue
                    else:
                        value: bool = bool(value)
                log_message.info(f'Valor del parametro {param!r} agregado: {value!r}')
                value_params.append(value)
                break
        return value_params

    @property
    def path_input(self):
        return self.__path_input

    @property
    def path_output(self):
        return self.__path_output

    @path_input.setter
    def path_input(self, path_input: str):
        self.__path_input = path_input

    @path_output.setter
    def path_output(self, path_output: str):
        self.__path_output = path_output


class ApplyAlgorithms(BaseApplyAlgorithms):
    r"""A class used to apply edge detection algorithms to images in the input folder

    Apply the edge detection algorithms to the images in the Input folder, and generate the results
    in the Output folder

    :param method_algorithms: List of callable which correspond to edge detection algorithms
    :type method_algorithms: List[callable]
    :param path_input: Path of the folder input
    :type path_input: str
    :param path_output: Path of the folder output
    :type path_output: str
    :key apply_consensus_ground: Enable or disable the use of consensus ground
    :type apply_consensus_ground: bool
    """

    def __init__(self, method_algorithms: List[callable], path_input: str = 'input',
                 path_output: str = 'output', **kwargs):
        super(ApplyAlgorithms, self).__init__(path_input, path_output)
        self.__apply_consensus_ground: bool = False
        if 'apply_consensus_ground' in kwargs:
            self.__apply_consensus_ground: bool = kwargs['apply_consensus_ground']
            if self.__apply_consensus_ground and len(method_algorithms) < 2:
                raise Exception('It is necessary to use at least two algorithms for the application '
                                'of edge detection for use the consensus ground.')
        self.__name_algorithms: List[str] = [method_algorithm.__name__ for method_algorithm in method_algorithms]
        self.__list_algorithms: List[callable] = method_algorithms
        self.create_folders(self.__name_algorithms, self.__apply_consensus_ground)

    def apply_algorithms(self) -> None:
        name_imgs: List[str] = [name_img for name_img in os.listdir(self.path_input)
                                if os.path.isfile(os.path.join(self.path_input, name_img))]
        for name_img in name_imgs:
            log_message.info('========Reading image==========')
            log_message.info(f'Image to use the algorithms: {name_img}.')
            log_message.info('Use consensus images?: ' + ('Yes.' if self.__apply_consensus_ground else 'No.'))
            img_original: np.ndarray = self.read_img(os.path.join(self.path_input, name_img))
            img_gray: np.ndarray = self.read_img(os.path.join(self.path_input, name_img), True)
            if self.__apply_consensus_ground:
                image_result_dict: dict = {}
            for index, algorithm_name in enumerate(self.__name_algorithms):
                algorithm_method: callable = self.__list_algorithms[index]
                name_args, type_args = self.get_name_and_type_of_args(algorithm_method)
                name_args, value_params = self.verify_and_add_np_array_args(name_args,
                                                                            type_args,
                                                                            img_gray,
                                                                            img_original)
                if len(name_args):
                    value_params += self.verify_and_add_params(algorithm_name, name_args, type_args)
                start = default_timer()
                try:
                    img_with_algorithm: Union[np.ndarray, List[np.ndarray]] = algorithm_method(*value_params)
                except Exception as ex1:
                    raise ex1
                else:
                    stop = default_timer()
                    log_message.info('Execution time: {0}'.format(stop - start))
                    delete_lowercase = str.maketrans('', '', string.ascii_lowercase)
                    name_method_abbreviated: str = algorithm_name.replace('_', ' ').title() \
                        .translate(delete_lowercase).replace(' ', '')
                    if isinstance(img_with_algorithm, dict):
                        # Imagen with title '*', will be to used for consensus ground
                        index_img_result = [index for index, title in enumerate(img_with_algorithm['title'])
                                            if title.find('*') != -1][0]
                        img_with_algorithm['title'][index_img_result] = img_with_algorithm['title'][index_img_result] \
                            .replace('*', '')
                        [self.save_img(img, os.path.join(self.path_output, algorithm_name,
                                                         f'{name_method_abbreviated}_{title}_{name_img}'))
                         for img, title in zip(img_with_algorithm['img'], img_with_algorithm['title'])]
                        if self.__apply_consensus_ground:
                            path_img_with_algorithm = os.path.join(self.path_output, algorithm_name,
                                                                   name_method_abbreviated + '_' +
                                                                   img_with_algorithm['title'][index_img_result] + '_'
                                                                   + name_img)
                            self.__consensus_ground(image_result_dict, True, path_img=path_img_with_algorithm,
                                                    algorithm_name=algorithm_name)
                    else:
                        path_img_with_algorithm = os.path.join(self.path_output, algorithm_name,
                                                               f'{name_method_abbreviated}_{name_img}')
                        self.save_img(img_with_algorithm, path_img_with_algorithm)
                        if self.__apply_consensus_ground:
                            self.__consensus_ground(image_result_dict, True, path_img=path_img_with_algorithm,
                                                    algorithm_name=algorithm_name)
            if self.__apply_consensus_ground:
                self.__consensus_ground(image_result_dict, False, name_img=name_img)

        # De-allocate any associated memory usage
        cv2.destroyAllWindows()

    def __consensus_ground(self, image_result_dict: dict, join_image_to_list: bool, **kwargs):
        if join_image_to_list:
            keys = list(kwargs.keys())
            log_message.info('Joining the result of the algorithm applied to the image '
                             'together with the other results.')
            if 'path_img' in keys and 'algorithm_name' in keys:
                try:
                    img_with_algorithm = Image.open(kwargs['path_img']).convert('1')
                except PIL.UnidentifiedImageError as ex:
                    log_message.error(ex)
                    log_message.error('The file was not generated correctly, in future updates this will be corrected, '
                                      'please try to use .jpg, .jpeg or .png extensions.')
                else:
                    image_result_dict.update({
                        kwargs['algorithm_name']: np.asarray(img_with_algorithm)
                    })
            else:
                raise Exception('Must define the path_img and the algorithm_name.')
        else:
            name_img = 'unknown'
            if 'name_img' in kwargs.keys():
                name_img = kwargs['name_img']
            log_message.info('Generate consensus ground truth for the image in order to compare '
                             'edge detection techniques.')
            algorithms_to_combine = []
            algorithms_used = list(image_result_dict.keys())
            log_message.info(f'Total No. of resulting images: {len(algorithms_used)}')
            for i in range(0, len(algorithms_used) + 1):
                for subset in itertools.combinations(algorithms_used, i):
                    if len(subset) > 1:
                        algorithms_to_combine.append(subset)
            log_message.info(f'Total No. of combinations using two image with algorithms applied: '
                             f'{len(algorithms_to_combine)}')
            combination_results = []
            for combinations in algorithms_to_combine:
                list_results_image: np.ndarray = np.array([[[]]])
                for algorithm in combinations:
                    if not list_results_image.size:
                        list_results_image = np.array([np.asarray(image_result_dict[algorithm])])
                    else:
                        list_results_image = np.append(list_results_image,
                                                       [np.asarray(image_result_dict[algorithm])], axis=0)
                combination_results.append(np.array(sum(list_results_image), dtype=bool))
            assert len(combination_results) == len(algorithms_to_combine), 'Wrong combinations'
            for index, combination in enumerate(combination_results):
                combination: np.ndarray = combination.astype(int) * 255
                path_img_combination = os.path.join(self.path_output, 'consensus_ground',
                                                    '_and_'.join(algorithms_to_combine[index]) + f'_{name_img}')
                self.save_img(combination, path_img_combination)
            log_message.info('Combination of images with applied edge detection algorithms, completed successfully.')

    @property
    def name_algorithms(self):
        return self.__name_algorithms

    @property
    def apply_consensus_ground(self):
        return self.__apply_consensus_ground

    @apply_consensus_ground.setter
    def apply_consensus_ground(self, apply_consensus_ground: bool):
        self.__apply_consensus_ground = apply_consensus_ground
