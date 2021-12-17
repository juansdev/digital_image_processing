import os
from setuptools import setup, find_packages
from os.path import join, dirname

# La ruta del build donde digital_image_processing esta siendo compilado
src_path = dirname(__file__)
print("La ruta actual es: {}".format(os.getcwd()))
print("Fuente y la inicialización de la carpeta build es: {}".format(src_path))

# __version__ es importado por exec, pero ayuda a linter a no quejarse
__version__ = None
with open(join(src_path, 'digital_image_processing', '_version.py'), encoding="utf-8") as f:
    exec(f.read())


def get_description():
    with open(join(dirname(__file__), 'README.md'), 'rb') as fileh:
        return fileh.read().decode("utf8").replace('\r\n', '\n')


setup(
    name='digital_image_processing',
    version=__version__,
    packages=find_packages(),
    package_data={
        "digital_image_processing": ["test_imgs/*.*"]
    },
    author='Juan Guillermo Serrano Ramirez & Sergio Orjuela',
    author_email='juanfater2017@gmail.com',
    url='https://github.com/juansdev',
    entry_points={
        'console_scripts': [
            'digital_image_processing = digital_image_processing.main:test_algorithms_main',
        ],
    },
    install_requires=[i.strip() for i in open("requirements.txt").readlines()],
    description="Digital Image Processing is a python package featuring Numpy/Scipy/OpenCV implementations of "
                "image edge detection and adaptive thresholding algorithms.",
    long_description=get_description(),
    long_description_content_type='text/markdown',
    license='MIT',
    keywords='digital image processing',
    download_url='https://github.com/juansdev/digital_image_processing/archive/refs/tags/0.2.zip',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python',
        'License :: OSI Approved :: MIT License'
    ],
)
