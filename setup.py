#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
from os import path

NAME = 'lpdr'
DESCRIPTION = 'License Plate Detector and Recognizer'
URL = 'https://github.com/szachovy/lpdr'
EMAIL = 'wjmaj98@gmail.com'
AUTHOR = 'Wiktor Maj'
VERSION = '0.1'

REQUIRED = [
    'numpy>=1.17.4',
    'opencv-python>=4.1.2.30',
    'Keras>=2.3.1',
    'pytesseract>=0.3.4',
]


with open(path.join(path.abspath(path.dirname(__file__)), 'README.md')) as f:
    long_description = f.read()

setup(
    name=NAME,
    version=VERSION,

    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',

    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
#    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*", "Images"]),
    install_requires=REQUIRED,
    include_package_data=True,
    package_data = {
        '': ['*.h5'],
        '': ['*.json']
    },
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
    ]
)
