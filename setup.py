#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
from os import path
#from pip.req import parse_requirements

NAME = 'lpdr'
DESCRIPTION = 'License Plate Detector and Recognizer'
URL = 'https://github.com/szachovy/lpdr'
EMAIL = 'wjmaj98@gmail.com'
AUTHOR = 'Wiktor Maj'
VERSION = '0.1'

REQUIRED = [
    'numpy>=1.17.4',
    'opencv-python==4.1.2.30',
    'Keras>=2.3.1',
    'pytesseract>=0.3.4'
]

#REQUIRED = parse_requirements('requirements.txt')

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
    #packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    packages=['lpdr'],
    install_requires=REQUIRED, #[str(ir.req) for ir in REQUIRED],
    include_package_data=True,
    package_data = {
        '': ['*.h5'],
        '': ['*.json']
    },
    license='MIT',
    data_files=[('lpdr/wpod_net', ['lpdr/wpod_net/wpod_net_update1.h5', 'lpdr/wpod_net/wpod_net_update1.json'])],
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3'
    ]
)




