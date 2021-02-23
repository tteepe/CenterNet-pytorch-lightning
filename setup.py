#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='CenterNet',
    version='0.0.0',
    description='Refactored version of CenterNet (Objects as Points). With PyTorch Lightning and imgaug.',
    author='Torben Teepe',
    author_email='torben@tee.pe',
    url='https://github.com/tteepe/CenterNet-pytorch-lightning',
    install_requires=['pytorch-lightning'],
    packages=find_packages(),
)

