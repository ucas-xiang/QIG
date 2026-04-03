#!/usr/bin/env python
from setuptools import setup
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="qmllm",
    version="0.1.0",
    description="Modality-Balanced Quantization",
    author="Shiyao Li",
    author_email="shiyao1620@gmail.com",
    packages=setuptools.find_packages(),
    license="MIT",
    long_description=long_description,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
