#!/usr/bin/env python3

import os

# Always prefer setuptools over distutils
import sys

from setuptools import find_packages, setup

long_description = open("README.md", "r", encoding="utf-8").read()

setup(
    name="hearpreprocess",
    version="2021.0.1",
    description="Holistic Evaluation of Audio Representations (HEAR) 2021 -- Preprocessing Pipeline",
    author="",
    author_email="",
    url="https://github.com/neuralaudio/hear-preprocess",
    download_url="https://github.com/neuralaudio/hear-preprocess",
    license="Apache-2.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    project_urls={
        "Bug Tracker": "https://github.com/neuralaudio/hear-preprocess/issues",
        "Source Code": "https://github.com/neuralaudio/hear-preprocess",
    },
    packages=find_packages(exclude=("tests",)),
    python_requires=">=3.7",
    entry_points={},
    install_requires=[
        "click",
        "luigi",
        "numpy==1.19.2",
        "pandas",
        "python-slugify",
        "requests",
        "soundfile",
        "spotty",
        "tqdm",
        "scikit-learn>=0.24.2",
    ],
    extras_require={
        "test": [
            "pytest",
            "pytest-cov",
            "pytest-env",
        ],
        "dev": [
            "pre-commit",
            "black",  # Used in pre-commit hooks
            "pytest",
            "pytest-cov",
            "pytest-env",
        ],
    },
    classifiers=[],
)
