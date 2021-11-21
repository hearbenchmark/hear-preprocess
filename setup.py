#!/usr/bin/env python3

import os

import sys

# Always prefer setuptools over distutils
from setuptools import find_packages, setup

long_description = open("README.md", "r").read()

setup(
    name="hearpreprocess",
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
        # One of the requirements pulls in librosa, I believe note_seq
        # So we need to pin these, otherwise librosa breaks
        "numpy==1.19.2",
        "numba==0.48",
        "pandas",
        "python-slugify",
        "requests",
        "soundfile",
        "spotty",
        "tensorflow-datasets",
        "tqdm",
        "scikit-learn>=0.24.2",
        "ffmpeg-python",
        "note_seq",
        "tensorflow>=2.0",
        "schema",
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
