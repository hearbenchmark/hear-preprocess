#!/usr/bin/env python3

# Always prefer setuptools over distutils
from setuptools import find_packages, setup

long_description = open("README.md", "r", encoding="utf-8").read()

setup(
    name="hearsecrettasks",
    version="2021.0.1",
    description="Holistic Evaluation of Audio Representations (HEAR)"
    + " 2021 -- Secret Tasks",
    author="",
    author_email="",
    url="https://github.com/neuralaudio/hear2021-secret-tasks",
    download_url="https://github.com/neuralaudio/hear2021-secret-tasks",
    license="Apache-2.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    project_urls={
        "Bug Tracker": "https://github.com/neuralaudio/hear2021-secret-tasks/issues",
        "Source Code": "https://github.com/neuralaudio/hear2021-secret-tasks",
    },
    packages=find_packages(exclude=("tests",)),
    python_requires=">=3.7",
    entry_points={"console_scripts": []},
    install_requires=["luigi", "pandas", "note_seq", "numpy", "patool"],
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
