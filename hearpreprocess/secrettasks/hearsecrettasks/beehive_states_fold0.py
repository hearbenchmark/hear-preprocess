#!/usr/bin/env python3
"""
Pre-processing pipeline for Beehive Dataset
Train/Val Set - Hive 1
Test Set - Hive 3
https://zenodo.org/record/2667806
"""

import logging
from pathlib import Path
from typing import Any, Dict

import hearpreprocess.pipeline as pipeline
import luigi
import pandas as pd
from hearpreprocess.pipeline import (
    TEST_PERCENTAGE,
    TRAIN_PERCENTAGE,
    TRAINVAL_PERCENTAGE,
    VALIDATION_PERCENTAGE,
)
from hearpreprocess.util.luigi import diagnostics

logger = logging.getLogger("luigi-interface")

# The train hive will be split into 90 percent train and 10 percent valid
# by the pipeline
TRAIN_HIVE = "hive1"
# The test hive will be used intact as the test set
TEST_HIVE = "hive3"

generic_task_config: Dict[str, Any] = {
    "task_name": "beehive_states_fold0",
    "version": "v2",
    "embedding_type": "scene",
    "prediction_type": "multiclass",
    "split_mode": "trainvaltest",
    # "mean": 587.75, "var": 891.81, "min": 85.91, "max": 601.6, "10th": 580.5,
    # "25th": 585.14, "50th": 592.0, "75th": 598.4, "90th": 598.4, "95th": 598.4
    "sample_duration": 60 * 10,
    # https://ieeexplore.ieee.org/document/8682981
    # AUCROC is the primary metric in the paper
    "evaluation": ["aucroc", "mAP", "d_prime", "top1_acc"],
    "download_urls": [
        {
            "split": "hive1",
            "name": "12062018",
            "url": "https://zenodo.org/record/2667806/files/Hive1_12_06_2018.rar?download=1",  # noqa: E501
            "md5": "c04cd7fd310a84f327eb7121e026b40a",
        },
        {
            "split": "hive1",
            "name": "31052018",
            "url": "https://zenodo.org/record/2667806/files/Hive1_31_05_2018.rar?download=1",  # noqa: E501
            "md5": "834c58cb1e4c9a15a332a16e106bfa11",
        },
        {
            "split": "hive3",
            "name": "14072017",
            "url": "https://zenodo.org/record/2667806/files/Hive3_14_07_2017.rar?download=1",  # noqa: E501
            "md5": "c1fc5c4d2e110aa2bc8687393db7d55d",
        },
        {
            "split": "hive3",
            "name": "28072017",
            "url": "https://zenodo.org/record/2667806/files/Hive3_28_07_2017.rar?download=1",  # noqa: E501
            "md5": "3fe9c0488a2207379f410a81d17c5d18",
        },
    ],
    # Total duration - 576 * 600 = 345600 secs = 5760 mins = 96 hours
    # So default mode is set to 5 hour
    "default_mode": "5h",
    "modes": {
        "5h": {
            # This sample duration allows us to use all 576 audio files
            "sample_duration": 30,
            # No more than 5 hours of audio (training + validation)
            "max_task_duration_by_split": {
                "train": 3600 * 5 * TRAIN_PERCENTAGE / TRAINVAL_PERCENTAGE,
                "valid": 3600 * 5 * VALIDATION_PERCENTAGE / TRAINVAL_PERCENTAGE,
                "test": 3600 * 5 * TEST_PERCENTAGE / TRAINVAL_PERCENTAGE,
            },
        },
        "full": {
            "max_task_duration_by_split": {"train": None, "valid": None, "test": None},
        },
        "small": {
            "download_urls": [
                {
                    "split": "hive1",
                    "name": "12062018",
                    "url": "https://github.com/kmarufraj/s-task/raw/main/hive1_12062018.zip",  # noqa: E501
                    "md5": "461c77d29e5c2c4ac51deb0b87c82d5c",
                },
                {
                    "split": "hive1",
                    "name": "31052018",
                    "url": "https://github.com/kmarufraj/s-task/raw/main/hive1_31052018.zip",  # noqa: E501
                    "md5": "c0e0b4c33879b865de022413138a61da",
                },
                {
                    "split": "hive3",
                    "name": "14072017",
                    "url": "https://github.com/kmarufraj/s-task/raw/main/hive3_14072017.zip",  # noqa: E501
                    "md5": "2650a40a1e38088391181451e0183ca6",
                },
                {
                    "split": "hive3",
                    "name": "28072017",
                    "url": "https://github.com/kmarufraj/s-task/raw/main/hive3_28072017.zip",  # noqa: E501
                    "md5": "51d43e9621b8ad7681a44355f70329a2",
                },
            ],
            "sample_duration": 2,
            "max_task_duration_by_split": {"train": None, "valid": None, "test": None},
        },
    },
}


class ExtractMetadata(pipeline.ExtractMetadata):
    hive1_12062018 = luigi.TaskParameter()
    hive1_31052018 = luigi.TaskParameter()
    hive3_14072017 = luigi.TaskParameter()
    hive3_28072017 = luigi.TaskParameter()

    def requires(self):
        return {
            requires_task: getattr(self, requires_task)
            for requires_task in [
                "hive1_12062018",
                "hive1_31052018",
                "hive3_14072017",
                "hive3_28072017",
            ]
        }

    @property
    def train_hive(self):
        return TRAIN_HIVE

    @property
    def test_hive(self):
        return TEST_HIVE

    def get_split(self, requires_key: str):
        hive_id: str = requires_key.split("_")[0]
        assert hive_id in ["hive1", "hive3"]
        assert self.train_hive != self.test_hive
        if hive_id == self.train_hive:
            split = "train"
        elif hive_id == self.test_hive:
            split = "test"
        else:
            raise ValueError(
                f"Hive ID should either be {self.train_hive} or {self.test_hive} "
                f"Found {hive_id}"
            )
        return split

    @staticmethod
    def get_label(relpath):
        """Returns label from the relpath"""
        return "NOQUEEN" if "NO" in relpath.name else "QUEEN"

    def get_requires_metadata(self, requires_key: str) -> pd.DataFrame:
        logger.info(f"Preparing metadata for {requires_key}")

        diagnostics.info(
            f"{self.longname} Train Hive - {self.train_hive} "
            f"Test Hive - {self.test_hive}"
        )

        # Loads and prepares the metadata for a specific requires key
        split_path = Path(self.requires()[requires_key].workdir).joinpath(
            *requires_key.split("_")
        )
        # Get all the audio files in the split folder
        audio_files = list(split_path.rglob("*.wav"))
        metadata: pd.DataFrame = pd.DataFrame(audio_files, columns=["relpath"]).assign(
            label=lambda df: df["relpath"].apply(self.get_label),
            split=self.get_split(requires_key),
        )

        return metadata


def extract_metadata_task(task_config: Dict[str, Any]) -> pipeline.ExtractMetadata:
    # Build the dataset pipeline with the custom metadata configuration task
    download_tasks = pipeline.get_download_and_extract_tasks(task_config)

    return ExtractMetadata(
        outfile="process_metadata.csv", task_config=task_config, **download_tasks
    )
