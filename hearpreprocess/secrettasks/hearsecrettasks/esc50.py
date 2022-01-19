#!/usr/bin/env python3
"""
Pre-processing pipeline for ESC-50 Dataset
"""

import logging
from pathlib import Path
from typing import Any, Dict

import luigi
import pandas as pd

import hearpreprocess.pipeline as pipeline

logger = logging.getLogger("luigi-interface")

generic_task_config = {
    "task_name": "esc50",
    "version": "v2.0.0",
    "embedding_type": "scene",
    "prediction_type": "multiclass",
    # All audio is of length 5 seconds
    "sample_duration": 5,
    # Folds in ESC-50 are pre-defined. We use those folds for k-fold.
    "split_mode": "presplit_kfold",
    "nfolds": 5,
    "evaluation": ["top1_acc", "mAP", "d_prime", "aucroc"],
    "download_urls": [
        {
            "split": "all_folds",
            "url": "https://github.com/karoldvl/ESC-50/archive/master.zip",
            "md5": "7771e4b9d86d0945acce719c7a59305a",
        },
    ],
    "default_mode": "full",
    # Different modes for preprocessing this dataset
    "modes": {
        "full": {
            # For tasks with folds, the max_task_duration_by_fold has to be
            # defined for each fold explicitly. It is set to None, so that the full
            # fold can be selected without any max cap
            "max_task_duration_by_fold": None
        },
        "small": {
            "download_urls": [
                {
                    "split": "all_folds",
                    "url": "https://github.com/kmarufraj/s-task/raw/main/esc50-all_folds-small.zip",  # noqa: E501
                    "md5": "08bb240e414ce96529087aba52983b1c",
                }
            ],
            "max_task_duration_by_fold": None,
            "sample_duration": 2,
        },
    },
}


class ExtractMetadata(pipeline.ExtractMetadata):
    all_folds = luigi.TaskParameter()

    def requires(self):
        return {"all_folds": self.all_folds}

    def get_requires_metadata(self, requires_key: str):

        split_path = Path(self.requires()[requires_key].workdir).joinpath(
            requires_key, "ESC-50-master"
        )
        audio_path = split_path.joinpath("audio")
        metadata_path = split_path.joinpath("meta", "esc50.csv")
        src_metadata: pd.DataFrame = pd.read_csv(metadata_path)

        metadata = src_metadata.assign(
            relpath=src_metadata["filename"].apply(
                lambda path: audio_path.joinpath(path)
            ),
            split=src_metadata["fold"].apply(
                lambda fold_num: f"fold{(fold_num - 1):02d}"
            ),
            label=src_metadata["category"],
        )

        return metadata


def extract_metadata_task(task_config: Dict[str, Any]) -> pipeline.ExtractMetadata:
    # Build the dataset pipeline with the custom metadata configuration task
    download_tasks = pipeline.get_download_and_extract_tasks(task_config)

    return ExtractMetadata(
        outfile="process_metadata.csv", task_config=task_config, **download_tasks
    )
