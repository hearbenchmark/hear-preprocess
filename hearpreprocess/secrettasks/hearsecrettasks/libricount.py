#!/usr/bin/env python3
"""
Pre-processing pipeline for LibriCount
https://zenodo.org/record/1216072#.YZC0ir1Bzt2

Task details -
    * This is a novel speaker count extimation task
    * The original test set is not available, so this does a 5-fold
        cross validation, modelled as a classification, to classify
        the number of speakers ( 0 - 10 ) in each audio

"""

import logging
from pathlib import Path
from typing import Any, Dict

import luigi
import pandas as pd

import hearpreprocess.pipeline as pipeline

logger = logging.getLogger("luigi-interface")

generic_task_config: Dict[str, Any] = {
    "task_name": "libricount",
    "version": "v1.0.0-hear2021",
    "embedding_type": "scene",
    "prediction_type": "multiclass",
    "split_mode": "new_split_kfold",
    # This is standard in literature
    "nfolds": 5,
    # All sound are 5 seconds
    "sample_duration": 5.0,
    "evaluation": ["top1_acc", "mAP", "d_prime", "aucroc"],
    "download_urls": [
        {
            "split": "train",
            "url": "https://zenodo.org/record/1216072/files/LibriCount10-0dB.zip?download=1",  # noqa: E501
            "md5": "30c8f844dc59fa65d216d53db9dc37e2",
        }
    ],
    # Total duration: 5.0 * 5720 = 28600 s = 476 mins = 7.94 hrs so default mode is
    # set to full
    "default_mode": "full",
    "modes": {
        "full": {
            "max_task_duration_by_fold": None,
        },
        "small": {
            "download_urls": [
                {
                    "split": "train",
                    "url": "https://github.com/kmarufraj/s-task/raw/main/libricount-train-small.zip",  # noqa: E501
                    "md5": "25d1bd1e71800440d635171a8f41a5f7",
                }
            ],
            "max_task_duration_by_fold": None,
            "sample_duration": 2,
        },
    },
}


class ExtractMetadata(pipeline.ExtractMetadata):
    train = luigi.TaskParameter()

    def requires(self):
        return {"train": self.train}

    def get_requires_metadata(self, split: str) -> pd.DataFrame:
        logger.info(f"Preparing metadata for {split}")

        # Loads and prepares the metadata for a specific split
        split_path = Path(self.requires()[split].workdir).joinpath(split, "test")

        # Get all the audio files in the split folder
        audio_files = list(split_path.rglob("*.wav"))
        metadata: pd.DataFrame = pd.DataFrame(audio_files, columns=["relpath"]).assign(
            label=lambda df: df["relpath"].apply(
                lambda relpath: relpath.name.split("_")[0]
            ),
            split="train",
        )

        return metadata


def extract_metadata_task(task_config: Dict[str, Any]) -> pipeline.ExtractMetadata:
    # Build the dataset pipeline with the custom metadata configuration task
    download_tasks = pipeline.get_download_and_extract_tasks(task_config)

    return ExtractMetadata(
        outfile="process_metadata.csv", task_config=task_config, **download_tasks
    )
