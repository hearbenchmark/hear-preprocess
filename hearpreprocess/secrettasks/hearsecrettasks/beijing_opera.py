#!/usr/bin/env python3
"""
Pre-processing pipeline for Beijing Opera Percussion Instrument Dataset
https://zenodo.org/record/1285212

Task details -
    * This is a novel low resource percussion instrument detection task
    * The actual dataset was used in an onset detection challenge,
        however, the test set was not released, so this task uses the
        orginal train set and does 5 fold cross validation to do
        low resource instrument classification
"""

import logging
from pathlib import Path
from typing import Any, Dict

import luigi
import pandas as pd

import hearpreprocess.pipeline as pipeline

logger = logging.getLogger("luigi-interface")

generic_task_config: Dict[str, Any] = {
    "task_name": "beijing_opera",
    "version": "v1.0-hear2021",
    "embedding_type": "scene",
    "prediction_type": "multiclass",
    "split_mode": "new_split_kfold",
    "nfolds": 5,
    #   "min": 0.07, "max": 8.94, "10th": 0.21, "25th": 0.32, "50th": 1.16,
    #   "75th": 2.54, "90th": 4.32, "95th": 4.77
    "sample_duration": 4.77,
    "evaluation": ["top1_acc", "mAP", "d_prime", "aucroc"],
    "download_urls": [
        {
            "split": "train",
            "url": "https://zenodo.org/record/1285212/files/beijing_opera_percussion_instrument_1.0.zip?download=1",  # noqa: E501
            "md5": "51dc70d02a06b9d96befadf320a6996f",
        }
    ],
    "default_mode": "full",
    "modes": {
        "full": {
            # 236 * 4.77 = 1125.72 secs
            "max_task_duration_by_fold": None,
        },
        "small": {
            "download_urls": [
                {
                    "split": "train",
                    "url": "https://github.com/kmarufraj/s-task/raw/main/beijing_opera-train-small.zip",  # noqa: E501
                    "md5": "391b086d4a81c525d8b052ac50a6a8c0",
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
        split_path = Path(self.requires()[split].workdir).joinpath(
            split, "beijing_opera_percussion_instrument_1.0"
        )

        # Get all the audio files in the split folder
        audio_files = list(split_path.glob("*.wav"))
        metadata: pd.DataFrame = pd.DataFrame(audio_files, columns=["relpath"]).assign(
            label=lambda df: df["relpath"].apply(
                lambda relpath: relpath.name.split("__")[-1].split("-")[-2]
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
