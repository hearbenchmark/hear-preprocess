#!/usr/bin/env python3
"""
Pre-processing pipeline for Mridangam Stroke Dataset
Stroke Prediction
Strokes - Bheem, Cha, Dheem, Dhin, Num, Ta, Tha, Tham, Thi, Thom
https://zenodo.org/record/4068196/
"""

import logging
from pathlib import Path
from typing import Any, Dict

import luigi
import pandas as pd

import hearpreprocess.pipeline as pipeline

logger = logging.getLogger("luigi-interface")

generic_task_config: Dict[str, Any] = {
    "task_name": "mridangam_stroke",
    "version": "v1.5",
    "embedding_type": "scene",
    "prediction_type": "multiclass",
    "split_mode": "new_split_kfold",
    "nfolds": 5,
    #   "mean": 0.35, "var": 0.04, "min": 0.02, "max": 1.58, "10th": 0.19,
    #   "25th": 0.21, "50th": 0.31, "75th": 0.42, "90th": 0.66, "95th": 0.81
    "sample_duration": 0.81,
    "evaluation": ["top1_acc", "mAP", "d_prime", "aucroc"],
    "download_urls": [
        {
            "split": "train",
            "url": "https://zenodo.org/record/4068196/files/mridangam_stroke_1.5.zip?download=1",  # noqa: E501
            "md5": "39af55b2476b94c7946bec24331ec01a",
        }
    ],
    # Total duration: 0.81 * 6977 = 5651 secs = 94.1 mins so default mode is
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
                    "url": "https://github.com/kmarufraj/s-task/raw/main/mridangam-train-small.zip",  # noqa: E501
                    "md5": "8269ea439beacacf8db19d6c26dd1e71",
                }
            ],
            "max_task_duration_by_fold": None,
            "sample_duration": 1,
        },
    },
}


class ExtractMetadata(pipeline.ExtractMetadata):
    train = luigi.TaskParameter()

    def requires(self):
        return {"train": self.train}

    @staticmethod
    def get_label(relpath):
        # Filename format - <StrokeName>_<Tonic>_<InstanceNumber>.wav
        # <StrokeName> = {Bheem, Cha, Dheem, Dhin, Num, Ta, Tha, Tham, Thi, Thom}
        # idx -3 gives the stroke
        stroke = relpath.name.split("__")[-1].split("-")[-3]
        assert stroke in [
            "bheem",
            "cha",
            "dheem",
            "dhin",
            "num",
            "ta",
            "tha",
            "tham",
            "thi",
            "thom",
        ], "Unexpected Stroke"
        return stroke

    def get_requires_metadata(self, split: str) -> pd.DataFrame:
        logger.info(f"Preparing metadata for {split}")

        # Loads and prepares the metadata for a specific split
        split_path = Path(self.requires()[split].workdir).joinpath(
            split, "mridangam_stroke_1.5"
        )
        # Get all the audio files in the split folder
        audio_files = list(split_path.rglob("*.wav"))
        metadata: pd.DataFrame = pd.DataFrame(audio_files, columns=["relpath"]).assign(
            label=lambda df: df["relpath"].apply(self.get_label),
            split="train",
        )

        return metadata


def extract_metadata_task(task_config: Dict[str, Any]) -> pipeline.ExtractMetadata:
    # Build the dataset pipeline with the custom metadata configuration task
    download_tasks = pipeline.get_download_and_extract_tasks(task_config)

    return ExtractMetadata(
        outfile="process_metadata.csv", task_config=task_config, **download_tasks
    )
