#!/usr/bin/env python3
"""
Pre-processing pipeline for VoxLingua107
http://bark.phon.ioc.ee/voxlingua107/

Task details -
    * This is a novel low resource language identification task
    * The actual data is huge, so we are using
        all the audio from the top 10 most frequent languages from the
        small dev set (1600 utterences) to do low resource language identification
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import luigi
import pandas as pd

import hearpreprocess.pipeline as pipeline
from hearpreprocess.util.luigi import diagnostics

logger = logging.getLogger("luigi-interface")

# Select the number of languages to include in the dataset.
# A lot of languages have too few labels
NUMBER_OF_LANGUAGES: int = 10

generic_task_config: Dict[str, Any] = {
    "task_name": "vox_lingua_top10",
    # We are selecting top 10 from the dev set of vox lingua 107 and
    # hence it is a new task
    "version": "hear2021",
    "embedding_type": "scene",
    "prediction_type": "multiclass",
    "split_mode": "new_split_kfold",
    "nfolds": 5,
    # "min": 1.75, "max": 19.98, "10th": 4.31, "25th": 6.19, "50th": 9.18,
    # "75th": 14.07, "90th": 17.06, "95th": 18.64
    "sample_duration": 18.64,
    "evaluation": ["top1_acc", "mAP", "d_prime", "aucroc"],
    "download_urls": [
        {
            "split": "train",
            "url": "http://bark.phon.ioc.ee/voxlingua107/dev.zip",
            "md5": "07fbd4732c97f28c10ec8dcfed4f600e",
        }
    ],
    # 18.64 * 972 (after selecting top 10 language) = 18118 secs = 302 mins
    "default_mode": "full",
    "modes": {
        "full": {
            "max_task_duration_by_fold": None,
        },
        "small": {
            "download_urls": [
                {
                    "split": "train",
                    "url": "https://github.com/kmarufraj/s-task/raw/main/vox_lingua-train-small.zip",  # noqa: E501
                    "md5": "b94370873fbbdc34241a6e4f7eb1952b",
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

    def select_top_k_categories(self, k, metadata) -> pd.DataFrame:
        """Get the k most frequent labels in the train set. All the splits will
        be filtered for these labels"""
        k_frequent_labels: List[str]
        k_frequent_labels = (
            metadata[metadata["split"] == "train"]["label"]
            .value_counts()[:k]
            .index.tolist()
        )
        diagnostics.info(
            f"{self.longname} Selecting top {k}/{metadata['label'].nunique()}"
        )
        diagnostics.info(
            f"{self.longname} Total samples before selecting top k labels: "
            f"{json.dumps(metadata.groupby('split')['relpath'].nunique().to_dict())}"
        )
        metadata = metadata.loc[metadata["label"].isin(k_frequent_labels)]
        diagnostics.info(
            f"{self.longname} Total samples after selecting top k labels: "
            f"{json.dumps(metadata.groupby('split')['relpath'].nunique().to_dict())}"
        )

        return metadata

    def get_requires_metadata(self, split: str) -> pd.DataFrame:
        logger.info(f"Preparing metadata for {split}")

        # Loads and prepares the metadata for a specific split
        split_path = Path(self.requires()[split].workdir).joinpath(split)

        # Get all the audio files in the split folder
        audio_files = list(split_path.rglob("*.wav"))

        metadata: pd.DataFrame = pd.DataFrame(audio_files, columns=["relpath"]).assign(
            label=lambda df: df["relpath"].apply(lambda relpath: relpath.parent.name),
            split="train",
        )
        metadata = self.select_top_k_categories(
            k=NUMBER_OF_LANGUAGES, metadata=metadata
        )

        return metadata


def extract_metadata_task(task_config: Dict[str, Any]) -> pipeline.ExtractMetadata:
    # Build the dataset pipeline with the custom metadata configuration task
    download_tasks = pipeline.get_download_and_extract_tasks(task_config)

    return ExtractMetadata(
        outfile="process_metadata.csv", task_config=task_config, **download_tasks
    )
