#!/usr/bin/env python3
"""
Pre-processing pipeline for Free Spoken Digit dataset
"""
from typing import Any, Dict

import luigi

import hearpreprocess.pipeline as pipeline
import hearpreprocess.tfds_pipeline as tfds_pipeline

generic_task_config = {
    "task_name": "spoken_digit",
    "version": "1.0.9",
    "embedding_type": "scene",
    "prediction_type": "multiclass",
    "split_mode": "new_split_kfold",
    "nfolds": 5,
    # mean: 0.41s, 75th: 0.47s, 90th: 0.56s, max: 2.28s
    "sample_duration": 0.56,
    "evaluation": ["top1_acc"],
    "tfds_task_name": "spoken_digit",
    "tfds_task_version": "1.0.9",
    "extract_splits": ["train"],
    "default_mode": "full",
    "modes": {
        "full": {"max_task_duration_by_fold": None},
        "small": {
            "download_urls": [
                {
                    "split": "train",
                    "url": "https://github.com/hearbenchmark/hear2021-open-tasks-downsampled/raw/main/spoken_digit-small.zip",  # noqa: E501
                    "md5": "69d50c15805ea11beb12d9a4db1d4c2a",
                }
            ],
            "max_task_duration_by_fold": None,
        },
    },
}


class ExtractMetadata(pipeline.ExtractMetadata):
    """
    Defining the ExtractMetadata Task since `train` is the only available split
    Please refer to docstring of `tfds_pipeline.ExtractMetadata` for more
    details.
    """

    train = luigi.TaskParameter()

    def requires(self):
        return {"train": self.train}

    # Override the get_requires_metadata method to handle the train split
    get_requires_metadata = tfds_pipeline.ExtractMetadata.get_requires_metadata


def extract_metadata_task(task_config: Dict[str, Any]) -> pipeline.ExtractMetadata:
    """
    Receives a task_config dictionary, downloads and extracts the correct files from
    TFDS and prepares the ExtractMetadata task class for the train split
    """
    if task_config["mode"] == "small":
        # Small mode uses a sampled version of TF dataset - downloads directly
        # from a URL as opposed to using the tfds methods.
        download_tasks = pipeline.get_download_and_extract_tasks(task_config)
    else:
        download_tasks = tfds_pipeline.get_download_and_extract_tasks_tfds(task_config)

    return ExtractMetadata(
        outfile="process_metadata.csv", task_config=task_config, **download_tasks
    )
