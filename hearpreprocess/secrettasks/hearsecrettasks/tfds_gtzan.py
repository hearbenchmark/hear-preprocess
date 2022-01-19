#!/usr/bin/env python3
"""
Pre-processing pipeline for GTZAN Dataset
"""
from typing import Any, Dict

import luigi

import hearpreprocess.pipeline as pipeline
import hearpreprocess.tfds_pipeline as tfds_pipeline

generic_task_config = {
    "task_name": "tfds_gtzan",
    "version": "1.0.0",
    "embedding_type": "scene",
    "prediction_type": "multiclass",
    # Splits will be made by deterministically partitionin the
    # data
    "split_mode": "new_split_kfold",
    # Doing k fold validation is standard in literature
    "nfolds": 10,
    # All audio duration is 30 seconds
    "sample_duration": 30.0,
    "evaluation": ["top1_acc", "mAP", "d_prime", "aucroc"],
    "tfds_task_name": "gtzan",
    "tfds_task_version": "1.0.0",
    "extract_splits": ["train"],
    "default_mode": "full",
    "modes": {
        "full": {
            # Total duration 30.0*1000 seconds: 500 minutes = 8.3 hours
            # Max task duration for each fold is set to None to fetch the full fold
            "max_task_duration_by_fold": None
        },
        "small": {
            "download_urls": [
                {
                    "split": "train",
                    "url": "https://github.com/kmarufraj/s-task/raw/main/gtzan-small.zip",  # noqa: E501
                    "md5": "c3ac045c759cde4c5f44d2e7daf5d43d",
                }
            ],
            "max_task_duration_by_fold": None,
            "sample_duration": 2,
        },
    },
}


class ExtractMetadata(pipeline.ExtractMetadata):
    """
    Defining the ExtractMetadata Task since `train` is the only available split
    Please refer to docstring of `tfds_pipeline.ExtractMetadata` for more
    details
    """

    train = luigi.TaskParameter()

    def requires(self):
        return {"train": self.train}

    get_requires_metadata = tfds_pipeline.ExtractMetadata.get_requires_metadata


def extract_metadata_task(task_config: Dict[str, Any]) -> tfds_pipeline.ExtractMetadata:
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
