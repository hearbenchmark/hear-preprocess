#!/usr/bin/env python3
"""
Pre-processing pipeline for GTZAN Music Speech Dataset
"""
from typing import Any, Dict

import luigi

import hearpreprocess.pipeline as pipeline
import hearpreprocess.tfds_pipeline as tfds_pipeline

generic_task_config = {
    "task_name": "tfds_gtzan_music_speech",
    "version": "1.0.0",
    "embedding_type": "scene",
    "prediction_type": "multiclass",
    # Split the full dataset in k folds ( k = 10 )
    "split_mode": "new_split_kfold",
    # Doing 10 fold cross validation is standard in literature
    "nfolds": 10,
    # All audio is of 30 seconds
    "sample_duration": 30.0,
    "evaluation": ["top1_acc", "mAP", "d_prime", "aucroc"],
    "tfds_task_name": "gtzan_music_speech",
    "tfds_task_version": "1.0.0",
    "extract_splits": ["train"],
    "default_mode": "full",
    "modes": {
        "full": {
            # Total duration 30.0*120 seconds = 3600 seconds = 1 hour
            # Max task duration for each fold is set to None to fetch the full fold
            "max_task_duration_by_fold": None
        },
        "small": {
            "download_urls": [
                {
                    "split": "train",
                    "url": "https://github.com/kmarufraj/s-task/raw/main/gtzan_music_speech-small.zip",  # noqa: E501
                    "md5": "11d0448e2c497ce1a3bd1a9ee5710aee",
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


def extract_metadata_task(task_config: Dict[str, Any]) -> ExtractMetadata:
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
