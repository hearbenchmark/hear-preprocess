#!/usr/bin/env python3
from typing import Any, Dict

import hearpreprocess.tfds_pipeline as tfds_pipeline
from hearpreprocess.pipeline import (
    TRAIN_PERCENTAGE,
    TRAINVAL_PERCENTAGE,
    VALIDATION_PERCENTAGE,
)

generic_task_config = {
    # Define the task name and the version
    "task_name": "tfds_speech_commands",
    "version": "0.0.2",
    "embedding_type": "scene",
    "prediction_type": "multiclass",
    "split_mode": "trainvaltest",
    "sample_duration": 1.0,
    "evaluation": ["top1_acc"],
    # This task uses tfds which doesn't require the download paths,
    # but rather the tfds dataset name and version. For speech commands
    # the tf dataset has all the splits, and so we will not be
    # doing any custom splitting. All the splits will be extracted
    # from tfds builder.
    "tfds_task_name": "speech_commands",
    "tfds_task_version": "0.0.2",
    # By default all the splits for the above task and version will
    # be downloaded, the below key helps to select the splits to extract
    "extract_splits": ["train", "test", "valid"],
    "default_mode": "5h",
    "modes": {
        # Different modes for different max task duration
        "5h": {
            # No more than 5 hours of audio (training + validation)
            "max_task_duration_by_split": {
                "train": 3600 * 5 * TRAIN_PERCENTAGE / TRAINVAL_PERCENTAGE,
                "valid": 3600 * 5 * VALIDATION_PERCENTAGE / TRAINVAL_PERCENTAGE,
                # The test set is 1.33 hours, so we use the entire thing
                "test": None,
            }
        },
        "full": {
            "max_task_duration_by_split": {"train": None, "valid": None, "test": None},
        },
    },
}


def extract_metadata_task(task_config: Dict[str, Any]) -> tfds_pipeline.ExtractMetadata:
    # Build the dataset pipeline with the custom metadata configuration task.
    # Please note the tfds download and extract tasks are used to download and the
    # extract the tensorflow data splits below
    download_tasks = tfds_pipeline.get_download_and_extract_tasks_tfds(task_config)

    return tfds_pipeline.ExtractMetadata(
        outfile="process_metadata.csv", task_config=task_config, **download_tasks
    )
