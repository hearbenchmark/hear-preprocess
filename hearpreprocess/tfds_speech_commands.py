#!/usr/bin/env python3
from typing import Dict, Any

import hearpreprocess.tfds_pipeline as tfds_pipeline

generic_task_config = {
    "task_name": "tfds_speech_commands",
    "version": "0.0.2",
    "embedding_type": "scene",
    "prediction_type": "multiclass",
    "sample_duration": 1.0,
    "evaluation": ["top1_acc"],
    "default_mode": "tfds_full",
    "modes": {
        # This is tfds mode which doesn't require the path but the tfds dataset name
        # and version. For speech commands the tf dataset has all the splits, and so
        # we will not be doing any custom splitting. All the splits will be extracted
        # from tfds builder.
        "tfds_full": {
            "tfds_task_name": "speech_commands",
            "tfds_task_version": "0.0.2",
            # By default all the splits will be downloaded, the below key
            # helps to select the splits to extract
            "extract_splits": ["train", "test", "valid"],
            # The test set is 1.33 hours, so we use the entire thing
            "max_task_duration_by_split": {"train": None, "valid": None, "test": None},
        }
    },
}


def extract_metadata_task(task_config: Dict[str, Any]) -> tfds_pipeline.ExtractMetadata:
    # Build the dataset pipeline with the custom metadata configuration task.
    # Please note the tfds download and extract tasks are used to download and the
    # extract the tensorflow data splits here
    download_tasks = tfds_pipeline.get_download_and_extract_tasks_tfds(task_config)

    return tfds_pipeline.ExtractMetadata(
        outfile="process_metadata.csv", task_config=task_config, **download_tasks
    )
