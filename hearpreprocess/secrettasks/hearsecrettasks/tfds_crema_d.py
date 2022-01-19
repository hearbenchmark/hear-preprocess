#!/usr/bin/env python3
"""
Pre-processing pipeline for Creme D Dataset
"""

from typing import Any, Dict

import hearpreprocess.pipeline as pipeline
import hearpreprocess.tfds_pipeline as tfds_pipeline

generic_task_config = {
    "task_name": "tfds_crema_d",
    "version": "1.0.0",
    "embedding_type": "scene",
    "prediction_type": "multiclass",
    # The tf dataset has pre-defined splits, but examples in the literature use
    # 5-fold cross validation and average scores across folds.
    "split_mode": "new_split_kfold",
    "nfolds": 5,
    # full dataset stats - 75th %tile: 2.84, 90th %tile: 3.17, max: 5
    "sample_duration": 5,  # in seconds
    # Original Dataset Paper - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4313618/pdf/nihms-596618.pdf # noqa: E501
    # Dataset Application Paper - https://arxiv.org/pdf/2002.05039v1.pdf - f1_score
    # Dataset Application Ppaer - https://www.researchgate.net/publication/342678861_Dual-Modal_Emotion_Recognition_Using_Discriminant_Correlation_Analysis - top1_acc # noqa: E501
    "evaluation": ["top1_acc", "mAP", "d_prime", "aucroc"],
    # This task uses tfds which doesn't require the download paths,
    # but rather the tfds dataset name and version.
    "tfds_task_name": "crema_d",
    "tfds_task_version": "1.0.0",
    "extract_splits": ["train", "test", "valid"],
    "default_mode": "full",
    "modes": {
        "full": {
            # 1488 or 1487 samples per fold @ 5 seconds
            # Total duration of all folds is 10.3hrs
            "max_task_duration_by_fold": None,
        },
        "small": {
            "download_urls": [
                {
                    "split": "train",
                    "url": "https://github.com/kmarufraj/s-task/raw/main/crema_d-train-small.zip",  # noqa: E501
                    "md5": "a42147d0bed93d579250a064c6a46000",
                },
                {
                    "split": "test",
                    "url": "https://github.com/kmarufraj/s-task/raw/main/crema_d-test-small.zip",  # noqa: E501
                    "md5": "f7f9d662d43de2ecd6ed2cc75adcf315",
                },
                {
                    "split": "valid",
                    "url": "https://github.com/kmarufraj/s-task/raw/main/crema_d-valid-small.zip",  # noqa: E501
                    "md5": "d1bdddb23004d421218533b72e64737c",
                },
            ],
            "max_task_duration_by_fold": None,
            "sample_duration": 2,
        },
    },
}


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

    return tfds_pipeline.ExtractMetadata(
        outfile="process_metadata.csv", task_config=task_config, **download_tasks
    )
