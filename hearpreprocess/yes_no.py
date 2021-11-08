#!/usr/bin/env python3
"""
Pre-processing pipeline for YesNo dataset
"""

generic_task_config = {
    "task_config": "yes_no",
    "version": "1.0.0",
    "embedding_type": "scene",
    "prediction_type": "multiclass",
    "split_mode": "new_split_kfold",
    "nfolds": 5,
    "sample_duration": 10.0,
    "evaluation": ["top1_acc"],
    "tfds_task_name": "yes_no",
    "tfds_task_version": "1.0.0",
    "extract_splits": "train",
    "default_mode": "full",
    "modes" : {

    }


}