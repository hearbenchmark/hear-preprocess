#!/usr/bin/env python3
"""
Pre-processing pipeline for Beehive Dataset
Train/Val Set - Hive 3
Test Set - Hive 1
https://zenodo.org/record/2667806
"""

import copy
import logging
from typing import Any, Dict

import hearpreprocess.pipeline as pipeline

from . import beehive_states_fold0

logger = logging.getLogger("luigi-interface")

TRAIN_HIVE = "hive3"
TEST_HIVE = "hive1"

generic_task_config: Dict[str, Any] = copy.deepcopy(
    beehive_states_fold0.generic_task_config
)
generic_task_config["task_name"] = "beehive_states_fold1"


class ExtractMetadata(beehive_states_fold0.ExtractMetadata):
    @property
    def train_hive(self):
        return TRAIN_HIVE

    @property
    def test_hive(self):
        return TEST_HIVE


def extract_metadata_task(task_config: Dict[str, Any]) -> pipeline.ExtractMetadata:
    # Build the dataset pipeline with the custom metadata configuration task
    download_tasks = pipeline.get_download_and_extract_tasks(task_config)

    return ExtractMetadata(
        outfile="process_metadata.csv", task_config=task_config, **download_tasks
    )
