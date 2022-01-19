#!/usr/bin/env python3
"""
Pre-processing pipeline for Mridangam Stroke Dataset
Tonic Prediction -
Tonics - b, c, csh, d, dsh, e
https://zenodo.org/record/4068196/
"""

import logging
from typing import Any, Dict
import copy

import hearpreprocess.pipeline as pipeline
from . import mridangam_stroke

logger = logging.getLogger("luigi-interface")

generic_task_config = copy.deepcopy(mridangam_stroke.generic_task_config)
# Rename the task name to be mridangam_tonic
generic_task_config["task_name"] = "mridangam_tonic"


class ExtractMetadata(mridangam_stroke.ExtractMetadata):
    @staticmethod
    def get_label(relpath):
        # <StrokeName>_<Tonic>_<InstanceNumber>.wav
        # <Tonic> = {b, c, csh, d, dsh, e}
        # idx -2 gives the tonic
        tonic = relpath.name.split("__")[-1].split("-")[-2]
        assert tonic in ["b", "c", "csh", "d", "dsh", "e"], f"Unexpected Tonic {tonic}"
        return tonic


def extract_metadata_task(task_config: Dict[str, Any]) -> pipeline.ExtractMetadata:
    # Build the dataset pipeline with the custom metadata configuration task
    download_tasks = pipeline.get_download_and_extract_tasks(task_config)

    return ExtractMetadata(
        outfile="process_metadata.csv", task_config=task_config, **download_tasks
    )
