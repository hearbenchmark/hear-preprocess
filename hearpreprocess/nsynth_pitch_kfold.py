#!/usr/bin/env python3
"""
Pre-processing pipeline for NSynth pitch detection
"""

from nsynth_pitch import *

generic_task_config["split_mode"] = "new_split_kfold"
generic_task_config["nfolds"] = 5

for mode in generic_task_config["modes"]:
    generic_task_config["modes"]["max_task_duration_by_fold"] = generic_task_config[
        "modes"
    ]["max_task_duration_by_split"]["test"]
    del generic_task_config["modes"]["max_task_duration_by_split"]


# This works on train/val/test, which for kfold might b0rk.
# Worst case we just use the massive train set for our kfold,
# which should be fine given the amount we downsample.
# (For kfold we will naturally have different test sets than the original
# anyway, and nsynth splits are split in the same way we split
# (by instrument)
# class ExtractMetadata(pipeline.ExtractMetadata):
#    train = luigi.TaskParameter()
#    test = luigi.TaskParameter()
#    valid = luigi.TaskParameter()
