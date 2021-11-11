#!/usr/bin/env python3
"""
Pre-processing pipeline for NSynth pitch detection, but with 10-fold
as a way of checking kfold versus 80/10/10 train/val/test split.
"""

from hearpreprocess.nsynth_pitch import *

generic_task_config["task_name"] = "nsynth_pitch_kfold"
generic_task_config["split_mode"] = "new_split_kfold"
# train/val/test is 80/10/10, so 10-fold is identical in split proportions
generic_task_config["nfolds"] = 10

# generic_task_config["modes"]["5h"] = {"max_task_duration_by_fold": 3600 * 5 * TEST_PERCENTAGE / TRAINVAL_PERCENTAGE}
# generic_task_config["modes"]["50h"] = {"max_task_duration_by_fold": 36000 * 5 * TEST_PERCENTAGE / TRAINVAL_PERCENTAGE}
# generic_task_config["modes"]["small"]["max_task_duration_by_fold"] = None
# del generic_task_config["modes"]["small"]["max_task_duration_by_split"]

# Keep the test length (same as validation) for every fold the
# same as our standard 80/10/10 train/val/test nsynth split.
for mode in list(generic_task_config["modes"].keys()):
    generic_task_config["modes"][mode][
        "max_task_duration_by_fold"
    ] = generic_task_config["modes"][mode]["max_task_duration_by_split"]["test"]
    del generic_task_config["modes"][mode]["max_task_duration_by_split"]


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
