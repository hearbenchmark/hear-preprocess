"""
Task Config Validator for hear preprocess tasks
"""

from typing import Dict, Any, List
from schema import Schema, And, Optional, Or, Forbidden
import copy

SPLITS = ["train", "test", "valid"]


def validate_generic_task_config(
    generic_task_config: Dict[str, Any], ignore_extra_keys: bool = True
):
    """Validates a task config to be compatible with the hearpreprocess pipeline
    Args:
        task_config: Task config to be used with the pipeline
        ignore_extra_keys: Flag for ignoring extra keys in the task configuration.
            Defaults to True
    Raises:
        schema.SchemaError: If the schema doesnot match, this error is raised
            with the information on what didnot match
    """
    assert "split_mode" in generic_task_config, "split_mode key should be defined"
    split_mode: str = generic_task_config["split_mode"]

    assert "modes" in generic_task_config, "modes key should be defined"
    # value against the `modes` key in the task_config should be a dictionary
    Schema(dict).validate(generic_task_config["modes"])
    task_modes: List = list(generic_task_config["modes"].keys())

    # Validate the generic task config for each mode
    for task_mode in task_modes:
        print(
            f"Validating for '{task_mode}' mode for the "
            f"{generic_task_config['task_name']} task. "
            "If an error occurs in the schema please check the value against "
            "the keys in this task mode."
        )

        task_config: Dict[str, Any] = copy.deepcopy(generic_task_config)
        task_config.update(dict(task_config["modes"][task_mode]))
        del task_config["modes"]

        # GENERIC Taks keys
        schema: Dict[str, Any] = {
            "task_name": str,
            "version": str,
            "embedding_type": Or("scene", "event"),
            "prediction_type": Or("multiclass", "multilabel"),
            "split_mode": Or("trainvaltest", "presplit_kfold", "new_split_kfold"),
            "sample_duration": Or(float, int, None),
            "evaluation": Schema([str]),
            "default_mode": Or("5h", "50h", "full"),
        }
        # DOWNLOAD Source specific keys
        # If the source of data for the task is a tensorflow dataset
        if "tfds_task_name" in task_config:
            schema.update(
                {
                    # The tfds dataset name
                    "tfds_task_name": str,
                    # The tfds dataset name
                    "tfds_task_version": str,
                    # Defines which splits to extract from the tfds dataset source
                    "extract_splits": Schema([Or(*SPLITS)]),
                }
            )
        # If the source of data for the task is source download urls
        else:
            # ignore_extra_keys is true for the download_urls dict
            # as the download source might require a different set of keys
            # and a modified get_download_and_extract_tasks function
            # to be defined for the task in task specific scripts
            schema.update(
                {
                    "download_urls": Schema(
                        [
                            # A set of optional keys are still defined here
                            {
                                Optional("split"): str,
                                Optional("name"): str,
                                Optional("url"): str,
                                Optional("md5"): str,
                            }
                        ],
                        ignore_extra_keys=True,
                    )
                }
            )
        # SPLIT Mode specific keys
        if split_mode == "trainvaltest":
            schema.update(
                {
                    # max_task_duration_by_split duration is optional
                    # If not defined, the default max_task_duration_by_split for each
                    # of the train, test and valid will be used
                    "max_task_duration_by_split": Schema(
                        {split: Or(int, float, None) for split in SPLITS}
                    ),
                    # nfolds is invalid for this split mode
                    Forbidden(
                        "nfolds",
                        error=(
                            f"nfolds should not be defined for {split_mode} "
                            "split mode"
                        ),
                    ): object,
                    # max_task_duration_by_fold is invalid for this split mode.
                    Forbidden(
                        "max_task_duration_by_fold",
                        error="max_task_duration_by_fold should not be defined for "
                        "{split_mode} split mode",
                    ): object,
                }
            )
        elif split_mode in ["presplit_kfold", "new_split_kfold"]:

            assert (
                "nfolds" in task_config
            ), "nfolds should be defined for {split_mode} split mode."
            nfolds: int = task_config["nfolds"]
            schema.update(
                {
                    # nfolds defines the number of folds
                    # If the split_mode is new_split_kfold nfolds number of folds
                    # will be made
                    "nfolds": And(int, lambda nfolds: nfolds > 3),
                    # max_task_duration_by_fold should be defined for each fold
                    # It can have a value of None if the full fold is required
                    "max_task_duration_by_fold": Schema(
                        {
                            "fold{:02d}".format(i): Or(int, float, None)
                            for i in range(nfolds)
                        }
                    ),
                    # max_task_duration_by_split is invalid for this mode
                    Forbidden(
                        "max_task_duration_by_split",
                        error=(
                            "max_task_duration_by_split should not be defined for "
                            "{split_mode} split_mode"
                        ),
                    ): object,
                }
            )
        else:
            raise ValueError("Invalid split_mode")

        Schema(schema, ignore_extra_keys=ignore_extra_keys).validate(task_config)
        print("Successfully validated")
