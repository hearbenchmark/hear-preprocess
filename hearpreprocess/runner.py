#!/usr/bin/env python3
"""
Runs a luigi pipeline to build a dataset
"""

import copy
import logging
import multiprocessing
from typing import List, Optional

import click

import hearpreprocess.dcase2016_task2 as dcase2016_task2
import hearpreprocess.nsynth_pitch as nsynth_pitch
import hearpreprocess.nsynth_pitch_kfold as nsynth_pitch_kfold
import hearpreprocess.pipeline as pipeline
import hearpreprocess.speech_commands as speech_commands
import hearpreprocess.spoken_digit as spoken_digit
import hearpreprocess.tfds_speech_commands as tfds_speech_commands
from hearpreprocess.util.task_config import validate_generic_task_config

logger = logging.getLogger("luigi-interface")
# Currently the runner is only allowed to run for open tasks
# The secret tasks module will be not be available for the participants
try:
    from hearpreprocess.secrettasks import hearsecrettasks

    secret_tasks = hearsecrettasks.tasks

except ImportError as e:
    print(e)
    logger.info(
        "The hearsecrettask submodule is not installed. "
        "If you are a participant, this is an expected behaviour as the "
        "secret tasks are not made available to you. "
    )
    secret_tasks = {}

tasks = {
    "tfds_speech_commands": [tfds_speech_commands],
    "speech_commands": [speech_commands],
    "nsynth_pitch": [nsynth_pitch],
    "nsynth_pitch_kfold": [nsynth_pitch_kfold],
    "dcase2016_task2": [dcase2016_task2],
    "spoken_digit": [spoken_digit],
    "open": [
        speech_commands,
        nsynth_pitch,
        nsynth_pitch_kfold,
        dcase2016_task2,
        spoken_digit,
    ],
    "all": [
        speech_commands,
        nsynth_pitch,
        nsynth_pitch_kfold,
        dcase2016_task2,
        spoken_digit,
    ]
    + secret_tasks.get("all-secret", []),
    # Add the task config for the secrets task if the secret task config was found.
    # Not available for participants
    **secret_tasks,
}


@click.command()
@click.argument("tasklist", nargs=-1, required=True)
@click.option(
    "--num-workers",
    default=None,
    help="Number of CPU workers to use when running. "
    "If not provided all CPUs are used.",
    type=int,
)
@click.option(
    "--sample-rate",
    default=None,
    help="Perform resampling only to this sample rate. "
    "By default we resample to 16000, 22050, 32000, 44100, 48000.",
    type=int,
)
@click.option(
    "--tmp-dir",
    default="_workdir",
    help="Temporary directory to save all the "
    "intermediate tasks (will not be deleted afterwords). "
    "(default: _workdir/)",
    type=str,
)
@click.option(
    "--tasks-dir",
    default="tasks",
    help="Directory to save the final task output (default: tasks/)",
    type=str,
)
@click.option(
    "--tar-dir",
    default=".",
    help="Directory to save the tar'ed output (default: .)",
    type=str,
)
@click.option(
    "--mode",
    default="default",
    help="default, all, or small mode for each task.",
    type=str,
)
def run(
    tasklist: List[str],
    num_workers: Optional[int] = None,
    sample_rate: Optional[int] = None,
    tmp_dir: Optional[str] = "_workdir",
    tasks_dir: Optional[str] = "tasks",
    tar_dir: Optional[str] = ".",
    mode: str = "default",
):
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
        logger.info(f"Using {num_workers} workers")

    if sample_rate is None:
        sample_rates = [16000, 22050, 32000, 44100, 48000]
    else:
        sample_rates = [sample_rate]

    tasks_to_run = []
    for task in tasklist:
        for task_module in tasks[task]:
            # Validate the generic task configuration defined for the task

            validate_generic_task_config(task_module.generic_task_config)  # type: ignore # noqa: E501
            if mode == "default":
                task_modes = [task_module.generic_task_config["default_mode"]]  # type: ignore # noqa: E501
            elif mode == "small":
                task_modes = ["small"]
            elif mode in task_module.generic_task_config["modes"]:  # type: ignore
                task_modes = [mode]
            elif mode == "all":
                task_modes = [
                    task_mode
                    for task_mode in task_module.generic_task_config["modes"].keys()  # type: ignore # noqa: E501
                    if task_mode != "small"
                ]
                assert task_modes is not [], f"Task {task} has no modes besides 'small'"
            else:
                raise ValueError(f"mode {mode} unknown")
            for task_mode in task_modes:
                task_config = copy.deepcopy(task_module.generic_task_config)  # type: ignore # noqa: E501
                if task_mode == "small" and "small" not in task_config["modes"]:
                    print(
                        f"No small mode found in {task_config['task_name']} task"
                        "Skipping the task, Please add the small mode for the task to "
                        "run it in small mode"
                    )
                    continue

                task_config.update(dict(task_config["modes"][task_mode]))
                task_config["tmp_dir"] = tmp_dir
                task_config["mode"] = task_mode
                del task_config["modes"]

                # The `splits` key has to be initialised outside the pipeline,
                # since the splits are used in defining the required tasks
                # for example, in requires method for ResampleSubcorpuses
                if "split_mode" not in task_config:
                    raise ValueError("split_mode is a required config for all tasks")

                if task_config["split_mode"] == "trainvaltest":
                    # Dataset will be partitioned into train/validation/test splits
                    task_config["splits"] = pipeline.SPLITS
                elif task_config["split_mode"] in ["new_split_kfold", "presplit_kfold"]:
                    # Dataset will be partitioned in k-folds, either using
                    # predefined folds or with using folds defined in the pipeline
                    n_folds = task_config["nfolds"]
                    assert isinstance(n_folds, int)
                    task_config["splits"] = [
                        "fold{:02d}".format(i) for i in range(n_folds)
                    ]
                else:
                    raise ValueError(
                        f"Unknown split_mode received: {task_config['split_mode']}, "
                        "expected 'trainvaltest, 'new_split_kfold', or 'presplit_kfold'"
                    )

                metadata_task = task_module.extract_metadata_task(task_config)  # type: ignore # noqa: E501
                final_task = pipeline.FinalizeCorpus(
                    sample_rates=sample_rates,
                    tasks_dir=tasks_dir,
                    tar_dir=tar_dir,
                    metadata_task=metadata_task,
                    task_config=task_config,
                )
                tasks_to_run.append(final_task)

    pipeline.run(
        tasks_to_run,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    run()
