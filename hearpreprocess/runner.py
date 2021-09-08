#!/usr/bin/env python3
"""
Runs a luigi pipeline to build a dataset
"""

import copy
import logging
import multiprocessing
from typing import Optional

import click

import hearpreprocess.dcase2016_task2 as dcase2016_task2
import hearpreprocess.nsynth_pitch as nsynth_pitch
import hearpreprocess.pipeline as pipeline
import hearpreprocess.speech_commands as speech_commands

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
    "speech_commands": [speech_commands],
    "nsynth_pitch": [nsynth_pitch],
    "dcase2016_task2": [dcase2016_task2],
    "all": [speech_commands, nsynth_pitch, dcase2016_task2]
    + secret_tasks.get("all-secret", []),
    # Add the task config for the secrets task if the secret task config was found.
    # Not available for participants
    **secret_tasks,
}


@click.command()
@click.argument("task")
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
    task: str,
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
    for task_module in tasks[task]:
        if mode == "default":
            modes = [task_module.generic_task_config["default_mode"]]
        elif mode == "small":
            modes = ["small"]
        elif mode == "all":
            modes = [
                mode
                for mode in task_module.generic_task_config["modes"].keys()
                if mode != "small"
            ]
            assert modes is not [], f"Task {task} has no modes besides 'small'"
        else:
            raise ValueError, f"mode {mode} unknown"
        for mode in modes:
            task_config = copy.deepcopy(task_module.generic_task_config)
            task_config.update(dict(task_config["modes"][mode]))
            task_config["tmp_dir"] = tmp_dir
            # Postpend the mode to the version number
            task_config["version"] = task_config["version"] + "-" + mode
            task_config["mode"] = mode
            del task_config["modes"]
            metadata_task = task_module.extract_metadata_task(task_config)
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
