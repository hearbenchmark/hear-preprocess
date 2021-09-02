"""
Common Luigi classes and functions for evaluation tasks
"""

import hashlib
import os
import os.path
from functools import partial
from pathlib import Path

import luigi
import numpy as np
import requests
from tqdm.auto import tqdm


class WorkTask(luigi.Task):
    """
    We assume following conventions:
        * Each luigi Task will have a name property
        * The "output" of each task is a touch'ed file,
        indicating that the task is done. Each .run()
        method should end with this command:
            `_workdir/{task_subdir}{task_id}.done`
            task_id unique identifies the task by a combination of name and
            input parameters
            * Optionally, working output of each task will go into:
            `_workdir/{task_subdir}{name}`

    Downstream dependencies should be cautious of automatically
    removing the working output, unless they are sure they are the
    only downstream dependency of a particular task (i.e. no
    triangular dependencies).
    """

    # Class attribute sets the task name for all inheriting luigi tasks
    task_config = luigi.DictParameter(
        visibility=luigi.parameter.ParameterVisibility.PRIVATE
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add versioned task name to the task id, so that each task is unique across
        # different datasets. This helps in tracking task in the pipeline's DAG.
        self.task_id = f"{self.task_id}_{self.versioned_task_name}"

    @property
    def name(self):
        return type(self).__name__

    def output(self):
        """
        Outfile to mark a task as complete.
        """
        output_name = f"{self.stage_number:02d}-{self.task_id}.done"
        output_file = self.task_subdir.joinpath(output_name)
        return luigi.LocalTarget(output_file)

    def mark_complete(self):
        """Touches the output file, marking this task as complete"""
        self.output().open("w").close()

    @property
    def workdir(self):
        """Working directory"""
        d = self.task_subdir.joinpath(f"{self.stage_number:02d}-{self.name}")
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def task_subdir(self):
        """Task specific subdirectory"""
        return Path(self.task_config.get("luigi_dir", "_workdir")).joinpath(
            str(self.versioned_task_name)
        )

    @property
    def versioned_task_name(self):
        """
        Versioned Task name contains the provided name in the
        data config and the version
        """
        return f"{self.task_config['task_name']}-{self.task_config['version']}"

    @property
    def stage_number(self):
        """
        Numerically sort the DAG tasks.
        This stage number will go into the name.

        This should be overridden as 0 by any task that has no
        requirements.
        """
        if isinstance(self.requires(), WorkTask):
            return 1 + self.requires().stage_number
        elif isinstance(self.requires(), list):
            return 1 + max([task.stage_number for task in self.requires()])
        elif isinstance(self.requires(), dict):
            parentasks = []
            for task in list(self.requires().values()):
                if isinstance(task, list):
                    parentasks.extend(task)
                else:
                    parentasks.append(task)
            return 1 + max([task.stage_number for task in parentasks])
        else:
            raise ValueError(f"Unknown requires: {self.requires()}")


def download_file(url, local_filename, expected_md5):
    """
    The downside of this approach versus `wget -c` is that this
    code does not resume.

    The benefit is that we are sure if the download completely
    successfuly, otherwise we should have an exception.
    From: https://stackoverflow.com/a/16696317/82733
    """
    # NOTE the stream=True parameter below
    with requests.get(url, stream=True, allow_redirects=True) as r:
        r.raise_for_status()
        total_length = r.headers.get("content-length")
        total_length = int(total_length)
        with open(local_filename, "wb") as f:
            pbar = tqdm(total=total_length)
            chunk_size = 8192
            for chunk in r.iter_content(chunk_size=chunk_size):
                # If you have chunk encoded response uncomment if
                # and set chunk_size parameter to None.
                f.write(chunk)
                pbar.update(chunk_size)
            pbar.close()
    assert (
        md5sum(local_filename) == expected_md5
    ), f"md5sum for url: {url} is: {md5sum(local_filename)}"
    "It should be {expected_md5}"
    return local_filename


def new_basedir(filename, basedir):
    """
    Rewrite .../filename as basedir/filename
    """
    return os.path.join(basedir, os.path.split(filename)[1])


def md5sum(filename):
    """
    NOTE: Identical hash value as running md5sum from the command-line.
    """
    with open(filename, mode="rb") as f:
        with tqdm(total=os.path.getsize(filename)) as pbar:
            d = hashlib.md5()
            for buf in iter(partial(f.read, 32768), b""):
                d.update(buf)
                pbar.update(32768)
    return d.hexdigest()
