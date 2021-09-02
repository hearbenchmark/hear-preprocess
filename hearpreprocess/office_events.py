#!/usr/bin/env python3
"""
Pre-processing pipeline for DCASE 2016 task 2 task (sound event
detection).

The HEAR 2021 variation of DCASE 2016 Task 2 is that we ignore the
monophonic training data. We mix the dev and eval data, and
re-partition ourselves, to reduce variance between the partitions.

We also allow training data outside this task.
"""

import logging
from pathlib import Path
from typing import List

import hearpreprocess.pipeline as pipeline
import luigi
import pandas as pd
from slugify import slugify

logger = logging.getLogger("luigi-interface")

task_config = {
    "task_name": "office_events",
    "version": "hear2021",
    "embedding_type": "event",
    "prediction_type": "multilabel",
    "sample_duration": 120.0,
    # DCASE2016 task 2 used the segment-based total error rate as
    # their main score and then the onset only event based F1 as
    # their secondary score.
    # However, we announced that onset F1 would be our primary score.
    "evaluation": ["event_onset_200ms_fms", "segment_1s_er"],
    "download_urls": [
        {
            "split": "train",
            "name": "dev",
            "url": "https://archive.org/download/dcase2016_task2_train_dev/dcase2016_task2_train_dev.zip",  # noqa: E501
            "md5": "4e1b5e8887159193e8624dec801eb9e7",
        },
        {
            "split": "train",
            "name": "eval",
            "url": "https://archive.org/download/dcase2016_task2_test_public/dcase2016_task2_test_public.zip",  # noqa: E501
            "md5": "ac98768b39a08fc0c6c2ddd15a981dd7",
        },
    ],
    "small": {
        "download_urls": [
            {
                "split": "train",
                "name": "dev",
                "url": "https://github.com/neuralaudio/hear2021-open-tasks-downsampled/raw/main/dcase2016_task2_train_dev-small.zip",  # noqa: E501
                "md5": "aa9b43c40e9d496163caab83becf972e",
            },
            {
                "split": "train",
                "name": "eval",
                "url": "https://github.com/neuralaudio/hear2021-open-tasks-downsampled/raw/main/dcase2016_task2_test_public-small.zip",  # noqa: E501
                "md5": "14539d85dec03cb7ac75eb62dd1dd21e",
            },
        ],
        "version": "hear2021-small",
    },
}


class ExtractMetadata(pipeline.ExtractMetadata):
    train_dev = luigi.TaskParameter()
    train_eval = luigi.TaskParameter()

    def requires(self):
        return {"train_eval": self.train_eval, "train_dev": self.train_dev}

    @staticmethod
    def slugify_file_name(relative_path: str):
        """
        Overriding slugify method for dcase files. Normally, slugify will remove
        extra hyphens(-) and this will create ambiguity for positive and negative
        decibles in dcase file names as they also have decible information in the
        file names (example 6 and -6 will be slugified to being 6, which is not
        expected)
        """
        slug_text = str(Path(relative_path).stem)
        # Replace all the '-' to '_negative_'
        slug_text = slug_text.replace("-", "_negative_")
        return f"{slugify(slug_text)}"

    """
    DCASE 2016 uses funny pathing, so we just hardcode the desired
    (paths)
    """
    requires_key_to_path_str = {
        "train_dev": "train/dev/dcase2016_task2_train_dev/dcase2016_task2_dev/",
        "train_eval": "train/eval/dcase2016_task2_test_public/",
    }

    def get_requires_metadata(self, requires_key: str) -> pd.DataFrame:
        logger.info(f"Preparing metadata for {requires_key}")

        assert requires_key.startswith("train_")

        requires_path = Path(self.requires()[requires_key].workdir).joinpath(
            self.requires_key_to_path_str[requires_key]
        )

        metadatas = []
        for annotation_file in requires_path.glob("annotation/*.txt"):
            metadata = pd.read_csv(
                annotation_file,
                sep="\t",
                header=None,
                names=["start", "end", "label"],
            )
            # Convert start and end times to milliseconds
            metadata["start"] *= 1000
            metadata["end"] *= 1000
            sound_file = (
                str(annotation_file)
                .replace("annotation", "sound")
                .replace(".txt", ".wav")
            )
            metadata = metadata.assign(
                relpath=sound_file,
                split=lambda df: "train",
            )

            metadatas.append(metadata)

        return pd.concat(metadatas).reset_index(drop=True)


def main(
    sample_rates: List[int],
    tmp_dir: str,
    tasks_dir: str,
    small: bool = False,
):
    if small:
        task_config.update(dict(task_config["small"]))  # type: ignore
    task_config.update({"tmp_dir": tmp_dir})

    # Build the dataset pipeline with the custom metadata configuration task
    download_tasks = pipeline.get_download_and_extract_tasks(task_config)

    extract_metadata = ExtractMetadata(
        outfile="process_metadata.csv", task_config=task_config, **download_tasks
    )
    final_task = pipeline.FinalizeCorpus(
        sample_rates=sample_rates,
        tasks_dir=tasks_dir,
        metadata_task=extract_metadata,
        task_config=task_config,
    )
    return final_task
