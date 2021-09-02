#!/usr/bin/env python3
"""
Pre-processing pipeline for NSynth pitch detection
"""

import logging
from functools import partial
from pathlib import Path
from typing import List

import hearpreprocess.pipeline as pipeline
import luigi
import pandas as pd

logger = logging.getLogger("luigi-interface")


task_config = {
    "task_name": "nsynth_pitch",
    "version": "v2.2.3",
    "embedding_type": "scene",
    "prediction_type": "multiclass",
    "sample_duration": 4.0,
    "pitch_range_min": 21,
    "pitch_range_max": 108,
    "evaluation": ["pitch_acc", "chroma_acc"],
    "download_urls": [
        {
            "split": "train",
            "url": "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-train.jsonwav.tar.gz",  # noqa: E501
            "md5": "fde6665a93865503ba598b9fac388660",
        },
        {
            "split": "valid",
            "url": "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-valid.jsonwav.tar.gz",  # noqa: E501
            "md5": "87e94a00a19b6dbc99cf6d4c0c0cae87",
        },
        {
            "split": "test",
            "url": "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz",  # noqa: E501
            "md5": "5e6f8719bf7e16ad0a00d518b78af77d",
        },
    ],
    "small": {
        "download_urls": [
            {
                "split": "train",
                "url": "https://github.com/neuralaudio/hear2021-open-tasks-downsampled/raw/main/nsynth-train-small.zip",  # noqa: E501
                "md5": "c17070e4798655d8bea1231506479ba8",
            },
            {
                "split": "valid",
                "url": "https://github.com/neuralaudio/hear2021-open-tasks-downsampled/raw/main/nsynth-valid-small.zip",  # noqa: E501
                "md5": "e36722262497977f6b945bb06ab0969d",
            },
            {
                "split": "test",
                "url": "https://github.com/neuralaudio/hear2021-open-tasks-downsampled/raw/main/nsynth-test-small.zip",  # noqa: E501
                "md5": "9a98e869ed4add8ba9ebb0d7c22becca",
            },
        ],
        "version": "v2.2.3-small",
    },
}


class ExtractMetadata(pipeline.ExtractMetadata):
    train = luigi.TaskParameter()
    test = luigi.TaskParameter()
    valid = luigi.TaskParameter()

    def requires(self):
        return {
            "train": self.train,
            "test": self.test,
            "valid": self.valid,
        }

    @staticmethod
    def get_rel_path(root: Path, item: pd.DataFrame) -> Path:
        # Creates the relative path to an audio file given the note_str
        audio_path = root.joinpath("audio")
        filename = f"{item}.wav"
        return audio_path.joinpath(filename)

    @staticmethod
    def get_split_key(df: pd.DataFrame) -> pd.Series:
        """
        The instrument is the split key.
        """
        return df["slug"].apply(lambda slug: slug.split("-")[0])

    def get_requires_metadata(self, split: str) -> pd.DataFrame:
        logger.info(f"Preparing metadata for {split}")

        # Loads and prepares the metadata for a specific split
        split_path = Path(self.requires()[split].workdir).joinpath(split)
        split_path = split_path.joinpath(f"nsynth-{split}")

        metadata = pd.read_json(split_path.joinpath("examples.json"), orient="index")

        metadata = (
            # Filter out pitches that are not within the range
            metadata.loc[
                metadata["pitch"].between(
                    task_config["pitch_range_min"], task_config["pitch_range_max"]
                )
                # Assign metadata columns
            ].assign(
                relpath=lambda df: df["note_str"].apply(
                    partial(self.get_rel_path, split_path)
                ),
                label=lambda df: df["pitch"],
                split=lambda df: split,
            )
        )

        return metadata


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
