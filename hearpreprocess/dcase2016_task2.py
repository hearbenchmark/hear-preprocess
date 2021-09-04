#!/usr/bin/env python3
"""
Pre-processing pipeline for DCASE 2016 task 2 task (sound event
detection).

The HEAR 2021 variation of DCASE 2016 Task 2 is that we ignore the
monophonic training data and use the dev data for train.
We also allow training data outside this task.
"""

import logging
from pathlib import Path
from typing import List

import luigi
import pandas as pd

import hearpreprocess.pipeline as pipeline

logger = logging.getLogger("luigi-interface")

task_config = {
    "task_name": "dcase2016_task2",
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
            "url": "https://archive.org/download/dcase2016_task2_train_dev/dcase2016_task2_train_dev.zip",  # noqa: E501
            "md5": "4e1b5e8887159193e8624dec801eb9e7",
        },
        {
            "split": "test",
            "url": "https://archive.org/download/dcase2016_task2_test_public/dcase2016_task2_test_public.zip",  # noqa: E501
            "md5": "ac98768b39a08fc0c6c2ddd15a981dd7",
        },
    ],
    "small": {
        "download_urls": [
            {
                "split": "train",
                "url": "https://github.com/neuralaudio/hear2021-open-tasks-downsampled/raw/main/dcase2016_task2_train_dev-small.zip",  # noqa: E501
                "md5": "aa9b43c40e9d496163caab83becf972e",
            },
            {
                "split": "test",
                "url": "https://github.com/neuralaudio/hear2021-open-tasks-downsampled/raw/main/dcase2016_task2_test_public-small.zip",  # noqa: E501
                "md5": "14539d85dec03cb7ac75eb62dd1dd21e",
            },
        ],
        "version": "hear2021-small",
    },
}


class ExtractMetadata(pipeline.ExtractMetadata):
    train = luigi.TaskParameter()
    test = luigi.TaskParameter()

    def requires(self):
        return {"train": self.train, "test": self.test}

    """
    DCASE 2016 uses funny pathing, so we just hardcode the desired
    (paths)
    Note that for our training data, we only use DCASE 2016 dev data.
    Their training data is short monophonic events.
    """
    split_to_path_str = {
        "train": "dcase2016_task2_train_dev/dcase2016_task2_dev/",
        "test": "dcase2016_task2_test_public/",
    }

    def get_requires_metadata(self, split: str) -> pd.DataFrame:
        logger.info(f"Preparing metadata for {split}")

        split_path = (
            Path(self.requires()[split].workdir)
            .joinpath(split)
            .joinpath(self.split_to_path_str[split])
        )

        metadatas = []
        for annotation_file in split_path.glob("annotation/*.txt"):
            metadata = pd.read_csv(
                annotation_file, sep="\t", header=None, names=["start", "end", "label"]
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
                split=lambda df: split,
            )

            metadatas.append(metadata)

        return pd.concat(metadatas).reset_index(drop=True)

    def split_train_test_val(self, metadata: pd.DataFrame):
        """
        Because the training set is so small, manually partition it into a
        50/50 dev set that captures all hyperparameters in train and test.
        """

        splits_present = metadata["split"].unique()
        assert set(splits_present) == {"train", "test"}

        # Manually split into train + dev to try to balance the hyperparams
        train_stems = {
            "dev_1_ebr_-6_nec_1_poly_0",
            "dev_1_ebr_-6_nec_3_poly_0",
            "dev_1_ebr_-6_nec_4_poly_1",
            "dev_1_ebr_6_nec_2_poly_0",
            "dev_1_ebr_6_nec_3_poly_1",
            "dev_1_ebr_6_nec_5_poly_1",
            "dev_1_ebr_0_nec_2_poly_0",
            "dev_1_ebr_0_nec_3_poly_1",
            "dev_1_ebr_0_nec_4_poly_1",
        }

        valid_stems = {
            "dev_1_ebr_-6_nec_2_poly_0",
            "dev_1_ebr_-6_nec_3_poly_1",
            "dev_1_ebr_-6_nec_5_poly_1",
            "dev_1_ebr_6_nec_1_poly_0",
            "dev_1_ebr_6_nec_3_poly_0",
            "dev_1_ebr_6_nec_4_poly_1",
            "dev_1_ebr_0_nec_1_poly_0",
            "dev_1_ebr_0_nec_3_poly_0",
            "dev_1_ebr_0_nec_5_poly_1",
        }
        assert len(train_stems) + len(valid_stems) == len(train_stems | valid_stems)

        # Gross, let's never do this again
        if self.task_config["version"].split("-")[-1] == "small":
            assert train_stems | valid_stems >= set(
                metadata[metadata.split == "train"]["unique_filestem"].unique()
            )
        else:
            assert train_stems | valid_stems == set(
                metadata[metadata.split == "train"]["unique_filestem"].unique()
            )
        metadata.reset_index(drop=True, inplace=True)
        metadata.loc[metadata["unique_filestem"].isin(valid_stems), "split"] = "valid"
        if self.task_config["version"].split("-")[-1] == "small":
            assert train_stems >= set(
                metadata[metadata.split == "train"]["unique_filestem"].unique()
            )
            assert valid_stems >= set(
                metadata[metadata.split == "valid"]["unique_filestem"].unique()
            )
        else:
            assert train_stems == set(
                metadata[metadata.split == "train"]["unique_filestem"].unique()
            )
            assert valid_stems == set(
                metadata[metadata.split == "valid"]["unique_filestem"].unique()
            )
        return metadata


def main(
    sample_rates: List[int],
    tmp_dir: str,
    tasks_dir: str,
    tar_dir: str,
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
        tar_dir=tar_dir,
        metadata_task=extract_metadata,
        task_config=task_config,
    )
    return final_task
