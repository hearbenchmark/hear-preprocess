#!/usr/bin/env python3
"""
Pre-processing pipeline for Vocal Imitation Dataset
https://zenodo.org/record/1340763
"""

import logging
from typing import Any, Dict

import luigi
import pandas as pd

import hearpreprocess.pipeline as pipeline

logger = logging.getLogger("luigi-interface")

generic_task_config: Dict[str, Any] = {
    "task_name": "vocal_imitation",
    # Zenodo URL identifies this as 1.1.3 twice (in the title and
    # filename) and as 2.0 once (in the sidebar)
    "version": "v1.1.3",
    "embedding_type": "scene",
    "prediction_type": "multiclass",
    "split_mode": "new_split_kfold",
    "nfolds": 3,
    # "mean": 6.01, "var": 13.31, "min": 0.09, "max": 21.74, "10th": 1.62,
    # "25th": 2.93, "50th": 5.46, "75th": 8.96, "90th": 10.4, "95th": 11.26
    "sample_duration": 11.26,
    "evaluation": ["mAP", "d_prime", "aucroc", "top1_acc"],
    "download_urls": [
        {
            "split": "train",
            "url": "https://zenodo.org/record/1340763/files/VocalImitationSet_v1.1.3.zip?download=1",  # noqa: E501
            "md5": "386e7b1487fe0800ade4916c344086bc",
        }
    ],
    # Total duration: 11.26 * 5601 = 63067 s = 1051 mins = 17.52 hrs
    "default_mode": "full",
    "modes": {
        "full": {
            "max_task_duration_by_fold": None,
        },
        "small": {
            "download_urls": [
                {
                    "split": "train",
                    "url": "https://github.com/kmarufraj/s-task/raw/main/vocal_imitation-train-small.zip",  # noqa: E501
                    "md5": "08283695cafda4ba42a7f42f75f568d1",
                }
            ],
            "max_task_duration_by_fold": None,
            "sample_duration": 2,
        },
    },
}


class ExtractMetadata(pipeline.ExtractMetadata):
    train = luigi.TaskParameter()

    def requires(self):
        return {"train": self.train}

    def get_requires_metadata(self, split: str) -> pd.DataFrame:
        logger.info(f"Preparing metadata for {split}")

        # Loads and prepares the metadata for a specific split
        requires_path = self.requires()[split].workdir
        split_path = requires_path.joinpath(
            split, "VocalImitationSet", "vocal_imitations", "included"
        )

        source_metadata_path = requires_path.joinpath(
            split, "VocalImitationSet", "vocal_imitations.txt"
        )
        source_metadata = pd.read_csv(source_metadata_path, sep="\t")

        # Select audio which are in the included set
        source_metadata["included"] = source_metadata["included"].astype(bool)
        source_metadata = source_metadata.loc[source_metadata["included"]]

        audio_files = list(split_path.rglob("*.wav"))
        # Participant ID is not used as a split key, as it makes the folds unstable
        # due to its inconsistent distribution across the dataset
        metadata = (
            pd.DataFrame(audio_files, columns=["relpath"])
            .assign(
                filename=lambda df: df["relpath"].apply(lambda relpath: relpath.name)
            )
            .merge(
                source_metadata,
                left_on="filename",
                right_on="imitation_filename",
                how="inner",
                validate="1:1",
            )
            .assign(
                # The reference filename is used as the label
                # Given the imitation, the reference file will be predicted
                label=lambda df: df["reference_filename"],
                split="train",
            )
        )[["relpath", "label", "split"]]

        assert len(audio_files) == len(
            metadata
        ), "Some audio files donot have the corresponding metadata"

        return metadata


def extract_metadata_task(task_config: Dict[str, Any]) -> pipeline.ExtractMetadata:
    # Build the dataset pipeline with the custom metadata configuration task
    download_tasks = pipeline.get_download_and_extract_tasks(task_config)

    return ExtractMetadata(
        outfile="process_metadata.csv", task_config=task_config, **download_tasks
    )
