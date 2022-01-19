#!/usr/bin/env python3
"""
Pre-processing pipeline for FSD50K Dataset
"""
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Set
from urllib.parse import urlparse

import luigi
import pandas as pd
from slugify import slugify

import hearpreprocess.pipeline as pipeline
from hearpreprocess.pipeline import (
    DownloadCorpus,
    ExtractArchive,
)
from hearpreprocess.util.luigi import WorkTask, download_file

logger = logging.getLogger("luigi-interface")

generic_task_config = {
    "task_name": "fsd50k",
    "version": "v1.0",
    "embedding_type": "scene",
    # One file can have multiple labels
    "prediction_type": "multilabel",
    "split_mode": "trainvaltest",
    # Paper Link - https://arxiv.org/pdf/2010.00475.pdf
    # Page 12 Discussion B Section 1 -
    # The audio duration varies from 0.3 seconds to 30 seconds
    # The labels are weak labels and event can be anywhere in the
    "sample_duration": None,  # in seconds
    # Paper link - https://arxiv.org/pdf/2010.00475.pdf
    # Metrics used: mAP, d_prime, lwrap for evaluation [Page 15]
    # Metrics used: pr_auc for training [Page 16]
    "evaluation": ["mAP", "d_prime", "aucroc", "top1_acc"],
    "download_urls": [
        {
            "split": "train",
            "multipart_zip_urls": [
                {
                    "url": "https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z01?download=1",  # noqa: E501
                    "md5": "faa7cf4cc076fc34a44a479a5ed862a3",
                },
                {
                    "url": "https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z02?download=1",  # noqa: E501
                    "md5": "8f9b66153e68571164fb1315d00bc7bc",
                },
                {
                    "url": "https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z03?download=1",  # noqa: E501
                    "md5": "1196ef47d267a993d30fa98af54b7159",
                },
                {
                    "url": "https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z04?download=1",  # noqa: E501
                    "md5": "d088ac4e11ba53daf9f7574c11cccac9",
                },
                {
                    "url": "https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z05?download=1",  # noqa: E501
                    "md5": "81356521aa159accd3c35de22da28c7f",
                },
                {
                    "url": "https://zenodo.org/record/4060432/files/FSD50K.dev_audio.zip?download=1",  # noqa: E501
                    "md5": "c480d119b8f7a7e32fdb58f3ea4d6c5a",
                },
            ],
            "zipname": "FSD50K.dev_audio.zip",
        },
        {
            "split": "test",
            "multipart_zip_urls": [
                {
                    "url": "https://zenodo.org/record/4060432/files/FSD50K.eval_audio.z01?download=1",  # noqa: E501
                    "md5": "3090670eaeecc013ca1ff84fe4442aeb",
                },
                {
                    "url": "https://zenodo.org/record/4060432/files/FSD50K.eval_audio.zip?download=1",  # noqa: E501
                    "md5": "6fa47636c3a3ad5c7dfeba99f2637982",
                },
            ],
            "zipname": "FSD50K.eval_audio.zip",
        },
        {
            "name": "ground_truth",
            "url": "https://zenodo.org/record/4060432/files/FSD50K.ground_truth.zip?download=1",  # noqa: E501
            "md5": "ca27382c195e37d2269c4c866dd73485",
        },
        {
            "name": "source_metadata",
            "url": "https://zenodo.org/record/4060432/files/FSD50K.metadata.zip?download=1",  # noqa: E501
            "md5": "b9ea0c829a411c1d42adb9da539ed237",
        },
    ],
    "default_mode": "full",
    # Different modes for preprocessing this dataset
    "modes": {
        "full": {
            "max_task_duration_by_split": {"test": None, "train": None, "valid": None}
        },
        "small": {
            "download_urls": [
                {
                    "split": "train",
                    "url": "https://github.com/kmarufraj/s-task/raw/main/fsd50k-train-small.zip",  # noqa: E501
                    "md5": "75fc0ae63467ada263ead31afff0978a",
                },
                {
                    "split": "test",
                    "url": "https://github.com/kmarufraj/s-task/raw/main/fsd50k-test-small.zip",  # noqa: E501
                    "md5": "1b3c32e6183b6f991b32ad2ee793a2c5",
                },
                {
                    "split": "ground_truth",
                    "url": "https://github.com/kmarufraj/s-task/raw/main/fsd50k-ground_truth-small.zip",  # noqa: E501
                    "md5": "e730b34c0d71a4607338c8fea1b3c3d2",
                },
                {
                    "split": "source_metadata",
                    "url": "https://github.com/kmarufraj/s-task/raw/main/fsd50k-source_metadata-small.zip",  # noqa: E501
                    "md5": "df52ffa58ffd99b3d2ac151b36c7e71c",
                },
            ],
            "max_task_duration_by_split": {"test": None, "train": None, "valid": None},
            "sample_duration": None,
        },
    },
}


class DownloadMultipartCorpus(WorkTask):
    """
    Downloads multipart urls and verifies the corresponding md5s.
    Following this the multipart zip files are merged into one zip file
    Args:
        urls: List of multipart zip file urls
        expected_md5s: corresponding md5 for each url
        zipname:
            Format of the multipart zip file names are
            somename.z01, somename.z02, ... somename.zip -
            The last name is the zipname which is the actual zip file
            present as splits. This will be used to construct the
            command to combine the zip files.
        outfile: filename of the output unsplit zip file
    """

    urls = luigi.ListParameter()
    expected_md5s = luigi.ListParameter()
    zipname = luigi.Parameter()
    outfile = luigi.Parameter()

    def run(self):
        for url, expected_md5 in zip(self.urls, self.expected_md5s):
            local_filename: str = os.path.basename(urlparse(url).path)
            download_file(url, self.workdir.joinpath(local_filename), expected_md5)
        # Command to combine the zip files into one zip file
        command = (
            f"zip -s 0 {self.workdir.joinpath(self.zipname)} "
            f"--out {self.workdir.joinpath(self.outfile)}"
        )
        print("Unsplitting the multi part zip files. Please wait")
        print(f"Running command: {command}")
        ret = subprocess.call(command.split(" "), stdout=subprocess.DEVNULL)
        assert ret == 0
        print(f"Successfully combined the zip files to {self.outfile}")

        self.mark_complete()

    @property
    def stage_number(self) -> int:
        return 0


def get_download_and_extract_tasks(task_config: Dict):
    """
    Iterates over the dowload urls and builds download and extract
    tasks for them

    This is redefined in this task (fsd50k), and this can handle the
    multipart zip files as well.
    """

    tasks = {}
    outdirs: Set[str] = set()
    for urlobj in task_config["download_urls"]:
        if "multipart_zip_urls" in urlobj:
            assert "url" not in urlobj, "For multipart zip urls, the url field should "
            "not be provided. Rather all the url should be in the multipart_zip_urls"
            "along with the corresponding md5s."
            zipname = urlobj["zipname"]
            urls, md5s = zip(
                *[
                    (zip_url["url"], zip_url["md5"])
                    for zip_url in urlobj["multipart_zip_urls"]
                ]
            )
            filename = f"combined_{zipname}"
            download_task = DownloadMultipartCorpus(
                urls=urls,
                zipname=zipname,
                outfile=filename,
                expected_md5s=md5s,
                task_config=task_config,
            )
            outdir = urlobj["split"]
        else:
            url, md5 = urlobj["url"], urlobj["md5"]
            filename = os.path.basename(urlparse(url).path)
            download_task = DownloadCorpus(
                url=url, outfile=filename, expected_md5=md5, task_config=task_config
            )
            outdir = urlobj["name"]

        assert outdir is not None
        assert outdir not in outdirs, f"{outdir} in {outdirs}. If you are downloading "
        "multiple archives into one split, they should have different 'name's."

        outdirs.add(outdir)
        task = ExtractArchive(
            download=download_task,
            infile=filename,
            outdir=outdir,
            task_config=task_config,
        )
        tasks[slugify(outdir, separator="_")] = task

    return tasks


class ExtractMetadata(pipeline.ExtractMetadata):
    train = luigi.TaskParameter()
    test = luigi.TaskParameter()
    ground_truth = luigi.TaskParameter()
    source_metadata = luigi.TaskParameter()

    def requires(self):
        return {
            "train": self.train,
            "test": self.test,
            "ground_truth": self.ground_truth,
            "source_metadata": self.source_metadata,
        }

    requires_key_to_path_str = {
        "train": {
            "audio": "train/FSD50K.dev_audio",
            "ground_truth": "ground_truth/FSD50K.ground_truth/dev.csv",
            "source_meta": "source_metadata/FSD50K.metadata/dev_clips_info_FSD50K.json",
        },
        "test": {
            "audio": "test/FSD50K.eval_audio",
            "ground_truth": "ground_truth/FSD50K.ground_truth/eval.csv",
            "source_meta": "source_metadata/FSD50K.metadata/eval_clips_info_FSD50K.json",  # noqa: E501
        },
    }

    @staticmethod
    def get_split_key(df: pd.DataFrame) -> pd.Series:
        """The uploader information is use as split key for this dataset"""
        assert "uploader" in df, "The uploader information is not present"
        return df["uploader"]

    def get_requires_metadata(self, requires_key: str):
        logger.info(f"Preparing metadata for {requires_key}")

        requires_workdir: Path = self.requires()[requires_key].workdir
        paths: Dict[str, str] = self.requires_key_to_path_str[requires_key]

        audio_dir_path: Path = requires_workdir.joinpath(paths["audio"])
        label_path: Path = requires_workdir.joinpath(paths["ground_truth"])
        src_metadata_path: Path = requires_workdir.joinpath(paths["source_meta"])

        raw_label_df: pd.DataFrame = pd.read_csv(label_path)
        src_metadata_df: pd.DataFrame = pd.read_json(src_metadata_path).T

        # For test split the split column is not present
        if "split" not in raw_label_df:
            raw_label_df["split"] = requires_key

        # Merge the label df with the source metadata to get the uploader information.
        # This uploader will be used as the split key. Please check the
        # `get_split_key` function overrided in this class for clarity
        label_df = raw_label_df.merge(
            src_metadata_df[["uploader"]],
            left_on="fname",
            right_index=True,
            validate="1:1",
        )
        assert len(raw_label_df) == len(
            label_df
        ), "Merge with the uploader information did not create same number of rows"

        # These 12 train files have length >= 30sec (due to incorrect
        # metadata in Freesound), and thus should be removed (Fonseca, p.c.)
        if requires_key == "train":
            too_long_files = set(
                [
                    83299,
                    83298,
                    121426,
                    121351,
                    121472,
                    121471,
                    124796,
                    397150,
                    124797,
                    124800,
                    124834,
                    124858,
                ]
            )
            new_label_df = label_df[~label_df["fname"].isin(too_long_files)]
            assert len(new_label_df) + len(too_long_files) == len(label_df), (
                f"Had {len(label_df)} files before filtering {len(too_long_files)} "
                + f"too long files, but {len(new_label_df)} after"
            )
            label_df = new_label_df

        metadata: pd.DataFrame = (
            label_df.assign(
                relpath=lambda df: df["fname"].apply(
                    lambda fname: audio_dir_path.joinpath(f"{fname}.wav")
                ),
                label=lambda df: df["labels"].str.split(","),
                # map `val` to consistent key `valid`
                split=label_df["split"].replace({"val": "valid"}),
                # Add the `uploader` column in return so that the partition can be made
            ).filter(["relpath", "label", "split", "uploader"])
            # Explode the label to handle multilabel entry
            # Each row will correspond to one entry
            .explode("label")
        )

        return metadata

    def get_all_metadata(self) -> pd.DataFrame:
        metadata = pd.concat(
            [
                self.get_requires_metadata_check(requires_key)
                for requires_key in ["train", "test"]
            ]
        ).reset_index(drop=True)
        return metadata


def extract_metadata_task(task_config: Dict[str, Any]) -> pipeline.ExtractMetadata:
    # Build the dataset pipeline with the custom metadata configuration task

    if task_config["mode"] == "small":
        # Small mode uses a small version of the data which is not present as
        # multipart corpus
        download_tasks = pipeline.get_download_and_extract_tasks(task_config)
    else:
        download_tasks = get_download_and_extract_tasks(task_config)

    return ExtractMetadata(
        outfile="process_metadata.csv", task_config=task_config, **download_tasks
    )
