#!/usr/bin/env python3
"""
Pre-processing pipeline for Gunshot Triangulation
https://zenodo.org/record/3997406#.YaIaoPFBzt0
"""

import logging
import os
from typing import Any, Dict, List
from urllib.parse import urlparse

import librosa
import luigi
import numpy as np
import pandas as pd
import soundfile as sf
from numba import jit
from tqdm import tqdm

import hearpreprocess.pipeline as pipeline
import hearpreprocess.util.luigi as luigi_util

logger = logging.getLogger("luigi-interface")

generic_task_config: Dict[str, Any] = {
    "task_name": "gunshot_triangulation",
    # Zenodo has no version?
    "version": "v1.0",
    "embedding_type": "scene",
    "prediction_type": "multiclass",
    "split_mode": "new_split_kfold",
    # 7 folds, one corresponding to each firearmID, there are total 7 firearms
    "nfolds": 7,
    # 22 shots, each of 1.5 seconds is partitioned out from each audio
    "sample_duration": 1.5,
    "evaluation": ["top1_acc", "mAP", "d_prime", "aucroc"],
    "download_urls": [
        {
            "split": "train",
            "url": "https://zenodo.org/record/3997406/files/mic1raw.wav?download=1",  # noqa: E501
            "md5": "e2f4acad742278c973e9c190fee03b5f",
        },
        {
            "split": "train",
            "url": "https://zenodo.org/record/3997406/files/mic2raw16b.wav?download=1",  # noqa: E501
            "md5": "184d888f457f331f592eeb4ee5fa12e2",
        },
        {
            "split": "train",
            "url": "https://zenodo.org/record/3997406/files/mic3raw.wav?download=1",  # noqa: E501
            "md5": "db2ffa4fdf8b42b98c208670f4578b4b",
        },
        {
            "split": "train",
            "url": "https://zenodo.org/record/3997406/files/mic4raw.wav?download=1",  # noqa: E501
            "md5": "571da84eecf397bf0356a24400769ced",
        },
    ],
    "default_mode": "full",
    "max_task_duration_by_fold": None,
    "modes": {
        "full": {},
        "small": {
            "download_urls": [
                {
                    "split": "train",
                    "url": "https://github.com/kmarufraj/s-task/raw/main/gunshot_triangulation.zip",  # noqa: E501
                    "md5": "c24bfa3b60fc9a2f51169fdb9bc336fa",
                }
            ],
            "sample_duration": 1.5,
        },
    },
}


class GenerateTrainDataset(luigi_util.WorkTask):
    """
    Task to breakdown audio from each mic into 22 gun shots
    Total Mics - 4 [1, 2, 3, 4]
    Total Shots/Mic - 22 [0, 1, 2, ...]
    Total FirearmIDs - 7 (Each of the 22 shot has a corresponding FirearmID) [1, 2, ...]

    Shot to Firearm ID mapping from the table in readme -
    https://zenodo.org/record/3997406#.YaIaoPFBzt0

    Data correspoding to each Firearm ID is used to create one fold by
    setting n_folds as 7 and using the FirearmID as the split id

    Partitioning of the audio files in 22 shots is Based on matlab script at -
    https://zenodo.org/record/3997406/files/create_data.m?download=1
    """

    mic1raw = luigi.TaskParameter()
    mic2raw16b = luigi.TaskParameter()
    mic3raw = luigi.TaskParameter()
    mic4raw = luigi.TaskParameter()
    outdir = luigi.Parameter()

    @property
    def output_path(self):
        return self.workdir.joinpath(self.outdir)

    def requires(self):
        return {
            name: getattr(self, name)
            for name in ["mic1raw", "mic2raw16b", "mic3raw", "mic4raw"]
        }

    @staticmethod
    @jit
    def find_shots(audio, min_samples_between_shots, num_shots):
        """
        Returns a list of shot start times (sample indices) from an audio buffer.
        Identifies a shot onset as corresponding to a sample with
            - an abs value greater than 0.8 and
            - at least min_sample_between_shots in distance from the previous onset.
        This is the same method used in matlab file provided in the original dataset
        """
        in_shot = False
        shot_starts = []
        for i, sample in enumerate(audio):
            if not in_shot and np.abs(sample) >= 0.8 and len(shot_starts) < num_shots:
                shot_starts.append(i)
                in_shot = True

            if in_shot and i - shot_starts[-1] > min_samples_between_shots:
                in_shot = False

        return shot_starts

    def run(self):
        self.output_path.mkdir(parents=True, exist_ok=True)

        micoffsets: List[int] = [15764451, 13764742, 9334032, 6794611]
        recordings_names: List[str] = ["mic1raw", "mic2raw16b", "mic3raw", "mic4raw"]
        # For each mic, the first 4 identified shots corresponds to Firearm ID of 1,
        # followed by 3 shots for each consecutive id - 2, 3, 4, 5, 6, 7
        # https://zenodo.org/record/3997406#.YaIaoPFBzt0
        shot_firearm_ids: List[int] = [1] + [(shot) // 3 + 1 for shot in range(21)]
        sr = 48000

        recordings = [
            self.requires()[recording_name].workdir.joinpath(f"{recording_name}.wav")
            for recording_name in recordings_names
        ]

        audio = []
        for i, recording in enumerate(recordings):
            samples, _ = librosa.load(recording, sr=sr)
            # Clip beginning and align audio from each mic
            samples = samples[micoffsets[i] - sr :]
            audio.append(samples)

        # Find the shot start frames for the first audio
        shot_starts = self.find_shots(audio[0], 25000, 22)

        # Trim all audio clips based on the shot start frames from the first mic
        # since audio from each mic is already aligned
        for j, starts in enumerate(shot_starts):
            for i in tqdm(range(len(audio))):
                # Start the clip for the shot before 1/8 seconds of the
                # identified start time for the shot
                start_time = int(starts - sr / 8)
                # Each shot duration should be 1.5 secs
                shot = audio[i][start_time : start_time + sr + int(sr / 2)]
                mic_id: int = i + 1  # Mics are 1, 2, 3, 4
                shot_id: int = j
                firearm_id: int = shot_firearm_ids[shot_id]

                audio_file = self.output_path.joinpath(
                    f"mic{mic_id}_shot{shot_id}_fid{firearm_id}.wav"
                )
                sf.write(audio_file, shot, samplerate=sr)
        self.mark_complete()


def get_download_and_extract_tasks(task_config: Dict) -> Dict[str, luigi_util.WorkTask]:
    """
    Iterates over the dowload urls downloads them

    Redefined for this task as -
        * the files are provided as wavfiles and donot need any extraction.
        * the downloaded data is passed in GenerateTrainDataset to split the audio
            into desired training set
    """

    download_tasks = {}
    for urlobj in task_config["download_urls"]:
        url, md5 = urlobj["url"], urlobj["md5"]
        filename = os.path.basename(urlparse(url).path)
        task = pipeline.DownloadCorpus(
            url=url, outfile=filename, expected_md5=md5, task_config=task_config
        )
        download_tasks[filename.split(".")[0]] = task

    tasks = {
        "train": GenerateTrainDataset(
            **download_tasks, task_config=task_config, outdir="train"
        )
    }
    return tasks


class ExtractMetadata(pipeline.ExtractMetadata):
    train = luigi.TaskParameter()

    def requires(self):
        return {"train": self.train}

    @staticmethod
    def get_split_key(df: pd.DataFrame) -> pd.Series:
        """
        The firearm ID is used as the split key
        There are total 7 firearm IDs. With num_fold set to 7, audio corresponding
        to each firearm will get into one split
        """
        assert "firearm_id" in df, "The uploader information is not present"
        return df["firearm_id"]

    def get_requires_metadata(self, split: str) -> pd.DataFrame:
        logger.info(f"Preparing metadata for {split}")

        # Loads and prepares the metadata for a specific split
        split_path = self.requires()[split].workdir.joinpath(split)

        # Get all the audio files in the split folder
        audio_files = list(split_path.rglob("*.wav"))

        metadata = pd.DataFrame(audio_files, columns=["relpath"])
        metadata = metadata.assign(
            # <MIC_ID>_<SHOT_ID>_<FIREARM_ID>
            # Predicting the mic is the label
            label=metadata["relpath"].apply(
                lambda relpath: relpath.name.split(".")[0].split("_")[-3]
            ),
            firearm_id=metadata["relpath"].apply(
                lambda relpath: relpath.name.split(".")[0].split("_")[-1]
            ),
            split="train",
        )
        assert all(
            metadata["label"].isin(f"mic{id}" for id in range(1, 5))
        ), "Label is the micId and should be be one of mic1-mic4"
        assert all(
            metadata["firearm_id"].isin(f"fid{id}" for id in range(1, 8))
        ), "Firearm IDs should be from fid1-fid7"

        return metadata


def extract_metadata_task(task_config: Dict[str, Any]) -> pipeline.ExtractMetadata:
    # Build the dataset pipeline with the custom metadata configuration task
    if task_config["mode"] == "small":
        # If small mode, download the prepartitioned (into 22 shots) data
        download_tasks = pipeline.get_download_and_extract_tasks(task_config)
    else:
        download_tasks = get_download_and_extract_tasks(task_config)

    return ExtractMetadata(
        **download_tasks,
        outfile="process_metadata.csv",
        task_config=task_config,
    )
