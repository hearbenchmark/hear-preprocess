#!/usr/bin/env python3
"""
Pre-processing pipeline for Google speech_commands, using tensorflow-datasets
"""
import os
import re
from pathlib import Path
from typing import List

# https://github.com/tensorflow/datasets/issues/1441#issuecomment-581660890
import resource

low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
high = min(high, 10000)
resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

import luigi
import pandas as pd
import soundfile as sf
import tensorflow_datasets as tfds
from tqdm import tqdm

import hearpreprocess.pipeline as pipeline
import hearpreprocess.util.luigi as luigi_util

# WORDS = ["down", "go", "left", "no", "off", "on", "right", "stop", "up", "yes"]
# BACKGROUND_NOISE = "_background_noise_"
# UNKNOWN = "_unknown_"
# SILENCE = "_silence_"

task_config = {
    "task_name": "tf_speech_commands",
    "version": "v0.0.2",
    "embedding_type": "scene",
    "prediction_type": "multiclass",
    "sample_duration": 1.0,
    "evaluation": ["top1_acc"],
    # The test set is 1.33 hours, so we use the entire thing
    "max_task_duration_by_split": {"train": None, "valid": None, "test": None},
    "download_urls": [],
    "small": {
        "download_urls": [],
        "version": "v0.0.2-small",
    },
}

split_to_tf_split = {
    "train": "train",
    "valid": "validation",
    "test": "test",
}


class TensorflowDatasetLoad(luigi_util.WorkTask):
    """ """

    @property
    def stage_number(self) -> int:
        return 0

    @property
    def tfpath(self) -> Path:
        return self.workdir.joinpath("tensorflow-datasets")

    def run(self):
        for split in split_to_tf_split.values():
            # WARNING: This appear to say it shuffles???
            ds = tfds.load(
                "speech_commands",
                split=split,
                shuffle_files=False,
                as_supervised=False,
                data_dir=self.tfpath,
            )
        self.mark_complete()


class ExtractSplit(luigi_util.WorkTask):
    """ """

    split = luigi.Parameter()
    download: TensorflowDatasetLoad = luigi.TaskParameter()

    def requires(self):
        return {"download": self.download}

    @property
    def output_path(self):
        return self.workdir.joinpath(self.outdir)

    def run(self):
        ds = tfds.load(
            "speech_commands",
            split=split_to_tf_split[self.split],
            shuffle_files=False,
            as_supervised=False,
            data_dir=self.requires()["download"].tfpath,
        )
        import IPython

        ipshell = IPython.embed
        ipshell(banner1="ipshell")


#        self.mark_complete()


class ExtractMetadata(pipeline.ExtractMetadata):
    train = luigi.TaskParameter()
    test = luigi.TaskParameter()

    def requires(self):
        return {
            "train": self.train,
            "test": self.test,
        }

    @staticmethod
    def relpath_to_unique_filestem(relpath: str) -> str:
        """
        Include the label (parent directory) in the filestem.
        """
        # Get the parent directory (label) and the filename
        name = "_".join(Path(relpath).parts[-2:])
        # Remove the suffix
        name = os.path.splitext(name)[0]
        return str(name)

    @staticmethod
    def speaker_hash(unique_filestem: str) -> str:
        """Get the speaker hash as the Split key for speech_commands"""
        hsh = re.sub(r"_nohash_.*$", "", unique_filestem)
        return hsh

    @staticmethod
    def get_split_key(df: pd.DataFrame) -> pd.Series:
        """Get the speaker hash as the split key for speech_commands"""
        return df["unique_filestem"].apply(ExtractMetadata.speaker_hash)

    @staticmethod
    def relpath_to_label(relpath: Path):
        label = os.path.basename(os.path.dirname(relpath))
        if label not in WORDS and label != SILENCE:
            label = UNKNOWN
        return label

    def get_split_paths(self):
        """
        Splits the dataset into train/valid/test files using the same method as
        described in by the TensorFlow dataset:
        https://www.tensorflow.org/datasets/catalog/speech_commands
        """
        # Test files
        test_path = Path(self.requires()["test"].workdir).joinpath("test")
        test_df = pd.DataFrame(test_path.glob("*/*.wav"), columns=["relpath"]).assign(
            split=lambda df: "test"
        )

        # All silence paths to add to the train and validation
        train_path = Path(self.requires()["train"].workdir)
        all_silence = list(train_path.glob(f"{SILENCE}/*.wav"))

        # Validation files
        with open(os.path.join(train_path, "validation_list.txt"), "r") as fp:
            validation_paths = fp.read().strip().splitlines()
        validation_rel_paths = [os.path.join(train_path, p) for p in validation_paths]

        # There are no silence files marked explicitly for validation. We add all
        # the running_tap.wav samples to the silence class for validation.
        # https://github.com/tensorflow/datasets/blob/e24fe9e6b03053d9b925d299a2246ea167dc85cd/tensorflow_datasets/audio/speech_commands.py#L183
        val_silence = list(train_path.glob(f"{SILENCE}/running_tap*.wav"))
        validation_rel_paths.extend(val_silence)
        validation_df = pd.DataFrame(validation_rel_paths, columns=["relpath"]).assign(
            split=lambda df: "valid"
        )

        # Train-test files.
        with open(os.path.join(train_path, "testing_list.txt"), "r") as fp:
            train_test_paths = fp.read().strip().splitlines()
        audio_paths = [
            str(p.relative_to(train_path)) for p in train_path.glob("[!_]*/*.wav")
        ]

        # The final train set is all the audio files MINUS the files marked as
        # test / validation files in testing_list.txt or validation_list.txt
        train_paths = list(
            set(audio_paths) - set(train_test_paths) - set(validation_paths)
        )
        train_rel_paths = [os.path.join(train_path, p) for p in train_paths]

        # Training silence is all the generated silence / background noise samples
        # minus those marked for validation.
        train_silence = list(set(all_silence) - set(val_silence))
        train_rel_paths.extend(train_silence)
        train_df = pd.DataFrame(train_rel_paths, columns=["relpath"]).assign(
            split=lambda df: "train"
        )
        assert len(train_df.merge(validation_df, on="relpath")) == 0

        return pd.concat([test_df, validation_df, train_df]).reset_index(drop=True)

    def get_all_metadata(self) -> pd.DataFrame:
        metadata = self.get_split_paths()
        metadata = metadata.assign(
            label=lambda df: df["relpath"].apply(self.relpath_to_label),
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

    #    # Build the dataset pipeline with the custom metadata configuration task
    #    download_tasks = pipeline.get_download_and_extract_tasks(task_config)

    load = TensorflowDatasetLoad(
        task_config=task_config,
    )
    #    for split in
    return ExtractSplit(
        split="valid",
        download=load,
        task_config=task_config,
    )
    """
    extract_metadata = ExtractMetadata(
        train=generate,
        test=download_tasks["test"],
        outfile="process_metadata.csv",
        task_config=task_config,
    )

    final_task = pipeline.FinalizeCorpus(
        sample_rates=sample_rates,
        tasks_dir=tasks_dir,
        tar_dir=tar_dir,
        metadata_task=extract_metadata,
        task_config=task_config,
    )
    return final_task
    """
