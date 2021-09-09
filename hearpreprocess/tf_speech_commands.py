#!/usr/bin/env python3
"""
Pre-processing pipeline for Google speech_commands, using tensorflow-datasets
"""
import os
import re
from pathlib import Path
from typing import List, Dict, Any

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

task_config = {
    "task_name": "tf_speech_commands",
    "version": "v0.0.2",
    "embedding_type": "scene",
    "prediction_type": "multiclass",
    "sample_duration": 1.0,
    "evaluation": ["top1_acc"],
    # The test set is 1.33 hours, so we use the entire thing
    "max_task_duration_by_split": {"train": None, "valid": None, "test": None},
    "mode": "full",
    "modes": {
        # Only full mode is supported for tfds now
        "full": {
            # By default all the splits will be downloaded, the below key
            # helps to select the splits to extract
            "extract_splits": ["train", "test", "valid"]
        }
    },
}


class DownloadTFDS(luigi_util.WorkTask):
    "Downloads the the tensorflow dataset and prepairs the split"

    @property
    def stage_number(self) -> int:
        return 0

    def get_tfds_builder(self):
        tfdspath = self.workdir.joinpath("tensorflow-datasets")
        builder = tfds.builder("speech_commands", data_dir=tfdspath)
        return builder

    def run(self):
        builder = self.get_tfds_builder()
        builder.download_and_prepare()
        self.mark_complete()


split_to_tf_split = {
    "train": "train",
    "valid": "validation",
    "test": "test",
}


class ExtractTFDS(luigi_util.WorkTask):

    outdir = luigi.Parameter()
    download: DownloadTFDS = luigi.TaskParameter()

    def requires(self):
        return {"download": self.download}

    @property
    def output_path(self):
        return self.workdir.joinpath(self.outdir)

    def run(self):
        builder = self.requires()["download"].get_tfds_builder()
        split = split_to_tf_split["valid"]
        ds = builder.as_dataset(split="test", shuffle_files=False)
        ds.take(5)
        info = builder._info()

        label_idx_map = {
            label_idx: label
            for label_idx, label in enumerate(info.features["label"].names)
        }
        ds_sample_rate = info.features["audio"].sample_rate

        audio_dir = self.output_path.joinpath('audio')
        audio_dir.mkdir(exist_ok = True, parents = True)
        file_labels = []
        for file_idx, example in enumerate(tqdm(tfds.as_numpy(ds))):
            numpy_audio = example["audio"].astype('int32')
            label = label_idx_map[example["label"]]
            audio_path = audio_dir.joinpath(f"tf_data_idx_{file_idx}.wav")
            sf.write(audio_path, numpy_audio, ds_sample_rate)
            file_labels.append((audio_path, label))

        file_labels_df = pd.DataFrame(file_labels, columns = ["path", "label"])
        file_labels_path = self.output_path.joinpath(f"{split}_labels.csv")
        file_labels_df.to_csv(file_labels_path, index = False)


def get_download_and_extract_tasks(task_config: Dict) -> Dict[str, luigi_util.WorkTask]:
    tasks = {}
    outdirs: Set[str] = set()
    for split in task_config["extract_splits"]:
        outdir = split
        task = ExtractTFDS(
            download=DownloadCorpus(task_config=task_config),
            outdir=outdir,
            task_config=task_config,
        )
        tasks[outdir] = task

    return tasks


class ExtractMetadata(pipeline.ExtractMetadata):
    train = luigi.TaskParameter()
    test = luigi.TaskParameter()
    valid = luigi.TaskParameter()

    def requires(self):
        return {"train": self.train, "test": self.test, "valid": self.valid}

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
                    self.task_config["pitch_range_min"],
                    self.task_config["pitch_range_max"],
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


def extract_metadata_task(task_config: Dict[str, Any]) -> pipeline.ExtractMetadata:
    # Build the dataset pipeline with the custom metadata configuration task
    download_tasks = get_download_and_extract_tasks(task_config)

    return ExtractMetadata(
        **download_tasks,
        outfile="process_metadata.csv",
        task_config=task_config,
    )


if __name__ == "__main__":
    luigi.build(
        [
            ExtractTFDS(
                download=DownloadTFDS(task_config=task_config), task_config=task_config, outdir = "test"
            )
        ],
        workers=5,
        local_scheduler=True,
        log_level="INFO",
    )
