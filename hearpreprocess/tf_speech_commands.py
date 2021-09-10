#!/usr/bin/env python3
"""
Pre-processing pipeline for Google speech_commands, using tensorflow-datasets
"""
import logging
import os
import re
from pathlib import Path
from typing import List, Dict, Any

import luigi
import pandas as pd
import soundfile as sf
import tensorflow as tf
import tensorflow_datasets as tfds
from slugify import slugify
from tqdm import tqdm

import hearpreprocess.pipeline as pipeline
import hearpreprocess.util.luigi as luigi_util

logger = logging.getLogger("luigi-interface")

generic_task_config = {
    "task_name": "tf_speech_commands",
    "version": "v0.0.2",
    "embedding_type": "scene",
    "prediction_type": "multiclass",
    "sample_duration": 1.0,
    "evaluation": ["top1_acc"],
    # The test set is 1.33 hours, so we use the entire thing
    "max_task_duration_by_split": {"train": None, "valid": None, "test": None},
    "default_mode": "tfds",
    "modes": {
        # This is tfds mode which doesn't require the path but the tfds dataset name
        # and version. For speech commands the tf dataset has all the splits, and so
        # we will not be doing any custom splitting. All the splits will be extracted
        # from tfds builder.
        "tfds": {
            "tf_task_name": "speech_commands",
            "tf_task_version": "0.0.2",
            # By default all the splits will be downloaded, the below key
            # helps to select the splits to extract
            "extract_splits": ["train", "test", "valid"],
        }
    },
}

split_to_tf_split = {
    "train": "train",
    "valid": "validation",
    "test": "test",
}


class DownloadTFDS(luigi_util.WorkTask):
    "Download and build the tensorflow dataset"

    @property
    def stage_number(self) -> int:
        return 0

    def get_tfds_builder(self):
        tf_task_name = self.task_config["tf_task_name"]
        tf_task_version = self.task_config["tf_task_version"]
        tfds_path = self.workdir.joinpath("tensorflow-datasets")

        builder = tfds.builder(
            name=tf_task_name, version=tf_task_version, data_dir=tfds_path
        )
        return builder

    def run(self):
        builder = self.get_tfds_builder()
        builder.download_and_prepare()
        self.mark_complete()


class ExtractTFDS(luigi_util.WorkTask):
    """Extracts the downloaded tfds dataset for the split"""

    outdir = luigi.Parameter()
    split = luigi.Parameter()
    download: DownloadTFDS = luigi.TaskParameter()

    def requires(self):
        return {"download": self.download}

    @property
    def output_path(self):
        return self.workdir.joinpath(self.outdir)

    @staticmethod
    def load_tfds(builder , **as_dataset_kwargs) -> tf.data.Dataset:
        """
        This loads the dataset from the builder. Specifically this function returns
        a dataset which will also contain the tfds_id which uniquely determines 
        each example in the dataset

        https://github.com/tensorflow/datasets/blob/master/docs/determinism.ipynb
        """
        read_config = as_dataset_kwargs.pop("read_config", tfds.ReadConfig())
        read_config.add_tfds_id = True  # Set `True` to return the 'tfds_id' key
        return builder.as_dataset(read_config=read_config, **as_dataset_kwargs)

    def run(self):
        # Get the tfds builder from the download task. From the builder info the 
        # label to idx map and the dataset sample can be extracted as well
        builder = self.requires()["download"].get_tfds_builder()
        label_idx_map = {
            label_idx: label
            for label_idx, label in enumerate(builder.info.features["label"].names)
        }
        # Get the datset sample rate. This will be used to save the audio
        dataset_sample_rate = builder.info.features["audio"].sample_rate

        # Map the split with the tensorflow version of the split name
        split = split_to_tf_split[self.split]
        # Get the dataset for the split
        dataset: tf.data.Dataset = self.load_tfds(builder, split=split, shuffle_files=False)
        dataset = dataset.take(2)
        assert isinstance(dataset, tf.data.Dataset)

        audio_dir = self.output_path.joinpath("audio")
        audio_dir.mkdir(exist_ok=True, parents=True)
        filename_labels = []
        for file_idx, example in enumerate(tqdm(tfds.as_numpy(dataset))):
            # The format was int64, so converted to int32 because soundfile required
            # format int32 to save the audio
            numpy_audio = example["audio"].astype("int32")
            tfds_id = example["tfds_id"]

            # The label in tfds is the index of the label. Get the corresponding
            # label name from the label_idx_map
            label = label_idx_map[example["label"]]

            # Since the audio name is not available, the audio is given a name
            # according to its index in the ds. Since shuffle is set to false,
            # the idx corresponding to one audio will not change
            audio_filename = f"tfds_id_{slugify(tfds_id)}.wav"

            sf.write(audio_dir.joinpath(audio_filename), numpy_audio, dataset_sample_rate)
            filename_labels.append((audio_filename, label))

        # Save the audio filename and the corresponding label in
        # a dataframe in the split folder
        file_labels_df = pd.DataFrame(filename_labels, columns=["filename", "label"])
        file_labels_path = self.output_path.joinpath(f"{split}_labels.csv")
        file_labels_df.to_csv(file_labels_path, index=False)

        self.mark_complete()


def get_download_and_extract_tasks_tfds(task_config: Dict) -> Dict[str, luigi_util.WorkTask]:
    tasks = {}
    outdirs: Set[str] = set()
    for split in task_config["extract_splits"]:
        outdir = split
        task = ExtractTFDS(
            download=DownloadTFDS(task_config=task_config),
            outdir=outdir,
            split = split,
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
    download_tasks = get_download_and_extract_tasks_tfds(task_config)

    return ExtractMetadata(
        **download_tasks,
        outfile="process_metadata.csv",
        task_config=task_config,
    )


if __name__ == "__main__":
    task_mode = "tfds"
    task_config['mode'] = task_mode
    task_config.update(dict(task_config["modes"][task_mode]))
    download_tasks = get_download_and_extract_tasks_tfds(task_config)
    luigi.build(
        [
            download_tasks["valid"]
        ],
        workers=7,
        local_scheduler=True,
        log_level="INFO",
    )
