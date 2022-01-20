#!/usr/bin/env python3
"""
Custom Preprocessing pipeline for tensorflow dataset

Tfds audio datasets can be preprocessed with the hear-preprocess pipeline by defining
the generic_task_config dict and optionally overriding the extract metadata in this
file
See example tfds_speech_commands.py for a sample way to configure this for a tfds data

Tasks in this file helps to download and extract the tfds as wav files, followed
by overriding the extract metadata function to consume the extracted audio files and
labels. This is connected to downstream tasks from the main pipeline.

"""
import logging
from pathlib import Path
from typing import Any, Dict

import luigi
import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf
import tensorflow_datasets as tfds
from slugify import slugify
from tqdm import tqdm

import hearpreprocess.pipeline as pipeline
import hearpreprocess.util.luigi as luigi_util

logger = logging.getLogger("luigi-interface")

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
        """
        Returns the tfds builder which can be used to download and prepare the
        data (in `DownloadTFDS`)
        The tfds builder will also be used to load data for a split (in `ExtractTFDS`)
        """
        tf_task_name = self.task_config["tfds_task_name"]
        tf_task_version = self.task_config["tfds_task_version"]
        tfds_path = self.workdir.joinpath("tensorflow-datasets")

        # Define the builder
        builder = tfds.builder(
            name=tf_task_name, version=tf_task_version, data_dir=tfds_path
        )
        return builder

    def run(self):
        builder = self.get_tfds_builder()
        # Download and prepare the data in the task folder
        builder.download_and_prepare()
        self.mark_complete()


class ExtractTFDS(luigi_util.WorkTask):
    """
    Extracts the downloaded tfds dataset for a split
    If a split is not present, data for that split will
    be deterministically sampled from the train split, by the downstream
    pipeline (Specifically `ExtractMetadata.split_train_test_val`)
    """

    outdir = luigi.Parameter()
    split = luigi.Parameter()
    download: DownloadTFDS = luigi.TaskParameter()

    def requires(self):
        return {"download": self.download}

    @property
    def output_path(self):
        return self.workdir.joinpath(self.outdir)

    @staticmethod
    def load_tfds(builder, **as_dataset_kwargs) -> tf.data.Dataset:
        """
        This loads the dataset from the builder. Specifically this function returns
        a dataset which will also contain the tfds_id to uniquely determine
        each example in the dataset

        https://github.com/tensorflow/datasets/blob/master/docs/determinism.ipynb
        """
        read_config = as_dataset_kwargs.pop("read_config", tfds.ReadConfig())
        read_config.add_tfds_id = True  # Set `True` to return the 'tfds_id' key
        return builder.as_dataset(read_config=read_config, **as_dataset_kwargs)

    @staticmethod
    def save_audio_labels(
        dataset: tf.data.Dataset,
        audio_dir: Path,
        dataset_sample_rate: int,
        audio_labels_df_path: Path,
        label_idx_map: Dict[int, Any],
    ):
        """
        Iterates over the tfds dataset and saves the audio as wavfile and
        the audio labels as dataframe keyed on the tfds_id of the file.
        The sample rate of the audio is required to save the numpy array
        as wavfiles
        """
        filename_labels: Dict[str, Any] = {}
        # Track label indices for all the audio files
        all_label_idx: set = set()
        for example in tqdm(tfds.as_numpy(dataset)):
            # If the audio returned by TFDS is an integer type, then convert it to
            # an int16 type. TFDS returns 16-bit audio in int64, and for these to be
            # saved correctly by soundfile they need to be cast as int16.
            # The amplitude will be incorrect otherwise.
            numpy_audio = example["audio"]
            if np.issubdtype(numpy_audio.dtype, np.integer):
                assert np.max(numpy_audio.abs()) <= 32767, (
                    f"Was a expecting 16bit audio but the max value found "
                    f"exceeds the range for int16."
                )
                numpy_audio = numpy_audio.astype("int16")

            # Formats that work for writing in
            assert numpy_audio.dtype in [np.float32, np.float64, np.int16], (
                f"The audio's numpy array datatype: {numpy_audio.dtype} cannot be "
                "saved with soundfile"
            )

            # Since the audio name is not available in tfds, the unique tfds_id
            # is used to define the audio filename
            # https://www.tensorflow.org/datasets/determinism
            tfds_id = example["tfds_id"]
            audio_filename = f"tfds_id_{slugify(tfds_id)}.wav"
            audio_filepath = audio_dir.joinpath(audio_filename)
            assert audio_filename not in filename_labels, "Filenames are not unique "
            "for each example in the tfds data"
            sf.write(audio_filepath, numpy_audio, dataset_sample_rate)
            assert audio_filepath.exists(), "Audio file was not saved"

            # The label for an audio in tfds is the index of the actual label.
            # The mapping of this label idx to label is taken from the build info
            # and passed into this function
            label_idx: int = int(example["label"])
            all_label_idx.add(label_idx)
            label = label_idx_map[label_idx]
            filename_labels[audio_filename] = label

        # Since labels from tfds are supposed to be indices they should be continuous
        # integers. eg [0, 1, 2, ...]
        assert all_label_idx == set(range(len(all_label_idx))), (
            "TFDS labels for audio are not conitnuous integers. "
            "All Label indices: {all_label_idx}"
        )

        # Save the audio filename and the corresponding label as
        # a dataframe in the split folder
        filename_labels_df = pd.DataFrame(
            filename_labels.items(), columns=["filename", "label"]
        )
        filename_labels_df.to_csv(audio_labels_df_path, index=False)

    def run(self):
        # Get the tfds builder from the download task. Builder also provides info
        # about the label to idx map and the dataset sample rate
        # The dataset sample rate will be used to save the audio file
        builder = self.requires()["download"].get_tfds_builder()
        label_idx_map = {
            label_idx: label
            for label_idx, label in enumerate(builder.info.features["label"].names)
        }
        dataset_sample_rate = builder.info.features["audio"].sample_rate

        # Map the split with the tensorflow version of the split name
        split = split_to_tf_split[self.split]
        # Get the dataset for the split
        dataset: tf.data.Dataset = self.load_tfds(
            builder, split=split, shuffle_files=False
        )
        assert isinstance(dataset, tf.data.Dataset)
        # dataset = dataset.take(300)  # Remove me. Only for testing

        audio_dir = self.output_path.joinpath("audio")
        audio_dir.mkdir(exist_ok=True, parents=True)
        audio_labels_df_path = self.output_path.joinpath(f"{self.split}_labels.csv")
        self.save_audio_labels(
            dataset, audio_dir, dataset_sample_rate, audio_labels_df_path, label_idx_map
        )

        self.mark_complete()


def get_download_and_extract_tasks_tfds(
    task_config: Dict,
) -> Dict[str, luigi_util.WorkTask]:
    """Gets all the download and extract tasks for tensorflow dataset"""
    tasks = {}
    for split in task_config["extract_splits"]:
        outdir = split
        task = ExtractTFDS(
            download=DownloadTFDS(task_config=task_config),
            outdir=outdir,
            split=split,
            task_config=task_config,
        )
        tasks[outdir] = task

    return tasks


class ExtractMetadata(pipeline.ExtractMetadata):
    """
    All the splits are present in the tfds data set by default.
    If not, please override this `ExtractMetadata`, rather than using it
    as it is to extract metadata for the splits present in the data set.
    In this case, the not found splits will be automatically sampled from the
    train set in the `ExtractMetadata.split_train_test_val`.
    """

    train = luigi.TaskParameter()
    test = luigi.TaskParameter()
    valid = luigi.TaskParameter()

    def requires(self):
        return {"train": self.train, "test": self.test, "valid": self.valid}

    def get_requires_metadata(self, split: str) -> pd.DataFrame:
        logger.info(f"Preparing metadata for {split}")

        split_path = self.requires()[split].workdir.joinpath(split)
        # The directory where all the audio for the split was saved after extracting
        # from tfds
        audio_dir = split_path.joinpath("audio")
        # The metadata helps in getting the label associated with the audio samples.
        # This was also saved while extracting the audio from the tfds in ExtractTFDS
        metadata = pd.read_csv(split_path.joinpath(f"{split}_labels.csv"))

        metadata = metadata.assign(
            relpath=lambda df: df["filename"].apply(
                lambda filename: audio_dir.joinpath(filename)
            ),
            label=lambda df: df["label"],
            split=split,
        )

        return metadata
