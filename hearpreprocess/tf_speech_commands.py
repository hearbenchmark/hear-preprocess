#!/usr/bin/env python3
"""
Pre-processing pipeline for Google speech_commands, using tensorflow-datasets
"""
import os
import re
from pathlib import Path
from typing import List, Dict, Any

# # https://github.com/tensorflow/datasets/issues/1441#issuecomment-581660890
# import resource

# low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
# high = min(high, 10000)
# resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

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

    "default_mode": "tfds",
    "modes": {
        # This is a tfds mode which doesnot require the path but the tfds dataset name
        "tfds": {
            "tf_task_name": "speech_commands",
            "tf_task_version": "0.0.2",
            # By default all the splits will be downloaded, the below key
            # helps to select the splits to extract
            "extract_splits": ["train", "test", "valid"]
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

        builder = tfds.builder(name = tf_task_name, version = tf_task_version, data_dir=tfds_path)
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

    def run(self):
        #Get the tfds builder from the download task
        builder = self.requires()["download"].get_tfds_builder()
        #Map the split with the tensorflow version of the split name
        split = split_to_tf_split[self.split]

        #Get the dataset corresponding to the split
        dataset = builder.as_dataset(split=split, shuffle_files=False)
        info = builder._info()
        print("----------------", info.features)

        #With the info build the label to idx map
        label_idx_map = {
            label_idx: label
            for label_idx, label in enumerate(info.features["label"].names)
        }
        #Get the datset sample rate. This will be used while saving the audio
        dataset_sample_rate = info.features["audio"].sample_rate

        audio_dir = self.output_path.joinpath('audio')
        audio_dir.mkdir(exist_ok = True, parents = True)
        file_labels = []
        for file_idx, example in enumerate(tqdm(tfds.as_numpy(ds))):
            #The format was int64, so converted to int32 because soundfile required
            #format int32
            numpy_audio = example["audio"].astype('int32')
            #The label here is the index of the label. Get the corresponding label name
            label = label_idx_map[example["label"]]
            audio_path = audio_dir.joinpath(f"tf_data_idx_{file_idx}.wav")
            sf.write(audio_path, numpy_audio, ds_sample_rate)
            #Append the audio path and the corresponding label
            file_labels.append((audio_path, label))

        file_labels_df = pd.DataFrame(file_labels, columns = ["path", "label"])
        file_labels_path = self.output_path.joinpath(f"{split}_labels.csv")
        file_labels_df.to_csv(file_labels_path, index = False)


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
