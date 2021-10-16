"""
Generic pipelines for datasets
"""

import json
import os
import random
import shutil
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Union
from urllib.parse import urlparse

import luigi
import pandas as pd
from slugify import slugify
from tqdm import tqdm

import hearpreprocess.util.audio as audio_util
from hearpreprocess import __version__
from hearpreprocess.util.luigi import (
    WorkTask,
    diagnostics,
    download_file,
    new_basedir,
    str2int,
)

INCLUDE_DATESTR_IN_FINAL_PATHS = False

SPLITS = ["train", "valid", "test"]
# This percentage should not be changed as this decides
# the data in the split and hence is not a part of the data config
# We use a 80/10/10 split.
VALIDATION_PERCENTAGE = 10
TEST_PERCENTAGE = 10
TRAIN_PERCENTAGE = 100 - VALIDATION_PERCENTAGE - TEST_PERCENTAGE
TRAINVAL_PERCENTAGE = TRAIN_PERCENTAGE + VALIDATION_PERCENTAGE

# We want no more than 5 hours of audio (training + validation) per task.
# This can be overriden in the task config.
# e.g. speech_commands test set.
# If None, no limit is used.
MAX_TASK_DURATION_BY_SPLIT = {
    "train": 3600 * 5 * TRAIN_PERCENTAGE / TRAINVAL_PERCENTAGE,
    "valid": 3600 * 5 * VALIDATION_PERCENTAGE / TRAINVAL_PERCENTAGE,
    "test": 3600 * 5 * TEST_PERCENTAGE / TRAINVAL_PERCENTAGE,
}


def _diagnose_split_labels(taskname: str, event_str: str, df: pd.DataFrame):
    """Makes split and label diagnostics"""
    assert "split" in df.columns
    assert "label" in df.columns

    split_file_count = df.groupby("split")["relpath"].nunique().to_dict()
    # Get fraction of rows with a particular label for each split
    split_label_frac = {
        split: split_df["label"].value_counts(normalize=True)
        for split, split_df in df.groupby("split")
    }
    # Get frequency of a particular label for each split
    split_label_freq = {
        split: list((round(split_df, 3)).to_dict().items())
        for split, split_df in split_label_frac.items()
    }
    # Get labels which are missing for a particular split
    split_label_missing = {
        split: set(df["label"].unique()) - set(labels)
        for split, labels in df.groupby("split")["label"].apply(set).to_dict().items()
    }
    for split in SPLITS:
        if split in split_file_count:
            diagnostics.info(
                "{} {} file count {:5s}: {}".format(
                    taskname, event_str, split, split_file_count[split]
                )
            )
    for split in SPLITS:
        if split in split_label_freq:
            diagnostics.info(
                "{} {} label freq (descending) {:5s}: {}".format(
                    taskname, event_str, split, split_label_freq[split]
                )
            )
    for split in SPLITS:
        if split in split_label_freq:
            diagnostics.info(
                "{} {} label freq (alphabetical) {:5s}: {}".format(
                    taskname, event_str, split, sorted(split_label_freq[split])
                )
            )
    for split in SPLITS:
        if split in split_label_missing and split_label_missing[split]:
            diagnostics.info(
                "{} {} MISSING LABELS {:5s}: {}".format(
                    taskname, event_str, split, split_label_missing[split]
                )
            )


class DownloadCorpus(WorkTask):
    """
    Downloads from the url and saveds it in the workdir with name
    outfile
    """

    url = luigi.Parameter()
    outfile = luigi.Parameter()
    expected_md5 = luigi.Parameter()

    def run(self):
        download_file(self.url, self.workdir.joinpath(self.outfile), self.expected_md5)
        self.mark_complete()

    @property
    def stage_number(self) -> int:
        return 0


class ExtractArchive(WorkTask):
    """
    Extracts the downloaded file in the workdir(optionally in subdir inside
    workdir)

    Parameter
        infile: filename which has to be extracted from the
            download task working directory
        download(DownloadCorpus): task which downloads the corpus to be
            extracted
    Requires:
        download(DownloadCorpus): task which downloads the corpus to be
            extracted
    """

    infile = luigi.Parameter()
    download = luigi.TaskParameter(
        visibility=luigi.parameter.ParameterVisibility.PRIVATE
    )
    # outdir is the sub dir inside the workdir to extract the file.
    outdir = luigi.Parameter()

    def requires(self):
        return {"download": self.download}

    @property
    def output_path(self):
        return self.workdir.joinpath(self.outdir)

    def run(self):
        archive_path = self.requires()["download"].workdir.joinpath(self.infile)
        archive_path = archive_path.absolute()
        shutil.unpack_archive(archive_path, self.output_path)

        self.mark_complete()


def get_download_and_extract_tasks(task_config: Dict) -> Dict[str, WorkTask]:
    """
    Iterates over the dowload urls and builds download and extract
    tasks for them
    """

    tasks = {}
    outdirs: Set[str] = set()
    for urlobj in task_config["download_urls"]:
        split, name, url, md5 = (
            urlobj["split"],
            urlobj.get("name", None),
            urlobj["url"],
            urlobj["md5"],
        )
        filename = os.path.basename(urlparse(url).path)
        if name is not None:
            outdir = f"{split}/{name}"
        else:
            outdir = f"{split}"
        assert outdir not in outdirs, f"{outdir} in {outdirs}. If you are downloading "
        "multiple archives into one split, they should have different 'name's."
        outdirs.add(outdir)
        task = ExtractArchive(
            download=DownloadCorpus(
                url=url, outfile=filename, expected_md5=md5, task_config=task_config
            ),
            infile=filename,
            outdir=outdir,
            task_config=task_config,
        )
        tasks[slugify(outdir, separator="_")] = task

    return tasks


class ExtractMetadata(WorkTask):
    """
    This is an abstract class that extracts metadata (including labels)
    over the full dataset.
    If we detect that the original dataset doesn't have a full
    train/valid/test split, we extract 20% validation/test if it
    is missing.

    We create a metadata csv file that will be used by downstream
    luigi tasks to curate the final dataset.

    The metadata columns are:
        * relpath
            (Possible variable) location of this audio
            file relative to the Python working directory.
            WARNING: Don't use this for hashing e.g. for splitting,
            because it may vary depending upon our choice of _workdir.
            Use datapath instead.
        * datapath [DISABLED]
            Fixed unique location of this audio file
            relative to the dataset root. This can be used for hashing
            and generating splits.
        * split
            Split of this particular audio file: ['train' 'valid', 'test']
            # TODO: Verify this
        * label
            Label for the scene or event. For multilabel, if
            there are multiple labels, they will be on different rows
            of the df.
        * start, end
            Start time in milliseconds for the event with
            this label. Event prediction tasks only, i.e. timestamp
            embeddings.
        * unique_filestem
            These filenames are used for final HEAR audio files,
            WITHOUT extension. This is because the audio might be
            in different formats, but ultimately we will convert
            it to wav.
            They must be unique across all relpaths, over the
            *entire* corpus. (Thus they imply a particular split.)
            They should be fixed across each run of this
            preprocessing pipeline.
        * split_key - See get_split_key [TODO: Move here]
    """

    outfile = luigi.Parameter()

    """
    You should define one for every (split, name) task.
    `ExtractArchive` is usually enough.

    However, custom downstream processing may be required. For
    example, `speech_commands.GenerateTrainDataset` adds silence
    and background noise instances to the train split.  Custom
    downstream tasks beyond ExtractArchive should have `output_path`
    property, like `self.ExtractArchive` or
    `speech_commands.GenerateTrainDataset`

    e.g.
    """
    # train = luigi.TaskParameter()
    # test = luigi.TaskParameter()

    def requires(self):
        # You should have one for each TaskParameter above. e.g.
        # return { "train": self.train, "test": self.test }
        ...

    @staticmethod
    def relpath_to_unique_filestem(relpath: str) -> str:
        """
        Convert a relpath to a unique filestem.
        Default: The relpath's filestem.
        Override: e.g. for speech commands, we include the command
        (parent directory) name so as not to clobber filestems.
        """
        return Path(relpath).stem

    @staticmethod
    def get_split_key(df: pd.DataFrame) -> pd.Series:
        """
        Gets the split key for each audio file.

        A file should only be in one split, i.e. we shouldn't spread
        file events across splits. This is the default behavior, and
        the split key is the filename itself.
        We use unique_filestem because it is fixed for a particular
        archive.
        (We could also use datapath.)

        Override: For some corpora:
        * An instrument cannot be split (nsynth)
        * A speaker cannot be split (speech_commands)
        """
        return df["unique_filestem"]

    def get_requires_metadata(self, split: str) -> pd.DataFrame:
        """
        For a particular key in the task requires (e.g. "train", or "train_eval"),
        return a metadata dataframe with the following columns (see above):
            * relpath
            * split
            * label
            * start, end: Optional
        """
        raise NotImplementedError("Deriving classes need to implement this")

    def get_all_metadata(self) -> pd.DataFrame:
        """
        Combine all metadata for this task. Should have the same
        columns described in `self.get_requires_metadata`.

        By default, we do one required task at a time and then
        concat them.

        Override: When a split cannot be computed
        using just one dataset path, and multiple datasets
        must be combined (see speech_commands).
        If you override this, make sure to `.reset_index(drop=True)`
        on the final df. You won't need to override
        `get_requires_metadata`.
        """
        metadata = pd.concat(
            [
                self.get_requires_metadata_check(requires_key)
                for requires_key in list(self.requires().keys())
            ]
        ).reset_index(drop=True)
        return metadata

    # ################  You don't need to override anything else

    def postprocess_all_metadata(self, metadata: pd.DataFrame) -> pd.DataFrame:
        """
        * Assign columns unique_filestem and split_key
        * Check uniqueness of unique_filestem
        * Deterministically shuffle the metadata rows
        * If --small, keep only metadata belonging to audio
        files in the small corpus.
        """

        # tqdm.pandas()
        metadata = metadata.assign(
            # This one apply is slow for massive datasets like nsynth
            # So we disable it because datapath isn't currently used.
            # datapath=lambda df: df.relpath.progress_apply(self.relpath_to_datapath),
            unique_filestem=lambda df: df.relpath.apply(
                self.relpath_to_unique_filestem
            ),
            split_key=self.get_split_key,
        )

        # No slashes can be present in the filestems. They are files, not dirs.
        assert not metadata["unique_filestem"].str.contains("/", regex=False).any()

        # Check if one unique_filestem is associated with only one relpath.
        assert metadata["relpath"].nunique() == metadata["unique_filestem"].nunique(), (
            f'{metadata["relpath"].nunique()} != '
            + f'{metadata["unique_filestem"].nunique()}'
        )
        # Also implies there is a one to one correspondence between relpath
        # and unique_filestem.
        #  1. One unique_filestem to one relpath -- the bug which
        #    we were having is one unique_filestem for two relpath(relpath
        #    with -6 as well as +6 having the same unique_filestem),
        #    groupby by unique_filestem and see if one relpath is
        #    associated with one unique_filestem - this is done in the
        #    assert statement.
        #  2. One relpath to one unique_filestem -- always the case
        #  3. relpath.nunique() == unique_filestem.nunique(), automatically
        # holds if the above two holds.
        assert (
            metadata.groupby("unique_filestem")["relpath"].nunique() == 1
        ).all(), "One unique_filestem is associated with more than one relpath "
        "Please make sure unique_filestems are unique"

        if "datapath" in metadata.columns:
            # If you use datapath, previous assertions check its
            # uniqueness wrt relpaths.
            assert metadata["relpath"].nunique() == metadata["datapath"].nunique()
            assert (
                metadata.groupby("datapath")["relpath"].nunique() == 1
            ).all(), "One datapath is associated with more than one relpath "
            "Please make sure datapaths are unique"

        # First, put the metadata into a deterministic order.
        if "start" in metadata.columns:
            metadata.sort_values(
                ["unique_filestem", "start", "end", "label"],
                inplace=True,
                kind="stable",
            )
        else:
            metadata.sort_values(
                ["unique_filestem", "label"], inplace=True, kind="stable"
            )

        # Now, deterministically shuffle the metadata
        # If we are going to drop things or subselect, we don't
        # want to do it according to alphabetical or filesystem order.
        metadata = metadata.sample(
            frac=1, random_state=str2int("postprocess_all_metadata")
        ).reset_index(drop=True)

        # Filter the files which actually exist in the data
        exists = metadata["relpath"].apply(lambda relpath: Path(relpath).exists())

        # If any of the audio files in the metadata is missing, raise an error for the
        # regular dataset. However, in case of small dataset, this is expected and we
        # need to remove those entries from the metadata
        if sum(exists) < len(metadata):
            if self.task_config["mode"] == "small":
                print(
                    "All files in metadata do not exist in the dataset. This is "
                    "expected behavior when small task is running.\n"
                    f"Removing {len(metadata) - sum(exists)} entries in the "
                    "metadata"
                )
                metadata = metadata.loc[exists]
                metadata.reset_index(drop=True, inplace=True)
                assert len(metadata) == sum(exists)
            else:
                raise FileNotFoundError(
                    "Files in the metadata are missing in the directory"
                )
        return metadata

    def split_train_test_val(self, metadata: pd.DataFrame):
        """
        This functions splits the metadata into test, train and valid from train
        split if any of test or valid split is not found. We split
        based upon the split_key (see above).

        If there is any data specific split, that will already be
        done in get_all_metadata. This function is for automatic
        splitting if the splits are not found.

        Note that all files are shuffled and we pick exactly as
        many as we want for each split. Unlike using modulus of the
        hash of the split key (Google `which_set` method), the
        filename does not uniquely determine the split, but the
        entire set of audio data determines the split.
        * The downside is that if a later version of the
        dataset is released with more files, this method will not
        preserve the split across dataset versions.
        * The benefit is that, for small datasets, it correctly
        stratifies the data according to the desired percentages.
        For small datasets, unless the splits are predetermined
        (e.g. in a key file), using the size of the data set to
        stratify is unavoidable. If we do want to preserve splits
        across versions, we can create key files for audio files
        that were in previous versions.

        Three cases might arise -
        1. Validation split not found - Train will be split into valid and train
        2. Test split not found - Train will be split into test and train
        3. Validation and Test split not found - Train will be split into test, train
            and valid
        """

        splits_present = metadata["split"].unique()

        # The metadata should at least have the train split
        # test and valid if not found in the metadata can be sampled
        # from the train
        assert "train" in splits_present, "Train split not found in metadata"
        splits_to_sample = set(SPLITS).difference(splits_present)
        diagnostics.info(
            f"{self.longname} Splits not already present in the dataset, "
            + f"now sampled with split key are: {splits_to_sample}"
        )

        if "split_percentage" in self.task_config:
            orig_train_percentage = self.task_config["split_percentage"]["train"]
            orig_valid_percentage = self.task_config["split_percentage"]["valid"]
            orig_test_percentage = self.task_config["split_percentage"]["test"]
        else:
            orig_train_percentage = TRAIN_PERCENTAGE
            orig_valid_percentage = VALIDATION_PERCENTAGE
            orig_test_percentage = TEST_PERCENTAGE

        train_percentage: float
        valid_percentage: float
        test_percentage: float

        # If we want a 60/20/20 split, but we already have test and don't
        # to partition one, we want to do a 75/25/0 split. i.e. we
        # keep everything summing to one and the proportions the same.
        if splits_to_sample == set():
            return metadata
        if splits_to_sample == set(["valid"]):
            tot = (orig_train_percentage + orig_valid_percentage) / 100
            train_percentage = orig_train_percentage / tot
            valid_percentage = orig_valid_percentage / tot
            test_percentage = 0
        elif splits_to_sample == set(["test"]):
            tot = (orig_train_percentage + orig_test_percentage) / 100
            train_percentage = orig_train_percentage / tot
            valid_percentage = 0
            test_percentage = orig_test_percentage / tot
        else:
            assert splits_to_sample == set(["valid", "test"])
            train_percentage = orig_train_percentage
            valid_percentage = orig_valid_percentage
            test_percentage = orig_test_percentage
        assert (
            train_percentage + valid_percentage + test_percentage == 100
        ), f"{train_percentage + valid_percentage + test_percentage} != 100"

        diagnostics.info(
            f"{self.longname} Split percentage for splitting the train set to "
            "generate other sets: "
            "{}".format(
                {
                    "test": test_percentage,
                    "train": train_percentage,
                    "valid": valid_percentage,
                }
            )
        )

        # Deterministically sort all unique split_keys.
        split_keys = sorted(metadata[metadata.split == "train"]["split_key"].unique())
        # Deterministically shuffle all unique split_keys.
        rng = random.Random("split_train_test_val")
        rng.shuffle(split_keys)
        n = len(split_keys)

        n_valid = int(round(n * valid_percentage / 100))
        n_test = int(round(n * test_percentage / 100))
        assert n_valid > 0 or valid_percentage == 0
        assert n_test > 0 or test_percentage == 0
        valid_split_keys = set(split_keys[:n_valid])
        test_split_keys = set(split_keys[n_valid : n_valid + n_test])
        metadata.loc[metadata["split_key"].isin(valid_split_keys), "split"] = "valid"
        metadata.loc[metadata["split_key"].isin(test_split_keys), "split"] = "test"
        return metadata

    def trim_event_metadata(self, metadata: pd.DataFrame, duration: float):
        """
        This modifies the event metadata to
        retain only events which start before the sample duration
        and trim them if they extend beyond the sample duration
        """
        # Since the duration in the task config is in seconds convert to milliseconds
        duration_ms = duration * 1000.0
        assert "start" in metadata.columns
        assert "end" in metadata.columns

        # Drop the events starting at or after the sample duration
        trimmed_metadata = metadata.loc[metadata["start"] < duration_ms]
        events_dropped = len(metadata) - len(trimmed_metadata)

        # Trim the events starting before but extending beyond the sample duration
        events_trimmed = len(trimmed_metadata.loc[metadata["end"] > duration_ms])
        trimmed_metadata.loc[metadata["end"] > duration_ms, "end"] = duration_ms

        assert (trimmed_metadata["start"] < duration_ms).all()
        assert (trimmed_metadata["end"] <= duration_ms).all()
        assert len(trimmed_metadata) <= len(metadata)
        assert (
            metadata["relpath"].nunique() == trimmed_metadata["relpath"].nunique()
        ), "File are getting removed while trimming. This is "
        "unexpected and only events from the end of the files should be removed"

        diagnostics.info(
            f"{self.longname} - Events dropped count {events_dropped} "
            "percentage {}%".format(round(events_dropped / len(metadata) * 100.0, 2))
        )
        diagnostics.info(
            f"{self.longname} - Events trimmed count {events_trimmed} "
            "percentage {}%".format(round(events_dropped / len(metadata) * 100.0, 2))
        )
        return trimmed_metadata

    def get_requires_metadata_check(self, requires_key: str) -> pd.DataFrame:
        df = self.get_requires_metadata(requires_key)
        assert "relpath" in df.columns
        assert "split" in df.columns
        assert "label" in df.columns
        if self.task_config["embedding_type"] == "event":
            assert "start" in df.columns
            assert "end" in df.columns
        return df

    def run(self):
        # Output stats for every input directory
        for key, requires in self.requires().items():
            stats = audio_util.get_audio_dir_stats(
                in_dir=requires.output_path,
                out_file=self.workdir.joinpath(f"{key}_stats.json"),
            )
            diagnostics.info(f"{self.longname} extractdir {key} stats {stats}")

        # Get all metadata to be used for the task
        metadata = self.get_all_metadata()
        print(f"metadata length = {len(metadata)}")

        _diagnose_split_labels(self.longname, "original", metadata)
        metadata = self.postprocess_all_metadata(metadata)
        _diagnose_split_labels(self.longname, "postprocessed", metadata)

        # Split the metadata to create valid and test set from train if they are not
        # created explicitly in get_all_metadata
        metadata = self.split_train_test_val(metadata)

        # Each split should have unique files and no file should be across splits
        assert (
            metadata.groupby("unique_filestem")["split"].nunique() == 1
        ).all(), "One unique_filestem is associated with more than one split"

        _diagnose_split_labels(self.longname, "split", metadata)

        if self.task_config["embedding_type"] == "scene":
            # Multiclass predictions should only have a single label per file
            if self.task_config["prediction_type"] == "multiclass":
                label_count = metadata.groupby("unique_filestem")["label"].aggregate(
                    len
                )
                assert (label_count == 1).all()
        elif self.task_config["embedding_type"] == "event":
            # Remove the events starting after the sample duration, and trim
            # the events starting before but extending beyond the sample
            # duration
            # sample duration is specified in the task config.
            # The specified sample duration is in seconds
            metadata = self.trim_event_metadata(
                metadata, duration=self.task_config["sample_duration"]
            )
        else:
            raise ValueError(
                "%s embedding_type unknown" % self.task_config["embedding_type"]
            )

        metadata.to_csv(
            self.workdir.joinpath(self.outfile),
            index=False,
        )

        # Save the label count for each split
        for split, split_df in metadata.groupby("split"):
            json.dump(
                split_df["label"].value_counts(normalize=True).to_dict(),
                self.workdir.joinpath(f"labelcount_{split}.json").open("w"),
                indent=True,
            )

        self.mark_complete()

    # UNUSED
    def relpath_to_datapath(self, relpath: Path) -> Path:
        """
        Given the path to this audio file from the Python working
        directory, strip all output_path from each required task.

        This filename directory is a little fiddly and gnarly.
        """
        # Find all possible base paths into which audio was extracted
        base_paths = [t.output_path for t in self.requires().values()]
        assert len(base_paths) == len(set(base_paths)), (
            "You seem to have duplicate (split, name) in your task "
            + f"config. {len(base_paths)} != {len(set(base_paths))}"
        )
        datapath = Path(relpath)
        relatives = 0
        for base_path in base_paths:
            if datapath.is_relative_to(base_path):  # type: ignore
                datapath = datapath.relative_to(base_path)
                relatives += 1
        assert relatives == 1, f"relatives {relatives}. " + f"base_paths = {base_paths}"
        assert datapath != relpath, (
            f"datapath in {relpath} not found. " + f"base_paths = {base_paths}"
        )
        return datapath


class MetadataTask(WorkTask):
    """
    Abstract WorkTask that wants to have access to the metadata
    from the entire dataset.
    """

    metadata_task: ExtractMetadata = luigi.TaskParameter()
    _metadata: Optional[pd.DataFrame] = None

    @property
    def metadata(self):
        if self._metadata is None:
            self._metadata = pd.read_csv(
                self.metadata_task.workdir.joinpath(self.metadata_task.outfile)
            )
        return self._metadata


class SplitTask(MetadataTask):
    """
    Abstract MetadataTask that has a split directory of workdir, and
    split parameter.
    """

    split = luigi.Parameter()

    @property
    def splitdir(self) -> Path:
        return self.workdir.joinpath(self.split)

    def createsplit(self):
        # Would be nice to have this happen automatically
        if self.splitdir.exists():
            assert self.splitdir.is_dir()
            shutil.rmtree(self.splitdir)
        self.splitdir.mkdir()


class SubsampleSplit(SplitTask):
    """
    For large datasets, we may want to restrict each split to a
    certain number of minutes.
    This subsampler acts on a specific split, and ensures we are under
    our desired audio length threshold for this split.

    Parameters:
        split: name of the split for which subsampling has to be done
    """

    def requires(self):
        return {
            "metadata": self.metadata_task,
        }

    def run(self):
        self.createsplit()

        split_metadata = self.metadata[
            self.metadata["split"] == self.split
        ].reset_index(drop=True)
        # Get all unique_filestem in this split, deterministically sorted
        split_filestem_relpaths = (
            split_metadata[["unique_filestem", "relpath"]]
            .drop_duplicates()
            .sort_values("unique_filestem")
        )

        assert split_filestem_relpaths["relpath"].nunique() == len(
            split_filestem_relpaths
        )
        # Deterministically shuffle the filestems
        split_filestem_relpaths = split_filestem_relpaths.sample(
            frac=1, random_state=str2int(f"SubsampleSplit({self.split})")
        ).values

        # This might round badly for small corpora with long audio :\
        # But we aren't going to use audio that is more than a couple
        # minutes or the timestamp embeddings will explode
        sample_duration = self.task_config["sample_duration"]

        # Get the max split duration from the task config.
        # If not defined use defaults from MAX_TASK_DURATION_BY_SPLIT
        if "max_task_duration_by_split" in self.task_config:
            max_task_duration_by_split = self.task_config["max_task_duration_by_split"]
            # Check if all the splits are specified in
            # max_task_duration_by_split
            assert set(max_task_duration_by_split.keys()) == set(SPLITS)
            max_split_duration = max_task_duration_by_split[self.split]
        else:
            # Get the default values for the split
            max_split_duration = MAX_TASK_DURATION_BY_SPLIT[self.split]

        # If max_split_duration is not None set the max_files so that
        # the total duration of all the audio files after subsampling
        # comes around (less than) max_split_duration
        if max_split_duration is not None:
            max_files = int(max_split_duration / sample_duration)
        else:
            # Otherwise set max_files to select all the files in the split
            max_files = len(split_filestem_relpaths)

        diagnostics.info(
            f"{self.longname} "
            f"Max number of files to sample in split {self.split}: "
            f"{max_files}"
        )

        diagnostics.info(
            f"{self.longname} "
            f"Files in split {self.split} before resampling: "
            f"{len(split_filestem_relpaths)}"
        )
        split_filestem_relpaths = split_filestem_relpaths[:max_files]
        diagnostics.info(
            f"{self.longname} "
            f"Files in split {self.split} after resampling: "
            f"{len(split_filestem_relpaths)}"
        )

        for unique_filestem, relpath in split_filestem_relpaths:
            audiopath = Path(relpath)
            newaudiofile = Path(
                # Add the current filetype suffix (mp3, webm, etc)
                # to the unique filestem.
                self.splitdir.joinpath(unique_filestem + audiopath.suffix)
            )
            assert not newaudiofile.exists(), f"{newaudiofile} already exists! "
            "We shouldn't have two files with the same name. If this is happening "
            "because luigi is overwriting an incomplete output directory "
            "we should write code to delete the output directory "
            "before this tasks begins."
            "If this is happening because different data dirs have the same "
            "audio file name, we should include the data dir in the symlinked "
            "filename."
            newaudiofile.symlink_to(audiopath.resolve())

        self.mark_complete()


class MonoWavSplit(SplitTask):
    """
    Converts the files to wav and mono encoding.

    This task ensures that the audio is converted to an uncompressed
    format so that any downstream operation on the audio results
    in precise outputs (like trimming and padding)
    https://stackoverflow.com/questions/54153364/ffmpeg-being-inprecise-when-trimming-mp3-files # noqa: E501

    Requires:
        corpus (SubsampleSplit): Split that was subsampled.
    """

    def requires(self):
        return {
            "corpus": SubsampleSplit(
                split=self.split,
                metadata_task=self.metadata_task,
                task_config=self.task_config,
            )
        }

    def run(self):
        self.createsplit()

        for audiofile in tqdm(list(self.requires()["corpus"].splitdir.iterdir())):
            if audiofile.suffix == ".json":
                continue
            newaudiofile = self.splitdir.joinpath(f"{audiofile.stem}.wav")
            audio_util.mono_wav(str(audiofile), str(newaudiofile))

        self.mark_complete()


class TrimPadSplit(SplitTask):
    """
    Trims and pads the wav audio files

    Requires:
        corpus (MonoWavSplit): task which converts the audio to wav file
    """

    def requires(self):
        return {
            "corpus": MonoWavSplit(
                split=self.split,
                metadata_task=self.metadata_task,
                task_config=self.task_config,
            )
        }

    def run(self):
        self.createsplit()

        for audiofile in tqdm(list(self.requires()["corpus"].splitdir.iterdir())):
            newaudiofile = self.splitdir.joinpath(f"{audiofile.stem}.wav")
            audio_util.trim_pad_wav(
                str(audiofile),
                str(newaudiofile),
                duration=self.task_config["sample_duration"],
            )

        self.mark_complete()


class SubcorpusData(MetadataTask):
    """
    Aggregates subsampling of all the splits into a single task as dependencies.

    Requires:
        splits (list(SplitTask)): final task over each split.
            This is a TrimPadSplit.
    """

    def requires(self):
        # Perform subsampling on each split independently
        splits = {
            split: TrimPadSplit(
                split=split,
                metadata_task=self.metadata_task,
                task_config=self.task_config,
            )
            for split in SPLITS
        }
        return splits

    def run(self):
        workdir = Path(self.workdir)
        if workdir.exists():
            if workdir.is_dir():
                workdir.rmdir()
            else:
                workdir.unlink()

        # We need to link the workdir of the requires, they will all be the same
        # for all the requires so just grab the first one.
        key = list(self.requires().keys())[0]
        workdir.symlink_to(Path(self.requires()[key].workdir).absolute())
        for key2 in self.requires().keys():
            assert self.requires()[key].workdir == self.requires()[key2].workdir

        # Output stats for every input directory
        for split in SPLITS:
            stats = audio_util.get_audio_dir_stats(
                in_dir=self.workdir.joinpath(split),
                out_file=self.workdir.joinpath(f"{split}_stats.json"),
            )
            diagnostics.info(f"{self.longname} {split} stats {stats}")

        self.mark_complete()


class SubcorpusMetadata(MetadataTask):
    """
    Find the metadata for the subcorpus, based upon which audio
    files are in each subcorpus split.

    Requires
        data (SubcorpusData): which produces the subcorpus data.
    """

    def requires(self):
        return {
            "data": SubcorpusData(
                metadata_task=self.metadata_task, task_config=self.task_config
            ),
        }

    def run(self):
        split_label_dfs = []
        for split in SPLITS:
            split_path = self.requires()["data"].workdir.joinpath(split)
            audiodf = pd.DataFrame(
                [(a.stem, a.suffix) for a in list(split_path.glob("*.wav"))],
                columns=["unique_filestem", "ext"],
            )
            assert len(audiodf) != 0, f"No audio files found in: {split_path}"
            assert (
                not audiodf["unique_filestem"].duplicated().any()
            ), "Duplicate files in: {split_path}"
            assert len(audiodf["ext"].drop_duplicates()) == 1
            assert audiodf["ext"].drop_duplicates().values[0] == ".wav"

            # Get the label from the metadata with the help
            # of the unique_filestem of the filename
            audiolabel_df = (
                self.metadata.merge(audiodf, on="unique_filestem")
                .assign(unique_filename=lambda df: df["unique_filestem"] + df["ext"])
                .drop("ext", axis=1)
            )

            if self.task_config["embedding_type"] == "scene":
                # Create a dictionary containing a list of metadata
                # keyed on the unique_filestem.
                audiolabel_json = (
                    audiolabel_df[["unique_filename", "label"]]
                    .groupby("unique_filename")["label"]
                    .apply(list)
                    .to_dict()
                )

            elif self.task_config["embedding_type"] == "event":
                # For event labeling each file will have a list of metadata
                audiolabel_json = (
                    audiolabel_df[["unique_filename", "label", "start", "end"]]
                    .set_index("unique_filename")
                    .groupby(level=0)
                    .apply(lambda group: group.to_dict(orient="records"))
                    .to_dict()
                )
            else:
                raise ValueError("Invalid embedding_type in dataset config")

            # Save the json used for training purpose
            json.dump(
                audiolabel_json,
                self.workdir.joinpath(f"{split}.json").open("w"),
                indent=True,
            )

            # Save the slug and the label in as the split metadata
            audiolabel_df.to_csv(
                self.workdir.joinpath(f"{split}.csv"),
                index=False,
            )
            split_label_dfs.append(audiolabel_df)

        _diagnose_split_labels(
            self.longname, "", pd.concat(split_label_dfs).reset_index(drop=True)
        )
        self.mark_complete()


class MetadataVocabulary(MetadataTask):
    """
    Creates the vocabulary CSV file for a task.

    Requires
        subcorpus_metadata (SubcorpusMetadata): task which produces
            the subcorpus metadata
    """

    def requires(self):
        return {
            "subcorpus_metadata": SubcorpusMetadata(
                metadata_task=self.metadata_task, task_config=self.task_config
            )
        }

    def run(self):
        labelset = set()
        # Save statistics about each subcorpus metadata
        for split in SPLITS:
            labeldf = pd.read_csv(
                self.requires()["subcorpus_metadata"].workdir.joinpath(f"{split}.csv")
            )
            json.dump(
                labeldf["label"].value_counts(normalize=True).to_dict(),
                self.workdir.joinpath(f"labelcount_{split}.json").open("w"),
                indent=True,
            )
            split_labelset = set(labeldf["label"].unique().tolist())
            assert len(split_labelset) != 0
            labelset = labelset | split_labelset

        # Build the label idx csv and save it
        labelcsv = pd.DataFrame(
            list(enumerate(sorted(list(labelset)))),
            columns=["idx", "label"],
        )

        labelcsv.to_csv(
            os.path.join(self.workdir, "labelvocabulary.csv"),
            columns=["idx", "label"],
            index=False,
        )

        self.mark_complete()


class ResampleSubcorpus(MetadataTask):
    """
    Resamples one split in the subsampled corpus to a particular sampling rate
    Parameters
        split (str): The split for which the resampling has to be done
        sr (int): output sampling rate
    Requires
        data (SubcorpusData): task which produces the subcorpus data
    """

    sr = luigi.IntParameter()
    split = luigi.Parameter()

    def requires(self):
        return {
            "data": SubcorpusData(
                metadata_task=self.metadata_task, task_config=self.task_config
            )
        }

    def run(self):
        original_dir = self.requires()["data"].workdir.joinpath(str(self.split))
        resample_dir = self.workdir.joinpath(str(self.sr)).joinpath(str(self.split))
        resample_dir.mkdir(parents=True, exist_ok=True)
        for audiofile in tqdm(sorted(list(original_dir.glob("*.wav")))):
            resampled_audiofile = new_basedir(audiofile, resample_dir)
            audio_util.resample_wav(audiofile, resampled_audiofile, self.sr)

        self.mark_complete()


class ResampleSubcorpuses(MetadataTask):
    """
    Aggregates resampling of all the splits and sampling rates
    into a single task as dependencies.

    Requires:
        ResampleSubcorpus for all split and sr
    """

    sample_rates = luigi.ListParameter()

    def requires(self):
        # Perform resampling on each split and sampling rate independently
        resample_splits = [
            ResampleSubcorpus(
                sr=sr,
                split=split,
                metadata_task=self.metadata_task,
                task_config=self.task_config,
            )
            for sr in self.sample_rates
            for split in SPLITS
        ]
        return resample_splits

    def run(self):
        workdir = Path(self.workdir)
        workdir.rmdir()
        # We need to link the workdir of the requires, they will all be the same
        # for all the requires so just grab the first one.
        requires_workdir = Path(self.requires()[0].workdir).absolute()
        workdir.symlink_to(requires_workdir)
        self.mark_complete()


class FinalCombine(MetadataTask):
    """
    Create a final dataset, no longer in _workdir but in directory
    tasks_dir.

    Parameters:
        sample_rates (list(int)): The list of sampling rates in
            which the corpus is required.
        tasks_dir str: Directory to put the combined dataset.
    Requires:
        resample (List(ResampleSubCorpus)): task which resamples
            the entire subcorpus
        subcorpus_metadata (SubcorpusMetadata): task with the
            subcorpus metadata
    """

    sample_rates = luigi.ListParameter()
    tasks_dir = luigi.Parameter()

    def requires(self):
        # Will copy the resampled subsampled data, the subsampled metadata,
        # and the metadata_vocabulary
        return {
            "resample": ResampleSubcorpuses(
                sample_rates=self.sample_rates,
                metadata_task=self.metadata_task,
                task_config=self.task_config,
            ),
            "subcorpus_metadata": SubcorpusMetadata(
                metadata_task=self.metadata_task, task_config=self.task_config
            ),
            "metadata_vocabulary": MetadataVocabulary(
                metadata_task=self.metadata_task, task_config=self.task_config
            ),
        }

    # We overwrite workdir here, because we want the output to be
    # the finalized task directory
    @property
    def workdir(self):
        return Path(self.tasks_dir).joinpath(self.versioned_task_name)

    def run(self):
        if self.workdir.exists():
            shutil.rmtree(self.workdir)

        # Copy the resampled files
        shutil.copytree(self.requires()["resample"].workdir, self.workdir)

        # Copy labelvocabulary.csv
        shutil.copy2(
            self.requires()["metadata_vocabulary"].workdir.joinpath(
                "labelvocabulary.csv"
            ),
            self.workdir.joinpath("labelvocabulary.csv"),
        )
        # Copy the train test metadata jsons
        src = self.requires()["subcorpus_metadata"].workdir
        dst = self.workdir
        for item in sorted(os.listdir(src)):
            if item.endswith(".json"):
                # Based upon https://stackoverflow.com/a/27161799
                assert not dst.joinpath(item).exists()
                assert not src.joinpath(item).is_dir()
                shutil.copy2(src.joinpath(item), dst.joinpath(item))
        # Python >= 3.8 only
        # shutil.copytree(src, dst, dirs_exist_ok=True, \
        #        ignore=shutil.ignore_patterns("*.csv"))
        # Save the dataset config as a json file
        config_out = self.workdir.joinpath("task_metadata.json")
        with open(config_out, "w") as fp:
            json.dump(
                self.task_config, fp, indent=True, cls=luigi.parameter._DictParamEncoder
            )

        self.mark_complete()


class TarCorpus(MetadataTask):
    """
    Tar the final dataset at some sample rate.

    TODO: Secret tasks should go into another directory,
    so we don't accidentally copy them to the public bucket.

    Parameters:
        sample_rates (list(int)): The list of sampling rates in
            which the corpus is required.
        tasks_dir str: Directory to put the combined dataset.
        tar_dir str: Directory to put the tar-files.
    Requires:
        combined (FinalCombine): Final combined dataset.
    """

    sample_rate = luigi.IntParameter()
    combined_task = luigi.TaskParameter()
    tasks_dir = luigi.Parameter()
    tar_dir = luigi.Parameter()

    def requires(self):
        return {"combined": self.combined_task}

    def source_to_archive_path(
        self, source_path: Union[str, Path], datestr: str
    ) -> str:
        source_path = str(source_path)
        archive_path = source_path.replace(self.tasks_dir, "tasks").replace(
            "tasks//", "tasks/"
        )
        assert (
            self.tasks_dir in ("tasks", "tasks/") or archive_path != source_path
        ), f"{archive_path} == {source_path}"
        assert archive_path.startswith("tasks")
        archive_path = f"hear-{datestr}{__version__}/{archive_path}"
        return archive_path

    @staticmethod
    def tar_filter(tarinfo: tarfile.TarInfo, pbar: tqdm) -> Optional[tarfile.TarInfo]:
        """tarfile with progress bar"""
        pbar.update(1)
        return tarinfo

    def create_tar(self, sample_rate: int):
        if INCLUDE_DATESTR_IN_FINAL_PATHS:
            datestr = datetime.today().strftime("%Y%m%d") + "-"
        else:
            datestr = ""
        tarname = (
            f"hear-{datestr}{__version__}-"
            + f"{self.versioned_task_name}-{sample_rate}.tar.gz"
        )
        tarname_latest = f"hear-LATEST-{self.versioned_task_name}-{sample_rate}.tar.gz"
        source_dir = str(self.requires()["combined"].workdir)

        # Compute the audio files to be tar'ed
        files = set()
        for split in SPLITS:
            files |= set(
                json.load(open(os.path.join(source_dir, f"{split}.json"))).keys()
            )

        # tarfile is pure python and very slow
        # But it's easy to precisely control, so we use it
        tarfile_workdir = self.workdir.joinpath(tarname)
        assert not os.path.exists(tarfile_workdir)
        with tarfile.open(tarfile_workdir, "w:gz") as tar:
            # First, add all files in the task
            for source_file in Path(source_dir).glob("*"):
                if source_file.is_file():
                    tar.add(
                        source_file, self.source_to_archive_path(source_file, datestr)
                    )
            # Now add audio files for this sample rate
            sample_rate_source = os.path.join(source_dir, str(sample_rate))
            with tqdm(
                desc=f"tar {self.versioned_task_name} {sample_rate}", total=len(files)
            ) as pbar:
                tar.add(
                    sample_rate_source,
                    self.source_to_archive_path(sample_rate_source, datestr),
                    filter=lambda tarinfo: self.tar_filter(tarinfo, pbar),
                )
        shutil.copyfile(tarfile_workdir, Path(self.tar_dir).joinpath(tarname))
        shutil.copyfile(tarfile_workdir, Path(self.tar_dir).joinpath(tarname_latest))

    def run(self):
        self.create_tar(self.sample_rate)
        self.mark_complete()


class FinalizeCorpus(MetadataTask):
    """
    Finalize the corpus, simply create all tar files.

    Parameters:
        sample_rates (list(int)): The list of sampling rates in
            which the corpus is required.
        tasks_dir str: Directory to put the combined dataset.
        tar_dir str: Directory to put the tar-files.
    Requires:
        combined (FinalCombine): Final combined dataset.
    """

    sample_rates = luigi.ListParameter()
    tasks_dir = luigi.Parameter()
    tar_dir = luigi.Parameter()

    def requires(self):
        combined_task = FinalCombine(
            sample_rates=self.sample_rates,
            tasks_dir=self.tasks_dir,
            metadata_task=self.metadata_task,
            task_config=self.task_config,
        )
        return {
            str(sr): TarCorpus(
                sample_rate=sr,
                combined_task=combined_task,
                tasks_dir=self.tasks_dir,
                tar_dir=self.tar_dir,
                metadata_task=self.metadata_task,
                task_config=self.task_config,
            )
            for sr in self.sample_rates
        }

    def run(self):
        self.mark_complete()


def run(task: Union[List[luigi.Task], luigi.Task], num_workers: int):
    """
    Run a task / set of tasks

    Args:
        task: a single or list of luigi tasks
        num_workers: Number of CPU workers to use for this task
    """

    # If this is just a single task then add it to a list
    if isinstance(task, luigi.Task):
        task = [task]

    diagnostics.info("LUIGI START")
    luigi_run_result = luigi.build(
        task,
        workers=num_workers,
        local_scheduler=True,
        log_level="INFO",
        detailed_summary=True,
    )
    diagnostics.info("LUIGI END")
    assert luigi_run_result.status in [
        luigi.execution_summary.LuigiStatusCode.SUCCESS,
        luigi.execution_summary.LuigiStatusCode.SUCCESS_WITH_RETRY,
    ], f"Received luigi_run_result.status = {luigi_run_result.status}"
