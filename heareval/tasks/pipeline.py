"""
Generic pipelines for datasets
"""

import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Set, Union
from urllib.parse import urlparse

import luigi
import pandas as pd
from pandas import DataFrame, Series
from slugify import slugify
from tqdm import tqdm

import heareval.tasks.util.audio as audio_util
from heareval.tasks.util.luigi import (
    WorkTask,
    download_file,
    filename_to_int_hash,
    new_basedir,
    perform_metadata_subsampling,
)

SPLITS = ["train", "valid", "test"]
# This percentage should not be changed as this decides
# the data in the split and hence is not a part of the data config
VALIDATION_PERCENTAGE = 20
TEST_PERCENTAGE = 20
TRAIN_PERCENTAGE = 100 - VALIDATION_PERCENTAGE - TEST_PERCENTAGE

# We want no more than 5 hours of audio per task.
MAX_TASK_DURATION_BY_SPLIT = {
    "train": 3 * 3600,
    "valid": 1 * 3600,
    "test": 1 * 3600,
}


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
    # Outdir is the sub dir inside the workdir to extract the file.
    outdir = luigi.Parameter()

    def requires(self):
        return {"download": self.download}

    def run(self):
        archive_path = self.requires()["download"].workdir.joinpath(self.infile)
        archive_path = archive_path.absolute()
        output_path = self.workdir.joinpath(self.outdir)
        shutil.unpack_archive(archive_path, output_path)
        audio_util.audio_dir_stats_wav(
            in_dir=output_path,
            out_file=self.workdir.joinpath(f"{slugify(self.outdir)}_stats.json"),
        )

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
        * relpath - How you find the file path in the original dataset.
        * slug - This is the filename in our dataset. It should be
        unique, it should be obvious what the original filename
        was, and perhaps it should contain the label for audio scene
        tasks.
        * subsample_key - Hash or a tuple of hash to do subsampling
        * split - Split of this particular audio file.
        * label - Label for the scene or event.
        * start, end - Start and end time in seconds of the event,
        for event_labeling tasks.
    """

    outfile = luigi.Parameter()

    # This should have something like the following:
    # train = luigi.TaskParameter()
    # test = luigi.TaskParameter()

    def requires(self):
        ...
        # This should have something like the following:
        # return { "train": self.train, "test": self.test }

    @staticmethod
    def slugify_file_name(relative_path: str):
        """
        This is the filename in our dataset.

        It should be unique, it should be obvious what the original
        filename was, and perhaps it should contain the label for
        audio scene tasks.
        You can override this and simplify if the slugified filename
        for this dataset is too long.

        The basic version here takes the filename and slugifies it.
        """
        slug_text = str(Path(relative_path).stem)
        slug_text = slug_text.replace("-", "_negative_")
        return f"{slugify(slug_text)}"

    @staticmethod
    def get_subsample_key(df: DataFrame) -> Series:
        """
        Gets the subsample key.
        Subsample key is a unique hash at a audio file level used for subsampling.
        This is a hash of the slug. This is not recommended to be
        overridden.

        The data is first split by the split key and the subsample key is
        used to ensure stable sampling for groups which are incompletely
        sampled(the last group to be part of the subsample output)
        """
        assert "slug" in df, "slug column not found in the dataframe"
        return df["slug"].apply(str).apply(filename_to_int_hash)

    def get_all_metadata(self) -> pd.DataFrame:
        """
        Return a dataframe containing all metadata for this task.

        By default, we do one requires task at a time and then concat them.
        You might consider overriding this for some datasets (like
        Google Speech Commands) where you cannot process metadata
        on a per-split basis.
        """
        metadata = pd.concat(
            [
                self.get_requires_metadata(requires_key)
                for requires_key in list(self.requires().keys())
            ]
        ).reset_index(drop=True)
        return metadata

    def get_requires_metadata(self, requires_key: str) -> pd.DataFrame:
        raise NotImplementedError("Deriving classes need to implement this")

    def split_train_test_val(self, metadata: pd.DataFrame):
        """
        This functions splits the metadata into test, train and valid from train
        split if any of test or valid split is not found. We split
            based upon the relpath (filename), i.e. events in the same
        file go into the same split.

        If there is any data specific split, that will already be done in
        get_all_metadata. This function is for automatic splitting if the splits
        are not found.

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
        print(
            "Splits not already present in the dataset, "
            + f"now sampled with split key are: {splits_to_sample}"
        )

        train_percentage: float
        valid_percentage: float
        test_percentage: float

        # If we want a 60/20/20 split, but we already have test and don't
        # to partition one, we want to do a 75/25/0 split. i.e. we
        # keep everything summing to one and the proportions the same.
        if splits_to_sample == set():
            return metadata
        elif splits_to_sample == set(["valid"]):
            tot = (TRAIN_PERCENTAGE + VALIDATION_PERCENTAGE) / 100
            train_percentage = TRAIN_PERCENTAGE / tot
            valid_percentage = VALIDATION_PERCENTAGE / tot
            test_percentage = 0
        elif splits_to_sample == set(["test"]):
            tot = (TRAIN_PERCENTAGE + TEST_PERCENTAGE) / 100
            train_percentage = TRAIN_PERCENTAGE / tot
            valid_percentage = 0
            test_percentage = TEST_PERCENTAGE / tot
        else:
            assert splits_to_sample == set(["valid", "test"])
            train_percentage = TRAIN_PERCENTAGE
            valid_percentage = VALIDATION_PERCENTAGE
            test_percentage = TEST_PERCENTAGE
        assert (
            train_percentage + valid_percentage + test_percentage == 100
        ), f"{train_percentage + valid_percentage + test_percentage} != 100"

        relpaths = metadata[metadata.split == "train"]["relpath"].unique()
        rng = random.Random(0)
        rng.shuffle(relpaths)
        n = len(relpaths)

        n_valid = int(round(n * valid_percentage / 100))
        n_test = int(round(n * test_percentage / 100))
        assert n_valid > 0 or valid_percentage == 0
        assert n_test > 0 or test_percentage == 0
        valid_relpaths = set(relpaths[:n_valid])
        test_relpaths = set(relpaths[n_valid : n_valid + n_test])
        metadata.loc[metadata["relpath"].isin(valid_relpaths), "split"] = "valid"
        metadata.loc[metadata["relpath"].isin(test_relpaths), "split"] = "test"
        return metadata

    def run(self):
        # Process metadata gets all metadata to be used for the task
        metadata = self.get_all_metadata()

        # Deterministically shuffle the metadata
        metadata = metadata.sample(frac=1, random_state=0).reset_index(drop=True)

        metadata = metadata.assign(
            slug=lambda df: df.relpath.apply(self.slugify_file_name),
            subsample_key=self.get_subsample_key,
        )

        # Check if one slug is associated with only one relpath.
        # Also implies there is a one to one correspondence between relpath and slug.
        #  1. One slug to one relpath -- the bug which we were having is one slug for
        #   two relpath(relpath with -6 as well as +6 having the same slug), groupby
        #   by slug and see if one relpath is associated with one slug - this is done
        #   in the assert statement.
        #  2. One relpath to one slug -- always the case, because slugify is
        #   a deterministic function.
        #  3. relpath.nunique() == slug.nunique(), automatically holds if the above
        #   two holds.
        assert (
            metadata.groupby("slug")["relpath"].nunique() == 1
        ).all(), "One slug is associated with more than one file"
        "Please make sure slugs are unique at a file level"

        # Assertion sanity check -- one to one mapping between the relpaths and slugs
        assert metadata["relpath"].nunique() == metadata["slug"].nunique()

        # Filter the files which actually exist in the data
        exists = metadata["relpath"].apply(lambda relpath: Path(relpath).exists())

        # If any of the audio files in the metadata is missing, raise an error for the
        # regular dataset. However, in case of small dataset, this is expected and we
        # need to remove those entries from the metadata
        if sum(exists) < len(metadata):
            if self.task_config["version"].split("-")[-1] == "small":
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

        # Split the metadata to create valid and test set from train if they are not
        # created explicitly in get_all_metadata
        metadata = self.split_train_test_val(metadata)

        if self.task_config["embedding_type"] == "event":
            assert set(
                [
                    "relpath",
                    "slug",
                    "subsample_key",
                    "split",
                    "label",
                    "start",
                    "end",
                ]
            ).issubset(set(metadata.columns))
        elif self.task_config["embedding_type"] == "scene":
            assert set(
                [
                    "relpath",
                    "slug",
                    "subsample_key",
                    "split",
                    "label",
                ]
            ).issubset(set(metadata.columns))
            # Multiclass predictions should only have a single label per file
            if self.task_config["prediction_type"] == "multiclass":
                label_count = metadata.groupby("slug")["label"].aggregate(len)
                assert (label_count == 1).all()
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
                split_df["label"].value_counts().to_dict(),
                self.workdir.joinpath(f"labelcount_{split}.json").open("w"),
                indent=True,
            )

        self.mark_complete()


class SubsampleSplit(WorkTask):
    """
    A subsampler that acts on a specific split.

    Parameters:
        split: name of the split for which subsampling has to be done
        max_files: maximum files required from the subsampling
        metadata (ExtractMetadata): task which extracts corpus level metadata
    Requirements:
        metadata (ExtractMetadata): task which extracts corpus level metadata
    """

    split = luigi.Parameter()
    metadata: ExtractMetadata = luigi.TaskParameter()

    def requires(self):
        # The meta files contain the path of the files in the data
        # so we dont need to pass the extract as a dependency here.
        return {
            "metadata": self.metadata,
        }

    def get_metadata(self):
        return pd.read_csv(
            self.requires()["metadata"].workdir.joinpath(
                self.requires()["metadata"].outfile
            )
        )

    def get_subsample_metadata(self):
        metadata = self.get_metadata()[["split", "subsample_key", "slug", "relpath"]]

        if self.task_config["embedding_type"] == "scene":
            assert metadata["subsample_key"].nunique() == len(metadata)

        # Since event detection metadata will have duplicates, we de-dup
        subsample_metadata = (
            metadata.sort_values(by="subsample_key")
            # Drop duplicates as the subsample key is expected to be unique
            .drop_duplicates(subset="subsample_key", ignore_index=True)
            # Select the split to subsample
            .loc[lambda df: df["split"] == self.split]
        )
        return subsample_metadata

    def run(self):
        subsample_metadata = self.get_subsample_metadata()
        num_files = len(subsample_metadata)
        # This might round badly for small corpora with long audio :\
        # TODO: Issue to check for this
        sample_duration = self.task_config["sample_duration"]
        max_files = int(MAX_TASK_DURATION_BY_SPLIT[self.split] / sample_duration)
        if num_files > max_files:
            print(
                f"{num_files} audio files in corpus."
                f"Max files to subsample: {max_files}"
            )
            sampled_subsample_metadata = perform_metadata_subsampling(
                subsample_metadata, max_files
            )
            print(
                "Datapoints in split after resampling: "
                f"{len(sampled_subsample_metadata)}"
            )
            assert perform_metadata_subsampling(
                subsample_metadata.sample(frac=1), max_files
            ).equals(sampled_subsample_metadata), "The subsampling is not stable"
        else:
            sampled_subsample_metadata = subsample_metadata

        for _, audio in sampled_subsample_metadata.iterrows():
            audiofile = Path(audio["relpath"])
            # Add the original extension to the slug
            newaudiofile = Path(
                self.workdir.joinpath(f"{audio['slug']}{audiofile.suffix}")
            )
            # missing_ok is python >= 3.8
            assert not newaudiofile.exists(), f"{newaudiofile} already exists! "
            "We shouldn't have two files with the same name. If this is happening "
            "because luigi is overwriting an incomplete output directory "
            "we should write code to delete the output directory "
            "before this tasks begins."
            "If this is happening because different data dirs have the same "
            "audio file name, we should include the data dir in the symlinked "
            "filename."
            newaudiofile.symlink_to(audiofile.resolve())

        self.mark_complete()


class SubsampleSplits(WorkTask):
    """
    Aggregates subsampling of all the splits into a single task as dependencies.

    Parameter:
        metadata (ExtractMetadata): task which extracts a corpus level metadata
    Requires:
        subsample_splits (list(SubsampleSplit)): task subsamples each split
    """

    metadata: ExtractMetadata = luigi.TaskParameter()

    def requires(self):
        # Perform subsampling on each split independently
        subsample_splits = {
            split: SubsampleSplit(
                metadata=self.metadata,
                split=split,
                task_config=self.task_config,
            )
            for split in SPLITS
        }
        return subsample_splits

    def run(self):
        workdir = Path(self.workdir)
        workdir.rmdir()
        # We need to link the workdir of the requires, they will all be the same
        # for all the requires so just grab the first one.
        key = list(self.requires().keys())[0]
        workdir.symlink_to(Path(self.requires()[key].workdir).absolute())
        self.mark_complete()


class MonoWavTrimCorpus(WorkTask):
    """
    Converts the file to mono, changes to wav encoding,
    trims and pads the audio to be same length

    Parameters
        metadata (ExtractMetadata): task which extracts a corpus level metadata
    Requires:
        corpus (SubsampleSplits): task which aggregates the subsampling for each split
    """

    metadata: ExtractMetadata = luigi.TaskParameter()

    def requires(self):
        return {
            "corpus": SubsampleSplits(
                metadata=self.metadata, task_config=self.task_config
            )
        }

    def run(self):
        # TODO: this should check to see if the audio is already a mono wav at the
        #   correct length and just create a symlink if that is this case.
        for audiofile in tqdm(list(self.requires()["corpus"].workdir.iterdir())):
            newaudiofile = self.workdir.joinpath(f"{audiofile.stem}.wav")
            audio_util.mono_wav_and_fix_duration(
                audiofile, newaudiofile, duration=self.task_config["sample_duration"]
            )

        self.mark_complete()


class SplitData(WorkTask):
    """
    Go over the subsampled folder and pick the audio files. The audio files are
    saved with their slug names and hence the corresponding label can be picked
    up from the preprocess config. (These are symlinks.)

    Parameters
        metadata (ExtractMetadata): task which extracts a corpus level metadata
            the metadata helps to provide the split type of each audio file
    Requires
        corpus(MonoWavTrimCorpus): which processes the audio file and converts
            them to wav format
    """

    metadata: ExtractMetadata = luigi.TaskParameter()

    def requires(self):
        # The metadata helps in provide the split type for each
        # audio file
        return {
            "corpus": MonoWavTrimCorpus(
                metadata=self.metadata, task_config=self.task_config
            ),
            "metadata": self.metadata,
        }

    def run(self):
        meta = self.requires()["metadata"]
        metadata = pd.read_csv(
            os.path.join(meta.workdir, meta.outfile),
        )[["slug", "split"]]

        for audiofile in tqdm(list(self.requires()["corpus"].workdir.glob("*.wav"))):
            # Compare the filename with the slug.
            # Note that the slug does not have the extension of the file
            split = metadata.loc[metadata["slug"] == audiofile.stem, "split"].values[0]
            split_dir = self.workdir.joinpath(split)
            split_dir.mkdir(exist_ok=True)
            newaudiofile = new_basedir(audiofile, split_dir)
            os.symlink(os.path.realpath(audiofile), newaudiofile)

        self.mark_complete()


class SplitMetadata(WorkTask):
    """
    Splits the label dataframe, based upon which audio files are in this split.

    Parameters
        metadata (ExtractMetadata): task which extracts a corpus level metadata
    Requires
        data (SplitData): which produces the split level corpus
    """

    metadata: ExtractMetadata = luigi.TaskParameter()

    def requires(self):
        return {
            "data": SplitData(metadata=self.metadata, task_config=self.task_config),
            "metadata": self.metadata,
        }

    def get_metadata(self):
        metadata = pd.read_csv(
            self.requires()["metadata"].workdir.joinpath(
                self.requires()["metadata"].outfile
            )
        )
        return metadata

    def run(self):
        labeldf = self.get_metadata()

        for split_path in self.requires()["data"].workdir.iterdir():
            audiodf = pd.DataFrame(
                [(a.stem, a.suffix) for a in list(split_path.glob("*.wav"))],
                columns=["slug", "ext"],
            )
            assert len(audiodf) != 0, f"No audio files found in: {split_path}"
            assert (
                not audiodf["slug"].duplicated().any()
            ), "Duplicate files in: {split_path}"

            # Get the label from the metadata with the help of the slug of the filename
            audiolabel_df = (
                labeldf.merge(audiodf, on="slug")
                .assign(slug_path=lambda df: df["slug"] + df["ext"])
                .drop("ext", axis=1)
            )

            if self.task_config["embedding_type"] == "scene":
                # Create a dictionary containing a list of metadata keyed on the slug.
                audiolabel_json = (
                    audiolabel_df[["slug_path", "label"]]
                    .groupby("slug_path")["label"]
                    .apply(list)
                    .to_dict()
                )

            elif self.task_config["embedding_type"] == "event":
                # For event labeling each file will have a list of metadata
                audiolabel_json = (
                    audiolabel_df[["slug_path", "label", "start", "end"]]
                    .set_index("slug_path")
                    .groupby(level=0)
                    .apply(lambda group: group.to_dict(orient="records"))
                    .to_dict()
                )
            else:
                raise ValueError("Invalid embedding_type in dataset config")

            # Save the json used for training purpose
            json.dump(
                audiolabel_json,
                self.workdir.joinpath(f"{split_path.stem}.json").open("w"),
                indent=True,
            )

            # Save the slug and the label in as the split metadata
            audiolabel_df.to_csv(
                self.workdir.joinpath(f"{split_path.stem}.csv"),
                index=False,
            )

        self.mark_complete()


class MetadataVocabulary(WorkTask):
    """
    Creates the vocabulary CSV file for a task.

    Parameters
        metadata (ExtractMetadata): task which extracts a corpus level metadata
    Requires
        splitmeta (SplitMetadata): task which produces the split
            level metadata
    """

    metadata: ExtractMetadata = luigi.TaskParameter()

    def requires(self):
        return {
            "splitmeta": SplitMetadata(
                metadata=self.metadata, task_config=self.task_config
            )
        }

    def run(self):
        labelset = set()
        # Iterate over all the files in the split metadata and get the
        # split_metadata
        for split_metadata in list(self.requires()["splitmeta"].workdir.glob("*.csv")):
            labeldf = pd.read_csv(split_metadata)
            json.dump(
                labeldf["label"].value_counts().to_dict(),
                self.workdir.joinpath(f"labelcount_{split_metadata.stem}.json").open(
                    "w"
                ),
                indent=True,
            )
            labelset = labelset | set(labeldf["label"].unique().tolist())

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


class ResampleSubcorpus(WorkTask):
    """
    Resamples the Subsampled corpus in different sampling rate
    Parameters
        split(str): The split for which the resampling has to be done
        sr(int): output sampling rate
        metadata (ExtractMetadata): task which extracts corpus level metadata
    Requires
        data (SplitData): task which produces the split
            level corpus
    """

    sr = luigi.IntParameter()
    split = luigi.Parameter()
    metadata: ExtractMetadata = luigi.TaskParameter()

    def requires(self):
        return {"data": SplitData(metadata=self.metadata, task_config=self.task_config)}

    def run(self):
        original_dir = self.requires()["data"].workdir.joinpath(str(self.split))
        resample_dir = self.workdir.joinpath(str(self.sr)).joinpath(str(self.split))
        resample_dir.mkdir(parents=True, exist_ok=True)
        for audiofile in tqdm(list(original_dir.glob("*.wav"))):
            resampled_audiofile = new_basedir(audiofile, resample_dir)
            audio_util.resample_wav(audiofile, resampled_audiofile, self.sr)

        audio_util.audio_dir_stats_wav(
            in_dir=resample_dir,
            out_file=self.workdir.joinpath(str(self.sr)).joinpath(
                f"{self.split}_stats.json"
            ),
        )
        self.mark_complete()


class ResampleSubcorpuses(WorkTask):
    """
    Aggregates resampling of all the splits and sampling rates
    into a single task as dependencies.

    Parameter:
        metadata (ExtractMetadata): task which extracts corpus level metadata
    Requires:
        subsample_splits (list(SubsampleSplit)): task subsamples each split
    """

    sample_rates = luigi.ListParameter()
    metadata: ExtractMetadata = luigi.TaskParameter()

    def requires(self):
        # Perform resampling on each split and sampling rate independently
        resample_splits = [
            ResampleSubcorpus(
                sr=sr,
                split=split,
                metadata=self.metadata,
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


class FinalizeCorpus(WorkTask):
    """
    Create a final corpus, no longer in _workdir but in the top-level
    at directory config.TASKNAME.
    Parameters:
        sample_rates (list(int)): The list of sampling rates in which the corpus
            is required
        metadata (ExtractMetadata): task which extracts corpus level metadata
    Requires:
        resample (List(ResampleSubCorpus)): list of task which resamples each split
        splitmeta (SplitMetadata): task which produces the split
            level metadata
    """

    sample_rates = luigi.ListParameter()
    metadata: ExtractMetadata = luigi.TaskParameter()
    tasks_dir = luigi.Parameter()

    def requires(self):
        # Will copy the resampled data and the split metadata and the vocabmeta
        return {
            "resample": ResampleSubcorpuses(
                sample_rates=self.sample_rates,
                metadata=self.metadata,
                task_config=self.task_config,
            ),
            "splitmeta": SplitMetadata(
                metadata=self.metadata, task_config=self.task_config
            ),
            "vocabmeta": MetadataVocabulary(
                metadata=self.metadata, task_config=self.task_config
            ),
        }

    # We overwrite workdir here, because we want the output to be
    # the finalized top-level task directory
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
            self.requires()["vocabmeta"].workdir.joinpath("labelvocabulary.csv"),
            self.workdir.joinpath("labelvocabulary.csv"),
        )
        # Copy the train test metadata jsons
        src = self.requires()["splitmeta"].workdir
        dst = self.workdir
        for item in os.listdir(src):
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

    luigi_run_result = luigi.build(
        task,
        workers=num_workers,
        local_scheduler=True,
        log_level="INFO",
        detailed_summary=True,
    )
    assert luigi_run_result.status in [
        luigi.execution_summary.LuigiStatusCode.SUCCESS,
        luigi.execution_summary.LuigiStatusCode.SUCCESS_WITH_RETRY,
    ], f"Received luigi_run_result.status = {luigi_run_result.status}"
