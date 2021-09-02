"""
Generic pipelines for datasets
"""

import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Set, Union
from urllib.parse import urlparse

import hearpreprocess.util.audio as audio_util
import luigi
import pandas as pd
from hearpreprocess.util.luigi import WorkTask, download_file, new_basedir
from slugify import slugify
from tqdm import tqdm

SPLITS = ["train", "valid", "test"]
# This percentage should not be changed as this decides
# the data in the split and hence is not a part of the data config
VALIDATION_PERCENTAGE = 20
TEST_PERCENTAGE = 20
TRAIN_PERCENTAGE = 100 - VALIDATION_PERCENTAGE - TEST_PERCENTAGE

# We want no more than 5 hours of audio per task.
# This can be overriden in the task config.
# e.g. speech_commands test set.
# If None, no limit is used.
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
    # If set to None the file is extracted in the workdir without any
    # subdir
    outdir = luigi.Parameter(default=None)

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
        * split - Split of this particular audio file.
        * label - Label for the scene or event.
        * start, end - Start and end time in seconds of the event,
        for event_labeling tasks.
        * split_key - See get_split_key
        * slug - This is the filename in our dataset. It should be
        unique, it should be obvious what the original filename
        was, and perhaps it should contain the label for audio scene
        tasks.
    """

    outfile = luigi.Parameter()

    # This should have something like the following:
    # train = luigi.TaskParameter()
    # test = luigi.TaskParameter()

    def requires(self):
        ...
        # This should have something like the following:
        # return { "train": self.train, "test": self.test }

    def get_requires_metadata(self, requires_key: str) -> pd.DataFrame:
        """
        For a particular key in the task requires (e.g. "train", or "train_eval"),
        return a metadata dataframe with the following columns:
            * relpath - How you find the file path in the original dataset.
            * split - Split of this particular audio file.
            * label - Label for the scene or event.
            * start, end - Start and end time in seconds of the event,
            only for event_labeling tasks.
        """
        raise NotImplementedError("Deriving classes need to implement this")

    def get_requires_metadata_check(self, requires_key: str) -> pd.DataFrame:
        df = self.get_requires_metadata(requires_key)
        assert "relpath" in df.columns
        assert "split" in df.columns
        assert "label" in df.columns
        if self.task_config["embedding_type"] == "event":
            assert "start" in df.columns
            assert "end" in df.columns
        return df

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
                self.get_requires_metadata_check(requires_key)
                for requires_key in list(self.requires().keys())
            ]
        ).reset_index(drop=True)
        return metadata

    @staticmethod
    def slugify_file_name(relative_path: str) -> str:
        """
        This is the filename in our dataset, WITHOUT the extension.

        It should be unique, it should be obvious what the original
        filename was, and perhaps it should contain the label for
        audio scene tasks.
        """
        slug_text = str(Path(relative_path).stem)
        slug_text = slug_text.replace("-", "_negative_")
        return f"{slugify(slug_text)}"

    @staticmethod
    def get_split_key(df: pd.DataFrame) -> pd.Series:
        """
        Gets the split key.
        A file should only be in one split, i.e. we shouldn't spread
        file events across splits. This is the default behavior.
        For some corpora, we might want to be even more restrictive:
        * An instrument cannot be split.
        * A speaker cannot be split.
        """
        return df["relpath"]

    def split_train_test_val(self, metadata: pd.DataFrame):
        """
        This functions splits the metadata into test, train and valid from train
        split if any of test or valid split is not found. We split
        based upon the split_key (see above).

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

        # If we want a 60/20/20 split, but we already have test
        # then we want to do a 75/25/0 split so that train is still 3x validation
        if splits_to_sample == set():
            return metadata
        elif splits_to_sample == set("valid"):
            tot = TRAIN_PERCENTAGE + TEST_PERCENTAGE
            train_percentage = (
                TRAIN_PERCENTAGE + TRAIN_PERCENTAGE * VALIDATION_PERCENTAGE / tot
            )
            valid_percentage = 0
            test_percentage = (
                TEST_PERCENTAGE + TEST_PERCENTAGE * VALIDATION_PERCENTAGE / tot
            )
        elif splits_to_sample == set("test"):
            tot = TRAIN_PERCENTAGE + TEST_PERCENTAGE
            train_percentage = (
                TRAIN_PERCENTAGE + TRAIN_PERCENTAGE * TEST_PERCENTAGE / tot
            )
            valid_percentage = (
                VALIDATION_PERCENTAGE + VALIDATION_PERCENTAGE * TEST_PERCENTAGE / tot
            )
            test_percentage = 0
        else:
            train_percentage = TRAIN_PERCENTAGE
            valid_percentage = VALIDATION_PERCENTAGE
            test_percentage = TEST_PERCENTAGE
        assert (
            train_percentage + valid_percentage + test_percentage == 100
        ), f"{train_percentage + valid_percentage + test_percentage} != 100"

        split_keys = metadata["split_key"].unique()
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

    def run(self):
        # Process metadata gets all metadata to be used for the task
        metadata = self.get_all_metadata()

        # Deterministically shuffle the metadata
        metadata = metadata.sample(frac=1, random_state=0).reset_index(drop=True)

        metadata = metadata.assign(
            slug=lambda df: df.relpath.apply(self.slugify_file_name),
            split_key=self.get_split_key,
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

        if self.task_config["embedding_type"] == "scene":
            # Multiclass predictions should only have a single label per file
            if self.task_config["prediction_type"] == "multiclass":
                label_count = metadata.groupby("slug")["label"].aggregate(len)
                assert (label_count == 1).all()

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


class SubsampleSplit(MetadataTask):
    """
    For large datasets, we may want to restrict each split to a
    certain number of minutes.
    This subsampler acts on a specific split, and ensures we are under
    our desired audio length threshold for this split.

    Parameters:
        split: name of the split for which subsampling has to be done
    """

    split = luigi.Parameter()

    def requires(self):
        return {
            "metadata": self.metadata_task,
        }

    def run(self):
        split_metadata = self.metadata[self.metadata["split"] == self.split]
        relpaths = split_metadata["relpath"].unique()
        rng = random.Random("SubsampleSplit")
        rng.shuffle(relpaths)
        num_files = len(relpaths)

        # This might round badly for small corpora with long audio :\
        # But we aren't going to use audio that is more than a couple
        # minutes or the timestamp embeddings will explode
        sample_duration = self.task_config["sample_duration"]

        if (
            "max_task_duration_by_split" in self.task_config
            and self.split in self.task_config["max_task_duration_by_split"]
        ):
            max_split_duration = self.task_config["max_task_duration_by_split"][
                self.split
            ]
        else:
            max_split_duration = MAX_TASK_DURATION_BY_SPLIT[self.split]
        if max_split_duration is None:
            max_files = num_files
        else:
            max_files = int(MAX_TASK_DURATION_BY_SPLIT[self.split] / sample_duration)

        if num_files > max_files:
            print(
                f"{num_files} audio files in corpus."
                f"Max files to subsample: {max_files}"
            )
            subsampled_relpaths = set(relpaths[:max_files])
            print(f"Files in split after subsampling: f{len(subsampled_relpaths)}")
        else:
            subsampled_relpaths = relpaths

        for audiofile in subsampled_relpaths:
            audiopath = Path(audiofile)
            # Add the original extension to the slug
            newaudiofile = Path(
                self.workdir.joinpath(
                    "%s%s"
                    % (
                        self.metadata_task.slugify_file_name(audiofile),
                        audiopath.suffix,
                    )
                )
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


class SubsampleSplits(MetadataTask):
    """
    Aggregates subsampling of all the splits into a single task as dependencies.

    Requires:
        subsample_splits (list(SubsampleSplit)): task subsamples each split
    """

    def requires(self):
        # Perform subsampling on each split independently
        subsample_splits = {
            split: SubsampleSplit(
                metadata_task=self.metadata_task,
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


class MonoWavTrimSubcorpus(MetadataTask):
    """
    Converts the file to mono, changes to wav encoding,
    trims and pads the audio to be same length

    Requires:
        corpus (SubsampleSplits): task which aggregates all subsampled splits
    """

    def requires(self):
        return {
            "corpus": SubsampleSplits(
                metadata_task=self.metadata_task, task_config=self.task_config
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


class SubcorpusData(MetadataTask):
    """
    Go over the mono wav folder and symlink the audio files into split dirs.

    Requires
        corpus(MonoWavTrimSubcorpus): which processes the audio file and converts
            them to wav format
    """

    def requires(self):
        return {
            "corpus": MonoWavTrimSubcorpus(
                metadata_task=self.metadata_task, task_config=self.task_config
            ),
        }

    def run(self):
        for audiofile in tqdm(list(self.requires()["corpus"].workdir.glob("*.wav"))):
            # Compare the filename with the slug.
            # Note that the slug does not have the extension of the file
            split = self.metadata.loc[
                self.metadata["slug"] == audiofile.stem, "split"
            ].values[0]
            split_dir = self.workdir.joinpath(split)
            split_dir.mkdir(exist_ok=True)
            newaudiofile = new_basedir(audiofile, split_dir)
            os.symlink(os.path.realpath(audiofile), newaudiofile)

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
                self.metadata.merge(audiodf, on="slug")
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
        for subcorpus_metadata in list(
            self.requires()["subcorpus_metadata"].workdir.glob("*.csv")
        ):
            labeldf = pd.read_csv(subcorpus_metadata)
            json.dump(
                labeldf["label"].value_counts(normalize=True).to_dict(),
                self.workdir.joinpath(
                    f"labelcount_{subcorpus_metadata.stem}.json"
                ).open("w"),
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


class FinalizeCorpus(MetadataTask):
    """
    Create a final corpus, no longer in _workdir but in the top-level
    at directory config.TASKNAME.
    Parameters:
        sample_rates (list(int)): The list of sampling rates in which the corpus
            is required
    Requires:
        resample (List(ResampleSubCorpus)): task which resamples
                the entire subcorpus

        subcorpus_metadata (SubcorpusMetadata): task with the subcorpus metadata
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
            self.requires()["metadata_vocabulary"].workdir.joinpath(
                "labelvocabulary.csv"
            ),
            self.workdir.joinpath("labelvocabulary.csv"),
        )
        # Copy the train test metadata jsons
        src = self.requires()["subcorpus_metadata"].workdir
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
