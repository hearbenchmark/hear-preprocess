#!/usr/bin/env python3
"""
Runs a sampler to sample the downloaded dataset.

It resamples each task to have audio_sample_size samples.
TODO: Consider changing this to a certain number of seconds
(max_task_duration in the luigi pipeline).

Uses the same download and extract tasks to make sure
the same downloaded files can be used for sampling
Also uses the configs defined in the task files for making
it simple to scale across multiple dataset
"""

import logging
import multiprocessing
import random
import shutil
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import click
import luigi
from tqdm import tqdm
import ffmpeg

import hearpreprocess.pipeline as pipeline
from hearpreprocess import dcase2016_task2, nsynth_pitch, speech_commands
from hearpreprocess.util.luigi import WorkTask
import hearpreprocess.util.audio as audio_util

logger = logging.getLogger("luigi-interface")
# Currently the sampler is only allowed to run for open tasks
# The secret tasks module will not be available for participants
try:
    from hearpreprocess.secrettasks import hearsecrettasks

    secret_config = hearsecrettasks.sampler_config
except ModuleNotFoundError as e:
    print(e)
    logger.info(
        "The hearsecrettask submodule is not installed. "
        "If you are a participant, this is an expected behaviour as the "
        "secret tasks are not made available to you. "
    )
    secret_config = {}

logger = logging.getLogger("luigi-interface")

METADATAFORMATS = [".csv", ".json", ".txt", ".midi"]
AUDIOFORMATS = [".mp3", ".wav", ".ogg", ".webm"]


# Note: Necessary key helps to select audios with the necessary keys in there name
configs = {
    "dcase2016_task2": {
        "task_config": dcase2016_task2.generic_task_config,
        "audio_sample_size": 4,
        "necessary_keys": [],
    },
    "nsynth_pitch": {
        "task_config": nsynth_pitch.generic_task_config,
        "audio_sample_size": 100,
        "necessary_keys": [],
    },
    "speech_commands": {
        "task_config": speech_commands.generic_task_config,
        "audio_sample_size": 100,
        "necessary_keys": [],
    },
    # Add the sampler config for the secrets task if the secret task config was found.
    # Not available for participants
    **secret_config,
}


class RandomSampleOriginalDataset(WorkTask):
    necessary_keys = luigi.ListParameter()
    audio_sample_size = luigi.IntParameter()

    def requires(self):
        return pipeline.get_download_and_extract_tasks(self.task_config)

    @staticmethod
    def safecopy(src, dst):
        # Make sure the parent exists
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    @staticmethod
    def trimcopy_audio(src, dst, small_duration):
        """
        Trims and saves the audio file to minimise the size of the generated
        small dataset
        """
        # Make sure the parent exists
        dst.parent.mkdir(parents=True, exist_ok=True)
        # Trim the audio if it is greater than the small_duration
        audio_stats = audio_util.get_audio_stats(str(src))
        if audio_stats is not None and audio_stats["duration"] > small_duration:
            try:
                _ = (
                    ffmpeg.input(str(src))
                    .filter("atrim", end=small_duration)  # Trim
                    .output(str(dst))
                    .run(quiet=True)
                )
            except ffmpeg.Error as e:
                print(
                    "Please check the console output for ffmpeg to debug the "
                    "error in trimcopy_audio: ",
                    f"Error: {e}",
                )
                raise
        else:
            # else copy the audio file as it is
            shutil.copy2(src, dst)

    def sample(self, all_files):
        # All metadata files will be copied without any sampling
        metadata_files = list(
            filter(lambda file: file.suffix in METADATAFORMATS, all_files)
        )
        # If the file name has a necessary key
        necessary_files = list(
            filter(
                lambda file: any(key in str(file) for key in self.necessary_keys),
                all_files,
            )
        )
        audio_files_to_sample = list(
            # Filter all the audio files which are not in the necessary list.
            # Out of these audios audio_sample_size number of samples will be
            # selected
            filter(
                lambda file: file.suffix.lower() in map(str.lower, AUDIOFORMATS),
                [file for file in all_files if file not in necessary_files],
            )
        )

        rng = random.Random("RandomSampleOriginalDataset")
        rng.shuffle(audio_files_to_sample)
        sampled_audio_files = audio_files_to_sample[
            : max(0, self.audio_sample_size - len(necessary_files))
        ]

        return (metadata_files, necessary_files + sampled_audio_files)

    def run(self):
        for url_obj in self.task_config["modes"]["small"]["download_urls"]:
            # Sample a small subset to copy from all the files
            url_name = Path(urlparse(url_obj["url"]).path).stem
            split = url_obj["split"]
            copy_from = self.requires()[split].workdir.joinpath(split)
            all_files = [file.relative_to(copy_from) for file in copy_from.rglob("*")]
            copy_files, copy_audio = self.sample(all_files)

            # Copy and make a zip
            copy_to = self.workdir.joinpath(url_name)
            if copy_to.exists():
                shutil.rmtree(copy_to)

            for file in tqdm(copy_files):
                self.safecopy(src=copy_from.joinpath(file), dst=copy_to.joinpath(file))

            # Save all the audio after trimming them to small sample duration
            # The small sample duration is specified in the small mode of the
            # task_config
            small_duration = self.task_config["modes"]["small"][
                "sample_duration"
            ]  # in seconds
            for file in tqdm(copy_audio):
                self.trimcopy_audio(
                    src=copy_from.joinpath(file),
                    dst=copy_to.joinpath(file),
                    small_duration=small_duration,
                )
            shutil.make_archive(copy_to, "zip", copy_to)


@click.command()
@click.argument("task")
@click.option(
    "--num-workers",
    default=None,
    help="Number of CPU workers to use when running. "
    "If not provided all CPUs are used.",
    type=int,
)
def main(task: str, num_workers: Optional[int] = None):
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    logger.info(f"Using {num_workers} workers")
    config = configs[task]
    config["task_config"]["mode"] = config["task_config"]["default_mode"]
    sampler = RandomSampleOriginalDataset(
        task_config=config["task_config"],
        audio_sample_size=config["audio_sample_size"],
        necessary_keys=config["necessary_keys"],
    )
    pipeline.run(sampler, num_workers=num_workers)


if __name__ == "__main__":
    main()
