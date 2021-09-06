"""
Audio utility functions for evaluation task preparation
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

import numpy as np
import ffmpeg
from tqdm import tqdm


def mono_wav(in_file: str, out_file: str) -> None:
    """converts the audio to wav format with mono stream"""
    assert not Path(out_file).exists(), "File already exists"
    try:
        _ = (
            ffmpeg.input(in_file)
            .audio.output(out_file, f="wav", acodec="pcm_f32le", ac=1)
            .run(quiet=True)
        )
    except ffmpeg.Error as e:
        print(
            "Please check the console output for ffmpeg to debug the "
            "error in mono wav: ",
            f"Error: {e}",
        )
        raise

    # Check if the generated file is present and that ffmpeg can
    # read stats for the file to be used in subsequent processing steps
    assert Path(out_file).exists(), "wav file saved by ffmpeg was not found"
    assert (
        get_audio_stats(out_file)["ext"] is not None
    ), "Unable to get stats for the generated wav file"


def trim_pad_wav(in_file: str, out_file: str, duration: float) -> None:
    """
    Trims and pads the audio to the desired output duration
    If the audio is already of the desired duration, make a symlink
    """
    assert not Path(out_file).exists(), "File already exists"
    # If the audio is of the desired duration
    # move to the else part where we will just create a symlink
    if get_audio_stats(in_file)["duration"] != duration:
        # Trim and pad the audio
        try:
            _ = (
                ffmpeg.input(in_file)
                .audio.filter("apad", whole_dur=duration)  # Pad
                .filter("atrim", end=duration)  # Trim
                .output(out_file, f="wav", acodec="pcm_f32le", ac=1)
                .run(quiet=True)
            )
        except ffmpeg.Error as e:
            print(
                "Please check the console output for ffmpeg to debug the "
                "error in trim and pad wav: ",
                f"Error: {e}",
            )
            raise
        # Check if the file has been converted to the desired duration
        new_dur = get_audio_stats(out_file)["duration"]
        assert (
            new_dur == duration
        ), f"The new file is {new_dur} secs while expected is {duration} secs"
    else:
        Path(out_file).symlink_to(Path(in_file).absolute())


def resample_wav(in_file: str, out_file: str, out_sr: int) -> None:
    """
    Resample a wave file using SoX high quality mode
    If the audio is already of the desired sample rate, make a symlink
    """
    assert not Path(out_file).exists()
    # If the audio is of the desired sample rate
    # move to the else part where we will just create a symlink
    if get_audio_stats(in_file)["sample_rate"] != out_sr:
        try:
            _ = (
                ffmpeg.input(in_file)
                # Use SoX high quality mode
                .filter("aresample", resampler="soxr")
                .output(out_file, ar=out_sr)
                .run(quiet=True)
            )
        except ffmpeg.Error as e:
            print(
                "Please check the console output for ffmpeg to debug the "
                "error in resample wav: ",
                f"Error: {e}",
            )
            raise
        # Check if the file has been converted to the desired sampling rate
        new_sr = get_audio_stats(out_file)["sample_rate"]
        assert (
            new_sr == out_sr
        ), f"The new file is {new_sr} secs while expected is {out_sr} secs"
    else:
        # If the audio has the expected sampling rate, make a symlink
        Path(out_file).symlink_to(Path(in_file).absolute())


def get_audio_stats(in_file: Union[str, Path]) -> Union[Dict[str, Any], Any]:
    """Produces summary for a single audio file"""
    try:
        audio_stream = ffmpeg.probe(in_file, select_streams="a")["streams"][0]
        audio_stats = {
            "sample_rate": int(audio_stream["sample_rate"]),
            "samples": int(audio_stream["duration_ts"]),
            "mono": audio_stream["channels"] == 1,
            "duration": float(audio_stream["duration"]),
            "ext": Path(in_file).suffix,
        }
    except (ffmpeg.Error, KeyError):
        # Skipping audio file for stats calculation.
        return None
    return audio_stats


def get_audio_dir_stats(
    in_dir: Union[str, Path], out_file: str, exts: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Produce summary by recursively searching a directory for wav files"""
    MAX = 1000

    if exts is None:
        exts = [".wav", ".mp3", ".ogg", ".webm"]

    # Get all the audio files
    audio_paths = list(
        filter(
            lambda audio_path: audio_path.suffix.lower()
            in map(str.lower, exts),  # type: ignore
            Path(in_dir).absolute().rglob("*"),
        )
    )
    rng = random.Random(0)
    rng.shuffle(audio_paths)

    orig_count = len(audio_paths)
    audio_paths = audio_paths[:MAX]

    # Count the number of successful and failed statistics extraction to be
    # added the output stats file
    success_counter: Dict[str, int] = defaultdict(int)
    failure_counter: Dict[str, int] = defaultdict(int)

    # Iterate and get the statistics for each audio
    audio_dir_stats = []
    for audio_path in tqdm(audio_paths):
        audio_stats = get_audio_stats(audio_path)
        if audio_stats is not None:
            audio_dir_stats.append(audio_stats)
            success_counter[audio_path.suffix] += 1
        else:
            # update the failed counter if the extraction was not
            # succesful
            failure_counter[audio_path.suffix] += 1

    assert audio_dir_stats, "Stats was not calculated for any audio file. Please Check"
    " the formats of the audio file"
    durations = [stats["duration"] for stats in audio_dir_stats]
    unique_sample_rates = dict(
        Counter([stats["sample_rate"] for stats in audio_dir_stats])
    )
    mono_audio_count = sum(stats["mono"] for stats in audio_dir_stats)

    summary_stats: Dict[str, Any] = {"count": orig_count}
    if len(audio_paths) != orig_count:
        summary_stats.update({"count_sample": len(audio_paths)})

    duration = {
        "mean": round(np.mean(durations), 2),
        "var": round(np.var(durations), 2),
    }
    if np.var(durations) > 0.0:
        duration.update(
            {
                "min": round(np.min(durations), 2),
                "max": round(np.max(durations), 2),
                # Percentile duration of the audio
                **{
                    f"{p}th": round(np.percentile(durations, p), 2)
                    for p in [10, 25, 50, 75, 90]
                },
            }
        )
    summary_stats.update(
        {
            "duration": duration,
            "samplerates": unique_sample_rates,
            "count_mono": mono_audio_count,
            # Count of no of success and failure for audio summary extraction for each
            # extension type
            "summary": {
                "successfully_extracted": dict(success_counter),
                "failed_to_extract": dict(failure_counter),
            },
        }
    )

    json.dump(summary_stats, open(out_file, "w"), indent=True)
    return summary_stats
