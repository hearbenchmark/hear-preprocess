"""
Audio utility functions for evaluation task preparation
"""

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Optional, Union, Dict

import numpy as np
import ffmpeg
from tqdm import tqdm


def mono_wav_and_fix_duration(in_file: str, out_file: str, duration: float):
    """
    Convert to WAV file and trim to be equal to or less than a specific length

    If the audio is already mono and has the required duration, the step is skipped
    and a symlink is created to the original file
    """
    # Get the audio stats for the audio file
    audio_stats = get_audio_stats(in_file)
    # Get the filters to apply to the audio
    duration_filter_flag = audio_stats["duration"] != duration
    mono_filter_flag = not audio_stats["mono"]
    ext_filter_flag = audio_stats["ext"].lower() != ".wav"

    # If the audio has the desired duration and is already mono .wav all the flags will
    # be false, then we will move to the else part where we will just create a symlink
    if duration_filter_flag or mono_filter_flag or ext_filter_flag:
        # create a ffmpeg command chain to take the input
        cmd_chain = ffmpeg.input(in_file).audio
        # Add duration commands if required
        if duration_filter_flag:
            cmd_chain = cmd_chain.filter("apad", whole_dur=duration).filter(
                "atrim", end=duration
            )
        # Add mono flag if required
        if mono_filter_flag:
            cmd_chain = cmd_chain.output(out_file, f="wav", acodec="pcm_f32le", ac=1)
        else:
            cmd_chain = cmd_chain.output(out_file, f="wav", acodec="pcm_f32le")

        try:
            _ = cmd_chain.overwrite_output().run(quiet=True)
        except ffmpeg.Error as e:
            print(
                "Please check the console output for ffmpeg to debug the "
                "error in mono wav and fix duration: ",
                f"Error: {e}",
            )
            raise
    else:
        # If the file already has the desired duration and is mono wav, make a symlink
        if Path(out_file).exists():
            Path(out_file).unlink()
        Path(out_file).symlink_to(Path(in_file).absolute())


def resample_wav(in_file: str, out_file: str, out_sr: int):
    """
    Resample a wave file using SoX high quality mode
    """
    # Get the audio stats to get the sampling rate of the audio
    audio_stats = get_audio_stats(in_file)
    # If the sampling rate is the same as that of the original file, skip resampling
    if audio_stats["sample_rate"] != out_sr:
        try:
            _ = (
                ffmpeg.input(in_file)
                .filter("aresample", resampler="soxr")
                .output(out_file, ar=out_sr)
                .overwrite_output()
                .run(quiet=True)
            )
        except ffmpeg.Error as e:
            print(
                "Please check the console output for ffmpeg to debug the "
                "error in resample_wav: ",
                f"Error: {e}",
            )
            raise
    else:
        # If the audio has the expected sampling rate, make a synlink
        if Path(out_file).exists():
            Path(out_file).unlink()
        Path(out_file).symlink_to(Path(in_file).absolute())


def get_audio_stats(in_file: Union[str, Path]):
    try:
        audio_stream = ffmpeg.probe(in_file, select_streams="a")["streams"][0]
        audio_stats = {
            "sample_rate": int(audio_stream["sample_rate"]),
            "samples": int(audio_stream["duration_ts"]),
            "mono": True if audio_stream["channels"] == 1 else False,
            "duration": float(audio_stream["duration"]),
            "ext": Path(in_file).suffix,
        }
    except ffmpeg.Error as e:
        print(
            "Skipping audio file for stats calculation. "
            "Audio path: {file_path}"
            f"Error: {e}"
        )
        audio_stats = {}
    return audio_stats


def get_audio_dir_stats(
    in_dir: Union[str, Path], out_file: str, exts: Optional[List[str]] = None
):
    """Produce summary by recursively searching a directory for audio files"""
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

    # Count the number of successful and failed statistics extraction to be
    # added the output stats file
    success_counter: Dict[str, int] = defaultdict(int)
    failure_counter: Dict[str, int] = defaultdict(int)

    # Iterate and get the statistics for each audio
    audio_dir_stats = []
    for audio_path in tqdm(audio_paths):
        audio_stats = get_audio_stats(audio_path)
        if audio_stats:
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

    summary_stats = {
        # Count of no of success and failure for audio summary extraction for each
        # extension type
        "audio_summary_from": {
            "successfully_extracted": success_counter,
            "failed_to_extract": failure_counter,
        },
        "audio_samplerate_count": unique_sample_rates,
        "mono_audio_count": mono_audio_count,
        "audio_mean_dur(sec)": np.mean(durations),
        "audio_median_dur(sec)": np.median(durations),
        # Percentile duration of the audio
        **{
            f"{str(p)}th percentile dur(sec)": np.percentile(durations, p)
            for p in [10, 25, 75, 90]
        },
    }

    json.dump(summary_stats, open(out_file, "w"), indent=True)
