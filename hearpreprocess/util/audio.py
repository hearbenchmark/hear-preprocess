"""
Audio utility functions for evaluation task preparation
"""

import json
import os
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import soundfile as sf
from tqdm import tqdm


def mono_wav_and_fix_duration(in_file: str, out_file: str, duration: float):
    """
    Convert to WAV file and trim to be equal to or less than a specific length
    """
    ret = subprocess.call(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(in_file),
            "-filter_complex",
            f"apad=whole_dur={duration},atrim=end={duration}",
            "-ac",
            "1",
            "-c:a",
            "pcm_f32le",
            str(out_file),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Make sure the return code is 0 and the command was successful.
    assert ret == 0, f"ret = {ret}"


def convert_to_mono_wav(in_file: str, out_file: str):
    devnull = open(os.devnull, "w")
    # If we knew the sample rate, we could also pad/trim the audio file now, e.g.:
    # ffmpeg -i test.webm -filter_complex \
    #    apad=whole_len=44100,atrim=end_sample=44100 \
    #    -ac 1 -c:a pcm_f32le ./test.wav
    # print(" ".join(["ffmpeg", "-y", "-i", in_file,
    #    "-ac", "1", "-c:a", "pcm_f32le", out_file]))
    ret = subprocess.call(
        ["ffmpeg", "-y", "-i", in_file, "-ac", "1", "-c:a", "pcm_f32le", out_file],
        stdout=devnull,
        stderr=devnull,
    )
    # Make sure the return code is 0 and the command was successful.
    assert ret == 0


def resample_wav(in_file: str, out_file: str, out_sr: int):
    """
    Resample a wave file using SoX high quality mode
    """
    ret = subprocess.call(
        [
            "ffmpeg",
            "-y",
            "-i",
            in_file,
            # "-af",
            # "aresample=resampler=soxr",
            "-ar",
            str(out_sr),
            out_file,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Make sure the return code is 0 and the command was successful.
    assert ret == 0


def get_audio_stats(in_file: Union[str, Path]):
    try:
        audio_stream = ffmpeg.probe(in_file)["streams"][0]
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
            "Error: {e}"
        )
        audio_stats = {}
    return audio_stats


def get_audio_dir_stats(
    in_dir: Union[str, Path], out_file: str, exts: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Produce summary by recursively searching a directory for wav files"""
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
    success_counter = defaultdict(int)
    failure_counter = defaultdict(int)

    # Iterate and get the statistics for each audio
    audio_dir_stats = []
    for audio_path in tqdm(audio_paths):
        try:
            audio_dir_stats.append(get_audio_stats(audio_path))
            success_counter[audio_path.suffix] += 1
        except:
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
            "failure": failure_counter,
        },
        "audio_samplerate_count": unique_sample_rates,
        "mono_audio_count": mono_audio_count,
        "audio_mean_dur(sec)": np.mean(durations),
        "audio_median_dur(sec)": np.median(durations),
        # Percentile duration of the audio
        **{
            f"{str(p)}th percentile dur(sec)": np.percentile(durations, p)
            for p in [10, 25, 75, 90]
        }
    )
    json.dump(stats, open(out_file, "w"), indent=True)
    return stats
