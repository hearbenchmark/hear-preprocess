"""
Audio utility functions for evaluation task preparation
"""

import json
import os
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Optional, Union

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
    """Get statistics for audio files in any format(supported by ffmpeg)"""
    ret = subprocess.check_output(
        ["ffmpeg", "-i", in_file, "-f", "null", "-"], stderr=subprocess.STDOUT
    ).decode("utf-8")
    sample_rate = int(re.findall("([0-9]+) Hz", ret)[0])

    # Get the duration from the returned string with regex.
    h, m, s = re.findall(" time=([0-9:.]+)", ret)[0].split(":")
    duration = int(h) * 3600 + int(m) * 60 + float(s)

    # Get the Stream
    mono_flag = "mono" in re.findall("Stream (.+)", ret)[0]

    return {
        "samples": sample_rate * duration,
        "sample_rate": sample_rate,
        "duration": duration,
    }


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

    durations = [stats["duration"] for stats in audio_dir_stats]
    unique_sample_rates = dict(
        Counter([stats["sample_rate"] for stats in audio_dir_stats])
    )

    summary_stats = {
        # Count of no of success and failure for audio summary extraction for each
        # extension type
        "audio_summary_from": {
            "successfully_extracted": success_counter,
            "failure": failure_counter,
        },
        "audio_samplerate_count": unique_sample_rates,
        "audio_mean_dur(sec)": np.mean(durations),
        "audio_median_dur(sec)": np.median(durations),
        # Percentile duration of the audio
        **{
            f"{str(p)}th percentile dur(sec)": np.percentile(durations, p)
            for p in [10, 25, 75, 90]
        },
    }

    json.dump(summary_stats, open(out_file, "w"), indent=True)
