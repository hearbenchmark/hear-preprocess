#!/usr/bin/env python3
"""
Pre-processing pipeline for Maestro v3.0.0.
"""

import logging
from typing import Any, Dict, List

import luigi
import pandas as pd
from note_seq.midi_io import midi_file_to_note_sequence
from note_seq.sequences_lib import apply_sustain_control_changes
from tqdm import tqdm

import hearpreprocess.pipeline as pipeline
import hearpreprocess.util.audio as audio_utils

# from hearpreprocess.pipeline import (
#    TEST_PERCENTAGE,
#    TRAIN_PERCENTAGE,
#    TRAINVAL_PERCENTAGE,
#    VALIDATION_PERCENTAGE,
# )
from hearpreprocess.util.luigi import diagnostics

logger = logging.getLogger("luigi-interface")

generic_task_config = {
    "task_name": "maestro",
    "version": "v3.0.0",
    "embedding_type": "event",
    "prediction_type": "multilabel",
    "split_mode": "new_split_kfold",
    "nfolds": 5,
    # |train_test_valid| - 75th %tile: 686.33, 90th %tile: |1150.26|, max: 2628.99
    # Set to 120 seconds like dcase
    "sample_duration": 120.0,
    # Evaluation strategies adopted from
    # https://arxiv.org/pdf/1710.11153.pdf
    "evaluation": ["event_onset_50ms_fms", "event_onset_offset_50ms_20perc_fms"],
    # sed_eval is too slow, so we use the loss for early stopping
    # and grid point (hyperparameter) selection
    "use_scoring_for_early_stopping": False,
    # Skipping certain evaluations, which are standard in literature-
    # - Onset + Offset + Velocity: because this task is designed for simple multiclass
    #   prediction
    # - frame-based 10ms f1 - hop-size becomes be too large for the model
    "evaluation_params": {
        "event_postprocessing_grid": {
            # In preliminary tests, these smoothing parameters worked
            # well at optimizing onset fms.
            "median_filter_ms": [150],
            "min_duration": [50],
        },
        # Speed up training because each epoch is typically quite large,
        # 300K instances or more, can take 1 minute with torchcrepe
        # and longer with more fine-grained time resolution models
        "task_specific_param_grid": {
            "max_epochs": [50],
            "check_val_every_n_epoch": [1],
        },
        # Only needed with sed_eval
        # "task_specific_param_grid": {"check_val_every_n_epoch": [50]},
    },
    "download_urls": [
        {
            "split": "train_test_valid",
            "url": "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip",  # noqa: E501
            "md5": "41941abdcd786c8066d532002e3b79b9",
        }
    ],
    "default_mode": "5h",
    "modes": {
        "5h": {
            # Total duration (This depends on the sample duration)
            # In seconds |train_test_valid|: 1276*120.0 seconds
            # In hours |train_test_valid|: 42.53 hours
            # 3 train folds + 1 valid fold = 5h
            "max_task_duration_by_fold": 3600 * 5 / 4,
            # "max_task_duration_by_split": {
            #     "train": 3600 * 5 * TRAIN_PERCENTAGE / TRAINVAL_PERCENTAGE,
            #     "valid": 3600 * 5 * VALIDATION_PERCENTAGE / TRAINVAL_PERCENTAGE,
            #     "test": 3600 * 5 * TEST_PERCENTAGE / TRAINVAL_PERCENTAGE,
            # }
        },
        "small": {
            "download_urls": [
                {
                    "split": "train_test_valid",
                    "url": "https://github.com/kmarufraj/s-task/raw/main/maestro-v3.0.0-small.zip",  # noqa: E501
                    "md5": "852a3e0436d0f9d12e6566f9c61ed094",
                }
            ],
            "sample_duration": 5.0,
            # There are only 4 files in Maestro small
            "nfolds": 4,
            "max_task_duration_by_fold": None,
            #            "max_task_duration_by_split": {
            #                "train": None,
            #                "valid": None,
            #                "test": None,
            #            },
        },
    },
}


class ExtractMetadata(pipeline.ExtractMetadata):
    train_test_valid = luigi.TaskParameter()

    def requires(self):
        return {"train_test_valid": self.train_test_valid}

    @staticmethod
    def get_split(maestro_split: str):
        # Map the maestro split name to the hear split name
        return {"validation": "valid", "train": "train", "test": "test"}[maestro_split]

    def get_requires_metadata(self, split: str) -> pd.DataFrame:
        logger.info(f"Preparing metadata for {split}")

        # base path of the actual data with respect to the download tasks working dir
        base_path = self.requires()[split].workdir.joinpath(split, "maestro-v3.0.0")

        # Read the source metadata to get the files, their duration, and the
        # corresponding midi files to build the preprocessed data
        source_metadata_path = base_path.joinpath("maestro-v3.0.0.csv")
        source_metadata = pd.read_csv(source_metadata_path)
        # flag indicating if the task is running for a small version of the data
        # This is required to skip some assertions which might not be true for
        # the small data (Note - The small data is only for integration testing.
        # The assertions will still be done to ensure quality of the full dataset)
        is_small_version: bool = self.task_config["mode"] == "small"

        duration_stats = pd.DataFrame(
            columns=[
                "audio_path",
                "specified_duration",
                "midi_duration",
                "actual_duration",
            ],
        )
        metadatas: List[pd.DataFrame] = []
        # Iterate over the source metadata and prepare the task metadata
        for _, row in tqdm(list(source_metadata.iterrows())):
            audio_path = base_path.joinpath(row["audio_filename"])
            if not audio_path.exists():
                if is_small_version:
                    # If the dataset is the small version, this might happen as
                    # some audio has been removed form the original data to make it
                    # small
                    continue
                else:
                    # However, raise an error if this is the actual version of
                    # the data and files should not be missing in it
                    raise FileNotFoundError(f"{str(audio_path)} does not exists")
            # Get the midi and the audio file path from the source metadata
            midi_path = base_path.joinpath(row["midi_filename"])

            # 1. The note sequence is extracted from the midi file
            # with midi_file_to_note_sequence
            # From https://github.com/magenta/magenta/blob/main/magenta/models/onsets_frames_transcription/audio_label_data_utils.py # noqa: E501
            # Also referred https://github.com/BShakhovsky/PolyphonicPianoTranscription/blob/master/1%20Datasets%20Preparation.ipynb # noqa: E501

            # 2. followed by apply_sustain_control_changes.
            # From https://arxiv.org/pdf/1710.11153.pdf
            # Page 2, Section 2 "Dataset and Metrics":
            # ... we first translate “sustain pedal” control changes
            # into longer note durations. If a note is active when
            # sustain goes on, that note will be extended until either
            # sustain goes off or the same note is played again.

            # 3. As used in sequence_to_roll function in -
            # https://github.com/magenta/magenta/blob/85ef5267513f62f4a40b01b2a1ee488f90f64a13/magenta/models/onsets_frames_transcription/data.py#L232 # noqa: E501
            # The pitch is restricted in a range of 21 - 108 (Musical range of A0-C8)
            # onset_length_ms and offset_length_ms defines the duration
            # (in milliseconds) that the onsets and offsets should last
            # for when converting to a piano roll format (which is a frame-based
            # representation of note onsets and offsets) used in the original
            # Maestro work for training and frame-based evaluations. This
            # preprocessing retains the format as a sequence as opposed to a
            # piano roll, so these options are not applicable here.

            note_seq_raw = midi_file_to_note_sequence(midi_path)
            note_seq = apply_sustain_control_changes(note_seq_raw)
            # Each event has a start time, end time and label which is the pitch
            metadata = pd.DataFrame(
                [
                    # Start time and end time is converted to ms
                    (note.start_time * 1000.0, note.end_time * 1000.0, note.pitch)
                    for note in note_seq.notes
                    if 21 <= note.pitch <= 108
                ],
                columns=["start", "end", "label"],
            ).assign(relpath=str(audio_path), split=self.get_split(row["split"]))
            metadatas.append(metadata)

            duration_stat = {
                "audio_path": str(audio_path),
                # Get the duration specified in the metadata for the audio
                "specified_duration": float(row["duration"]),
                # Get the total time from the note sequence. This is the
                # end time of the last note from the midi file
                "midi_duration": note_seq_raw.total_time,
                # Get the duration of audio with ffmpeg
                "actual_duration": audio_utils.get_audio_stats(audio_path)["duration"],
            }
            duration_stats = duration_stats.append(duration_stat, ignore_index=True)

        duration_stats = duration_stats.assign(
            delta_spec_midi=duration_stats.specified_duration
            - duration_stats.midi_duration,
            delta_actual_midi=duration_stats.actual_duration
            - duration_stats.midi_duration,
            delta_actual_spec=duration_stats.actual_duration
            - duration_stats.specified_duration,
        )
        # Save the duration path for reference
        duration_stats.to_csv(self.workdir.joinpath("duration_stats.csv"), index=False)
        diagnostics.info(
            f"{self.longname} Duration stats (duration_stats.csv) with deltas for "
            "the durations has been saved in the ExtractMetadata Task folder"
        )

        # Assert if the average difference between the specified duration(in metadata)
        # and the end time of the last event in midi are within 50 ms.
        # This helps in verifying if the midi is getting the correct tempo for the audio
        avg_delta_spec_midi = duration_stats.delta_spec_midi.abs().mean()
        diagnostics.info(
            f"{self.longname} Avg. diff between Metadata Specified Duration "
            f"and End time of last MIDI event: {avg_delta_spec_midi}"
        )
        assert (
            avg_delta_spec_midi < 0.05
        ), f"avg_delta_spec_midi is {avg_delta_spec_midi} seconds"
        # Assert the same above fact with the actual duration calculated with ffmpeg
        # Do this only for the larger dataset since the audios in the small version
        # might have been trimmed off before the last midi event (midi_duration)
        # Also the audio might end after the last midi event, so the time
        # difference can be large. The threshold is set to 5 seconds for this
        if not is_small_version:
            avg_delta_actual_midi = duration_stats.delta_actual_midi.abs().mean()
            diagnostics.info(
                f"{self.longname} Avg. diff between Actual audio duration "
                f"and End time of last MIDI event: {avg_delta_actual_midi}"
            )
            assert (
                avg_delta_actual_midi < 5.0
            ), f"avg_delta_actual_midi is {avg_delta_actual_midi} seconds"

            # Also check if the specified duration and the actual duration (from ffmpeg)
            # are close
            avg_delta_actual_spec = duration_stats.delta_actual_spec.abs().mean()
            diagnostics.info(
                f"{self.longname} Avg. diff between Actual audio duration "
                f"and Metadata Specified Duration: {avg_delta_actual_spec}"
            )
            assert (
                avg_delta_actual_spec < 5.0
            ), f"avg_delta_actual_spec is {avg_delta_actual_spec} seconds"

        metadata = pd.concat(metadatas)
        return metadata


def extract_metadata_task(task_config: Dict[str, Any]) -> pipeline.ExtractMetadata:
    # Build the dataset pipeline with the custom metadata configuration task
    download_tasks = pipeline.get_download_and_extract_tasks(task_config)

    return ExtractMetadata(
        outfile="process_metadata.csv", task_config=task_config, **download_tasks
    )
