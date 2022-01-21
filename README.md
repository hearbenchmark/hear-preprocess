![HEAR2021](https://neuralaudio.ai/assets/img/hear-header-sponsor.jpg)
# hear-preprocess

Dataset preprocessing code for the HEAR 2021 NeurIPS competition.

Unless you are a HEAR organizer or want to contribute a task,
you won't need this repo. Use
[hear-eval-kit](https://github.com/neuralaudio/hear-eval-kit/) to
evaluate your embedding models on these tasks. 

Pre-processed datasets (@48kHz) for all HEAR 2021 tasks are available on 
[zenodo](https://doi.org/10.5281/zenodo.5802571).

This preprocessing is slow and disk-intensive but safe and careful.

## Cloud Usage

See [hear-eval's
README.spotty](https://github.com/neuralaudio/hear-eval-kit/blob/main/README.spotty.md)
for information on how to use spotty.

## Installation

```
pip3 install hearpreprocess
```

Tested with Python 3.7 and 3.8. Python 3.9 is not officially supported
because pip3 installs are very finicky, but it might work.

## Development

Clone repo:
```
git clone https://github.com/neuralaudio/hear-preprocess
cd hear-preprocess
```

Install in development mode:
```
pip3 install -e ".[dev]"
```

Make sure you have pre-commit hooks installed:
```
pre-commit install
```

Running tests:
```
python3 -m pytest
```

### Preprocessing

You probably don't need to do this unless you are implementing the
HEAR challenge.

If you want to run preprocessing yourself:
* You will need `ffmpeg>=4.2` installed (possibly from conda-forge).
* You will need `soxr` support, which might require package
libsox-fmt-ffmpeg or [installing from
source](https://github.com/neuralaudio/hear-eval-kit/issues/156#issuecomment-893151305).

These Luigi pipelines are used to preprocess the evaluation tasks
into a common format for downstream evaluation.

To run the preprocessing pipeline for all available tasks, with all
available modes for each task:
```
python3 -m hearpreprocess.runner all --mode all
``` 

You can instead just call a specific single task
```
python3 -m hearpreprocess.runner task1 --mode all
```
or specific multiple tasks:
```
python3 -m hearpreprocess.runner task1 task2 --mode all
```

#### Tasks
List of available tasks used in HEAR 2021:

| Task Name                 | Modes        |
|---------------------------|--------------|
| dcase2016_task2         | full       |
| nsynth_pitch            | 5h, 50h  |
| speech_commands         | 5h, full |
| beehive_states_fold0    | 5h, full |
| beehive_states_fold1    | 5h, full |
| beijing_opera           | full       |
| esc50                   | full       |
| fsd50k                  | full       |
| gunshot_triangulation   | full       |
| libricount              | full       |
| maestro                 | 5h         |
| mridangam_stroke        | full       |
| mridangam_tonic         | full       |
| tfds_crema_d            | full       |
| tfds_gtzan              | full       |
| tfds_gtzan_music_speech | full       |
| vocal_imitation         | full       |
| vox_lingua_top10        | full       |




#### Pipelines
Each pipeline will download and preprocess each dataset according
to the following DAG:
* DownloadCorpus
* ExtractArchive
* ExtractMetadata: Create splits over the entire corpus and find
the label metadata for them.
* SubcorpusSplit (subsample each split) => MonoWavSplit => TrimPadSplit => SubcorpusData (symlinks)
* SubcorpusData => {SubcorpusMetadata, ResampleSubcorpus}
* SubcorpusMetadata => MetadataVocabulary
* FinalCombine => TarCorpus => FinalizeCorpus

In terms of sampling:
* We create a 60/20/20 split if train/valid/test does not exist.
* We cap each split at 3/1/1/ hours of audio, defined as
* If further small sampling happens, that chooses a particular
number of audio samples per task.

These commands will download and preprocess the entire dataset. An
intermediary directory defined by the option `luigi-dir`(default
`_workdir`) will be created, and then a final directory defined by
the option `tasks-dir` (default `tasks`) will contain the completed
dataset.

Options:
```
Options:
  --num-workers INTEGER  Number of CPU workers to use when running. If not
                         provided all CPUs are used.
  --sample-rate INTEGER  Perform resampling only to this sample rate. By
                         default we resample to 16000, 22050, 44100, 48000.
  --tmp-dir TEXT         Temporary directory to save all the intermediate
                         tasks (will not be deleted afterwords). (default:
                         _workdir/)
  --tasks-dir TEXT       Directory to save the final task output (default:
                         tasks/)
  --tar-dir TEXT         Directory to save the tar'ed output (default: .)
  --mode TEXT            default, all, or small mode for each task.
  --help                 Show this message and exit.
```

To check the stats of an audio directory:
```
python3 -m hearpreprocess.audio_dir_stats {input folder} {output json file}
```
Stats include: audio_count, audio_samplerate_count, mean meadian
and certain (10, 25, 75, 90) percentile durations.  This is helpful
in getting a quick glance of the audio files in a folder and helps
in decideing the preprocessing configurations.

The pipeline will also generate some stats of the original and
preprocessed data sets, e.g.:
```
speech_commands-v0.0.2/01-ExtractArchive/test_stats.json
speech_commands-v0.0.2/01-ExtractArchive/train_stats.json
speech_commands-v0.0.2/03-ExtractMetadata/labelcount_test.json
speech_commands-v0.0.2/03-ExtractMetadata/labelcount_train.json
speech_commands-v0.0.2/03-ExtractMetadata/labelcount_valid.json
```

### Faster preprocessing, for development

The small flag runs the preprocessing pipeline on a small version
of each dataset stored at [Downsampled HEAR Open
Tasks](https://github.com/neuralaudio/hear2021-open-tasks-downsampled). This
is used for development and continuous integration tests for the
pipeline.

These small versions of the data can be generated
deterministically with the following command:
```
python3 -m hearpreprocess.sampler <taskname>
```

**_NOTE_** : `--mode small` is used to run the task on a
small version of the dataset for development.

### Breaking change for hear-eval

If the open tasks have changed enough to break the downstream CI,
(for example in the heareval repo), the [Preprocessed Downsampled HEAR Open
Tasks](https://github.com/neuralaudio/hear2021-open-tasks-downsampled/tree/main/preprocessed)
should be updated. An example of an obvious breaking changes can be modification of the task configuration.

The version should be bumped up in `hearpreprocess/__init__.py` and the pipeline should
be run for the open tasks with `--mode small` flag

Thereafter, the following command can be used to copy the tarred files produced by running the pipeline for the open tasks to the repo( Please clone the repo )

```
git clone git@github.com:neuralaudio/hear2021-open-tasks-downsampled.git
cp hear-LATEST-speech_commands-v0.0.2-small-44100.tar.gz ./hear2021-open-tasks-downsampled/preprocessed/
cp hear-LATEST-nsynth_pitch-v2.2.3-small-44100.tar.gz ./hear2021-open-tasks-downsampled/preprocessed/
cp hear-LATEST-dcase2016_task2-hear2021-small-44100.tar.gz ./hear2021-open-tasks-downsampled/preprocessed/
cp hear-2021.0.6-speech_commands-v0.0.2-small-44100.tar.gz ./hear2021-open-tasks-downsampled/preprocessed/
cp hear-2021.0.6-nsynth_pitch-v2.2.3-small-44100.tar.gz ./hear2021-open-tasks-downsampled/preprocessed/
cp hear-2021.0.6-dcase2016_task2-hear2021-small-44100.tar.gz ./hear2021-open-tasks-downsampled/preprocessed/
```
