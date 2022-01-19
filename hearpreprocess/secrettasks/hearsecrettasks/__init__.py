from hearpreprocess import pipeline, tfds_pipeline
from importlib import import_module
from itertools import chain
from typing import List, Dict, Any

# All the secrettasks
task_names: List[str] = [
    "coughvid",
    "maestro",
    "tfds_crema_d",
    "esc50",
    "tfds_gtzan",
    "tfds_gtzan_music_speech",
    "beijing_opera",
    "libricount",
    "vox_lingua_top10",
    "fsd50k",
    "mridangam_stroke",
    "mridangam_tonic",
    "vocal_imitation",
    "gunshot_triangulation",
    "beehive_states_fold0",
    "beehive_states_fold1",
]

# Import modules corresponding to the task names.
task_modules: Dict = {
    name: import_module(f".{name}", "hearpreprocess.secrettasks.hearsecrettasks")
    for name in task_names
}

# Create the tasks to be imported into the hearpreprocess runner
tasks: Dict[str, List] = {name: [task_modules[name]] for name in task_names}
tasks["all-secret"] = list(chain.from_iterable(tasks.values()))

sampler_config: Dict[str, Any] = {
    # Non-TFDS task sampler configuration
    **{
        name: {
            "task_config": task_modules[name].generic_task_config,
            "audio_sample_size": 20,
            "necessary_keys": [],
            "get_download_and_extract_tasks": pipeline.get_download_and_extract_tasks,
        }
        for name in [
            "esc50",
            "beijing_opera",
            "libricount",
            "vox_lingua_top10",
            "mridangam_stroke",
            "mridangam_tonic",
            "vocal_imitation",
            "beehive_states_fold0",
            "beehive_states_fold1",
        ]
    },
    # TFDS tasks sampler configuration
    **{
        name: {
            "task_config": task_modules[name].generic_task_config,
            "audio_sample_size": 20,
            "necessary_keys": [],
            "get_download_and_extract_tasks": tfds_pipeline.get_download_and_extract_tasks_tfds,  # noqa: E501
        }
        for name in ["tfds_crema_d", "tfds_gtzan", "tfds_gtzan_music_speech"]
    },
    # Sampler configuration should be specific for tasks which require
    # specific files ( from validation or train ) to be present in the
    # small dataset ( by including them in the necessary_key, see below).
    # download and extract tasks can also be task specific and can
    # necessitate sampler definition
    "coughvid": {
        "task_config": task_modules["coughvid"].generic_task_config,
        "audio_sample_size": 30,
        # Since coughvid has a condition of selecting audio files with 0.8 cough
        # detected probability, 5 strings from 5 valid audios are added to the
        # necessary keys. This will ensure these files are added to the small dataset
        "necessary_keys": [
            "5bfdff3895d3.webm",
            "dc04f2bd8cd4.webm",
            "15273e872dc3.webm",
            "12aa16912ea9.webm",
            "a8d14bc5a543.ogg",
        ],
        "get_download_and_extract_tasks": pipeline.get_download_and_extract_tasks,
    },
    "maestro": {
        "task_config": task_modules["maestro"].generic_task_config,
        "audio_sample_size": 7,
        "necessary_keys": [
            # One example from each of the set
            "MIDI-Unprocessed_17_R1_2006_01-06_ORIG_MID--AUDIO_17_R1_2006_04_Track04_wav.wav",  # valid # noqa: E501
            "MIDI-Unprocessed_02_R1_2009_03-06_ORIG_MID--AUDIO_02_R1_2009_02_R1_2009_04_WAV.wav",  # test # noqa: E501
            "MIDI-Unprocessed_Chamber3_MID--AUDIO_10_R3_2018_wav--1.wav",  # train
        ],
        "get_download_and_extract_tasks": pipeline.get_download_and_extract_tasks,
    },
    "fsd50k": {
        "task_config": task_modules["fsd50k"].generic_task_config,
        "audio_sample_size": 40,
        "necessary_keys": [],
        "get_download_and_extract_tasks": task_modules[
            "fsd50k"
        ].get_download_and_extract_tasks,
    },
    "gunshot_triangulation": {
        "task_config": task_modules["gunshot_triangulation"].generic_task_config,
        "audio_sample_size": 40,
        "necessary_keys": [],
        "get_download_and_extract_tasks": task_modules[
            "gunshot_triangulation"
        ].get_download_and_extract_tasks,
    },
}
