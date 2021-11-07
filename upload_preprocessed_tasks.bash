# This Script
# * clones the open task repo
# * runs the pipeline to generate the preprocessed open tasks
# * commits, adds and pushes the latest version of these tasks to the repo
# * deletes the repo
# * cleans up the pipeline outputs
git clone git@github.com:neuralaudio/hear2021-open-tasks-downsampled.git
python -m hearpreprocess.runner speech_commands --mode small
python -m hearpreprocess.runner nsynth_pitch --mode small
python -m hearpreprocess.runner dcase2016_task2 --mode small
cp hear-LATEST-speech_commands-v0.0.2-small-44100.tar.gz ./hear2021-open-tasks-downsampled/preprocessed/
cp hear-LATEST-nsynth_pitch-v2.2.3-small-44100.tar.gz ./hear2021-open-tasks-downsampled/preprocessed/
cp hear-LATEST-dcase2016_task2-hear2021-small-44100.tar.gz ./hear2021-open-tasks-downsampled/preprocessed/
git -C hear2021-open-tasks-downsampled/ add .
git -C hear2021-open-tasks-downsampled/ commit -m "Update Latest"
git -C hear2021-open-tasks-downsampled/ push
rm -Rf hear2021-open-tasks-downsampled
bash clean.sh