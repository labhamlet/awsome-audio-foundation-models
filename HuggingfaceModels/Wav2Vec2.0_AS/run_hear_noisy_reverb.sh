#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=MWMAE
#SBATCH --ntasks=1
#SBATCH --exclude=gcn118
#SBATCH --time=01:00:00
#SBATCH --output=hear_noisy_reverb/slurm_output_%A_%a.out
#SBATCH --array=0-5


cd ~/phd/awsome-audio-foundation-models/HuggingfaceModels/Wav2Vec2.0
module load 2023
module load Anaconda3/2023.07-2
source activate hear-other-models-eval
cd listen-eval-kit


# use fast grid.
# dcase2016_task2-hear2021-full


# Define arrays
task_dirs=(
    # libricount-v1.0.0-hear2021-full
    # mridangam_stroke-v1.5-full
    # mridangam_tonic-v1.5-full
    # nsynth_pitch-v2.2.3-5h
    # vox_lingua_top10-hear2021-full
    # speech_commands-v0.0.2-5h
    dcase2016_task2-hear2021-full
)

task_names=(
    -5
    0
    5
    10
    15
    20
)

num_task_names=${#task_names[@]}   # 6
num_task_dirs=${#task_dirs[@]}     # 5

# Calculate indices
task_name_idx=$((SLURM_ARRAY_TASK_ID % num_task_names))
task_dir_idx=$((SLURM_ARRAY_TASK_ID / num_task_names % num_task_dirs))

task_name=${task_names[$task_name_idx]}
task_dir=${task_dirs[$task_dir_idx]}

tasks_dir="/projects/prjs1338/create_noisy_reverb_hear/outputs/$task_dir"
embeddings_dir="/projects/prjs1338/Wav2Vec2.0Large/$task_dir"
score_dir="noisy_reverb_hear_wav2vec2.0_large/$task_dir"

model_name=hear_configs.wav2vec2_large

python3 -m heareval.embeddings.runner "$model_name" --tasks-dir $tasks_dir --task "$task_name" --embeddings-dir $embeddings_dir
python3 -m heareval.predictions.runner $embeddings_dir/$model_name/$task_name --grid fast

mkdir -p /projects/0/prjs1338/$score_dir/$model_name/$task_name

mv $embeddings_dir/$model_name/$task_name/test.predicted-scores.json  /projects/0/prjs1338/$score_dir/$model_name/$task_name
mv $embeddings_dir/$model_name/$task_name/*predictions.pkl /projects/0/prjs1338/$score_dir/$model_name/$task_name
mv $embeddings_dir/$model_name/$task_name/*embeddings.npy /projects/0/prjs1338/$score_dir/$model_name/$task_name

rm -r -d -f $embeddings_dir/$model_name/$task_name