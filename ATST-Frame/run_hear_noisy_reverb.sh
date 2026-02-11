#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=MWMAE
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --exclude=gcn118
#SBATCH --time=02:00:00
#SBATCH --output=hear/slurm_output_%A_%a.out
#SBATCH --array=0-41



cd ~/phd/awsome-audio-foundation-models/ATST-Frame
module load 2023
module load Anaconda3/2023.07-2
source activate atst-eval
cd listen-eval-kit


# Too big to do it in one go
# speech_commands-v0.0.2-5h
# Use fast grid
# dcase2016_task2-hear2021-full
# speech_commands-v0.0.2-5h


# Define arrays
task_dirs=(
    beijing_opera-v1.0-hear2021-full
    libricount-v1.0.0-hear2021-full
    mridangam_stroke-v1.5-full
    mridangam_tonic-v1.5-full
    tfds_crema_d-1.0.0-full
    nsynth_pitch-v2.2.3-5h
    vox_lingua_top10-hear2021-full
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

embeddings_dir="/projects/prjs1338/ATSTFrameEmbeddings/$task_dir"
score_dir="noisy_reverb_hear_atst_frame/$task_dir"

tasks_dir="/projects/prjs1338/create_noisy_reverb_hear/outputs/$task_dir"

model_name=hear_configs.atst_frame
model_size=base
model_options="{\"model_size\": \"$model_size\"}"

python3 -m heareval.embeddings.runner "$model_name" --tasks-dir $tasks_dir --task "$task_name" --embeddings-dir $embeddings_dir --model-options "$model_options"
source deactivate
source activate hear-lightning-eval
python3 -m heareval.predictions.runner $embeddings_dir/$model_name-model-size=$model_size/$task_name

mkdir -p /projects/0/prjs1338/$score_dir/$model_name-model-size=$model_size/$task_name

mv $embeddings_dir/$model_name-model-size=$model_size/$task_name/test.predicted-scores.json  /projects/0/prjs1338/$score_dir/$model_name-model-size=$model_size/$task_name
mv $embeddings_dir/$model_name-model-size=$model_size/$task_name/*predictions.pkl /projects/0/prjs1338/$score_dir/$model_name-model-size=$model_size/$task_name
mv $embeddings_dir/$model_name-model-size=$model_size/$task_name/*embeddings.npy /projects/0/prjs1338/$score_dir/$model_name-model-size=$model_size/$task_name

rm -r -d -f $embeddings_dir/$model_name-model-size=$model_size/$task_name