#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=MWMAE
#SBATCH --ntasks=1
#SBATCH --exclude=gcn118
#SBATCH --time=05:00:00
#SBATCH --output=hear_real_world/slurm_output_%A_%a.out
#SBATCH --array=1


cd ~/phd/awsome-audio-foundation-models/HuggingfaceModels/Wav2Vec2.0
module load 2023
module load Anaconda3/2023.07-2
source activate hear-other-models-eval
cd listen-eval-kit



task_names=(
    tau2021-v1.0.0-full
    starss23-v1.0.0-full
)

# Define arrays
task_dir=/projects/0/prjs1338/tasks_real_world

# FIXED: Swapped the index calculations
task_name_idx=$((SLURM_ARRAY_TASK_ID))
task_name=${task_names[$task_name_idx]}

embeddings_dir="/projects/prjs1338/Wav2Vec2.0Robust/$task_dir"
score_dir="noisy_reverb_hear_wav2vec2.0_robust/$task_dir"

model_name=hear_configs.wav2vec2_robust

python3 -m heareval.embeddings.runner "$model_name" --tasks-dir $task_dir --task "$task_name" --embeddings-dir $embeddings_dir
python3 -m heareval.predictions.runner $embeddings_dir/$model_name/$task_name --grid fast

# mkdir -p /projects/0/prjs1338/$score_dir/$model_name/$task_name

# mv $embeddings_dir/$model_name/$task_name/test.predicted-scores.json  /projects/0/prjs1338/$score_dir/$model_name/$task_name
# mv $embeddings_dir/$model_name/$task_name/*predictions.pkl /projects/0/prjs1338/$score_dir/$model_name/$task_name
# mv $embeddings_dir/$model_name/$task_name/*embeddings.npy /projects/0/prjs1338/$score_dir/$model_name/$task_name

# rm -r -d -f $embeddings_dir/$model_name/$task_name