#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=Localize
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=01:00:00
#SBATCH --output=slurm_output_%A_%a.out
#SBATCH --array=0-1



task_names=(
    esc50-v2.0.0-full
    speech_commands-v0.0.2-5h
)

cd ~/phd/hear-freq-models/Spatial-AST
module load 2023
module load Anaconda3/2023.07-2
source activate spatial-ast
cd listen-eval-kit

embeddings_dir=/projects/0/prjs1338/T60Embeddings
score_dir=hear_scores_t60_spatial_ast
tasks_dir=/projects/0/prjs1338/tasks_rt60_binaural

model_name=hear_configs.spatial_ast
model_options="{\"mode\": \"classification\"}"
task_name=${task_names[$SLURM_ARRAY_TASK_ID]}


python3 -m heareval.embeddings.runner "$model_name" --tasks-dir $tasks_dir --task "$task_name" --embeddings-dir $embeddings_dir --model-options "$model_options"
python3 -m heareval.predictions.runner $embeddings_dir/$model_name-mode=classification/$task_name --localization regression

mkdir -p /projects/0/prjs1338/$score_dir/$model_name-mode=classification/$task_name


mv $embeddings_dir/$model_name-mode=classification/$task_name/test.predicted-scores-localization.json  /projects/0/prjs1338/$score_dir/$model_name-mode=classification/$task_name
mv $embeddings_dir/$model_name-mode=classification/$task_name/*.pkl /projects/0/prjs1338/$score_dir/$model_name-mode=classification/$task_name
mv $embeddings_dir/$model_name-mode=classification/$task_name/*.npy /projects/0/prjs1338/$score_dir/$model_name-mode=classification/$task_name

rm -r -d -f $embeddings_dir/$model_name-mode=classification/$task_name