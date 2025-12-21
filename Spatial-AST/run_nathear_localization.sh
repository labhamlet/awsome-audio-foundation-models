#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=Localize
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=01:00:00
#SBATCH --output=slurm_output_%A_%a.out
#SBATCH --array=0-5


task_names=(
    esc50_5_10
    esc50_10_20
    esc50_20_40
    sc_5_10
    sc_10_20
    sc_20_40
)

cd ~/phd/hear-freq-models/Spatial-AST
module load 2023
module load Anaconda3/2023.07-2
source activate spatial-ast
cd listen-eval-kit

embeddings_dir=/projects/0/prjs1338/LocalizationEmbeddings
score_dir=hear_scores_localization_snr_spatial_ast
tasks_dir=/projects/0/prjs1338/tasks_spatial_binaural_snr

model_name=hear_configs.spatial_ast
model_options="{\"mode\": \"localization\"}"
task_name=${task_names[$SLURM_ARRAY_TASK_ID]}


python3 -m heareval.embeddings.runner "$model_name" --tasks-dir $tasks_dir --task "$task_name" --embeddings-dir $embeddings_dir --model-options "$model_options"
python3 -m heareval.predictions.runner $embeddings_dir/$model_name-mode=localization/$task_name --localization cartesian-regression

mkdir -p /projects/0/prjs1338/$score_dir/$model_name-mode=localization/$task_name


mv $embeddings_dir/$model_name-mode=localization/$task_name/test.predicted-scores-localization.json  /projects/0/prjs1338/$score_dir/$model_name-mode=localization/$task_name
mv $embeddings_dir/$model_name-mode=localization/$task_name/*.pkl /projects/0/prjs1338/$score_dir/$model_name-mode=localization/$task_name
mv $embeddings_dir/$model_name-mode=localization/$task_name/*.npy /projects/0/prjs1338/$score_dir/$model_name-mode=localization/$task_name

rm -r -d -f $embeddings_dir/$model_name-mode=localization/$task_name