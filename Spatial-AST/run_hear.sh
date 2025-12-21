#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=MWMAE
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --exclude=gcn118
#SBATCH --time=00:20:00
#SBATCH --output=slurm_output_%A_%a.out
#SBATCH --array=0

task_names=(dcase2016_task2-hear2021-full)

cd ~/phd/hear-other-models-niche/Spatial-AST
module load 2023
module load Anaconda3/2023.07-2
source activate spatial-ast
cd listen-eval-kit

embeddings_dir=/projects/0/prjs1338/BaselineEmbeddings
score_dir=hear_scores_baseline
tasks_dir=/projects/0/prjs1338/tasks

model_name=hear_configs.spatial_ast
model_options="{\"mode\": \"classification\"}"
task_name=${task_names[$SLURM_ARRAY_TASK_ID]}

python3 -m heareval.embeddings.runner "$model_name" --tasks-dir $tasks_dir --task "$task_name" --embeddings-dir $embeddings_dir --model-options "$model_options"
python3 -m heareval.predictions.runner $embeddings_dir/$model_name-mode=classification/$task_name --grid fast

mkdir -p /projects/0/prjs1338/$score_dir/$model_name-mode=classification/$task_name

mv $embeddings_dir/$model_name-mode=classification/$task_name/test.predicted-scores.json  /projects/0/prjs1338/$score_dir/$model_name-mode=classification/$task_name
mv $embeddings_dir/$model_name-mode=classification/$task_name/*.pkl /projects/0/prjs1338/$score_dir/$model_name-mode=classification/$task_name
mv $embeddings_dir/$model_name-mode=classification/$task_name/*.npy /projects/0/prjs1338/$score_dir/$model_name-mode=classification/$task_name

rm -r -d -f $embeddings_dir/$model_name-mode=classification/$task_name