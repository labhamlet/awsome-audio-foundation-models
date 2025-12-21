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

export MW_MAE_MODEL_DIR=~/phd/hear-other-models-niche/MWMAE


cd ~/phd/hear-other-models-niche/MWMAE
module load 2023
module load Anaconda3/2023.07-2
source activate mwmae-eval
cd listen-eval-kit


embeddings_dir=/projects/0/prjs1338/BaselineEmbeddings
score_dir=hear_scores_baseline
tasks_dir=/projects/0/prjs1338/tasks

model_name=hear_configs.mwmae_base_200_4x16_384d-8h-4l
task_name=${task_names[$SLURM_ARRAY_TASK_ID]}


python3 -m heareval.embeddings.runner "$model_name" --tasks-dir $tasks_dir --task "$task_name" --embeddings-dir $embeddings_dir
python3 -m heareval.predictions.runner $embeddings_dir/$model_name/$task_name --grid fast

mkdir -p /projects/0/prjs1338/$score_dir/$model_name/$task_name

mv $embeddings_dir/$model_name/$task_name/test.predicted-scores.json  /projects/0/prjs1338/$score_dir/$model_name/$task_name
mv $embeddings_dir/$model_name/$task_name/*predictions.pkl /projects/0/prjs1338/$score_dir/$model_name/$task_name
mv $embeddings_dir/$model_name/$task_name/*embeddings.npy /projects/0/prjs1338/$score_dir/$model_name/$task_name

rm -r -d -f $embeddings_dir/$model_name/$task_name