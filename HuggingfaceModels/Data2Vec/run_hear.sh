#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=Data2Vec
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --exclude=gcn118
#SBATCH --time=02:00:00
#SBATCH --output=hear/slurm_output_%A_%a.out
#SBATCH --array=1-2

grids=(
default
fast
default
default
default
default
default
default
default
default
)

task_dirs=(
/projects/0/prjs1338/tasks
/projects/0/prjs1338/tasks
/projects/0/prjs1261/tasks
/projects/0/prjs1338/tasks
/projects/0/prjs1338/tasks
/projects/0/prjs1338/tasks
/projects/0/prjs1338/tasks
/projects/0/prjs1338/tasks
/projects/0/prjs1338/tasks
/projects/0/prjs1338/tasks
/projects/0/prjs1338/tasks
)

task_names=(
beijing_opera-v1.0-hear2021-full
dcase2016_task2-hear2021-full
fsd50k-v1.0-full
esc50-v2.0.0-full
libricount-v1.0.0-hear2021-full
speech_commands-v0.0.2-5h
mridangam_stroke-v1.5-full
mridangam_tonic-v1.5-full
tfds_crema_d-1.0.0-full
nsynth_pitch-v2.2.3-5h
vox_lingua_top10-hear2021-full
)

cd ~/phd/awsome-audio-foundation-models/HuggingfaceModels/Data2Vec
module load 2023
module load Anaconda3/2023.07-2
source activate hear-other-models-eval
cd hear-eval-kit

embeddings_dir=/projects/0/prjs1338/Data2VecEmbeddingsHear
score_dir=hear_scores
task_name=${task_names[$SLURM_ARRAY_TASK_ID]}
task_dir=${task_dirs[$SLURM_ARRAY_TASK_ID]}
grid=${grids[$SLURM_ARRAY_TASK_ID]}

model_name=hear_configs.data2vec_base

python3 -m heareval.embeddings.runner "$model_name" --tasks-dir $task_dir --task "$task_name" --embeddings-dir $embeddings_dir
python3 -m heareval.predictions.runner $embeddings_dir/$model_name/$task_name --grid $grid

mkdir -p ~/phd/awsome-audio-foundation-models/$score_dir/$model_name/$task_name

mv $embeddings_dir/$model_name/$task_name/test.predicted-scores.json  ~/phd/awsome-audio-foundation-models/$score_dir/$model_name/$task_name

rm -r -d -f $embeddings_dir/$model_name/$task_name