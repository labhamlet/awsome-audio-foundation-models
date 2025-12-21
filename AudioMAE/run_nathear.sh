#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=MWMAE
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --exclude=gcn118
#SBATCH --time=02:00:00
#SBATCH --output=nathear/slurm_output_%A_%a.out
#SBATCH --array=0-10

cd ~/phd/hear-freq-models/AudioMAE
source env/bin/activate
cd listen-eval-kit


task_names=(fsd50k-v1.0-full
dcase2016_task2-hear2021-full
beijing_opera-v1.0-hear2021-full
esc50-v2.0.0-full
libricount-v1.0.0-hear2021-full
speech_commands-v0.0.2-5h
mridangam_stroke-v1.5-full
mridangam_tonic-v1.5-full
tfds_crema_d-1.0.0-full
nsynth_pitch-v2.2.3-5h
vox_lingua_top10-hear2021-full
)

tasks_dirs=(
/projects/0/prjs1261/tasks_noisy_ambisonics
/projects/0/prjs1261/tasks_noisy_ambisonics
/projects/0/prjs1338/tasks_noisy_ambisonics
/projects/0/prjs1338/tasks_noisy_ambisonics
/projects/0/prjs1338/tasks_noisy_ambisonics
/projects/0/prjs1338/tasks_noisy_ambisonics
/projects/0/prjs1338/tasks_noisy_ambisonics
/projects/0/prjs1338/tasks_noisy_ambisonics
/projects/0/prjs1338/tasks_noisy_ambisonics
/projects/0/prjs1338/tasks_noisy_ambisonics
/projects/0/prjs1338/tasks_noisy_ambisonics
)

task_name=${task_names[$SLURM_ARRAY_TASK_ID]}
tasks_dir=${tasks_dirs[$SLURM_ARRAY_TASK_ID]}

embeddings_dir=/projects/0/prjs1338/NoisyEmbeddings
score_dir=nathear_scores
model_name=hear_configs.MAE

python3 -m heareval.embeddings.runner "$model_name" --tasks-dir $tasks_dir --task "$task_name" --embeddings-dir $embeddings_dir
python3 -m heareval.predictions.runner $embeddings_dir/$model_name/$task_name

mkdir -p /projects/0/prjs1338/$score_dir/$model_name/$task_name


mv $embeddings_dir/$model_name/$task_name/test.predicted-scores.json  /projects/0/prjs1338/$score_dir/$model_name/$task_name
mv $embeddings_dir/$model_name/$task_name/*predictions.pkl /projects/0/prjs1338/$score_dir/$model_name/$task_name
mv $embeddings_dir/$model_name/$task_name/*embeddings.npy /projects/0/prjs1338/$score_dir/$model_name/$task_name

rm -r -d -f $embeddings_dir/$model_name/$task_name