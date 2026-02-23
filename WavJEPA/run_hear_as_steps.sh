#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=WavJEPA
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --exclude=gcn118
#SBATCH --time=02:00:00
#SBATCH --output=hear/slurm_output_%A_%a.out
#SBATCH --array=0-32   # 3 steps × 11 tasks = 33 jobs

steps=(
    50000
    100000
    200000
)

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

# --- Derive 2D indices from the flat SLURM array index ---
NUM_TASKS=${#task_names[@]}   # 11
STEP_IDX=$(( SLURM_ARRAY_TASK_ID / NUM_TASKS ))
TASK_IDX=$(( SLURM_ARRAY_TASK_ID % NUM_TASKS ))

step=${steps[$STEP_IDX]}
task_name=${task_names[$TASK_IDX]}
task_dir=${task_dirs[$TASK_IDX]}
grid=${grids[$TASK_IDX]}

if [[ -z "$step" ]]; then
    echo "ERROR: Could not resolve step for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "Running: step=$step | task=$task_name | grid=$grid"

cd ~/phd/awsome-audio-foundation-models/WavJEPA
module load 2023
module load Anaconda3/2023.07-2
source activate sjepa-eval
cd hear-eval-kit

embeddings_dir=/projects/0/prjs1338/WavJEPAEbeddingsHear$step

export MODEL_PATH="/gpfs/work5/0/prjs1261/saved_models_jepa_reproduce/SR=16000/LibriRatio=0.0/BatchSize=32/NrSamples=8/NrGPUs=2/ModelSize=base/LR=0.0004/Masking=time-inverse-masker/TargetProb=0.25/TargetLen=10/ContextLen=10/TopK=8/step=$step.ckpt"

model_name=hear_configs.WavJEPA_as
score_dir=hear_scores_$step

python3 -m heareval.embeddings.runner "$model_name" \
    --tasks-dir "$task_dir" \
    --task "$task_name" \
    --embeddings-dir "$embeddings_dir"

python3 -m heareval.predictions.runner \
    "$embeddings_dir/$model_name/$task_name" \
    --grid "$grid"

out_dir=~/phd/awsome-audio-foundation-models/$score_dir/$model_name/$task_name
mkdir -p "$out_dir"

mv "$embeddings_dir/$model_name/$task_name/test.predicted-scores.json" "$out_dir"

rm -rf "$embeddings_dir/$model_name/$task_name"