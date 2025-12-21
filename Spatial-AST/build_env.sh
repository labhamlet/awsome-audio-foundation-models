#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=Localize
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=00:20:00
#SBATCH --output=slurm_output_%A_%a.out
#SBATCH --array=0

cd ~/phd/hear-other-models-niche/Spatial-AST
module load 2023
module load Anaconda3/2023.07-2

conda env create -f environment.yml
source activate spatial-ast
bash timm_patch/patch.sh