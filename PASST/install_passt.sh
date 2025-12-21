#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=InstallReq
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=00:20:00
#SBATCH --output=slurm_output_%A.out

cd ~/phd/hear-other-models/passt
HYDRA_FULL_ERROR=1

module load 2023
module load Anaconda3/2023.07-2
conda create -n passt-eval python=3.9
source activate passt-eval

python3 -m pip install hear21passt
