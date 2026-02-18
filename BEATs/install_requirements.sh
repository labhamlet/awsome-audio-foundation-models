#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=InstallReq
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:20:00
#SBATCH --output=slurm_output_%A.out

cd ~/phd/awsome-audio-foundation-models/BEATs

module load 2023
module load Anaconda3/2023.07-2
conda create -n beats-eval python=3.9
source activate beats-eval


python3 -m pip install -r requirements.txt