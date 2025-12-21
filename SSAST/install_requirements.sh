#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=Classify
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=00:30:00
#SBATCH --output=slurm_output_%A_%a.out

cd ~/phd/ssast

module load 2023
module load Anaconda3/2023.07-2

conda create -n ssast python=3.9 -y
source activate ssast

python3 -m pip install -r req.txt