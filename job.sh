#!/usr/bin/env bash

#SBATCH -A NAISS2024-22-1122 -p alvis

#SBATCH -N 1 --gpus-per-node=A100:1

#SBATCH -t 3-00:00:00

#SBATCH -J "MNMG PyTorch"

module purge
module load virtualenv/20.23.1-GCCcore-12.3.0
source myvenv/bin/activate
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1

module load matplotlib/3.7.2-gfbf-2023a

python main_imagenet.py