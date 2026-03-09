#!/bin/bash
#SBATCH --job-name=cantilever_vi
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=24:00:00

# Activate conda environment
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate /mnt/hpc_data/users/ZDimitrov/neuraluq_cantilever/env

cd /mnt/hpc_data/users/ZDimitrov/neuraluq_cantilever/examples

python -u cantilever_vi_coeff.py
