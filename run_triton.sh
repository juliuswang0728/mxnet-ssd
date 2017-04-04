#!/bin/bash
# Request 1 GPU
#SBATCH --gres=gpu:m2090:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -p gpushort
#SBATCH --mem-per-cpu 4G
#SBATCH -t 00-04
#SBATCH --mail-user=juliuswang0728@gmail.com
#SBATCH --mail-type=ALL

python train.py
