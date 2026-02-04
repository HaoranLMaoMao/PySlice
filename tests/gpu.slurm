#!/bin/bash
#SBATCH -A birthright
#SBATCH --partition=burst
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -G v100:1
#SBATCH --output=gpu.sbout
#SBATCH --error=gpu.sberr

module load cuda/11.4.0
module load gcc
source deactivate base
conda activate /home/qwe/.conda/envs/pyslice_env

bash runAllTests.sh torchgpu
