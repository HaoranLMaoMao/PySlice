#!/bin/bash
#SBATCH -A birthright
#SBATCH --partition=batch
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --output=cpu.sbout
#SBATCH --error=cpu.sberr

module load cuda/11.4.0
module load gcc
source deactivate base
conda activate /home/qwe/.conda/envs/pyslice_env

bash runAllTests.sh torchcpu


