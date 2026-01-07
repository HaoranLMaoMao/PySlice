#!/bin/bash
#SBATCH -A birthright
#SBATCH --partition=batch
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --output=test-cpu.sbout
#SBATCH --error=test-cpu.sberr
#module load pytorch
#module load conda
#conda init

conda activate /home/qwe/.conda/envs/pyslice_env
bash runAllTests.sh torchcpu
