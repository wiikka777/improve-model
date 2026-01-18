#!/bin/bash
#SBATCH --job-name=prep_KuaiComt
#SBATCH --partition=scc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=05:00:00
#SBATCH --mem=256G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --gpus=A100:1

set -e
set -x   
module purge
module load gcc/13.2.0
module load python/3.11.9

cd /user/zhuohang.yu/u24922/csp/src

$HOME/hpc_gpu_venv/bin/python -u prepare_data.py --group_num 60 --windows_size 3 --eps 0.5 --dat_name KuaiComt --is_load 0

echo "✅ Data preparation completed successfully."
