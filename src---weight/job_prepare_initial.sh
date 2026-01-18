#!/bin/bash
# One-time script to generate KuaiComt_subset.csv from raw data
#SBATCH --job-name=prep_initial
#SBATCH --partition=scc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=05:00:00
#SBATCH --mem=256G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --gpus=A100:1

PROJECT_DIR="/user/zhuohang.yu/u24922/csp"
VENV_PYTHON="$HOME/hpc_gpu_venv/bin/python"

module purge
module load gcc/13.2.0
module load python/3.11.9

cd "$PROJECT_DIR/src"

echo "One-time initialization: Generating KuaiComt_subset.csv from raw data..."
$VENV_PYTHON prepare_data.py \
    --group_num 60 \
    --windows_size 3 \
    --eps 0.5 \
    --dat_name KuaiComt \
    --is_load 0

if [ $? -eq 0 ]; then
    echo "✅ Initial dataset generated successfully."
    ls -lh ../rec_datasets/WM_KuaiComt/KuaiComt_subset.csv
else
    echo "❌ Failed to generate initial dataset"
    exit 1
fi
