#!/bin/bash
# =================================================================
# SLURM 资源请求配置 - 数据准备步骤
# =================================================================
#SBATCH --job-name=Prepare_KuaiComt_Data
#SBATCH --partition=scc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --mem=128G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# =================================================================
# 1. 路径和变量定义
# =================================================================
# Ensure PROJECT_DIR is defined
if [ -z "$PROJECT_DIR" ]; then
    PROJECT_DIR="/user/zhuohang.yu/u24922/.project/dir.project"
fi

export DATASET_DIR="$PROJECT_DIR/rec_datasets"

PROJECT_DIR_PATH="/user/zhuohang.yu/u24922/csp"

# 明确定义 VENV Python 解释器路径 (解决 ModuleNotFoundError)
VENV_PYTHON="$HOME/hpc_gpu_venv/bin/python"

# 明确定义要运行的数据准备脚本的绝对路径
PREPARE_SCRIPT="/user/zhuohang.yu/u24922/csp/src/prepare_data.py"

# 定义参数 (与 run.sh 一致)
dataname="KuaiComt"
windows_size=3
eps=0.5
group_num=60
randseed=61

# =================================================================
# 2. 软件环境加载
# =================================================================
module purge
module load gcc/13.2.0
module load python/3.11.9
module load cuda/11.8.0

# =================================================================
# 3. 切换到工作目录
# =================================================================
cd $PROJECT_DIR_PATH

echo "Starting job on compute node: $(hostname)"
echo "Python Interpreter: $VENV_PYTHON"
echo "-------------------------------------"

# =================================================================
# 4. 第一步：数据准备 (运行 prepare_data.py)
# =================================================================
echo ""
echo "Step 1: Starting data preparation..."
echo "Dataset: ${dataname}"
echo "Data path: ${DATASET_DIR}/WM_KuaiComt"
echo "========================================="

$VENV_PYTHON -u $PREPARE_SCRIPT \
    --group_num ${group_num} \
    --windows_size ${windows_size} \
    --eps ${eps} \
    --dat_name ${dataname} \
    --is_load 0

# 检查 Python 脚本的退出码
if [ $? -eq 0 ]; then
    echo "✅ Data preparation completed successfully."
else
    echo "❌ Data preparation failed. Check slurm-${SLURM_JOB_ID}.err"
    exit 1
fi
