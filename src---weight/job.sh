#!/bin/bash
# -----------------------------------------------------------------------------
# SLURM 资源请求配置,跑LCU
# -----------------------------------------------------------------------------
#SBATCH --job-name=InternVL_DCN_PyTorch # 更新作业名称以反映实际运行的 Python 脚本
#SBATCH --partition=scc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16              # 增加 CPU 核心数，以匹配 128G 内存
#SBATCH --time=05:00:00                 # 运行时间限制：5 小时
#SBATCH --mem=128G                      # 🔴 关键修正：增大内存请求至 128GB (解决 mmap/RAM 限制)
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --gpus=A100:1                   # 关键修正：使用 GRES 语法请求 1 块 A100 GPU

# =================================================================
# 1. 路径和变量定义
# =================================================================
# 明确定义项目路径 (用于 cd)
PROJECT_DIR_PATH="/user/zhuohang.yu/u24922/csp"

# 明确定义 VENV Python 解释器路径 (解决 ModuleNotFoundError)
VENV_PYTHON="$HOME/hpc_gpu_venv/bin/python"

# 明确定义要运行的主 Python 脚本的绝对路径
MAIN_SCRIPT="/user/zhuohang.yu/u24922/csp/src/main.py"
PREPARE_SCRIPT="/user/zhuohang.yu/u24922/csp/src/prepare_data.py"

# 明确定义 Python 脚本的输出目录 (用于 --fout 参数)
OUTPUT_DIR="/user/zhuohang.yu/u24922/csp/rec_datasets/WM_KuaiComt/4DCN_WLR_0.001_0.1_test_40_2_61"

# 定义训练参数 (与 run.sh 一致)
gpu_id=0
lambda1=0.001
lambda2=0.1
randseed=61
dataname="KuaiComt"
windows_size=3
eps=0.5
c_inv=40
sigma=2
epo_nm=1
groupnum=30
modelname="DCN"
labelname="WLR"
label1name="user_clicked"
label2name="comments_score"

# =================================================================
# 2. 软件环境加载
# =================================================================
module purge
module load gcc/13.2.0
module load python/3.11.9
module load cuda/11.8.0 

# =================================================================
# 3. 运行您的 PyTorch 应用程序 (直接调用 VENV Python)
# =================================================================

echo "Starting job on compute node: $(hostname)"
echo "CUDA Version loaded: $(which nvcc)"
echo "Python Interpreter: $VENV_PYTHON"
echo "-------------------------------------"

# 切换到项目目录 (用于处理相对路径和日志输出)
cd $PROJECT_DIR_PATH

echo "Starting job on compute node: $(hostname)"
echo "CUDA Version loaded: $(which nvcc)"
echo "Python Interpreter: $VENV_PYTHON"
echo "-------------------------------------"

# =================================================================
# 4. 第一步：数据预处理 (已完成，跳过)
# =================================================================
echo ""
echo "Step 1: Using pre-generated dataset (KuaiComt_subset.csv)..."

# =================================================================
# 5. 第二步：模型训练 (运行 main.py)
# =================================================================
echo ""
echo "Step 2: Starting model training..."
CUDA_VISIBLE_DEVICES=${gpu_id} $VENV_PYTHON $MAIN_SCRIPT \
    --fout $OUTPUT_DIR \
    --dat_name ${dataname} \
    --model_name ${modelname} \
    --label_name ${labelname} \
    --randseed ${randseed} \
    --load_to_eval 0 \
    --epoch_num ${epo_nm} \
    --label1_name ${label1name} \
    --label2_name ${label2name} \
    --lambda1 ${lambda1} \
    --lambda2 ${lambda2}

# 检查 Python 脚本的退出码
if [ $? -eq 0 ]; then
    echo "✅ Model training completed successfully."
else
    echo "❌ Model training failed. Check slurm-${SLURM_JOB_ID}.err"
    exit 1
fi