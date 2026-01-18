#!/bin/bash
# 方案 B: Exponential Decay [1.0, 0.5, 0.25, 0.12, 0.06, 0.0]
#SBATCH --job-name=DCN_SchemeB
#SBATCH --partition=scc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=05:00:00
#SBATCH --mem=128G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --gpus=A100:1

module purge
module load gcc/13.2.0
module load python/3.11.9
module load cuda/11.8.0

PROJECT_DIR_PATH="/user/zhuohang.yu/u24922/csp"
VENV_PYTHON="$HOME/hpc_gpu_venv/bin/python"
MAIN_SCRIPT="$PROJECT_DIR_PATH/src/main.py"
PREPARE_SCRIPT="$PROJECT_DIR_PATH/src/prepare_data.py"

OUTPUT_DIR="$PROJECT_DIR_PATH/rec_datasets/WM_KuaiComt/SchemeB_ExponentialDecay_DCN_WLR_0.001_0.1_test_40_2_61"

gpu_id=0
lambda1=0.001
lambda2=0.1
randseed=61
dataname="KuaiComt"
windows_size=3
eps=0.5
modelname="DCN"
labelname="WLR"
label1name="user_clicked"
label2name="comments_score"
epo_nm=1

cd $PROJECT_DIR_PATH

echo "=========================================="
echo "Scheme B: Exponential Decay"
echo "Weights: [1.0, 0.5, 0.25, 0.12, 0.06, 0.0]"
echo "=========================================="
echo ""

# 注意：运行此脚本前，需要修改 pre_kuaicomt.py 中的权重方案
# 将 rank_map = [1.0, 0.1, 0.1, 0.1, 0.1, 0.1] 改为
# rank_map = [1.0, 0.5, 0.25, 0.12, 0.06, 0.0]

echo "⚠️  IMPORTANT: Please modify pre_kuaicomt.py first!"
echo "Change rank_map to: [1.0, 0.5, 0.25, 0.12, 0.06, 0.0]"
echo ""

# 第一步：生成权重数据 (从 src 目录运行以支持相对路径)
echo "Step 1: Generating dataset with Scheme B weights..."
cd $PROJECT_DIR_PATH/src
$VENV_PYTHON prepare_data.py --group_num 60 --windows_size ${windows_size} --eps ${eps} --dat_name ${dataname} --is_load 0
if [ $? -ne 0 ]; then
    echo "❌ Data preparation failed."
    exit 1
fi
echo "✅ Data preparation completed."

# 第二步：模型训练 (从项目根目录运行)
cd $PROJECT_DIR_PATH
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

if [ $? -eq 0 ]; then
    echo "✅ Model training completed successfully."
else
    echo "❌ Model training failed."
    exit 1
fi
