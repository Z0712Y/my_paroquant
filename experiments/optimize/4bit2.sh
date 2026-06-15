#!/bin/bash
set -e

# 设置环境变量使用镜像站
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=$(pwd)
# 缓解显存碎片化问题
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# [新增] 强制 datasets/transformers 离线模式，直接使用本地缓存，避免联网请求超时
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

model_path=$1   # 接收第一个参数：模型路径
shards=$2       # 接收第二个参数：cache shards（如果为空，下面设为1）

if [ -z "$shards" ]; then
    shards=1    # 如果没传，默认是 1
fi

# ==================== 配置变量（只需修改这里）====================
# 这些变量会被结构化记录到日志中
BATCH_SIZE=16
SEQLEN=2048
NUM_ROTATIONS=8
GROUP_SIZE=128
N_BIT=4
EPOCHS="10 10"
TRAIN_SIZE=1024
VALIDATION_SIZE=64
DATASETS="wikitext2 c4 redpajama"
VAL_DATASET="pileval"
PARAMS1="channel_scales:0.05,angles:0.05"
PARAMS2="weight:1e-5,quantizer:1e-6"
RESUME_FLAG="是"
CHECKPOINTING_FLAG="是"  # 设置为"是"启用 --checkpointing，"否"则禁用
SEED=0

# channel_scales:0.025,angles:0.025" "weight:5e-6,quantizer:5e-7
# channel_scales:0.05,angles:0.05" "weight:1e-5,quantizer:1e-6
# --bacth size 16 \         # 改成4
# --seqlen 2048 \           # 改为 512
# --checkpointing true \    # 新增
# --num-rotations 8 \       # 改成4
# =============================================================

# ==================== 变量校验 ====================
if [ -z "$model_path" ]; then
    echo "错误: 未提供模型路径" >&2
    echo "用法: $0 <model_path> [shards]" >&2
    exit 1
fi

if [ -z "$GROUP_SIZE" ]; then
    echo "错误: GROUP_SIZE 未设置" >&2
    exit 1
fi

if [ -z "$N_BIT" ]; then
    echo "错误: N_BIT 未设置" >&2
    exit 1
fi

if [ -z "$NUM_ROTATIONS" ]; then
    echo "错误: NUM_ROTATIONS 未设置" >&2
    exit 1
fi
# =================================================

# 获取 GPU 信息
GPU_INFO=${CUDA_VISIBLE_DEVICES:-"未指定(默认使用GPU 0)"}

# 如果有 LOG_DIR（通过 run_4bit_with_log.sh 传入），则写入结构化配置到 info.txt
if [ -n "$LOG_DIR" ] && [ -f "$LOG_DIR/info.txt" ]; then
    cat >> "$LOG_DIR/info.txt" << EOF

========================================
结构化配置参数
========================================
【模型配置】
  模型路径:      $model_path
  Batch Size:    $BATCH_SIZE
  Sequence Len:  $SEQLEN
  Cache Shards:  $shards

【优化参数】
  Epochs:        $EPOCHS
  Group Size:    $GROUP_SIZE
  N-bit:         $N_BIT
  Num Rotations: $NUM_ROTATIONS
  Params:        $PARAMS1
                 $PARAMS2

【数据集】
  Train:         $DATASETS
  Val:           $VAL_DATASET
  Train Size:    $TRAIN_SIZE
  Val Size:      $VALIDATION_SIZE

【其他】
  GPU:           $GPU_INFO
  Resume:        $RESUME_FLAG
  Seed:          $SEED
========================================
EOF
fi

# 输出配置信息到 stderr（同时会进入 run.log）
cat << EOF

========================================
运行配置参数 (来自 4bit.sh)
========================================
【模型配置】
  模型路径:      $model_path
  Batch Size:    $BATCH_SIZE
  Sequence Len:  $SEQLEN
  Cache Shards:  $shards

【优化参数】
  Epochs:        $EPOCHS
  Group Size:    $GROUP_SIZE
  N-bit:         $N_BIT
  Num Rotations: $NUM_ROTATIONS

【数据集】
  Train:         $DATASETS
  Val:           $VAL_DATASET

【环境】
  GPU:           $GPU_INFO
  Resume:        $RESUME_FLAG
========================================

EOF

# 构建数据集参数数组
DATASET_ARGS=()
for ds in $DATASETS; do
    DATASET_ARGS+=("--datasets" "$ds")
done

# 构建 checkpointing 参数
CHECKPOINTING_ARG=""
if [ "$CHECKPOINTING_FLAG" = "是" ]; then
    CHECKPOINTING_ARG="--checkpointing"
fi

# 执行 Python 命令（所有参数都使用变量）
# 注意：--epochs 直接使用 $EPOCHS（空格分隔），simple_parsing 才能正确解析为 list[int]
#     --output-dir ./output \
python3 optimize.py \
    --model "$model_path" \
    --params "$PARAMS1" "$PARAMS2" \
    --epochs $EPOCHS \
    --group-size "$GROUP_SIZE" \
    --n-bit "$N_BIT" \
    --num-rotations "$NUM_ROTATIONS" \
    "${DATASET_ARGS[@]}" \
    --val-dataset "$VAL_DATASET" \
    --train-size "$TRAIN_SIZE" \
    --validation-size "$VALIDATION_SIZE" \
    --batch-size "$BATCH_SIZE" \
    --seqlen "$SEQLEN" \
    --cache-shards "$shards" \
    --output-dir ./output \
    --resume \
    --seed "$SEED" \
    $CHECKPOINTING_ARG
