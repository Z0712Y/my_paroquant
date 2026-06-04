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
BATCH_SIZE=4
SEQLEN=1024
NUM_ROTATIONS=8
GROUP_SIZE=128
N_BIT=4
EPOCHS="10 10"
TRAIN_SIZE=1024
VALIDATION_SIZE=64
# ===== 三合一方案：双路径数据集配置 =====
# Short path: 非推理任务校准数据（模板化、短序列）
SHORT_DATASETS="wikitext2 c4"
SHORT_PREFIX="Question: (A) (B) (C) (D) Answer:"
# Long path: 推理任务校准数据（长思维链风格）
LONG_DATASETS="redpajama pileval"
LONG_PREFIX="Let's think step by step."
# 双路径损失权重（short 设更高以补足非推理任务短板）
TASK_LAMBDA_SHORT=3.0
TASK_LAMBDA_LONG=1.0
# Prefix-Activation-Aware 显著性融合系数（方案1）
PREFIX_ALPHA=1.0
PREFIX_BETA=0.5
# 任务自适应开关（方案5）
ENABLE_TASK_ADAPTIVE_FLAG="是"
# ==========================================
# 保留原始 datasets（兼容旧逻辑，optimize.py 的 --datasets 仍为 required）
DATASETS="wikitext2 c4 redpajama"
VAL_DATASET="pileval"
# "channel_scales:0.025,angles:0.025" "channel_scales:0.05,angles:0.05" 
# "weight:5e-6,quantizer:5e-7" "weight:1e-5,quantizer:1e-6
PARAMS1="channel_scales:0.025,angles:0.025"
PARAMS2="weight:5e-6,quantizer:5e-7"
RESUME_FLAG="是"
CHECKPOINTING_FLAG="否"  # 设置为"是"启用 --checkpointing，"否"则禁用
SEED=0
# [新增] 通道对选择策略: random(原随机洗牌) / greedy(基于显著性贪心选择)
# 注意：greedy 模式下采用"显著性差异最大化"配对（高-低通道配对），而非简单相加
SELECTION_MODE="greedy"
# [新增] 显著性分数指标: l2 / maxabs / var，仅在 SELECTION_MODE=greedy 时生效
METRIC="l2"


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
  Selection Mode:$SELECTION_MODE
  Metric:        $METRIC

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
  Selection Mode:$SELECTION_MODE
  Metric:        $METRIC

【数据集】
  Train:         $DATASETS
  Val:           $VAL_DATASET

【环境】
  GPU:           $GPU_INFO
  Resume:        $RESUME_FLAG
========================================

EOF

# 构建数据集参数数组（兼容旧逻辑）
# 注意：simple_parsing 对 list[str] 不支持重复 flag（会被覆盖），
# 必须将多个值以空格分隔放在同一个 flag 后面。
DATASET_ARGS=(--datasets $DATASETS)
SHORT_DATASET_ARGS=(--short-datasets $SHORT_DATASETS)
LONG_DATASET_ARGS=(--long-datasets $LONG_DATASETS)

# 构建 checkpointing 参数
CHECKPOINTING_ARG=""
if [ "$CHECKPOINTING_FLAG" = "是" ]; then
    CHECKPOINTING_ARG="--checkpointing"
fi

# 构建任务自适应参数
ENABLE_TASK_ADAPTIVE_ARG=""
if [ "$ENABLE_TASK_ADAPTIVE_FLAG" = "是" ]; then
    ENABLE_TASK_ADAPTIVE_ARG="--enable-task-adaptive"
fi

# 执行 Python 命令（所有参数都使用变量）
# 注意：--epochs 直接使用 $EPOCHS（空格分隔），simple_parsing 才能正确解析为 list[int]
python3 optimize.py \
    --model "$model_path" \
    --params "$PARAMS1" "$PARAMS2" \
    --epochs $EPOCHS \
    --group-size "$GROUP_SIZE" \
    --n-bit "$N_BIT" \
    --num-rotations "$NUM_ROTATIONS" \
    "${DATASET_ARGS[@]}" \
    "${SHORT_DATASET_ARGS[@]}" \
    "${LONG_DATASET_ARGS[@]}" \
    --val-dataset "$VAL_DATASET" \
    --train-size "$TRAIN_SIZE" \
    --validation-size "$VALIDATION_SIZE" \
    --batch-size "$BATCH_SIZE" \
    --seqlen "$SEQLEN" \
    --cache-shards "$shards" \
    --output-dir ./output \
    --resume \
    --seed "$SEED" \
    --selection-mode "$SELECTION_MODE" \
    --metric "$METRIC" \
    --short-prefix "$SHORT_PREFIX" \
    --long-prefix "$LONG_PREFIX" \
    --task-lambda-short "$TASK_LAMBDA_SHORT" \
    --task-lambda-long "$TASK_LAMBDA_LONG" \
    --prefix-alpha "$PREFIX_ALPHA" \
    --prefix-beta "$PREFIX_BETA" \
    $ENABLE_TASK_ADAPTIVE_ARG \
    $CHECKPOINTING_ARG