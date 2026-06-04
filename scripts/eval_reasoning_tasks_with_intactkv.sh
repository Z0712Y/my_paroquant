#!/usr/bin/env bash
#============intactKV：新建文件=====
# 这是为了 intactKV 新建的文件，目的是提供与原版 reasoning.sh 类似的
# 多随机种子循环评估入口，支持在 transformers_backend 上跑 MMLU-PRO、
# AIME-2024/2025、GPQA-Diamond 等推理任务，并可开关 IntactKV。
#============intactKV：新建文件=====

set -e

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH}"

# 参数解析
MODEL_PATH=$1
SEED=$2
# 第3个参数开始是数据集列表
datasets="${@:3}"

#============intactKV：路径处理与存在性检查=====
# 如果传入的是相对路径，先尝试基于 ROOT_DIR 解析为绝对路径
if [ ! -d "$MODEL_PATH" ] && [ ! -f "$MODEL_PATH" ]; then
    ABS_PATH="${ROOT_DIR}/${MODEL_PATH}"
    if [ -d "$ABS_PATH" ]; then
        MODEL_PATH="$ABS_PATH"
        echo "[INFO] 模型路径已自动解析为绝对路径: $MODEL_PATH"
    fi
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "[ERROR] 模型路径不存在: $MODEL_PATH" >&2
    echo "[HINT] 请确认:" >&2
    echo "  1. 真实量化模型路径: models/xxx-PARO (由 real_quant.py 生成)" >&2
    echo "  2. 假量化模型路径:  models/xxx-PARO-pseudo (由 pseudo_quant.py 生成)" >&2
    echo "  3. 当前是否在 paroquant-3 根目录执行?" >&2
    exit 1
fi

if [ ! -f "${MODEL_PATH}/config.json" ]; then
    echo "[ERROR] 模型目录下缺少 config.json: ${MODEL_PATH}/config.json" >&2
    exit 1
fi
#============intactKV：路径处理与存在性检查=====

# 如果未提供数据集，使用默认值
if [ -z "$datasets" ]; then
    datasets=("MMLU-PRO" "AIME-2024" "AIME-2025" "GPQA-Diamond")
else
    datasets=($datasets)
fi

#============intactKV：IntactKV 开关配置（修改这里即可）=====
# 设置为 "true" 启用 IntactKV，"false" 则为 baseline
USE_INTACTKV="true"
# Pivot Token 保护长度
INTACTKV_PIVOT_LEN=32
# 是否启用 thinking 模式（Qwen3 等推理模型建议开启）
ENABLE_THINKING="true"
# 最大生成 token 数
MAX_NEW_TOKENS=32768
#============intactKV：IntactKV 开关配置=====

# 构建 IntactKV 参数
INTACTKV_ARGS=""
if [ "$USE_INTACTKV" = "true" ]; then
    INTACTKV_ARGS="--use-intactkv --intactkv-pivot-len $INTACTKV_PIVOT_LEN"
    echo "[IntactKV] 已启用，pivot_len=$INTACTKV_PIVOT_LEN"
else
    echo "[IntactKV] 已禁用（Baseline 模式）"
fi

THINKING_ARG=""
if [ "$ENABLE_THINKING" = "true" ]; then
    THINKING_ARG="--enable-thinking"
fi

# 创建输出目录
OUTPUT_DIR="${ROOT_DIR}/outputs/intactkv_inference"
mkdir -p "$OUTPUT_DIR"

MODEL_NAME=$(basename "$MODEL_PATH")

# 遍历数据集执行评估
for dataset in "${datasets[@]}"; do
    echo "========================================"
    echo "Task: $dataset"
    echo "Model: $MODEL_NAME"
    echo "Seed: $SEED"
    echo "========================================"

    OUTPUT_FILE="${OUTPUT_DIR}/${dataset}_${MODEL_NAME}_seed${SEED}"
    if [ "$USE_INTACTKV" = "true" ]; then
        OUTPUT_FILE="${OUTPUT_FILE}_intactkv.json"
    else
        OUTPUT_FILE="${OUTPUT_FILE}_baseline.json"
    fi

    python3 "${ROOT_DIR}/scripts/eval_reasoning_tasks_with_intactkv.py" \
        --model "$MODEL_PATH" \
        --dataset "$dataset" \
        --seed "$SEED" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --output-file "$OUTPUT_FILE" \
        $INTACTKV_ARGS \
        $THINKING_ARG

done

echo ""
echo "全部任务完成，结果保存在: $OUTPUT_DIR"
