#!/usr/bin/env bash
# 用法: ./run_reasoning_nohup.sh <model_path> <seed> [datasets...]
# 示例: ./run_reasoning_nohup.sh models/Qwen3-14B-PARO-pseudo 42 MMLU-PRO

cd "$(dirname "$0")" || exit 1

MODEL="${1:-models/Qwen3-8B-PARO-pseudo}"
SEED="${2:-42}"
shift 2 || true
DATASETS=("$@")

# 如果没传数据集，默认用 reasoning.sh 里的默认值
if [ ${#DATASETS[@]} -eq 0 ]; then
    DATASETS=("MMLU-PRO")
fi

DATASETS_STR="${DATASETS[*]}"

# 日志目录
LOG_DIR="./outputs/inference/logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/run.log"
PID_FILE="$LOG_DIR/pid.txt"

# 保存基本信息
{
    echo "命令: ./experiments/tasks/reasoning.sh $MODEL $SEED $DATASETS_STR"
    echo "工作目录: $(pwd)"
    echo "启动时间: $(date '+%Y-%m-%d %H:%M:%S')"
} > "$LOG_DIR/info.txt"

echo "========================================"
echo "日志目录: $LOG_DIR"
echo "模型: $MODEL"
echo "数据集: $DATASETS_STR"
echo "启动时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

# nohup 执行
nohup ./experiments/tasks/reasoning.sh "$MODEL" "$SEED" "${DATASETS[@]}" > "$LOG_FILE" 2>&1 &
PID=$!
echo $PID > "$PID_FILE"

echo ""
echo "任务已在后台运行，PID: $PID"
echo "可以安全关闭终端"
echo ""
echo "查看方式:"
echo "  实时日志:   tail -f $LOG_FILE"
echo "  查看状态:   ls $LOG_DIR/"
echo "  结束时间:   cat $LOG_DIR/end_time.txt (任务结束后生成)"
echo "  退出码:     cat $LOG_DIR/exit_code.txt (任务结束后生成)"
echo ""

# 等待任务结束并记录
wait $PID
EXIT_CODE=$?

date '+%Y-%m-%d %H:%M:%S' > "$LOG_DIR/end_time.txt"
echo $EXIT_CODE > "$LOG_DIR/exit_code.txt"

{
    echo ""
    echo "========================================"
    echo "任务结束，退出码: $EXIT_CODE"
    echo "结束时间: $(cat $LOG_DIR/end_time.txt)"
    echo "========================================"
} >> "$LOG_FILE"

exit $EXIT_CODE
