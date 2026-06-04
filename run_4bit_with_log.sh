#!/bin/bash
# 脚本运行之后，再次修改脚本运行不会影响原程序运行。
# cd /home/hhw/zy/paroquant && ./run_4bit_with_log.sh

# ==================== 配置区域 ====================
# 修改以下变量即可更换模型或脚本
# 外部配置优先级 > 脚本内部配置，但只执行脚本时会安全地使用默认值。
SCRIPT="experiments/optimize/4bit2.sh"  # 要运行的脚本
MODEL="/home/hhw/zy/models/Qwen3-4B-Base"            # 模型路径
# GPU_ID="2"                             # 指定使用的 GPU
# ==================================================

# 设置 GPU
# export CUDA_VISIBLE_DEVICES=$GPU_ID

# 如果没有NOHUP环境变量，说明是第一次运行，需要用nohup重新启动
if [ -z "$_IN_NOHUP" ]; then
    export _IN_NOHUP=1
    
    LOG_DIR="/home/hhw/zy/paroquant/experiments/optimize/logs/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$LOG_DIR"
    
    echo "========================================"
    echo "日志目录: $LOG_DIR"
    echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "命令: bash $SCRIPT $MODEL"
    echo "========================================"
    
    # 保存基本信息
    {
        echo "命令: bash $SCRIPT $MODEL"
        echo "工作目录: $(pwd)"
        echo "启动时间: $(date '+%Y-%m-%d %H:%M:%S')"
    } > "$LOG_DIR/info.txt"
    
    date '+%Y-%m-%d %H:%M:%S' > "$LOG_DIR/start_time.txt"
    
    # 将日志目录传给子脚本，以便子脚本可以写入结构化配置
    export LOG_DIR
    
    # 使用nohup启动自己，传入日志目录
    nohup "$0" "$LOG_DIR" > /dev/null 2>&1 &
    PID=$!
    
    echo $PID > "$LOG_DIR/pid.txt"
    echo ""
    echo "任务已在后台运行，PID: $PID"
    echo "现在可以安全关闭终端"
    echo ""
    echo "重新打开终端后查看:"
    echo "  实时日志:   tail -f $LOG_DIR/run.log"
    echo "  查看状态:   ls $LOG_DIR/"
    echo "  查看结果:   cat $LOG_DIR/exit_code.txt"
    echo ""
    
    exit 0
fi

# ===== 以下是nohup环境下的实际执行代码 =====

LOG_DIR="$1"

# 记录实际的PID（不是nohup的PID）
echo $$ > "$LOG_DIR/pid.txt"

# 运行实际命令，所有输出（包括脚本中的配置信息）都写入 run.log
# 显式使用 bash 执行脚本，确保正确解析
stdbuf -oL bash "$SCRIPT" "$MODEL" > "$LOG_DIR/run.log" 2>&1

EXIT_CODE=$?

# 记录结束时间和退出码
date '+%Y-%m-%d %H:%M:%S' > "$LOG_DIR/end_time.txt"
echo $EXIT_CODE > "$LOG_DIR/exit_code.txt"

# 在日志末尾添加结束标记
{
    echo ""
    echo "========================================"
    echo "任务结束，退出码: $EXIT_CODE"
    echo "结束时间: $(cat $LOG_DIR/end_time.txt)"
    echo "========================================"
} >> "$LOG_DIR/run.log"

exit $EXIT_CODE
