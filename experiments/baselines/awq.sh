#!/usr/bin/env bash
# 参考: https://github.com/casper-hansen/AutoAWQ
# Commit: d6e797a42b9ef7778de8ee2352116e0f48a78d61
set -e

# ==========================================
# 模块一：环境依赖检查与安装
# 说明：检查当前环境是否已安装 autoawq，未安装则自动安装。
# ==========================================
pip freeze | grep autoawq > /dev/null || pip install autoawq[kernels]

# ==========================================
# 模块二：输入参数解析
# $1: Hugging Face 模型路径（如 Qwen/Qwen3-4B）
# $2: 量化位宽（如 4）
# $3: 评估时使用的序列长度（如 2048）
# ==========================================
model_path="$1"
bits="$2"
seqlen="$3"
model_name=$(echo $model_path | awk -F'/' '{print $2}')

# ==========================================
# 模块三：工程目录与输出路径配置
# project_dir: 当前 baseline 实验的根目录
# 【实验可替换】awq_cache 保存路径可自定义，以匹配不同实验组织方式。
# ==========================================
project_dir=baselines/autoawq

# ==========================================
# 模块四：执行 AWQ 量化
# 调用 autoawq_cli.py 对指定模型进行量化并保存。
# 【实验可替换】
#   --quant_name: 自定义量化模型命名规则
#   --q_group_size: 更改为 64/256 等其他 group size
#   --w_bit: 更改为 3/8 等其他位宽
#   --zero_point / --no-zero_point: 切换是否启用 zero point
#   --version: 更改量化 kernel 版本（如 GEMV）
# ==========================================
python experiments/baselines/autoawq_cli.py \
    --hf_model_path $model_path \
    --quant_name $model_name-w$bits-g128-quant \
    --local_save_path $project_dir/awq_cache/$model_name-w$bits-g128-quant \
    --zero_point \
    --q_group_size 128 \
    --w_bit $bits

# ==========================================
# 模块五：量化模型 PPL 评估
# 调用 eval_ppl.py 评估量化后模型的困惑度（PPL）。
# ==========================================
python scripts/eval_ppl.py \
    --model $project_dir/awq_cache/$model_name-w$bits-g128-quant \
    --seed 0 \
    --seqlen $seqlen \
