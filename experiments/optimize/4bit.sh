set -e

export PYTHONPATH=$(pwd)

model_path=$1
shards=$2

if [ -z $shards ]; then
    shards=1
fi

python3 optimize.py \
    --model $model_path \
    --params "channel_scales:0.05,angles:0.05" "weight:1e-5,quantizer:1e-6" \
    --epochs 10 10 \
    --group-size 128 \
    --n-bit 4 \
    --num-rotations 8 \
    --datasets wikitext2 c4 redpajama \
    --val-dataset pileval \
    --train-size 2048 \
    --validation-size 64 \
    --batch-size 16 \
    --seqlen 2048 \
    --cache-shards $shards \
    --output-dir ./output \
    --resume \
    --seed 0

#============intactKV：新增启用 Pivot-Aware Loss 的示例命令=====
# 如需启用 IntactKV 的 Pivot-Aware 逐层校准损失，可追加以下参数：
#     --pivot-aware-loss \
#     --pivot-loss-weight 2.0 \
#     --pivot-mode prefix \
#     --pivot-len 16 \
# 完整示例：
# python3 optimize.py \
#     --model $model_path \
#     --params "channel_scales:0.05,angles:0.05" "weight:1e-5,quantizer:1e-6" \
#     --epochs 10 10 \
#     --group-size 128 --n-bit 4 --num-rotations 8 \
#     --datasets wikitext2 c4 redpajama \
#     --val-dataset pileval \
#     --train-size 2048 --validation-size 64 \
#     --batch-size 16 --seqlen 2048 \
#     --cache-shards $shards --output-dir ./output \
#     --pivot-aware-loss --pivot-loss-weight 2.0 \
#     --pivot-mode prefix --pivot-len 16 \
#     --resume --seed 0
#============intactKV：新增启用 Pivot-Aware Loss 的示例命令=====
