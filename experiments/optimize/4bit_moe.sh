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
    --skipped-modules "mlp.gate" \
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

# 混合专家模型MoE
# 跳过mlp.gate模块的量化，保持其为全精度，以确保专家选择机制的准确性和模型性能的稳定性。
