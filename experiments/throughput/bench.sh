#!/usr/bin/env bash

set -e

HF_MODEL=$1
#============intactKV：接收可选的 IntactKV 参数=====
USE_INTACTKV=${2:-""}
PIVOT_LEN=${3:-""}

INTACTKV_ARGS=""
if [ "$USE_INTACTKV" = "--use-intactkv" ]; then
    INTACTKV_ARGS="--use-intactkv"
    if [ -n "$PIVOT_LEN" ]; then
        INTACTKV_ARGS="$INTACTKV_ARGS --intactkv-pivot-len $PIVOT_LEN"
    fi
fi
#============intactKV：接收可选的 IntactKV 参数=====

# Benchmark batch size = 1 decoding throughput
python scripts/bench_model.py \
  --model "$HF_MODEL" \
  --prefill-len 256 \
  --decode-len 512 \
  #============intactKV：追加 IntactKV 参数=====
  $INTACTKV_ARGS
  #============intactKV：追加 IntactKV 参数=====
