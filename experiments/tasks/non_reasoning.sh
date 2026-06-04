#!/usr/bin/env bash
model=$1
extra_args="${@:2}"
# tasks=arc_challenge,arc_easy,boolq,hellaswag
# tasks = arc_challenge,arc_easy,boolq,hellaswag,winogrande,piqa,openbookqa
tasks=winogrande,piqa,openbookqa

accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained=$model,enable_thinking=False,dtype=float16 \
    --tasks $tasks \
    --batch_size 32 \
    $extra_args

#============intactKV：新增使用 transformers_backend 进行 IntactKV 推理评估的示例=====
# 对于需要长思维链生成的评估（如 MMLU-PRO 的 step-by-step），
# 可以使用支持 IntactKV 的 TransformersGenerator：
# python3 -c "
# import asyncio
# import sys
# sys.path.insert(0, '.')
# from inference_engine.generation.transformers_backend import TransformersGenerator, GenerationParams
# gen = TransformersGenerator('$model', use_intactkv=True, intactkv_pivot_len=32)
# params = GenerationParams(max_new_tokens=2048, temperature=0.6, top_p=0.95)
# async def eval():
#     result = await gen.generate([{'role': 'user', 'content': 'What is the capital of France? Think step by step.'}], params)
#     print(result.output_text)
# asyncio.run(eval())
# "
#============intactKV：新增使用 transformers_backend 进行 IntactKV 推理评估的示例=====
