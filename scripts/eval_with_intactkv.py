#============intactKV：新建文件=====
# 这是为了 intactKV 新建的文件，目的是提供一个便捷的命令行评估脚本，
# 用于在 ParoQuant 量化模型上使用 IntactKV 进行推理，验证长思维链场景下
# Pivot Token KV 保护对误差累积的缓解效果。
#============intactKV：新建文件=====

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from inference_engine.generation.transformers_backend import (
    TransformersGenerator,
    GenerationParams,
)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ParoQuant model with IntactKV injection"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to quantized model checkpoint"
    )
    parser.add_argument(
        "--prompt", type=str, default=None, help="Single prompt text for generation"
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="JSONL file with prompts (one JSON object per line with 'messages' field)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="intactkv_eval_results.jsonl",
        help="Output file for generation results",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32768,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.6, help="Sampling temperature"
    )
    parser.add_argument("--top-p", type=float, default=0.95, help="Sampling top_p")
    parser.add_argument(
        "--use-intactkv",
        action="store_true",
        help="Enable IntactKV: precompute pivot token KV with FP16 model",
    )
    parser.add_argument(
        "--intactkv-pivot-len",
        type=int,
        default=None,
        help="Number of initial tokens to protect (default: full prompt length)",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable thinking mode for reasoning models (e.g., Qwen3)",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile decode function for faster inference",
    )
    args = parser.parse_args()

    # Prepare prompts
    prompts = []
    if args.prompt:
        prompts.append([{"role": "user", "content": args.prompt}])
    elif args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                messages = data.get("messages", data.get("prompt"))
                if isinstance(messages, str):
                    messages = [{"role": "user", "content": messages}]
                prompts.append(messages)
    else:
        # Default test prompt for long CoT evaluation
        prompts.append([
            {
                "role": "user",
                "content": (
                    "Think step by step before answering. "
                    "What are the primary causes of the Industrial Revolution, "
                    "and how did they interact to produce rapid economic growth in Europe?"
                ),
            }
        ])

    print(f"Loading model from {args.model} ...")
    generator = TransformersGenerator(
        model=args.model,
        compile_decode=args.compile,
        enable_thinking=args.enable_thinking,
        use_intactkv=args.use_intactkv,
        intactkv_pivot_len=args.intactkv_pivot_len,
    )

    params = GenerationParams(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    results = []
    for idx, messages in enumerate(prompts):
        print(f"\n[{idx + 1}/{len(prompts)}] Generating with IntactKV={'ON' if args.use_intactkv else 'OFF'} ...")
        start = time.time()
        import asyncio

        async def run_gen():
            return await generator.generate(messages, params)

        result = asyncio.run(run_gen())
        elapsed = time.time() - start

        print(f"Generated {result.stats.token_count} tokens in {elapsed:.2f}s "
              f"({result.stats.tokens_per_second:.2f} tok/s)")
        print("-" * 40)
        print(result.output_text[:500] + "..." if len(result.output_text) > 500 else result.output_text)
        print("-" * 40)

        results.append({
            "messages": messages,
            "output": result.output_text,
            "token_count": result.stats.token_count,
            "tokens_per_second": result.stats.tokens_per_second,
            "total_time_s": result.stats.total_time_s,
            "ttft_s": result.stats.ttft_s,
            "intactkv": args.use_intactkv,
            "pivot_len": args.intactkv_pivot_len,
        })

    with open(args.output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()
