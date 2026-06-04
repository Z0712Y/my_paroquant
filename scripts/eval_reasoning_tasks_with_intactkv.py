#============intactKV：新建文件=====
# 这是为了 intactKV 新建的文件，目的是在 transformers_backend 上复现 MMLU-PRO、
# AIME-2024/2025、GPQA-Diamond 等推理任务的评估，支持 IntactKV 注入，
# 可与原版 vLLM 后端的 reasoning.sh 结果进行直接对比。
#============intactKV：新建文件=====

import argparse
import asyncio
import json
import os
import random
import re
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from inference_engine.generation.transformers_backend import (
    TransformersGenerator,
    GenerationParams,
)


def extract_boxed_answer(text: str) -> str:
    """从生成文本中提取 \boxed{answer}"""
    matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    if matches:
        return matches[-1].strip()
    return text.strip()


def extract_letter_answer(text: str) -> str:
    """从生成文本中提取 'Answer: X' 格式的答案"""
    lines = text.strip().split("\n")
    for line in reversed(lines):
        match = re.search(r"Answer:\s*([A-D])", line, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return ""


def load_task_dataset(dataset_name: str, cache_dir: str = None):
    """加载数据集，复用 reasoning.py 中的配置"""
    if dataset_name == "AIME-2024":
        ds = load_dataset("Maxwell-Jia/AIME_2024", "default", split="train", cache_dir=cache_dir)
        return ds, "aime24"
    elif dataset_name == "AIME-2025":
        ds = load_dataset("yentinglin/aime_2025", "default", split="train", cache_dir=cache_dir)
        return ds, "aime25"
    elif dataset_name == "GPQA-Diamond":
        local_path = os.path.join(os.path.dirname(__file__), "..", "experiments", "tasks", "reasoning", "local_datasets", "gpqa_diamond")
        if os.path.exists(local_path):
            ds = load_dataset(local_path, split="train", cache_dir=cache_dir)
        else:
            ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train", cache_dir=cache_dir, trust_remote_code=True)
        return ds, "gpqa"
    elif dataset_name == "MMLU-PRO":
        ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test", cache_dir=cache_dir)
        return ds, "mmlu_pro"
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def build_prompt(item, dataset_name: str, seed: int = 42):
    """构建 prompt，复用 reasoning.py 中的 prompt 模板逻辑"""
    random.seed(seed)

    if dataset_name in ("AIME-2024", "aime24"):
        query = f"{item['Problem']}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        gold = str(item["Answer"])
        return query, gold, "boxed"

    elif dataset_name in ("AIME-2025", "aime25"):
        query = f"{item['problem']}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        gold = str(item["answer"])
        return query, gold, "boxed"

    elif dataset_name in ("GPQA-Diamond", "gpqa"):
        gold_index = random.randint(0, 3)
        choices = [
            item["Incorrect Answer 1"],
            item["Incorrect Answer 2"],
            item["Incorrect Answer 3"],
        ]
        choices.insert(gold_index, item["Correct Answer"])
        query_template = (
            "Answer the following multiple choice question. The last line of your response should be "
            "of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. "
            "Think step by step before answering.\n\n{Question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}"
        )
        query = query_template.format(
            A=choices[0], B=choices[1], C=choices[2], D=choices[3], Question=item["Question"]
        )
        gold_letter = chr(ord("A") + gold_index)
        return query, gold_letter, "letter"

    elif dataset_name in ("MMLU-PRO", "mmlu_pro"):
        original_options = item["options"]
        original_answer_label = item["answer"]
        n = len(original_options)
        original_index = ord(original_answer_label) - ord("A")
        correct_answer_text = original_options[original_index]
        shuffled_options = original_options.copy()
        random.shuffle(shuffled_options)
        gold_index = shuffled_options.index(correct_answer_text)
        labels = [chr(ord("A") + i) for i in range(n)]
        options_str = "\n".join(f"{label}) {opt}" for label, opt in zip(labels, shuffled_options))
        valid_letters_str = "".join(labels)
        query_template = (
            "Answer the following multiple choice question. Think step by step before answering. "
            "The last line of your response should be of the following format: 'Answer: $LETTER' "
            "(without quotes) where LETTER is one of {valid_letters}.\n\n"
            "{Question}\n\n{Options}"
        )
        query = query_template.format(
            Question=item["question"], Options=options_str, valid_letters=valid_letters_str
        )
        gold_letter = chr(ord("A") + gold_index)
        return query, gold_letter, "letter"

    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")


async def evaluate_dataset(args):
    """主评估逻辑"""
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

    ds, task_type = load_task_dataset(args.dataset)
    if args.max_samples is not None:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    results = []
    correct = 0
    total = 0

    for idx in tqdm(range(len(ds)), desc=f"Evaluating {args.dataset}"):
        item = ds[idx]
        query, gold, answer_type = build_prompt(item, task_type, seed=args.seed)

        messages = [{"role": "user", "content": query}]
        result = await generator.generate(messages, params)
        output_text = result.output_text

        if answer_type == "boxed":
            pred = extract_boxed_answer(output_text)
            # 尝试数值比较
            is_correct = False
            try:
                if abs(float(pred) - float(gold)) < 1e-5:
                    is_correct = True
            except (ValueError, TypeError):
                is_correct = pred.strip() == gold.strip()
        else:
            pred = extract_letter_answer(output_text)
            is_correct = pred == gold

        if is_correct:
            correct += 1
        total += 1

        results.append({
            "idx": idx,
            "query": query,
            "gold": gold,
            "prediction": pred,
            "generated_text": output_text,
            "correct": is_correct,
        })

    accuracy = correct / total if total > 0 else 0.0
    print(f"\n{args.dataset} Accuracy: {correct}/{total} = {accuracy:.4f}")

    output = {
        "dataset": args.dataset,
        "model": args.model,
        "intactkv": args.use_intactkv,
        "pivot_len": args.intactkv_pivot_len,
        "seed": args.seed,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "details": results,
    }

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {args.output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate reasoning tasks (MMLU-Pro/AIME/GPQA) with IntactKV on transformers_backend"
    )
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["AIME-2024", "AIME-2025", "GPQA-Diamond", "MMLU-PRO"],
        help="Dataset to evaluate",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to evaluate")
    parser.add_argument("--max-new-tokens", type=int, default=32768, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="Sampling top_p")
    parser.add_argument("--output-file", type=str, default=None, help="Output JSON file")
    parser.add_argument("--use-intactkv", action="store_true", help="Enable IntactKV")
    parser.add_argument("--intactkv-pivot-len", type=int, default=None, help="Pivot token length")
    parser.add_argument("--enable-thinking", action="store_true", help="Enable thinking mode")
    parser.add_argument("--compile", action="store_true", help="Compile decode function")
    parser.add_argument("--cache-dir", type=str, default=None, help="HuggingFace cache dir")
    args = parser.parse_args()

    if args.output_file is None:
        model_name = Path(args.model).name
        suffix = "_intactkv" if args.use_intactkv else "_baseline"
        args.output_file = f"./outputs/{args.dataset}_{model_name}_seed{args.seed}{suffix}.json"

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    torch.set_grad_enabled(False)
    random.seed(args.seed)
    asyncio.run(evaluate_dataset(args))


if __name__ == "__main__":
    main()
