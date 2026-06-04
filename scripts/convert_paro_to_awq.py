#!/usr/bin/env python3
"""
将 ParoQuant 生成的 AWQ-LLM 格式权重转换为 AutoAWQ 兼容格式。
主要转换：qweight、scales、scaled_zeros -> qweight、qzeros、scales。
"""
import argparse
import json
import shutil
from pathlib import Path
from typing import Dict

import torch
from safetensors.torch import load_file, save_file
from safetensors import safe_open
from huggingface_hub import snapshot_download
from inference_engine.utils.awq_conversion_utils import (
    convert_awq_llm_module_to_autoawq,
)


# ==========================================
# 模块一：单模块权重转换
# ==========================================
def convert_one_module(
    tensors: Dict[str, torch.Tensor],
    prefix: str,
    w_bit: int,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    从 safetensors 字典中提取指定前缀模块的量化参数，
    并调用转换工具生成 AutoAWQ 所需的三个张量。
    """
    # 根据前缀拼接出 ParoQuant 格式的三个 key
    qweight_key = f"{prefix}qlinear.qweight"
    scales_key = f"{prefix}qlinear.scales"
    scaled_zeros_key = f"{prefix}qlinear.scaled_zeros"

    qweight = tensors[qweight_key]
    scales = tensors[scales_key]
    scaled_zeros = tensors[scaled_zeros_key]

    return convert_awq_llm_module_to_autoawq(
        qweight=qweight,
        scales=scales,
        scaled_zeros=scaled_zeros,
        w_bit=w_bit,
        group_size=group_size,
    )


# ==========================================
# 模块二：单个 safetensors 文件转换
# ==========================================
def convert_file(
    in_path: Path,
    out_path: Path,
    w_bit: int,
    group_size: int,
) -> None:
    """读取单个 safetensors 文件，遍历其中所有量化模块并完成格式转换后保存。"""
    # 加载输入文件中的所有张量
    tensors = load_file(in_path)
    # 输出张量先复制原始内容，再逐步替换/删除需要转换的 key
    out_tensors: Dict[str, torch.Tensor] = dict(tensors)

    # 通过匹配 key 后缀 "qlinear.qweight" 找到所有需要转换的模块前缀
    prefixes = []
    for key in tensors.keys():
        if key.endswith("qlinear.qweight"):
            prefixes.append(key[: -len("qlinear.qweight")])

    for prefix in prefixes:
        # 目标 AutoAWQ 格式的 key
        qzeros_key = f"{prefix}qlinear.qzeros"
        scaled_zeros_key = f"{prefix}qlinear.scaled_zeros"
        scales_key = f"{prefix}qlinear.scales"
        qweight_key = f"{prefix}qlinear.qweight"

        # 若已存在 qzeros，说明该模块已被转换过，跳过
        if qzeros_key in tensors:
            continue
        # 安全检查：缺失必要张量则跳过
        if scales_key not in tensors or scaled_zeros_key not in tensors:
            continue

        # 调用模块一完成单模块转换
        qweight_awq, qzeros, scales = convert_one_module(
            tensors, prefix, w_bit=w_bit, group_size=group_size
        )
        # 将转换后的张量写入输出字典
        out_tensors[qweight_key] = qweight_awq
        out_tensors[qzeros_key] = qzeros
        out_tensors[scales_key] = scales
        # 删除 ParoQuant 特有的 scaled_zeros，避免 AutoAWQ 加载时出错
        if scaled_zeros_key in out_tensors:
            del out_tensors[scaled_zeros_key]

    # 保留原始 safetensors 的 metadata（如格式版本信息）
    metadata = None
    with safe_open(in_path, framework="pt", device="cpu") as f:
        metadata = f.metadata()

    # 确保输出目录存在并保存转换后的文件
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(out_tensors, out_path, metadata=metadata)


# ==========================================
# 模块三：输入路径解析（支持本地目录和 HuggingFace Hub）
# ==========================================
def resolve_input_dir(input_path: str) -> Path:
    """若输入是本地目录则直接返回；否则从 HuggingFace Hub 下载到缓存目录后返回。"""
    path = Path(input_path)
    if path.exists() and path.is_dir():
        return path
    local_dir = snapshot_download(repo_id=input_path)
    return Path(local_dir)


# ==========================================
# 模块四：主入口与命令行参数解析
# ==========================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert ParoQuant AWQ-LLM weights to AutoAWQ-compatible tensors."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Local directory or HuggingFace repo id.",
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--w-bit", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    args = parser.parse_args()

    in_dir = resolve_input_dir(args.input)
    out_dir = Path(args.output_dir)

    # 创建输出目录
    out_dir.mkdir(parents=True, exist_ok=True)
    # 复制所有非 safetensors 文件（如 config.json、tokenizer 文件等）
    for item in in_dir.iterdir():
        if item.is_file() and item.suffix != ".safetensors":
            shutil.copy2(item, out_dir / item.name)

    # 查找所有 safetensors 文件并逐个转换
    files = sorted(in_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors found in {in_dir}")

    for in_path in files:
        out_path = out_dir / in_path.name
        convert_file(
            in_path,
            out_path,
            w_bit=args.w_bit,
            group_size=args.group_size,
        )

    # ==========================================
    # 模块五：更新 safetensors 索引文件
    # ==========================================
    index_in = in_dir / "model.safetensors.index.json"
    index_out = out_dir / "model.safetensors.index.json"
    if index_in.exists():
        with index_in.open("r", encoding="utf-8") as f:
            index_data = json.load(f)

        weight_map = index_data.get("weight_map", {})
        updated_map = dict(weight_map)
        # 将 scaled_zeros 的条目替换为 qzeros，保持 shard 映射正确
        for key, shard in weight_map.items():
            if key.endswith("qlinear.scaled_zeros"):
                new_key = key.replace("qlinear.scaled_zeros", "qlinear.qzeros")
                updated_map[new_key] = shard
                if key in updated_map:
                    del updated_map[key]

        index_data["weight_map"] = updated_map
        with index_out.open("w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
