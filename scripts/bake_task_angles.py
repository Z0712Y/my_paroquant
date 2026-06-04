#!/usr/bin/env python3
"""
将任务特定角度残差合并到 angles_grouped 中，生成任务专用的 .pt 文件。
这样后续的 real_quant.py 和推理引擎无需任何修改即可支持任务自适应。
"""
import argparse
import torch
from pathlib import Path


def bake_angles(pt_dir: str, task_type: str, output_dir: str):
    """
    Args:
        pt_dir: 训练输出目录（包含 layer_idx.name.pt 文件）
        task_type: 'short' | 'long'
        output_dir: 烘焙后的输出目录
    """
    pt_dir = Path(pt_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pt_files = sorted(pt_dir.glob("*.pt"))
    if not pt_files:
        raise ValueError(f"No .pt files found in {pt_dir}")

    for pt_file in pt_files:
        sd = torch.load(pt_file, map_location="cpu")

        has_residual = "angles_residual_short" in sd or "angles_residual_long" in sd
        if not has_residual:
            # 没有任务自适应参数，直接复制
            torch.save(sd, output_dir / pt_file.name)
            continue

        if task_type == "short" and "angles_residual_short" in sd:
            sd["angles_grouped"] = sd["angles_grouped"] + sd["angles_residual_short"]
        elif task_type == "long" and "angles_residual_long" in sd:
            sd["angles_grouped"] = sd["angles_grouped"] + sd["angles_residual_long"]

        # 删除残差参数以减小文件大小并避免推理引擎混淆
        sd.pop("angles_residual_short", None)
        sd.pop("angles_residual_long", None)
        # 删除任务类型标识（已无用）
        sd.pop("task_type_id", None)

        torch.save(sd, output_dir / pt_file.name)

    print(f"Successfully baked {len(pt_files)} files for task={task_type} -> {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bake task-specific angle residuals into angles_grouped for inference."
    )
    parser.add_argument(
        "--pt-dir",
        type=str,
        required=True,
        help="Path to the training output directory containing .pt files",
    )
    parser.add_argument(
        "--task-type",
        type=str,
        choices=["short", "long"],
        required=True,
        help="Which task residual to bake: 'short' (non-reasoning) or 'long' (reasoning)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for baked .pt files",
    )
    args = parser.parse_args()
    bake_angles(args.pt_dir, args.task_type, args.output_dir)
