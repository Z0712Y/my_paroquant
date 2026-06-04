#============guidedquant：这是为了guidedquant新建的文件，目的：实现End Loss Guidance中的saliency提取、缓存和加载=====
"""
Saliency extraction module for ParoQuant.
Inspired by GuidedQuant's End Loss Guidance (ICML 2025).

Purpose: Extract end-to-end loss gradients (saliency) from each Transformer layer's
output activations via one full forward+backward pass on calibration data. These
saliency vectors are then used to weight the per-layer reconstruction loss in
optimize_module, so that hidden dimensions more sensitive to final model loss receive
higher optimization priority.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Optional

from .util import get_blocks, logger


@torch.enable_grad()
def extract_layer_saliency(
    model: nn.Module,
    samples: torch.Tensor,
    batch_size: int,
    device: str = "cuda",
) -> Dict[int, torch.Tensor]:
    """
    在原始 FP16 模型上做完整的前向+反向传播，提取每层输出激活的梯度作为 saliency。

    Args:
        model: 原始 FP16 模型（完整模型）。
        samples: 校准数据 token ids，shape (N, T)。
        batch_size: 反向传播时的 batch size（建议 1~4，控制显存）。
        device: 计算设备。

    Returns:
        layer_saliency: dict, key 为 layer_idx (int), value 为 saliency 向量 (hidden_size,)
                        saliency[j] = E[|dL/dz_j|]，表示第 j 个 hidden dim 对最终损失的敏感度。
    """
    model.to(device)
    model.train()

    layer_saliency: Dict[int, list[torch.Tensor]] = {}
    hooks = []

    def make_forward_hook(layer_idx: int):
        def hook(module, inp, out):
            # out 可能是 tuple（如 Transformer layer 输出 hidden_states + present_kv_cache）
            hidden = out[0] if isinstance(out, tuple) else out

            def bw_hook(grad: torch.Tensor):
                # grad shape: (batch, seq_len, hidden_size)
                # 在 batch 和 seq 维度取绝对值后平均，得到每个 hidden dim 的敏感度
                sal = grad.abs().mean(dim=(0, 1)).detach().cpu()
                layer_saliency.setdefault(layer_idx, []).append(sal)

            hidden.register_hook(bw_hook)

        return hook

    blocks = get_blocks(model)
    for idx, block in enumerate(blocks):
        hooks.append(block.register_forward_hook(make_forward_hook(idx)))

    num_samples = samples.shape[0]
    for i in range(0, num_samples, batch_size):
        batch = samples[i : i + batch_size].to(device)
        # 完整前向传播
        outputs = model(batch, use_cache=False)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

        # 标准语言建模 loss（next token prediction）
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="mean",
        )
        # 反向传播，触发所有注册的 bw_hook，收集各层 saliency
        loss.backward()
        model.zero_grad(set_to_none=True)

    for h in hooks:
        h.remove()

    # 跨 batch 平均
    result = {
        idx: torch.stack(sals).mean(dim=0).to(device)
        for idx, sals in layer_saliency.items()
    }

    model.eval()
    model.cpu()
    torch.cuda.empty_cache()

    logger.info(
        f"Extracted saliency for {len(result)} layers using "
        f"{num_samples} samples (batch_size={batch_size})."
    )
    return result


def save_saliency(saliency: Dict[int, torch.Tensor], path: str):
    """将 saliency 字典保存到磁盘。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(saliency, path)
    logger.info(f"Saliency saved to {path}")


def load_saliency(path: str, device: str = "cuda") -> Dict[int, torch.Tensor]:
    """从磁盘加载 saliency 字典。"""
    saliency = torch.load(path, map_location=device)
    logger.info(f"Saliency loaded from {path}")
    return saliency
