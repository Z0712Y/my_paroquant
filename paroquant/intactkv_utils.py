#============intactKV：新建文件===== 
# 这是为了 intactKV 新建的文件，目的是提供 Pivot Token 识别、FP16 预计算 KV、
# 以及向 StaticCache 注入 IntactKV 的工具函数，用于在 ParoQuant 权重量化场景下
# 保护关键初始 token 的 KV 不受量化误差影响。
#============intactKV：新建文件=====

import torch
import torch.nn as nn
from typing import Optional, List, Union
from transformers import AutoModelForCausalLM, AutoTokenizer


def compute_pivot_token_mask(
    input_ids: torch.Tensor,
    tokenizer: AutoTokenizer,
    pivot_mode: str = "prefix",
    pivot_len: Optional[int] = None,
    pivot_texts: Optional[List[str]] = None,
) -> torch.Tensor:
    """
    计算 pivot token mask，标记序列中哪些位置是 pivot tokens。

    Args:
        input_ids: (batch_size, seq_len) 的 token id 张量
        tokenizer: 分词器
        pivot_mode: "prefix" | "bos" | "auto"
            - "prefix": 前 pivot_len 个 token 为 pivot
            - "bos": 只有 BOS token 为 pivot
            - "auto": BOS + 系统提示/指令模板（需传入 pivot_texts）
        pivot_len: 在 "prefix" 模式下指定前多少个 token 为 pivot
        pivot_texts: 在 "auto" 模式下，匹配这些文本对应的 token 作为 pivot

    Returns:
        mask: (batch_size, seq_len) 的 bool 张量，True 表示 pivot token
    """
    batch_size, seq_len = input_ids.shape
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=input_ids.device)

    if pivot_mode == "prefix":
        if pivot_len is None:
            pivot_len = min(16, seq_len // 4)
        pivot_len = min(pivot_len, seq_len)
        mask[:, :pivot_len] = True

    elif pivot_mode == "bos":
        bos_id = tokenizer.bos_token_id
        if bos_id is not None:
            mask = input_ids == bos_id
        else:
            mask[:, 0] = True

    elif pivot_mode == "auto":
        if pivot_texts is not None and len(pivot_texts) > 0:
            for text in pivot_texts:
                pivot_ids = tokenizer.encode(text, add_special_tokens=False)
                if len(pivot_ids) == 0:
                    continue
                for b in range(batch_size):
                    seq = input_ids[b].tolist()
                    for start in range(len(seq) - len(pivot_ids) + 1):
                        if seq[start:start + len(pivot_ids)] == pivot_ids:
                            mask[b, start:start + len(pivot_ids)] = True
        else:
            if tokenizer.bos_token_id is not None:
                mask = input_ids == tokenizer.bos_token_id
            else:
                mask[:, 0] = True
    else:
        raise ValueError(f"Unknown pivot_mode: {pivot_mode}")

    return mask


@torch.no_grad()
def compute_intactkv(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    prompt_texts: Union[str, List[str]],
    device: Union[str, torch.device] = "cuda",
    dtype: torch.dtype = torch.float16,
    max_length: Optional[int] = None,
) -> tuple[List[torch.Tensor], List[torch.Tensor], int]:
    """
    使用 FP16（未量化）模型对 prompt 进行 prefill，获取每一层的 KV cache。

    Args:
        model: FP16 模型（AutoModelForCausalLM）
        tokenizer: 分词器
        prompt_texts: 单个 prompt 字符串或列表
        device: 计算设备
        dtype: 数据类型
        max_length: 最大长度限制

    Returns:
        key_caches: 每层 key cache 的列表
        value_caches: 每层 value cache 的列表
        seq_len: prompt 的序列长度
    """
    if isinstance(prompt_texts, str):
        prompt_texts = [prompt_texts]

    inputs = tokenizer(
        prompt_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)

    model.eval()
    model.to(device).to(dtype)

    with torch.no_grad():
        outputs = model(**inputs, use_cache=True, output_hidden_states=False)

    past_key_values = outputs.past_key_values
    key_caches = []
    value_caches = []

    for layer_kv in past_key_values:
        key_caches.append(layer_kv[0].cpu().clone())
        value_caches.append(layer_kv[1].cpu().clone())

    seq_len = inputs.input_ids.shape[1]
    return key_caches, value_caches, seq_len


@torch.no_grad()
def inject_intactkv_to_static_cache(
    key_caches: List[torch.Tensor],
    value_caches: List[torch.Tensor],
    static_cache,
    pivot_len: int,
    batch_offset: int = 0,
):
    """
    将预计算的 IntactKV（DynamicCache 格式）注入到 StaticCache 中。

    Args:
        key_caches: 每层 key cache 列表，形状 (batch, num_heads, seq_len, head_dim)
        value_caches: 每层 value cache 列表
        static_cache: StaticCache 实例
        pivot_len: 要注入的 pivot token 数量
        batch_offset: batch 维度偏移（单 batch 时为 0）
    """
    for layer_idx in range(len(key_caches)):
        k_src = key_caches[layer_idx].to(static_cache.key_cache[layer_idx].device)
        v_src = value_caches[layer_idx].to(static_cache.value_cache[layer_idx].device)

        if k_src.dim() == 4:
            batch_size = k_src.shape[0]
            for b in range(batch_size):
                static_cache.key_cache[layer_idx][
                    batch_offset + b, :, :pivot_len, :
                ] = k_src[b, :, :pivot_len, :]
                static_cache.value_cache[layer_idx][
                    batch_offset + b, :, :pivot_len, :
                ] = v_src[b, :, :pivot_len, :]
        else:
            static_cache.key_cache[layer_idx][:, :, :pivot_len, :] = k_src[:, :, :pivot_len, :]
            static_cache.value_cache[layer_idx][:, :, :pivot_len, :] = v_src[:, :, :pivot_len, :]


@torch.no_grad()
def inject_intactkv_to_dynamic_cache(
    key_caches: List[torch.Tensor],
    value_caches: List[torch.Tensor],
    dynamic_cache,
):
    """
    将预计算的 IntactKV 注入到 DynamicCache 中。
    """
    for layer_idx in range(len(key_caches)):
        k = key_caches[layer_idx].to(dynamic_cache.key_cache[layer_idx].device)
        v = value_caches[layer_idx].to(dynamic_cache.value_cache[layer_idx].device)
        dynamic_cache.key_cache[layer_idx] = k
        dynamic_cache.value_cache[layer_idx] = v
    if hasattr(dynamic_cache, "_seen_tokens"):
        dynamic_cache._seen_tokens = key_caches[0].shape[2]
