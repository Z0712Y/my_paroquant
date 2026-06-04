import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
import os
import gc
from typing import Optional, TypeVar
import warnings
import random
import logging
import math
from tqdm import tqdm


# ==========================================
# 模块一：模型结构遍历工具
# ==========================================
def get_blocks(model: nn.Module) -> nn.ModuleList:
    """获取 Transformer 模型的 decoder layers（如 Llama/Qwen 系列）。"""
    model_class_name = model.__class__.__name__
    if model_class_name in (
        "Qwen2ForCausalLM",
        "Qwen3ForCausalLM",
        "Qwen3MoeForCausalLM",
        "LlamaForCausalLM",
    ):
        m = model.model
    else:
        raise NotImplementedError(type(model))

    return m.layers


_Linear_T = TypeVar("Linear", bound=nn.Module)


def get_named_linears(
    module: nn.Module, subclass: type[_Linear_T] = nn.Linear
) -> dict[str, _Linear_T]:
    """递归遍历模块，返回所有指定子类（默认 nn.Linear）的命名映射。"""
    return {name: m for name, m in module.named_modules() if isinstance(m, subclass)}


def get_module_by_name(module, module_name):
    """根据完整名称（如 'layers.0.mlp.gate_proj'）从模型中获取对应模块。"""
    for name, m in module.named_modules():
        if name == module_name:
            return m
    return None


def set_module_by_name(layer, name, new_module):
    """根据完整名称将子模块替换为新的模块实例。"""
    levels = name.split(".")
    if len(levels) > 1:
        mod_ = layer
        # 逐层向下遍历，直到倒数第二层
        for l_idx in range(len(levels) - 1):
            if levels[l_idx].isdigit():
                # 若层级名是数字，说明是 ModuleList/Sequential 的索引
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        # 替换最后一层
        setattr(mod_, levels[-1], new_module)
    else:
        setattr(layer, name, new_module)


# ==========================================
# 模块二：模型与分词器加载
# ==========================================
def load_model(
    model_path: str, device_map: str = None, dtype=torch.float32, **kwargs
) -> nn.Module:
    """从 Hugging Face 预训练路径加载因果语言模型。"""
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map=device_map, torch_dtype=dtype, **kwargs
    )
    return model


def load_tokenizer(model_path: str, **kwargs) -> AutoTokenizer:
    """加载分词器，并将 pad_token 设置为 eos_token（避免 padding 时出错）。"""
    tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# ==========================================
# 模块三：模型组件迁移与缓存清理
# ==========================================
def move_embed(model, device):
    """将模型的 embedding 层和 rotary_emb 移动到指定设备上。"""
    model_class_name = model.__class__.__name__
    if model_class_name in (
        "Qwen2ForCausalLM",
        "Qwen3ForCausalLM",
        "Qwen3MoeForCausalLM",
        "LlamaForCausalLM",
    ):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
        model.model.rotary_emb = model.model.rotary_emb.to(device)
    else:
        raise NotImplementedError(type(model))


def empty_cache():
    """强制触发 Python GC 和 CUDA 显存缓存清理，用于大模型量化时节省显存。"""
    gc.collect()
    torch.cuda.empty_cache()


# ==========================================
# 模块四：校准数据集加载
# ==========================================
def get_mixed_calib_dataset(
    datasets: list[str],
    *,
    tokenizer,
    n_samples: int,
    block_size: int,
    seed: int,
    split: str,
) -> list[torch.Tensor]:
    """从多个数据集中按比例采样、混合并打乱，构建校准数据集。"""
    per_dataset_len = n_samples // len(datasets)
    results = []
    for i, dataset in enumerate(datasets):
        # 最后一个数据集负责补齐剩余样本数，避免除不尽
        dataset_samples = (
            per_dataset_len if i < len(datasets) - 1 else n_samples - len(results)
        )
        results.extend(
            get_calib_dataset(
                data=dataset,
                tokenizer=tokenizer,
                n_samples=dataset_samples,
                block_size=block_size,
                seed=seed,
                split=split,
            )
        )
    assert (
        len(results) == n_samples
    ), f"Expected {n_samples} samples, got {len(results)}"

    rand = random.Random(seed)
    rand.shuffle(results)

    return results


# Adapted from awq-llm
def get_calib_dataset(
    data="pileval",
    *,
    tokenizer,
    n_samples: int,
    block_size: int,
    seed: int,
    split: str,
) -> list[torch.Tensor]:
    """加载单个校准数据集，支持 pileval/wikitext2/c4/redpajama。"""
    if data == "pileval":
        if split != "validation":
            warnings.warn("The split argument is ignored when data is 'pileval'.")
        dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
        dataset = dataset.shuffle(seed=seed)
    elif data == "wikitext2":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        dataset = dataset.shuffle(seed=seed)
    elif data == "c4":
        # 本地数据集路径（优先使用）
        local_c4_dir = "/home/hhw/zy/datasets/c4_local/en"
        # HuggingFace 缓存路径
        cache_dir = os.path.expanduser("~/.cache/huggingface/datasets/allenai___c4/en/1.0.0")

        if split == "train":
            # 优先使用下载的 train 文件 (00001, 00003 等)
            train_files = []
            if os.path.exists(local_c4_dir):
                for f in sorted(os.listdir(local_c4_dir)):
                    if f.startswith("c4-train.") and f.endswith(".json.gz"):
                        train_files.append(os.path.join(local_c4_dir, f))

            if train_files:
                print(f"Loading {len(train_files)} local C4 train files...")
                dataset = load_dataset("json", data_files={"train": train_files}, split="train")
            else:
                # 尝试使用旧的缓存
                train_file = os.path.join(cache_dir, "c4-train.00000-of-01024.json.gz")
                if os.path.exists(train_file):
                    dataset = load_dataset("json", data_files={"train": train_file}, split="train")
                else:
                    raise FileNotFoundError("No local C4 train files found. Please download from ModelScope first.")

        elif split == "validation":
            # 优先使用本地缓存的验证文件
            val_file = os.path.join(local_c4_dir, "c4-validation.00000-of-00008.json.gz")
            if not os.path.exists(val_file):
                val_file = os.path.join(cache_dir, "c4-validation.00000-of-00008.json.gz")

            if os.path.exists(val_file):
                dataset = load_dataset("json", data_files={"validation": val_file}, split="validation")
            else:
                raise FileNotFoundError("No local C4 validation files found.")
        else:
            raise ValueError(f"Invalid split: {split}")

        dataset = dataset.shuffle(seed=seed)
    elif data == "redpajama":
        test_split, val_split = 0.2, 0.1
        dataset = load_dataset(
            "liang2kl/RedPajama-Data-1T-Sample-Backup",
            split="train",
            trust_remote_code=True,
        )
        dataset = dataset.shuffle(seed=seed)
        # 按 70% / 10% / 20% 切分 train / validation / test
        test_size = int(len(dataset) * test_split)
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - test_size - val_size
        if split == "test":
            dataset = dataset.select(range(len(dataset) - test_size, len(dataset)))
        elif split == "validation":
            dataset = dataset.select(
                range(len(dataset) - test_size - val_size, len(dataset) - test_size)
            )
        elif split == "train":
            dataset = dataset.select(range(0, train_size))
        else:
            raise ValueError(f"Invalid split: {split}")
    else:
        raise NotImplementedError

    # 收集样本：逐条读取文本，encode 后拼接成超长序列，再按 block_size 切块
    samples = []
    total_len = 0
    for data in dataset:
        line = data["text"]
        line = line.strip()
        line_encoded = tokenizer.encode(line)
        # 跳过超过 block_size 的单个句子（避免切块时产生不自然边界）
        if len(line_encoded) > block_size:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        total_len += len(line_encoded)
        # 当总长度达到需求时提前停止
        if total_len >= n_samples * block_size:
            break
    # 将所有样本在序列维度上拼接（dim=1），再 squeeze 成 1D 长序列
    samples = torch.cat(samples, dim=1).squeeze(0)
    # 计算最终能切出多少个完整 block
    n_split = min(samples.shape[0] // block_size, n_samples)

    return [samples[i * block_size : (i + 1) * block_size] for i in range(n_split)]


# ==========================================
# 模块五：层输入捕获（用于逐层量化/优化）
# ==========================================
@torch.no_grad()
def catch_first_layer_input(
    model: nn.Module,
    layers: nn.ModuleList,
    samples: torch.Tensor,
    batch_size: Optional[int],
) -> tuple[torch.Tensor, dict]:
    """
    用 hooks 的思想捕获第一个 Transformer layer 的输入和 kwargs。
    实现方式是将 layers[0] 临时替换为一个 Catcher 包装器，
    在前向传播时拦截输入并抛出 ValueError 中断后续计算。
    """
    layer_kwargs = {}
    batched = batch_size is not None
    inps: list[torch.Tensor] = []

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            # 绕过 nn.Module 的 __setattr__，直接通过 object 设置属性
            # 防止触发模块的自定义 setattr 逻辑
            object.__setattr__(self, "module", module)

        def forward(self, inp, **kwargs):
            # 捕获输入张量
            inps.append(inp)
            # 只在第一次调用时保存 kwargs（如 attention_mask, position_ids 等）
            if len(layer_kwargs) == 0:
                layer_kwargs.update(kwargs)
            # 抛出异常以中断后续层的计算，节省显存和时间
            raise ValueError

        def __getattr__(self, name):
            # 对于未在 Catcher 中定义的属性，转发到被包装模块
            return getattr(self.module, name)

    # 临时替换第一层为 Catcher
    layers[0] = Catcher(layers[0])
    # 确定 batch_size：若未指定则使用样本总数
    batch_size = samples.shape[0] if not batched or batch_size <= 0 else batch_size
    num_batches = samples.shape[0] // batch_size
    samples_batch = samples.chunk(num_batches)
    for samples in samples_batch:
        try:
            # 执行前向传播，Catcher 会在第一层抛出 ValueError
            model(samples.to(next(model.parameters()).device))
        except ValueError:
            pass
    # 恢复原始的第一层模块
    layers[0] = layers[0].module
    # 如果非 batched，直接返回第一个捕获的输入（通常就是唯一一个）
    if not batched:
        inps = inps[0]

    # 禁用 KV Cache，删除与缓存相关的关键字参数
    layer_kwargs["use_cache"] = False
    if "past_key_value" in layer_kwargs:
        del layer_kwargs["past_key_value"]
    if "past_key_values" in layer_kwargs:
        del layer_kwargs["past_key_values"]

    return inps, layer_kwargs


# ==========================================
# 模块六：分片张量缓存（显存优化）
# ==========================================
class CachedTensorShards:
    """
    将大批次张量分成多个 shard，大部分 offload 到 CPU，
    仅将当前需要的一个 shard 加载到 target_device（如 GPU），
    用于大模型逐层优化时显存受限的场景。
    """
    def __init__(
        self,
        batches: list[torch.Tensor],
        num_shards: int,
        *,
        target_device: torch.device,
        offload_device: torch.device = torch.device("cpu"),
    ):
        assert len(batches) % num_shards == 0
        # 若数据不在 offload_device 上，先转移过去
        if batches[0].device != offload_device:
            self.batches = [b.to(offload_device) for b in batches]
        else:
            self.batches = batches
        self.num_shards = num_shards
        self.current_shard: int = None
        self.cached_shard: list[torch.Tensor] = None
        self.target_device = target_device

    def _switch_shard(self, shard_index: int) -> None:
        """切换当前加载到 target_device 的 shard。"""
        if self.current_shard == shard_index:
            return
        self.current_shard = shard_index
        start, end = self._get_shard_range(shard_index)
        self.cached_shard = self.batches[start:end]
        # 将该 shard 的所有 batch 加载到目标设备
        self.cached_shard = [b.to(self.target_device) for b in self.cached_shard]

    def _get_shard_range(self, index: int) -> tuple[int, int]:
        """计算给定 shard 索引对应的 batch 范围 [start, end)。"""
        if self.num_shards == 1:
            return 0, len(self.batches)
        shard_size = len(self.batches) // self.num_shards
        start = shard_size * index
        if index == self.num_shards - 1:
            # 最后一个 shard 负责剩余所有 batch（处理除不尽的情况）
            end = len(self.batches)
        else:
            end = shard_size * (index + 1)
        return start, end

    def __getitem__(self, index: int) -> torch.Tensor:
        """支持下标访问，自动完成 shard 切换。"""
        shard_len = len(self.batches) // self.num_shards
        shard_index = index // shard_len
        if self.current_shard != shard_index:
            self._switch_shard(shard_index)
        offset = index % shard_len
        return self.cached_shard[offset]

    def __iter__(self) -> "Iterator":
        return self.Iterator(self)

    def __len__(self) -> int:
        return len(self.batches)

    class Iterator:
        """自定义迭代器，兼容 __getitem__ 的 shard 切换逻辑。"""
        def __init__(self, batches: "CachedTensorShards"):
            self.batches = batches
            self.current_index = 0

        def __iter__(self):
            return self

        def __next__(self) -> torch.Tensor:
            if self.current_index >= len(self.batches):
                raise StopIteration
            result = self.batches[self.current_index]
            self.current_index += 1
            return result

        def __len__(self) -> int:
            return len(self.batches)


# ==========================================
# 模块七：STE（Straight-Through Estimator）工具函数
# ==========================================
def round_ste(x: torch.Tensor) -> torch.Tensor:
    """
    可导的四舍五入函数。
    前向传播时返回 round(x)，反向传播时梯度直接穿透（等价于 identity）。
    公式: (round(x) - x).detach() + x
          = round(x).detach() - x.detach() + x
    其中 detach() 部分在前向等价于 round(x)，但在反向时不产生梯度；
    最后的 +x 保证了梯度为 1。
    """
    return (x.round() - x).detach() + x


def clamp_ste(
    x: torch.Tensor,
    min: Optional[torch.Tensor] = None,
    max: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    可导的裁剪函数（Straight-Through Estimator）。
    前向传播返回 clamp(x, min, max)，反向传播梯度直接穿透。
    """
    return (x.clamp(min, max) - x).detach() + x


# ==========================================
# 模块八：余弦退火参数调度器
# ==========================================
class CosineAnnealingParam:
    """简化版的余弦退火调度器，用于逐层优化时动态调整学习率。"""
    def __init__(self, start_value: float, end_value: float, T_max: int):
        """
        Args:
            start_value (float): The initial value (equivalent to eta_max).
            end_value (float): The final value (equivalent to eta_min).
            T_max (int): Maximum number of steps.
        """
        self.start_value = start_value
        self.end_value = end_value
        self.T_max = T_max
        self._step = -1

    def step(self) -> float:
        self._step += 1

        if self._step >= self.T_max:
            return self.end_value

        # 标准余弦退火公式：
        # value = end + (start - end) * (1 + cos(pi * step / T_max)) / 2
        cos_val = math.cos(math.pi * self._step / self.T_max)
        return self.end_value + (self.start_value - self.end_value) * (1 + cos_val) / 2


# ==========================================
# 模块九：日志工具（兼容 tqdm 进度条）
# ==========================================
class TqdmLoggingHandler(logging.Handler):
    """自定义 logging Handler，确保日志输出不会破坏 tqdm 进度条的显示。"""
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            # 使用 tqdm.write 输出，避免与进度条冲突
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


def get_logger(name: str) -> logging.Logger:
    """获取配置好的 Logger 实例，使用 TqdmLoggingHandler。"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    handler = TqdmLoggingHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger


# 全局 logger 实例
logger = get_logger("ParoQuant")
