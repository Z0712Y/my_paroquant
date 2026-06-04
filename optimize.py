import torch
import torch.nn as nn
import simple_parsing
from dataclasses import dataclass, field
from tqdm import tqdm
from pathlib import Path
import json
from typing import Literal, Optional
import sys

sys.path.append(Path(__file__).parent.as_posix())

from paroquant.optimize import (
    optimize_module,
    get_random_rotation_pairs,
)
from paroquant.module import (
    PseudoQuantizedLinear,
    reset_angles_by_mask,
)
from paroquant.util import (
    set_module_by_name,
    load_model,
    move_embed,
    load_tokenizer,
    get_blocks,
    get_calib_dataset,
    get_mixed_calib_dataset,
    catch_first_layer_input,
    get_named_linears,
    empty_cache,
    logger,
    CachedTensorShards,
)
from paroquant.convert_utils import transform_to_kernel_data


@dataclass(kw_only=True)
class Config:
    # Huggingface model path.
    model: str
    # The parameters to optimize at each stage and the corresponding learning rates,
    # e.g., --params "channel_scales:0.05,angles:0.05" "weight:1e-5,quantizer:1e-6"
    params: list[str]
    # The number of epochs for each stage of optimization,
    # e.g., --epochs 10 10
    epochs: list[int]

    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-10
    # Loss function to use.
    loss: Literal["mse", "smooth_l1"] = "smooth_l1"

    # Quantization & rotation group size.
    group_size: int
    # Bit width.
    n_bit: int
    # Number of rotations.
    num_rotations: int

    skipped_modules: list[str] = field(default_factory=list)

    # Calibration datasets. If more than one dataset is provided,
    # they will be sampled evenly and shuffled.
    datasets: list[str]
    val_dataset: str
    train_size: int
    validation_size: int
    batch_size: int
    val_batch_size: Optional[int] = None  # Defaults to batch_size if not set.
    seqlen: int

    # Number of shards to cache the input/output tensors. At any time, only one shard
    # will be moved to GPU for optimization. The rest will be kept in CPU memory.
    # Increasing this reduces GPU memory usage but increases training time.
    cache_shards: int = 1

    # Directory to save state dicts of optimized linear layers.
    output_dir: str

    # Whether to resume from previously saved results in `output_dir`.
    resume: bool = False
    # Whether to enable gradient checkpointing.
    checkpointing: bool = False

    seed: int

    # 通道对选择策略: "random" 保留原随机洗牌; "greedy" 使用基于显著性分数的贪心选择
    selection_mode: str = "random"
    # 显著性分数计算指标: "l2" | "maxabs" | "var"，仅在 selection_mode="greedy" 时生效
    metric: str = "l2"

    # ===== 以下为三合一方案新增配置（方案1 + 方案3 + 方案5）=====
    # 双路径任务数据集
    short_datasets: list[str] = field(default_factory=lambda: ["wikitext2"])
    long_datasets: list[str] = field(default_factory=lambda: ["c4"])
    # Prefix 文本（会在每个校准样本前拼接）
    short_prefix: str = ""
    long_prefix: str = ""
    # 双路径损失权重（short 通常设为更高以补足非推理任务短板）
    task_lambda_short: float = 1.0
    task_lambda_long: float = 1.0
    # Prefix-Activation-Aware 显著性融合系数（方案1）
    prefix_alpha: float = 1.0
    prefix_beta: float = 0.0
    # 是否启用任务自适应角度残差（方案5）
    enable_task_adaptive: bool = False


def main():
    args = simple_parsing.parse(
        Config, add_option_string_dash_variants=simple_parsing.DashVariant.DASH
    )
    print(args)

    # Store the results in a subdirectory.
    model_name = args.model.split("/")[-1]
    output_dir = Path(args.output_dir)
    output_dir = output_dir / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Currently only support single GPU training.
    device = "cuda"

    # Determine which params to optimize.
    params_to_optimize: list[dict[str, float]] = []
    for params in args.params:
        params = params.strip().split(",")
        param_dict = {}
        for param in params:
            param, lr = param.strip().split(":")
            param_dict[param.strip()] = float(lr.strip())
        params_to_optimize.append(param_dict)
    print(f"Parameters to optimize: {params_to_optimize}")

    # Save args to output directory
    with open(output_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Load model.
    model = load_model(args.model, device_map="cpu", dtype=torch.float16)
    move_embed(model, device)
    tokenizer = load_tokenizer(args.model)
    blocks = get_blocks(model)

    # ===== 三合一方案：双路径校准数据加载与 Prefix 拼接（方案3）=====
    def concat_prefix_to_samples(samples, prefix_text, seqlen, tokenizer, device):
        """在校准样本前拼接 Prefix token，并截断/填充到 seqlen。"""
        if not prefix_text:
            return torch.stack(samples, dim=0).to(device)
        prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
        prefix = torch.tensor(prefix_ids, dtype=torch.long)  # 留在 CPU，sample 也在 CPU
        result = []
        for sample in samples:
            sample = sample.squeeze()
            max_orig_len = seqlen - len(prefix_ids)
            if max_orig_len <= 0:
                # Prefix 比 seqlen 还长，只取 prefix 前 seqlen
                result.append(prefix[:seqlen])
                continue
            if len(sample) > max_orig_len:
                sample = sample[:max_orig_len]
            combined = torch.cat([prefix, sample])
            if len(combined) < seqlen:
                pad_len = seqlen - len(combined)
                combined = torch.nn.functional.pad(combined, (0, pad_len), value=tokenizer.pad_token_id or 0)
            result.append(combined[:seqlen])
        return torch.stack(result).to(device)

    # Short path datasets (非推理任务)
    logger.info("Loading short-path calibration datasets...")
    samples_short = get_mixed_calib_dataset(
        args.short_datasets if args.short_prefix or args.enable_task_adaptive else args.datasets,
        tokenizer=tokenizer,
        n_samples=args.train_size,
        block_size=args.seqlen,
        seed=args.seed,
        split="train",
    )
    samples_short = concat_prefix_to_samples(samples_short, args.short_prefix, args.seqlen, tokenizer, device)

    val_samples_short = get_calib_dataset(
        args.val_dataset if not args.short_prefix else args.val_dataset,
        tokenizer=tokenizer,
        n_samples=args.validation_size,
        block_size=args.seqlen,
        seed=args.seed,
        split="validation",
    )
    val_samples_short = concat_prefix_to_samples(val_samples_short, args.short_prefix, args.seqlen, tokenizer, device)

    # Long path datasets (推理任务)
    logger.info("Loading long-path calibration datasets...")
    samples_long = get_mixed_calib_dataset(
        args.long_datasets if args.long_prefix or args.enable_task_adaptive else args.datasets,
        tokenizer=tokenizer,
        n_samples=args.train_size,
        block_size=args.seqlen,
        seed=args.seed + 1,  # 不同 seed 保证多样性
        split="train",
    )
    samples_long = concat_prefix_to_samples(samples_long, args.long_prefix, args.seqlen, tokenizer, device)

    val_samples_long = get_calib_dataset(
        args.val_dataset if not args.long_prefix else args.val_dataset,
        tokenizer=tokenizer,
        n_samples=args.validation_size,
        block_size=args.seqlen,
        seed=args.seed + 1,
        split="validation",
    )
    val_samples_long = concat_prefix_to_samples(val_samples_long, args.long_prefix, args.seqlen, tokenizer, device)

    # Capture first layer's input for both paths.
    logger.info("Capturing first layer input (short path)...")
    blocks[0].to(device)
    og_layer_input_batches_short, kwargs = catch_first_layer_input(
        model, blocks, samples_short, batch_size=args.batch_size,
    )
    val_batch_size = args.val_batch_size or args.batch_size
    og_layer_val_input_batches_short, _ = catch_first_layer_input(
        model, blocks, val_samples_short, batch_size=val_batch_size,
    )
    blocks[0].cpu()

    logger.info("Capturing first layer input (long path)...")
    blocks[0].to(device)
    og_layer_input_batches_long, _ = catch_first_layer_input(
        model, blocks, samples_long, batch_size=args.batch_size,
    )
    og_layer_val_input_batches_long, _ = catch_first_layer_input(
        model, blocks, val_samples_long, batch_size=val_batch_size,
    )
    blocks[0].cpu()

    del samples_short, samples_long
    empty_cache()

    @torch.no_grad()
    def forward_layer_batch(
        layer: nn.Module,
        input_batched: list[torch.Tensor],
        kwargs: dict,
        store_device: torch.device,
        dtype: torch.dtype = torch.float16,
        cast_to_dtype: torch.dtype = torch.float16,
    ) -> list[torch.Tensor]:
        output_batched = []

        layer.to(device)
        for input_batch in input_batched:
            output = layer(input_batch.to(dtype).to(device), **kwargs)
            if isinstance(output, tuple):
                output = output[0]
            if output.dtype != cast_to_dtype:
                output = output.to(cast_to_dtype)
            if output.device != store_device:
                output = output.to(store_device)
            output_batched.append(output)
        layer.cpu()

        empty_cache()
        return output_batched

    def set_checkpointing_enabled(module: nn.Module, enable: bool) -> None:
        for linear in get_named_linears(
            module, subclass=PseudoQuantizedLinear
        ).values():
            linear.enable_checkpoint = enable

    # Layerwise, multi-stage optimization.
    for layer_idx, layer in enumerate(tqdm(blocks)):
        empty_cache()
        logger.info(f"Capturing original layer output (dual path)...")
        # 分别计算 short / long 路径的原始输出
        og_layer_output_batches_short = forward_layer_batch(
            layer, og_layer_input_batches_short, kwargs, store_device="cpu"
        )
        og_layer_val_output_batches_short = forward_layer_batch(
            layer, og_layer_val_input_batches_short, kwargs, store_device="cpu"
        )
        og_layer_output_batches_long = forward_layer_batch(
            layer, og_layer_input_batches_long, kwargs, store_device="cpu"
        )
        og_layer_val_output_batches_long = forward_layer_batch(
            layer, og_layer_val_input_batches_long, kwargs, store_device="cpu"
        )

        # 误差传播：分别维护两套输入
        if layer_idx > 0:
            layer_input_batches_short = new_layer_output_batches_short
            layer_val_input_batches_short = new_layer_val_output_batches_short
            layer_input_batches_long = new_layer_output_batches_long
            layer_val_input_batches_long = new_layer_val_output_batches_long
        else:
            layer_input_batches_short = og_layer_input_batches_short
            layer_val_input_batches_short = og_layer_val_input_batches_short
            layer_input_batches_long = og_layer_input_batches_long
            layer_val_input_batches_long = og_layer_val_input_batches_long

        # 方案1：收集 Prefix 激活统计用于通道显著性融合
        act_stats_fused = None
        if args.prefix_beta > 0 and args.selection_mode == "greedy":
            try:
                # 从 short 路径第一个 batch 的前缀区域收集激活
                prefix_len_short = len(tokenizer.encode(args.short_prefix, add_special_tokens=False)) if args.short_prefix else 0
                prefix_len_long = len(tokenizer.encode(args.long_prefix, add_special_tokens=False)) if args.long_prefix else 0
                prefix_len = max(prefix_len_short, prefix_len_long, 1)

                # 取输入激活 (batch, seq_len, hidden_dim)
                prefix_input = layer_input_batches_short[0][:, :prefix_len, :]
                act_stats = prefix_input.abs().amax(dim=(0, 1))  # (hidden_dim,)

                # reshape 为 group 维度
                num_groups = act_stats.numel() // args.group_size
                act_stats_fused = act_stats.view(num_groups, args.group_size).to(device)

                # 若 long 路径也有 prefix，可进一步融合
                if args.long_prefix:
                    prefix_input_long = layer_input_batches_long[0][:, :prefix_len, :]
                    act_stats_long = prefix_input_long.abs().amax(dim=(0, 1))
                    act_stats_long_grouped = act_stats_long.view(num_groups, args.group_size).to(device)
                    act_stats_fused = torch.maximum(act_stats_fused, act_stats_long_grouped)
            except Exception as e:
                logger.warning(f"Failed to collect activation stats at layer {layer_idx}: {e}")
                act_stats_fused = None

        # 冻结所有参数
        for param in layer.parameters():
            param.requires_grad = False

        linear_modules = get_named_linears(layer)
        if args.resume:
            all_files_exist = True
            for name in linear_modules.keys():
                file_name = f"{layer_idx}.{name}.pt"
                file_path = output_dir / file_name
                if not file_path.exists() and name not in args.skipped_modules:
                    all_files_exist = False
                    break
        else:
            all_files_exist = False

        if not all_files_exist:
            logger.info(f"Initializing rotation parameters...")

        for name, old_module in linear_modules.items():
            if name in args.skipped_modules:
                continue

            if all_files_exist:
                existing_result_file = output_dir / f"{layer_idx}.{name}.pt"
                sd = torch.load(existing_result_file, map_location=device)
                new_module = PseudoQuantizedLinear.from_state_dict(sd)
                set_module_by_name(layer, name, new_module)
                continue

            old_module.to(device)

            num_pairs_factor = 0.5
            weight = old_module.weight.float()
            weight_grouped = weight.view(weight.shape[0], -1, args.group_size).permute(
                1, 0, 2
            )

            # 检查 activation_scores 是否匹配当前模块的权重分组
            # act_stats_fused 基于 layer 输入 (hidden_dim) 计算，仅适用于 in_features == hidden_dim 的模块
            # 例如 down_proj 的 in_features 通常为 intermediate_size (如 12288)，与 hidden_dim (4096) 不同
            module_act_stats = act_stats_fused
            if act_stats_fused is not None and act_stats_fused.shape[0] != weight_grouped.shape[0]:
                logger.warning(
                    f"Skipping activation_scores for {name} at layer {layer_idx}: "
                    f"act_stats_fused shape {act_stats_fused.shape} does not match "
                    f"weight_grouped groups {weight_grouped.shape[0]}"
                )
                module_act_stats = None

            all_pairs = get_random_rotation_pairs(
                weight_grouped,
                group_size=args.group_size,
                num_rotations=args.num_rotations,
                num_pairs_factor=num_pairs_factor,
                seed=args.seed + layer_idx,
                selection_mode=args.selection_mode,
                metric=args.metric,
                activation_scores=module_act_stats,
                alpha=args.prefix_alpha,
                beta=args.prefix_beta,
            )

            all_pairs = [
                torch.tensor(pairs, device="cpu", dtype=torch.int32)
                for pairs in all_pairs
            ]
            initial_angles = [
                torch.zeros(pairs.shape[0], device="cpu") for pairs in all_pairs
            ]
            initial_scales = torch.ones(
                1, weight.shape[1], dtype=torch.float16, device=device
            )

            npairs, angles, mask = transform_to_kernel_data(
                all_pairs,
                initial_angles,
                group_size=args.group_size,
            )
            npairs = npairs.to(device)
            angles = angles.to(device)
            mask = mask.to(device)
            rotation_pairs = [npairs, angles, mask]
            channel_scales = initial_scales

            new_module = PseudoQuantizedLinear(
                old_module,
                rotation_pairs,
                channel_scales,
                group_size=args.group_size,
                n_bits=args.n_bit,
                num_rotations=args.num_rotations,
                enable_task_adaptive=args.enable_task_adaptive,
            )

            set_module_by_name(layer, name, new_module)
            old_module.cpu()

        if not all_files_exist:
            layer.to(device).float()

            set_checkpointing_enabled(layer, args.checkpointing)
            for step, step_params_dict in enumerate(params_to_optimize):
                empty_cache()
                optim_params = []
                new_modules = get_named_linears(layer, subclass=PseudoQuantizedLinear)
                for new_module in new_modules.values():
                    new_module.set_optim_enabled(
                        **{param_name: True for param_name in step_params_dict.keys()},
                    )
                    for param_name, lr in step_params_dict.items():
                        optim_params.append(
                            dict(
                                params=new_module.get_optim_params(param_name),
                                lr=lr,
                                weight_decay=args.weight_decay,
                                betas=args.betas,
                                eps=args.eps,
                            )
                        )

                logger.info(
                    f"Optimizing layer {layer_idx}, step {step + 1}/{len(params_to_optimize)}: "
                    f"{', '.join([k for k in step_params_dict])}"
                )

                # 构造双路径数据
                train_short = (
                    CachedTensorShards(layer_input_batches_short, args.cache_shards, target_device=device),
                    CachedTensorShards(og_layer_output_batches_short, args.cache_shards, target_device=device),
                )
                train_long = (
                    CachedTensorShards(layer_input_batches_long, args.cache_shards, target_device=device),
                    CachedTensorShards(og_layer_output_batches_long, args.cache_shards, target_device=device),
                )
                val_short = (
                    [b.to(device) for b in layer_val_input_batches_short],
                    [b.to(device) for b in og_layer_val_output_batches_short],
                )
                val_long = (
                    [b.to(device) for b in layer_val_input_batches_long],
                    [b.to(device) for b in og_layer_val_output_batches_long],
                )

                # 判断是否启用双路径联合优化
                if args.enable_task_adaptive or args.task_lambda_short != args.task_lambda_long:
                    optimize_module(
                        layer,
                        [train_short, train_long],
                        [val_short, val_long],
                        kwargs,
                        optim_params,
                        loss_fn=args.loss,
                        n_iter=args.epochs[step],
                        early_stop=None,
                        post_optim_callback=reset_angles_by_mask,
                        task_lambdas=[args.task_lambda_short, args.task_lambda_long],
                    )
                else:
                    # 回退到单路径（兼容旧逻辑）
                    optimize_module(
                        layer,
                        (CachedTensorShards(layer_input_batches_short, args.cache_shards, target_device=device),
                         CachedTensorShards(og_layer_output_batches_short, args.cache_shards, target_device=device)),
                        val_short,
                        kwargs,
                        optim_params,
                        loss_fn=args.loss,
                        n_iter=args.epochs[step],
                        early_stop=None,
                        post_optim_callback=reset_angles_by_mask,
                    )

            set_checkpointing_enabled(layer, False)

            del (
                train_short, train_long, val_short, val_long,
            )
            empty_cache()

        else:
            logger.info(
                f"Skipping optimization for layer {layer_idx}: already been optimized."
            )

        layer.half().to(device)

        logger.info("Capturing new layer output (dual path)...")
        new_layer_output_batches_short = forward_layer_batch(
            layer,
            layer_input_batches_short,
            kwargs,
            store_device="cpu",
        )
        new_layer_val_output_batches_short = forward_layer_batch(
            layer,
            layer_val_input_batches_short,
            kwargs,
            store_device="cpu",
        )
        new_layer_output_batches_long = forward_layer_batch(
            layer,
            layer_input_batches_long,
            kwargs,
            store_device="cpu",
        )
        new_layer_val_output_batches_long = forward_layer_batch(
            layer,
            layer_val_input_batches_long,
            kwargs,
            store_device="cpu",
        )

        # 更新下一层的原始输入目标
        og_layer_input_batches_short = og_layer_output_batches_short
        og_layer_val_input_batches_short = og_layer_val_output_batches_short
        og_layer_input_batches_long = og_layer_output_batches_long
        og_layer_val_input_batches_long = og_layer_val_output_batches_long

        if all_files_exist:
            layer.cpu()
            continue

        # Save the optimized result (一套 state_dict 包含 shared + short + long 所有参数)
        for name, module in get_named_linears(
            layer, subclass=PseudoQuantizedLinear
        ).items():
            result_file = output_dir / f"{layer_idx}.{name}.pt"
            torch.save(
                module.state_dict(),
                result_file,
            )

        layer.cpu()


if __name__ == "__main__":
    main()
