import torch
import torch.nn as nn
from tqdm import tqdm
from copy import deepcopy
from typing import Literal, Optional, Iterator, Callable
import random

from .util import CosineAnnealingParam, logger


# ==========================================
# 模块一：独立通道对选择（贪婪算法）
# ==========================================
# 该函数用于在通道维度上挑选互不重叠的通道对 (i, j)，
# 确保在同一个 rotation 中，每个通道最多只出现一次。
def _get_independent_channel_pairs(
    pairs: torch.Tensor, dim: int, num_rotations: int, num_pairs_each: int
) -> list[list[tuple[int, int]]]:
    # 将候选对转为 CPU 列表以便处理
    pairs = pairs.cpu().tolist()
    rotations_pairs = []
    # 可用性矩阵：记录哪些通道对尚未被使用过
    available = torch.ones(dim, dim)
    # 对角线置 0，因为通道不能与自身配对
    available.fill_diagonal_(0)

    for _ in range(num_rotations):
        independent_pairs = []
        # 在当前 rotation 中维护一个可用性矩阵的副本，
        # 用于跟踪本 rotation 内已被占用的通道
        available_in_rotation = available.clone()
        # 贪婪遍历所有候选对
        for i, j in pairs:
            # 如果已经收集到足够的对数，提前结束
            if len(independent_pairs) == num_pairs_each:
                break
            # 若该对在当前 rotation 中不可用（已被占用），跳过
            if available_in_rotation[i, j] == 0:
                continue
            # 选中该对
            independent_pairs.append((i, j))
            # 标记 i 和 j 通道在当前 rotation 中已被占用：
            # 任何包含 i 或 j 的其他对都不能再被选中
            available_in_rotation[i, :] = 0
            available_in_rotation[j, :] = 0
            available_in_rotation[:, i] = 0
            available_in_rotation[:, j] = 0
            # 在全局可用性矩阵中也标记该对已被使用，
            # 但 i 和 j 仍可在后续 rotation 中与其他通道配对
            available[i, j] = 0
            available[j, i] = 0

        rotations_pairs.append(independent_pairs)

    return rotations_pairs


# ==========================================
# 模块二：通道显著性分数计算
# ==========================================
def compute_channel_scores(weight_grouped: torch.Tensor, metric: str = "l2") -> torch.Tensor:
    """
    为每个 group 的每个通道计算显著性分数。

    Args:
        weight_grouped: (group_num, out_features, group_size)
        metric: "l2" | "maxabs" | "var"

    Returns:
        scores: (group_num, group_size)
    """
    if metric == "l2":
        scores = weight_grouped.norm(dim=1)
    elif metric == "maxabs":
        scores = weight_grouped.abs().amax(dim=1)
    elif metric == "var":
        scores = weight_grouped.var(dim=1)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    return scores


# ==========================================
# 模块三：旋转对生成（随机或贪心显著性）
# ==========================================
# 基于敏感度输入，为每个 group 生成通道对，
# 并调用贪婪算法确保每个 rotation 内部通道不冲突。
def get_random_rotation_pairs(
    sensitivity_input: torch.Tensor,
    group_size: int,
    num_rotations: int,
    num_pairs_factor: float,
    seed: int,
    selection_mode: str = "random",
    metric: str = "l2",
) -> list[tuple[int, int]]:
    sorted_pairs: list[list[tuple[int, int]]] = []
    # group 的数量由 sensitivity_input 的第一维决定
    group_num = sensitivity_input.shape[0]

    # 贪心模式下预计算所有通道显著性；随机模式下保留原随机种子逻辑
    if selection_mode == "greedy":
        channel_scores = compute_channel_scores(sensitivity_input, metric=metric)
    else:
        rand = random.Random(seed)

    # 为每个 group 生成所有可能的通道对 (i, j) 并排序
    for group_idx in range(group_num):
        candidate_pairs = []
        for i in range(group_size):
            for j in range(i + 1, group_size):
                candidate_pairs.append((i, j))

        if selection_mode == "greedy":
            scores_i = channel_scores[group_idx]
            pair_scores = []
            for i, j in candidate_pairs:
                # [修复] 改为显著性差异最大化：高显著性通道与低显著性通道配对，
                # 旋转后才能有效打散权重、平衡分布。原方案用“和”会导致高-高聚集，量化崩溃。
                score = abs(scores_i[i].item() - scores_i[j].item())
                pair_scores.append((score, (i, j)))
            pair_scores.sort(key=lambda x: x[0], reverse=True)
            sorted_pairs.append([p for _, p in pair_scores])
        else:
            rand.shuffle(candidate_pairs)
            sorted_pairs.append(candidate_pairs)

    # 转为 tensor 并放到与输入相同的设备上
    sorted_pairs = torch.tensor(sorted_pairs, device=sensitivity_input.device)

    # 为每个 rotation 准备一个空列表
    pairs_k_groups = [[] for _ in range(num_rotations)]
    # 每个 group 在每个 rotation 中应选取的对数
    num_pairs_per_group = int(group_size * num_pairs_factor)

    # 遍历每个 group，获取其独立的旋转对
    for i in range(group_num):
        # 当前 group 在全局维度上的偏移量
        offset = i * group_size
        # 调用贪婪算法获取该 group 的独立通道对
        pairs_g_k_groups = _get_independent_channel_pairs(
            sorted_pairs[i], group_size, num_rotations, num_pairs_per_group
        )
        # 将局部索引转换为全局索引，并按 rotation 收集
        for r_idx in range(num_rotations):
            pairs_g = pairs_g_k_groups[r_idx]
            for j, (col1, col2) in enumerate(pairs_g):
                pairs_g[j] = (col1 + offset, col2 + offset)
            pairs_k_groups[r_idx].extend(pairs_g)

    return pairs_k_groups


# ==========================================
# 模块三：模块级优化（逐层/逐模块微调）
# ==========================================
# 使用 AdamW + AMP 自动混合精度 + 早停机制，
# 在训练集上优化模块参数，使量化后输出逼近原始输出。
def optimize_module(
    module: nn.Module,
    train_set_batches: tuple[Iterator[torch.Tensor], Iterator[torch.Tensor]],
    val_set_batches: tuple[Iterator[torch.Tensor], Iterator[torch.Tensor]],
    kwargs: dict,
    optim_params: list[dict],
    *,
    loss_fn: Literal["mse", "smooth_l1"],
    n_iter: int,
    early_stop: Optional[int],
    post_optim_callback: Optional[Callable[[nn.Module], None]] = None,
) -> None:
    # 解包训练集和验证集的输入/输出批次迭代器
    train_input_batches, train_output_batches = train_set_batches
    val_input_batches, val_output_batches = val_set_batches

    # 计算总训练步数 = 迭代轮数 × 每轮批次数量
    total_steps = n_iter * len(train_input_batches)
    # 为每个参数组创建余弦退火学习率调度器
    schedulers = [
        CosineAnnealingParam(
            start_value=param_group["lr"],
            end_value=param_group["lr"] / 20,
            T_max=total_steps,
        )
        for param_group in optim_params
    ]
    # 初始化 AdamW 优化器
    optimizer = torch.optim.AdamW(optim_params)
    # AMP 梯度缩放器，用于自动混合精度训练
    scaler = torch.amp.GradScaler()
    # 根据传入的 loss_fn 字符串选择对应的损失函数
    if loss_fn == "mse":
        loss = nn.MSELoss()
    elif loss_fn == "smooth_l1":
        loss = nn.SmoothL1Loss()
    else:
        raise ValueError(f"Unsupported loss function: {loss_fn}")

    # 初始化进度条
    progress_bar = tqdm(total=n_iter, unit="iter")

    # 定义一个辅助函数：前向传播并取第一个输出（兼容返回 tuple 的模块，如 Transformer 层）
    def module_output(input: torch.Tensor) -> torch.Tensor:
        out = module(input, **kwargs)
        if isinstance(out, tuple):
            out = out[0]
        return out

    # 定义一个无梯度的辅助函数：计算整个验证集/训练集上的平均损失
    @torch.no_grad()
    def loss_batches(
        input_batches: Iterator[torch.Tensor], output_batches: Iterator[torch.Tensor]
    ) -> torch.Tensor:
        total_loss = None
        for input_batch, output_batch in zip(input_batches, output_batches):
            output_q = module_output(input_batch)
            loss_value = loss(output_batch, output_q)
            if total_loss is None:
                total_loss = loss_value
            else:
                total_loss += loss_value
        return total_loss / len(input_batches)

    # 在优化开始前，先记录原始（未优化）的验证集损失
    with torch.no_grad():
        original_val_loss = loss_batches(val_input_batches, val_output_batches)

    # 初始化最佳验证损失和对应的状态字典
    best_val_loss = original_val_loss
    best_sd = deepcopy(module.state_dict())

    # 早停计数器
    early_stop_counter = 0

    # ==========================================
    # 训练循环
    # ==========================================
    for _ in range(n_iter):
        # 遍历训练批次
        for input_batch, output_batch in zip(train_input_batches, train_output_batches):
            optimizer.zero_grad()
            # 开启自动混合精度上下文
            with torch.amp.autocast("cuda"):
                output_q = module_output(input_batch)
                loss_value = loss(output_batch, output_q)

            # AMP 反向传播与参数更新
            scaler.scale(loss_value).backward()
            scaler.step(optimizer)
            scaler.update()

            # 更新每个参数组的学习率
            for i, scheduler in enumerate(schedulers):
                optimizer.param_groups[i]["lr"] = scheduler.step()

            # 若提供了后处理回调（如量化、裁剪等），在每个 batch 后执行
            if post_optim_callback:
                post_optim_callback(module)

        # ==========================================
        # 验证与早停判断
        # ==========================================
        with torch.no_grad():
            val_loss_value = loss_batches(val_input_batches, val_output_batches)

        # 若验证损失下降，保存最佳模型并重置早停计数器
        if val_loss_value < best_val_loss:
            best_val_loss = val_loss_value
            best_sd = deepcopy(module.state_dict())
            early_stop_counter = 0
        else:
            # 否则累加早停计数器
            early_stop_counter += 1
            # 若达到早停阈值，提前终止训练
            if early_stop is not None and early_stop_counter >= early_stop:
                break

        # 更新进度条显示
        progress_bar.set_postfix(
            val_loss=val_loss_value.item(),
            val_og_loss=original_val_loss.item(),
        )
        progress_bar.update(1)

    progress_bar.close()
    logger.info(
        f"Best val loss: {best_val_loss.item()}, Original val loss: {original_val_loss.item()}"
    )

    # 加载验证集上表现最好的模型状态
    module.load_state_dict(best_sd)
