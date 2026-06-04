import torch
import torch.nn as nn
from typing import Optional, Union
from torch.utils.checkpoint import checkpoint

from .util import clamp_ste, round_ste


# ==========================================
# 模块一：标度因子与零点计算
# ==========================================
# 根据权重矩阵的每个 group 内的最小值和最大值，
# 计算用于均匀仿射量化的 scale 和 zero_point。
def _calc_scales_and_zero_points(
    weight: torch.Tensor, group_size: int, qmin: int, qmax: int
) -> None:
    # 确保输入是二维矩阵（out_features × in_features）
    assert weight.dim() == 2, weight.shape
    # 若输入不是 float32，先转换为 float32 以保证数值稳定性
    if weight.dtype != torch.float32:
        weight = weight.float()
    # 将权重 reshape 为 (num_groups, group_size)，
    # 其中 num_groups = (out_features * in_features) / group_size
    x = weight.reshape(-1, group_size)
    # 沿 group_size 维度（dim=1）求最小值和最大值
    min_val = x.amin(dim=1, keepdim=True)
    max_val = x.amax(dim=1, keepdim=True)
    # scale = (max - min) / qmax，并用 clamp_ste 防止除零（最小值不小于 1e-5）
    scale = clamp_ste(max_val - min_val, min=1e-5) / qmax
    # zero_point = min / scale，表示量化零点对应的浮点值
    zero_point = min_val / scale

    # 安全检查：确保 zero_point 中没有 NaN
    assert torch.isnan(zero_point).sum() == 0, zero_point

    return scale, zero_point


# ==========================================
# 模块二：均匀仿射量化器（Uniform Affine Quantizer）
# ==========================================
# 改编自 EfficientQAT，支持 group-wise 均匀仿射量化，
# 并且将 scale 和 zero_point 作为可学习参数，以便在优化过程中微调。
class UniformAffineQuantizer(nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        n_bits: int,
        group_size: int,
    ):
        super().__init__()
        # 将 n_bits 和 group_size 转为 tensor 并注册为 buffer（不可学习但会随模型保存）
        if not isinstance(n_bits, torch.Tensor):
            n_bits = torch.tensor(n_bits)
        if not isinstance(group_size, torch.Tensor):
            group_size = torch.tensor(group_size)
        self.register_buffer("n_bits", n_bits)
        self.register_buffer("group_size", group_size)
        # 确保权重最后一维能被 group_size 整除
        assert weight.shape[-1] % group_size == 0

        # 基于初始权重计算 scale 和 zero_point
        scale, zero_point_float = _calc_scales_and_zero_points(
            weight, group_size, self.qmin, self.qmax
        )
        # 将 scale 和 zero_point 注册为可学习参数（nn.Parameter）
        self.scale = nn.Parameter(scale)
        self.zero_point_float = nn.Parameter(zero_point_float)

        # 是否启用梯度检查点（用于节省显存）
        self.enable_checkpoint = False

    # 量化最小值固定为 0（非对称量化）
    @property
    def qmin(self) -> int:
        return 0

    # 量化最大值由位宽决定：2^n_bits - 1
    @property
    def qmax(self) -> torch.Tensor:
        return 2**self.n_bits - 1

    # ==========================================
    # 模块三：前向传播（支持梯度检查点）
    # ==========================================
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 若处于训练模式且启用了梯度检查点，则使用 torch.utils.checkpoint
        # 以时间换空间，减少反向传播时的显存占用
        if torch.is_grad_enabled() and self.enable_checkpoint:
            return checkpoint(
                self.pseudo_quantize,
                x,
                self.n_bits,
                self.group_size,
                self.scale,
                self.zero_point_float,
                use_reentrant=False,
            )
        else:
            # 否则直接调用伪量化函数
            return self.pseudo_quantize(
                x,
                self.n_bits,
                self.group_size,
                self.scale,
                self.zero_point_float,
            )

    # 返回需要被优化的参数列表（scale 和 zero_point）
    def optim_params(self) -> list[nn.Parameter]:
        return [self.scale, self.zero_point_float]

    # 控制优化参数的 requires_grad 开关
    def set_optim_enabled(self, enabled: bool):
        for param in self.optim_params():
            param.requires_grad = enabled

    # ==========================================
    # 模块四：伪量化（核心量化/反量化逻辑）
    # ==========================================
    # 静态方法，实际执行 group-wise 的量化再反量化过程，
    # 模拟推理时的低精度计算，同时保持梯度可导（通过 STE）。
    @staticmethod
    def pseudo_quantize(
        x: torch.Tensor,
        n_bits: Union[int, torch.Tensor],
        group_size: Union[int, torch.Tensor],
        scale: Optional[torch.Tensor] = None,
        zero_point: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 记录原始数据类型，最终输出会转回该类型
        dtype = x.dtype
        # 量化前统一转为 float32，保证数值精度
        if x.dtype != torch.float32:
            x = x.float()
        # 安全检查：输入中不应存在 NaN
        assert torch.isnan(x).sum() == 0, x

        # 计算当前位宽下的量化范围
        qmin, qmax = 0, 2**n_bits - 1
        # 若外部未提供 scale 和 zero_point，则根据输入 x 动态计算
        if scale is None or zero_point is None:
            scale, zero_point = _calc_scales_and_zero_points(x, group_size, qmin, qmax)

        # 对 scale 做裁剪，防止过小（<1e-5）或过大（>1e5）导致数值异常
        scale = clamp_ste(scale, min=1e-5, max=1e5)
        # 对 zero_point 做四舍五入（STE）后再取反并裁剪到 [qmin, qmax]
        round_zero_point = clamp_ste(-round_ste(zero_point), qmin, qmax)
        # 记录原始矩阵形状，用于 reshape 恢复
        dim1, dim2 = x.shape
        # 按 group_size 分组
        x = x.reshape(-1, group_size)
        # 第一步：除以 scale（浮点值 → 整数域）
        x_int = round_ste(x / scale)
        # 安全检查
        assert torch.isnan(x_int).sum() == 0, (x_int, scale.min(), scale.max())
        # 第二步：加上 zero_point（完成偏移）
        x_int = x_int + round_zero_point
        # 第三步：裁剪到合法的量化整数范围 [qmin, qmax]
        x_int = clamp_ste(x_int, qmin, qmax)
        # 第四步：反量化（整数域 → 浮点值）
        x_dequant = x_int
        x_dequant = x_dequant - round_zero_point
        x_dequant = x_dequant * scale
        # 恢复原始形状
        x_dequant = x_dequant.reshape(dim1, dim2)
        # 安全检查：反量化结果不应包含 NaN 或 Inf
        assert torch.isnan(x_dequant).sum() == 0, x_dequant
        assert torch.isinf(x_dequant).sum() == 0, x_dequant
        # 转回原始 dtype 并返回
        return x_dequant.to(dtype)
