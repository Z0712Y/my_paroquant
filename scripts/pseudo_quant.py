import torch
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
import sys

# 将项目根目录加入 Python 路径，以便导入 paroquant 包
sys.path.append(Path(__file__).parent.parent.as_posix())

from paroquant.util import (
    load_model,
    load_tokenizer,
    get_blocks,
    get_named_linears,
)
from paroquant.module import PseudoQuantizedLinear


# ==========================================
# 模块一：命令行参数解析
# ==========================================
# 本脚本用于将优化后的逐层结果（.pt 文件）加载回原始模型，
# 生成一个“伪量化”后的稠密模型（权重是浮点，但已经过量化-反量化）。
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Hugging Face 模型路径或本地目录")
    parser.add_argument("--result-dir", type=str, required=True, help="逐层优化结果目录，内含 {layer_idx}.{linear_name}.pt 文件")
    parser.add_argument("--output-path", type=str, required=True, help="输出模型的保存路径")

    args = parser.parse_args()

    # ==========================================
    # 模块二：加载原始模型与分词器
    # ==========================================
    dtype = torch.float16
    device = "cuda"
    result_dir = Path(args.result_dir)

    # 加载原始模型（fp16）并移动到 GPU
    model = load_model(args.model, device_map=device, dtype=dtype)
    tokenizer = load_tokenizer(args.model)
    # 获取 Transformer 的 decoder layers
    blocks = get_blocks(model)

    # ==========================================
    # 模块三：逐层替换线性层权重为伪量化后的权重
    # ==========================================
    for i, layer in enumerate(tqdm(blocks)):
        # 将当前层移动到 GPU（支持 layer-by-layer CPU offloading 场景）
        layer = layer.to(device)
        # 获取当前层中所有 nn.Linear 模块
        for name, module in get_named_linears(layer).items():
            # 构造该线性层对应的优化结果文件路径
            result_file = result_dir / f"{i}.{name}.pt"
            if not result_file.exists():
                raise Exception(f"Result file not found: {result_file}")
            # 加载 PseudoQuantizedLinear 的状态字典
            sd = torch.load(result_file, weights_only=False, map_location=device)
            # 从状态字典重建 PseudoQuantizedLinear 对象
            qlayer = PseudoQuantizedLinear.from_state_dict(sd)
            # 获取伪量化后的浮点权重（即量化再反量化后的结果）
            weight = qlayer.pseudo_weight()
            # 将该权重复制到原始模型的对应 linear 层中
            module.weight.data.copy_(weight)

    # ==========================================
    # 模块四：保存伪量化后的稠密模型
    # ==========================================
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"Model saved to {args.output_path}")
