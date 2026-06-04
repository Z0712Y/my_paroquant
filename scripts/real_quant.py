import argparse
from transformers import AutoTokenizer
import sys
from pathlib import Path

# 将项目根目录加入 Python 路径，以便导入 inference_engine 工具
sys.path.append(Path(__file__).parents[1].as_posix())

from inference_engine.utils.checkpoint_utils import from_pt_to_ckpt


# ==========================================
# 模块一：命令行参数解析
# ==========================================
# 本脚本用于将训练/优化阶段输出的逐层 .pt 权重文件，
# 转换为可用于 inference_engine 推理的真实量化模型 checkpoint。
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert training output (.pt weights) to model checkpoint"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="原始 Hugging Face 模型路径（用于获取模型结构和 tokenizer）",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        required=True,
        help="path to the training output (.pt checkpoints) dir",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="checkpoint output path",
    )
    args = parser.parse_args()

    # ==========================================
    # 模块二：调用转换函数生成真实量化 checkpoint
    # ==========================================
    # from_pt_to_ckpt 会读取 result_dir 中的 .pt 文件，
    # 将其打包为 inference_engine 可加载的量化 checkpoint 格式。
    from_pt_to_ckpt(args.model, args.result_dir, args.output_path)

    # ==========================================
    # 模块三：复制 tokenizer 到输出目录
    # ==========================================
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.save_pretrained(args.output_path)
