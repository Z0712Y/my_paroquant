import argparse
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer


def main():
    # ==========================================
    # 模块一：命令行参数解析器初始化
    # ==========================================
    parser = argparse.ArgumentParser(
        description="CLI for model quantization and saving"
    )
    parser.add_argument(
        "--hf_model_path",
        type=str,
        required=True,
        help="Path to the Hugging Face model",
    )
    parser.add_argument(
        "--quant_name", type=str, required=True, help="Name of the quantized model"
    )
    parser.add_argument(
        "--local_save_path",
        type=str,
        required=True,
        help="Path to save the quantized model",
    )

    # ==========================================
    # 模块二：量化配置参数
    # ==========================================
    parser.add_argument(
        "--zero_point", action="store_true", help="Enable zero point for quantization"
    )
    parser.add_argument(
        "--no-zero_point",
        action="store_false",
        dest="zero_point",
        help="Disable zero point for quantization",
    )
    parser.add_argument(
        "--q_group_size", type=int, default=128, help="Quantization group size"
    )
    parser.add_argument("--w_bit", type=int, default=4, help="Weight bit width")
    parser.add_argument(
        "--version", type=str, default="GEMM", help="Quantization version"
    )

    # ==========================================
    # 模块三：模型加载配置参数
    # ==========================================
    parser.add_argument(
        "--device_map",
        type=str,
        default=None,
        help="Device map for loading the pretrained model",
    )

    # ==========================================
    # 模块四：量化校准参数
    # ==========================================
    parser.add_argument(
        "--max_calib_samples",
        type=int,
        default=128,
        help="Number of calibration samples.",
    )
    parser.add_argument(
        "--max_calib_seq_len",
        type=int,
        default=512,
        help="Calibration sample sequence length.",
    )

    # ==========================================
    # 模块五：解析参数并构建量化配置字典
    # ==========================================
    args = parser.parse_args()

    quant_config = {
        "zero_point": args.zero_point,
        "q_group_size": args.q_group_size,
        "w_bit": args.w_bit,
        "version": args.version,
    }

    # ==========================================
    # 模块六：加载预训练模型与分词器
    # ==========================================
    print(f"Loading model from: {args.hf_model_path}")
    model = AutoAWQForCausalLM.from_pretrained(
        args.hf_model_path,
        device_map=args.device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.hf_model_path, trust_remote_code=True
    )

    # ==========================================
    # 模块七：执行模型量化
    # ==========================================
    print(f"Quantizing model with config: {quant_config}")
    model.quantize(
        tokenizer,
        quant_config=quant_config,
        max_calib_samples=args.max_calib_samples,
        max_calib_seq_len=args.max_calib_seq_len,
    )

    # ==========================================
    # 模块八：保存量化后的模型与分词器
    # ==========================================
    print(f"Saving quantized model to: {args.local_save_path}")
    model.save_quantized(args.local_save_path)
    tokenizer.save_pretrained(args.local_save_path)

    print(f"Quantized model '{args.quant_name}' saved successfully.")


if __name__ == "__main__":
    main()
