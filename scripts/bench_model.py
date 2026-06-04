import argparse
import sys
import time
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend
from transformers import AutoTokenizer

# =============================================================================
# 模块1: 路径设置与模块导入
# =============================================================================
# 将项目根目录（scripts的父目录）添加到Python路径，以便导入自定义模块
sys.path.append(str(Path(__file__).resolve().parents[1]))

# 从inference_engine导入transformers后端的解码函数和模型加载函数
from inference_engine.generation.transformers_backend import (
    decode_one_token,      # 解码单个token的函数
    model_from_hf_path,    # 从HuggingFace路径加载模型的函数
    sample_next_token,     # 从logits中采样下一个token的函数
)
# 导入静态KV Cache工具类，用于存储注意力机制的key-value缓存
from inference_engine.model_executor.models.cache_utils import StaticCache


# =============================================================================
# 模块2: 全局配置
# =============================================================================
# 禁用梯度计算，因为推理阶段不需要反向传播，可以节省内存并加速计算
torch.set_grad_enabled(False)


# =============================================================================
# 模块3: 核心基准测试函数
# =============================================================================
@torch.no_grad()  # 装饰器方式禁用梯度计算
def benchmark(model, tokenizer, prefill_len, decode_len, past_kv):
    """
    执行单次基准测试，测量模型解码阶段的吞吐量。
    
    Args:
        model: 待测试的语言模型
        tokenizer: 对应的分词器
        prefill_len: 预填充（Prefill）阶段的输入长度
        decode_len: 解码阶段要生成的token数量
        past_kv: KV Cache对象，用于存储注意力缓存
    
    Returns:
        generated_ids: 生成的token ID序列
        texts: 解码后的文本结果
        decode_len / elapsed: 解码吞吐量（tokens/秒）
    """
    # -------------------------------------------------------------------------
    # 3.1 构建测试输入文本
    # -------------------------------------------------------------------------
    # 使用固定文本片段重复拼接，直到达到所需的prefill长度
    fixed_text = "The quick brown fox jumps over the lazy dog. "
    pieces = []  # 存储文本片段的列表
    while True:
        pieces.append(fixed_text)  # 添加文本片段
        # 将当前所有片段合并并编码，检查是否达到目标长度
        enc_full = tokenizer(
            "".join(pieces), return_tensors="pt", add_special_tokens=False
        )
        # 当输入token数量 >= 目标prefill长度时停止
        if enc_full.input_ids.shape[1] >= prefill_len:
            break

    # -------------------------------------------------------------------------
    # 3.2 准备模型输入
    # -------------------------------------------------------------------------
    # 将输入token IDs移动到GPU设备0
    input_ids = enc_full["input_ids"].to(0)
    # 获取batch大小和序列长度
    batch_size, seq_len = input_ids.shape

    # 创建缓存位置张量，表示当前处理的token位置（0到seq_len-1）
    cache_position = torch.arange(seq_len, device=0)
    
    # 预分配生成结果张量，大小为 [batch_size, prefill_len + decode_len]
    generated_ids = torch.empty(
        batch_size, seq_len + decode_len, dtype=torch.int, device=0
    )
    # 将原始输入token填入结果张量的前seq_len个位置
    generated_ids[:, :seq_len] = input_ids

    # -------------------------------------------------------------------------
    # 3.3 Prefill阶段（首次前向传播）
    # -------------------------------------------------------------------------
    # 将模型输入移动到GPU
    model_inputs = {key: value.to(0) for key, value in enc_full.items()}
    # 执行首次前向传播，计算所有输入token的表示，同时填充KV Cache
    out = model(
        **model_inputs,
        past_key_values=past_kv,      # 传入KV Cache
        cache_position=cache_position, # 指定缓存位置
        use_cache=True,               # 启用KV Cache
    )
    # 获取最后一个token的logits（预测下一个token的分数）
    logits = out[0]
    # 从logits中采样得到第一个生成的token
    next_token = sample_next_token(logits)
    # 将生成的token存入结果张量的第seq_len位置
    generated_ids[:, seq_len] = next_token

    # -------------------------------------------------------------------------
    # 3.4 Decode阶段（逐token生成，计时开始）
    # -------------------------------------------------------------------------
    # 设置解码位置指针，从seq_len+1开始（因为已经生成了第一个token）
    decode_position = torch.tensor([seq_len + 1], device=0)
    # 同步CUDA设备，确保所有之前的GPU操作完成
    torch.cuda.synchronize()
    # 记录开始时间
    start = time.time()

    # 循环生成剩余的decode_len-1个token
    for _ in range(1, decode_len):
        # 使用torch.nn.attention.sdpa_kernel上下文管理器指定注意力实现后端
        # FLASH_ATTENTION: 使用Flash Attention算法（最快）
        # MATH: 使用标准数学实现（兼容性最好）
        with torch.nn.attention.sdpa_kernel(
            backends=[SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]
        ):
            # 调用decode_one_token函数生成下一个token
            next_token, _ = decode_one_token(
                model,               # 模型
                next_token.clone(),  # 当前token（克隆避免原地修改）
                past_kv,             # KV Cache
                decode_position,     # 当前解码位置
                1.0,                 # temperature（温度参数，1.0表示不调整）
                None,                # top_k（None表示不限制）
                1.0,                 # top_p（1.0表示不限制）
            )
        # 将生成的token存入结果张量的对应位置
        generated_ids[:, decode_position] = next_token.int()
        # 解码位置指针加1
        decode_position += 1

    # -------------------------------------------------------------------------
    # 3.5 计算并返回结果
    # -------------------------------------------------------------------------
    # 同步CUDA设备，确保所有GPU操作完成
    torch.cuda.synchronize()
    # 计算经过的时间（秒）
    elapsed = time.time() - start

    # 使用tokenizer将token IDs解码为文本（仅用于验证输出）
    texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    # 返回生成的token IDs、解码后的文本、吞吐量（tokens/秒）
    return generated_ids, texts, decode_len / elapsed


# =============================================================================
# 模块4: 运行完整基准测试流程
# =============================================================================
def run_benchmark(hf_path, prefill_len, decode_len, empty_model, max_new_tokens):
    """
    运行完整的基准测试流程，包括模型加载、warmup和正式测试。
    
    Args:
        hf_path: HuggingFace模型路径或本地路径
        prefill_len: 预填充长度
        decode_len: 解码长度
        empty_model: 是否加载空模型（用于纯性能测试）
        max_new_tokens: 最大新生成token数（用于KV Cache大小设置）
    """
    # -------------------------------------------------------------------------
    # 4.1 加载模型和分词器
    # -------------------------------------------------------------------------
    # 从指定路径加载模型，返回模型对象和模型字符串标识
    model, model_str = model_from_hf_path(hf_path, empty_model=empty_model)
    # 加载对应的分词器
    tokenizer = AutoTokenizer.from_pretrained(model_str)
    # 将padding token设置为EOS token（某些模型需要）
    tokenizer.pad_token = tokenizer.eos_token

    # -------------------------------------------------------------------------
    # 4.2 初始化KV Cache
    # -------------------------------------------------------------------------
    # 创建StaticCache对象，用于存储注意力机制的key-value缓存
    past_kv = StaticCache(
        model.config,       # 模型配置
        1,                  # batch_size = 1（单样本测试）
        2 * max_new_tokens, # 缓存容量（两倍于max_new_tokens）
        device=0,           # GPU设备0
        dtype=model.dtype,  # 与模型相同的数据类型
    )

    # -------------------------------------------------------------------------
    # 4.3 Warmup运行（预热）
    # -------------------------------------------------------------------------
    # 在正式计时之前运行一次benchmark，用于：
    # 1. 预热GPU，加载数据和代码到缓存
    # 2. 触发CUDA graph捕获等一次性开销
    benchmark(model, tokenizer, 2, 8, past_kv)

    # -------------------------------------------------------------------------
    # 4.4 启用torch.compile优化
    # -------------------------------------------------------------------------
    # 打印提示信息，告知用户CUDA graph捕获可能需要时间
    print(
        "Capturing CUDA graphs, may take some time. If you are running a model over multiple GPUs, "
        "the first generation will be very slow due to compiling the model."
    )

    # 使用torch.compile对decode_one_token函数进行JIT编译优化
    # mode="max-autotune": 使用最大自动调优模式，牺牲编译时间换取最佳性能
    # fullgraph=True: 要求整个函数可以编译为单个图（不能有部分在Python中执行）
    global decode_one_token  # 声明全局变量以便修改
    decode_one_token = torch.compile(
        decode_one_token,
        mode="max-autotune",
        fullgraph=True,
    )

    # -------------------------------------------------------------------------
    # 4.5 正式基准测试
    # -------------------------------------------------------------------------
    # 再次warmup，这次使用编译后的函数
    benchmark(model, tokenizer, 16, 16, past_kv)
    # 执行正式的基准测试，获取吞吐量结果
    _, _, decode_tps = benchmark(model, tokenizer, prefill_len, decode_len, past_kv)
    # 打印最终的解码吞吐量结果
    print(
        f"\nDecoding throughput: {decode_tps:.02f} tokens/sec. Includes tokens generated after the EOS token.\n"
    )


# =============================================================================
# 模块5: 主程序入口与命令行参数解析
# =============================================================================
if __name__ == "__main__":
    # 创建ArgumentParser对象，用于解析命令行参数
    parser = argparse.ArgumentParser(
        description="Benchmark ParoQuant transformer backend decoding"
    )
    # --model: 模型路径（必需参数）
    parser.add_argument("--model", type=str, required=True, help="Path to checkpoint")
    # --empty-model: 是否加载空模型（仅用于性能测试，不加载实际权重）
    parser.add_argument(
        "--empty-model",
        action="store_true",
        help="Load empty model by config for benchmark",
    )
    # --prefill-len: 预填充长度（默认256）
    parser.add_argument("--prefill-len", type=int, default=256, help="Prefill length")
    # --decode-len: 解码长度（默认512，即生成的token数量）
    parser.add_argument("--decode-len", type=int, default=512, help="Decode length")
    # --max-new-tokens: 缓存大小提示（用于分配KV Cache，默认512）
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Cache sizing hint for benchmark",
    )
    # --disable-tf32: 禁用TF32（TensorFloat-32）精度
    parser.add_argument(
        "--disable-tf32",
        action="store_true",
        help="Disable TF32 for FP32 matmuls",
    )
    # 解析命令行参数
    cli_args = parser.parse_args()

    # -------------------------------------------------------------------------
    # 5.1 设置TF32精度（如果未禁用）
    # -------------------------------------------------------------------------
    # TF32是NVIDIA Ampere及更新架构GPU支持的精度格式
    # 它在保持接近FP32精度的同时，提供接近FP16的性能
    if not cli_args.disable_tf32:
        # 启用高精度模式（实际使用TF32进行FP32矩阵乘法）
        torch.set_float32_matmul_precision("high")

    # -------------------------------------------------------------------------
    # 5.2 执行基准测试
    # -------------------------------------------------------------------------
    run_benchmark(
        cli_args.model,
        cli_args.prefill_len,
        cli_args.decode_len,
        cli_args.empty_model,
        cli_args.max_new_tokens,
    )
