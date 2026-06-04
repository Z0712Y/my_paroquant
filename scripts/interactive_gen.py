import argparse
import asyncio
import sys
from pathlib import Path
import io
import contextlib
import warnings
import traceback
import os
from dataclasses import dataclass
from typing import Dict, List, Optional
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.theme import Theme

# 将项目根目录加入 sys.path，以便导入 inference_engine
sys.path.append(str(Path(__file__).parents[1]))

from inference_engine.generation import create_generator, GenerationParams


# ==========================================
# 模块一：配置数据结构
# ==========================================
@dataclass
class ChatAppConfig:
    """聊天应用配置，聚合命令行参数。"""
    model: str
    backend: str
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: Optional[int]
    gpu_memory_utilization: float = 0.8
    enable_thinking: bool = False
    debug: bool = False


# ==========================================
# 模块二：stderr 静默上下文管理器
# ==========================================
@contextlib.contextmanager
def _silence_stderr():
    """
    临时将 stderr 重定向到内存缓冲区，用于非 debug 模式下
    抑制 vLLM / transformers 等库的海量日志输出。
    """
    buffer = io.StringIO()
    with contextlib.redirect_stderr(buffer):
        yield


# ==========================================
# 模块三：核心聊天循环
# ==========================================
async def run_chat_app(config: ChatAppConfig):
    """初始化生成器并运行交互式终端聊天。"""
    # 若非 debug 模式，全局抑制各类库的日志和进度条
    if not config.debug:
        warnings.filterwarnings("ignore")
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        if config.backend == "vllm":
            os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
        try:
            from huggingface_hub import disable_progress_bars
            disable_progress_bars()
        except Exception:
            pass
        try:
            from transformers.utils import logging as transformers_logging
            transformers_logging.set_verbosity_error()
        except Exception:
            pass

    # 初始化 Rich 控制台，配置用户/助手/提示的配色主题
    console = Console(
        theme=Theme(
            {
                "user": "bold cyan",
                "assistant": "bold blue",
                "hint": "dim",
            }
        )
    )

    # 组装传递给生成器的额外参数
    kwargs = {"enable_thinking": config.enable_thinking}
    if config.backend == "vllm":
        kwargs["gpu_memory_utilization"] = config.gpu_memory_utilization

    console.print("[hint]Loading model...[/hint]")
    # 根据后端类型创建文本生成器实例
    generator = create_generator(config.backend, config.model, **kwargs)

    # 打印欢迎面板，展示当前 backend、model 及命令提示
    console.print(
        Panel.fit(
            f"[bold]ParoQuant Chat[/bold]\nBackend: [bold]{config.backend}[/bold]\nModel: [bold]{config.model}[/bold]\n\nType [bold]/quit[/bold] to exit, [bold]/clear[/bold] to reset history.",
            border_style="bright_blue",
        )
    )

    # 维护多轮对话历史（OpenAI 风格的 role/content 列表）
    history: List[Dict[str, str]] = []

    try:
        while True:
            try:
                # Rich Prompt.ask 会带颜色显示提示符并等待用户输入
                user_prompt = Prompt.ask("[user]You[/user]").strip()
            except (KeyboardInterrupt, EOFError):
                console.print("\n[hint]Session closed.[/hint]")
                break

            # 空输入则重新等待
            if not user_prompt:
                continue
            # 退出命令
            if user_prompt.lower() in {"/quit", "quit", "/exit", "exit"}:
                break
            # 清空历史
            if user_prompt.lower() == "/clear":
                history.clear()
                console.clear()
                console.print("[hint]Conversation history cleared.[/hint]")
                continue

            # 将用户输入加入历史
            history.append({"role": "user", "content": user_prompt})

            # 构建生成参数
            params = GenerationParams(
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
            )

            console.print("[assistant]Assistant[/assistant]: ", end="")
            # 根据 debug 模式决定是否静默 stderr
            generation_ctx = (
                contextlib.nullcontext() if config.debug else _silence_stderr()
            )
            with generation_ctx:
                # 流式生成：on_text 回调实时打印 token 到控制台
                result = await generator.generate(
                    history,
                    params,
                    on_text=lambda text: console.print(
                        text,
                        end="",
                        highlight=False,
                        soft_wrap=True,
                    ),
                )
            console.print()

            # 将模型输出加入历史
            history.append({"role": "assistant", "content": result.output_text})

            # 格式化并打印性能指标（TTFT、TPS、总 token 数等）
            stats = result.stats
            ttft = f"{stats.ttft_s * 1000:.2f}ms" if stats.ttft_s is not None else "n/a"
            metric_str = f"tokens={stats.token_count} | time={stats.total_time_s:.2f}s | ttft={ttft} | tps={stats.tokens_per_second:.2f}"
            console.print(f"Metrics: {metric_str}", style="hint", highlight=False)
            console.print()
    finally:
        # 确保生成器资源被正确释放（如关闭 vLLM engine）
        await generator.close()


# ==========================================
# 模块四：命令行参数解析
# ==========================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive terminal chat for ParoQuant models"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to checkpoint or HF model id"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=["vllm"],
        help="Generation backend (vLLM only)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16384,
        help="Maximum number of new tokens",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.6, help="Sampling temperature"
    )
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling")
    parser.add_argument("--top-k", type=int, default=32, help="Top-k sampling")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
        help="vLLM GPU memory utilization ratio",
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_false",
        dest="enable_thinking",
        help="Pass enable_thinking=False to chat template when supported",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Disable suppression and show backend logs/warnings",
    )
    return parser


# ==========================================
# 模块五：从命令行参数启动
# ==========================================
async def run_from_args(args: argparse.Namespace):
    config = ChatAppConfig(
        model=args.model,
        backend=args.backend,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_thinking=args.enable_thinking,
        debug=args.debug,
    )
    await run_chat_app(config)


if __name__ == "__main__":
    cli_args = build_parser().parse_args()
    asyncio.run(run_from_args(cli_args))
