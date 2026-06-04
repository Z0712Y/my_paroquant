import argparse
import os
import random
import sys
from pathlib import Path

# 添加项目根目录到路径，以便导入 PARO 的模型加载接口
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

# 使用 PARO 的模型加载接口，以正确加载伪量化/真量化模型
from inference_engine.generation.transformers_backend import model_from_hf_path


def get_wikitext2(seed, seqlen, tokenizer):
    """
    加载 WikiText-2 数据集的测试集部分。
    """
    # 从 Hugging Face Hub 加载 wikitext-2-raw-v1 的测试集
    # 使用本地缓存，避免联网
    testdata = load_dataset(
        "wikitext", "wikitext-2-raw-v1", split="test",
        download_mode="reuse_cache_if_exists",
    )
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    return testenc


def get_c4(seed, seqlen, tokenizer):
    """
    加载 C4 (Colossal Clean Crawled Corpus) 数据集的验证集部分。
    优先使用本地缓存文件，避免联网。
    """
    import os
    
    # 本地数据集路径（优先使用）
    local_c4_dir = "/home/hhw/zy/datasets/c4_local/en"
    # HuggingFace 缓存路径
    cache_dir = os.path.expanduser("~/.cache/huggingface/datasets/allenai___c4/en/1.0.0")
    
    # 优先使用本地缓存的验证文件
    val_file = os.path.join(local_c4_dir, "c4-validation.00000-of-00008.json.gz")
    if not os.path.exists(val_file):
        val_file = os.path.join(cache_dir, "c4-validation.00000-of-00008.json.gz")
    
    if os.path.exists(val_file):
        print(f"Loading C4 validation from local file: {val_file}")
        valdata = load_dataset("json", data_files={"validation": val_file}, split="validation")
    else:
        # 回退到 HuggingFace Hub
        print("Local C4 validation not found, trying to load from HuggingFace Hub...")
        valdata = load_dataset(
            "allenai/c4",
            "en",
            split="validation",
            download_mode="reuse_cache_if_exists",
        )

    random.seed(seed)
    valenc = []
    
    # 采样 256 个样本
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]["text"], return_tensors="pt")
            if tmp.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids

    valenc = TokenizerWrapper(valenc)

    return valenc


def get_test_tokens(name, seed, seqlen, tokenizer):
    if name == "wikitext2":
        return get_wikitext2(seed, seqlen, tokenizer).input_ids
    elif name == "c4":
        return get_c4(seed, seqlen, tokenizer).input_ids
    else:
        raise ValueError(f"Unknown dataset {name}")


def load_model_for_eval(path, device_map=None):
    """
    使用 PARO 的模型加载接口加载模型。
    """
    model, _ = model_from_hf_path(path, empty_model=False)
    return model


def main(args):
    # 加载模型，使用 PARO 的接口以正确处理量化模型
    model = load_model_for_eval(args.model, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)

    for dataset in args.datasets:
        print(f"\nEvaluating {dataset}...")
        input_tok = get_test_tokens(
            dataset, seed=args.seed, seqlen=args.seqlen, tokenizer=tokenizer
        )
        nsamples = input_tok.numel() // args.seqlen
        input_tok = input_tok[0, : (args.seqlen * nsamples)].view(nsamples, args.seqlen)

        loss_fct = torch.nn.CrossEntropyLoss().cuda()
        acc_loss = 0.0
        progress = tqdm(range(nsamples))
        for ii in progress:
            input = input_tok[ii, :].cuda().view(1, -1)
            output = model(
                input,
                use_cache=False,
                output_hidden_states=False,
                output_attentions=False,
            )
            output = output[0]
            shift_logits = output[:, :-1, :].contiguous()
            shift_labels = input[:, 1:]
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            acc_loss += loss.item()
            progress.set_description(f"avg_loss = {acc_loss/(ii+1)}")

        avg_loss = acc_loss / nsamples
        ppl = torch.exp(torch.tensor(avg_loss)).item()
        print(f"{dataset} perplexity: {ppl}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int, help="Random seed")
    parser.add_argument("--model", type=str, required=True, help="Path to the model")
    parser.add_argument("--seqlen", type=int, required=True, help="Sequence length")
    parser.add_argument(
        "--datasets", 
        nargs="+", 
        default=["wikitext2"],
        choices=["wikitext2", "c4"],
        help="Datasets to evaluate"
    )
    args = parser.parse_args()
    
    # 将相对路径转换为绝对路径
    args.model = os.path.abspath(args.model)
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model path not found: {args.model}")

    torch.set_grad_enabled(False)
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    main(args)
