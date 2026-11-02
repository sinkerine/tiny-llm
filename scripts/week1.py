"""Run the Week 1 CUDA model with a Hugging Face tokenizer."""

import argparse
import importlib

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument(
        "--solution", choices=["tiny_llm", "tiny_llm_ref", "ref"], default="tiny_llm"
    )
    parser.add_argument(
        "--prompt", default="Explain how attention works in one paragraph."
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=40)
    args = parser.parse_args()
    package = importlib.import_module(
        "tiny_llm_ref" if args.solution == "ref" else args.solution
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    hf_model = (
        AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
        .to(args.device)
        .eval()
    )
    model = package.Qwen3ModelWeek1(hf_model)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    package.simple_generate(
        model,
        tokenizer,
        prompt,
        package.make_sampler(args.temperature, args.top_p, args.top_k),
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
