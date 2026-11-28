"""Run Weeks 1, 2, or 3 on CUDA with a Hugging Face tokenizer."""

import argparse
import importlib

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.inference_mode()
def main(week=1):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--draft-model", default=None)
    parser.add_argument("--proposal-length", type=int, default=4)
    parser.add_argument(
        "--solution", choices=["tiny_llm", "tiny_llm_ref", "ref"], default="tiny_llm"
    )
    parser.add_argument(
        "--prompt", default="Explain how attention works in one paragraph."
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prefill-step", type=int, default=128)
    parser.add_argument("--page-size", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--sampler-temp", type=float, default=None)
    parser.add_argument("--sampler-top-p", type=float, default=None)
    parser.add_argument("--sampler-top-k", type=int, default=None)
    parser.add_argument(
        "--week2-checkpoint",
        default="selected",
        choices=[
            "kv-cache",
            "capacity-cache",
            "quantized-matvec",
            "simd-matmul",
            "rmsnorm",
            "rope",
            "swiglu",
            "tiled-prefill",
            "selected",
        ],
    )
    args = parser.parse_args()

    if args.max_tokens < 0:
        parser.error("--max-tokens must be a non-negative integer")
    if args.draft_model is not None and any(
        val is not None
        for val in (args.sampler_temp, args.sampler_top_p, args.sampler_top_k)
    ):
        parser.error("speculative decoding (--draft-model) does not support sampling")
    if args.max_tokens == 0:
        return

    package = importlib.import_module(
        "tiny_llm_ref" if args.solution == "ref" else args.solution
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    hf_model = (
        AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
        .to(args.device)
        .eval()
    )
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    if week == 3:
        model = package.dispatch_model(
            args.model, hf_model, week=3, page_size=args.page_size
        )
        if args.draft_model is not None:
            draft_tokenizer = AutoTokenizer.from_pretrained(args.draft_model)
            draft_hf = (
                AutoModelForCausalLM.from_pretrained(
                    args.draft_model, dtype=torch.bfloat16
                )
                .to(args.device)
                .eval()
            )
            draft_model = package.dispatch_model(
                args.draft_model, draft_hf, week=3, page_size=args.page_size
            )
            package.speculative_generate(
                draft_model,
                model,
                draft_tokenizer,
                tokenizer,
                prompt,
                proposal_length=args.proposal_length,
                max_tokens=args.max_tokens,
            )
            return
        results = package.batch_generate(
            model,
            tokenizer,
            [prompt],
            max_seq_len=args.max_seq_len,
            batch_size=args.batch_size,
            prefill_step=args.prefill_step,
        )
        for _, text in results:
            print(text)
        return

    model = (
        package.Qwen3ModelWeek1(hf_model)
        if week == 1
        else package.Qwen3ModelWeek2(hf_model, checkpoint=args.week2_checkpoint)
    )
    generate = package.simple_generate if week == 1 else package.cached_generate
    temp = args.sampler_temp if args.sampler_temp is not None else args.temperature
    top_p = args.sampler_top_p if args.sampler_top_p is not None else args.top_p
    top_k = args.sampler_top_k if args.sampler_top_k is not None else args.top_k
    generate(
        model,
        tokenizer,
        prompt,
        package.make_sampler(temp, top_p, top_k),
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
