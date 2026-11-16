"""Synchronized CUDA Week 2 comparison; no model download is performed."""

import argparse
import importlib
import json
import statistics
import time
import torch
from transformers import AutoModelForCausalLM
from .bench_request_metrics import (
    measure_request,
    PEAK_CUDA_SCOPE,
    REQUEST_ELAPSED_SCOPE,
)


class CUDAMemory:
    synchronize = staticmethod(torch.cuda.synchronize)
    reset_peak_memory = staticmethod(torch.cuda.reset_peak_memory_stats)
    get_peak_memory = staticmethod(torch.cuda.max_memory_allocated)


CHECKPOINTS = (
    "week1",
    "kv-cache",
    "capacity-cache",
    "quantized-matvec",
    "simd-matmul",
    "rmsnorm",
    "rope",
    "swiglu",
    "tiled-prefill",
    "selected",
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--solution", choices=("tiny_llm", "tiny_llm_ref"), default="tiny_llm_ref"
    )
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--checkpoint", action="append", choices=CHECKPOINTS)
    p.add_argument("--prefill-tokens", type=int, default=128)
    p.add_argument("--decode-tokens", type=int, default=16)
    p.add_argument("--prefill-logits", choices=("last", "all"), default="all")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--profile", help="optional PyTorch profiler Chrome trace path")
    args = p.parse_args()
    if min(args.prefill_tokens, args.decode_tokens, args.repeats) <= 0:
        p.error("token counts and repeats must be positive")
    return args


@torch.inference_mode()
def main():
    args = parse_args()
    if args.prefill_logits == "last" and "week1" in (args.checkpoint or ["week1"]):
        raise ValueError(
            "Week 1 computes all logits; use --prefill-logits all for matched comparisons"
        )
    package = importlib.import_module(args.solution)
    hf = (
        AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, local_files_only=True
        )
        .cuda()
        .eval()
    )
    prompt = (torch.arange(args.prefill_tokens, device="cuda") % hf.config.vocab_size)[
        None
    ]
    for checkpoint in args.checkpoint or [
        "week1",
        "kv-cache",
        "capacity-cache",
        "selected",
    ]:
        model = (
            package.Qwen3ModelWeek1(hf)
            if checkpoint == "week1"
            else package.Qwen3ModelWeek2(hf, checkpoint=checkpoint)
        )

        def request(model=model):
            cache = (
                None
                if checkpoint == "week1"
                else model.create_kv_cache(
                    capacity=args.prefill_tokens + args.decode_tokens
                    if model.use_bounded_kv_capacity
                    else None
                )
            )
            sequence = prompt
            torch.cuda.synchronize()
            start = time.perf_counter()
            if cache is None:
                logits = model(sequence)
            else:
                logits = model(
                    sequence,
                    0,
                    cache,
                    logits_to_keep=1 if args.prefill_logits == "last" else None,
                )
            torch.cuda.synchronize()
            prefill = time.perf_counter() - start
            offset = args.prefill_tokens
            start = time.perf_counter()
            for _ in range(args.decode_tokens):
                token = logits[:, -1].argmax(-1).reshape(1, 1)
                if cache is None:
                    sequence = torch.cat((sequence, token), 1)
                    logits = model(sequence)
                else:
                    logits = model(token, offset, cache, logits_to_keep=1)
                    offset += 1
            torch.cuda.synchronize()
            decode = time.perf_counter() - start
            counters = (
                {}
                if cache is None
                else {
                    name: sum(getattr(c, name) for c in cache)
                    for name in (
                        "logical_copy_bytes",
                        "physical_growth_copy_bytes",
                        "slice_write_bytes",
                    )
                }
            )
            if cache:
                for layer in cache:
                    layer.release()
            return prefill, decode, counters

        request()  # warmup includes JIT compilation; never report it as steady state
        measurements = [
            measure_request(lambda _: request(), None, memory=CUDAMemory())
            for _ in range(args.repeats)
        ]
        samples = [sample[0] for sample in measurements]
        report = {
            "checkpoint": checkpoint,
            "prefill_tokens": args.prefill_tokens,
            "decode_tokens": args.decode_tokens,
            "prefill_logits": args.prefill_logits,
            "prefill_ms": statistics.median(s[0] for s in samples) * 1000,
            "decode_ms": statistics.median(s[1] for s in samples) * 1000,
            "cache_bytes": samples[-1][2],
            "request_wall_ms": statistics.median(sample[1] for sample in measurements)
            * 1000,
            "request_elapsed_scope": REQUEST_ELAPSED_SCOPE,
            "request_peak_cuda_allocated_bytes": [sample[2] for sample in measurements],
            "peak_cuda_allocated_bytes": max(sample[2] for sample in measurements),
            "peak_cuda_scope": PEAK_CUDA_SCOPE,
        }
        report.update(
            model=args.model,
            gpu=torch.cuda.get_device_name(),
            torch_version=torch.__version__,
            repeats=args.repeats,
            warmups=1,
        )
        print(json.dumps(report), flush=True)
        if args.profile:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as prof:
                request()
            from pathlib import Path

            dest = Path(args.profile)
            dest.parent.mkdir(parents=True, exist_ok=True)
            prof.export_chrome_trace(
                str(dest.with_name(f"{dest.stem}-{checkpoint}{dest.suffix}"))
            )
        del model


if __name__ == "__main__":
    main()
