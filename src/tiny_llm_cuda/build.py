"""Build the Week 2 native CUDA extension outside the source tree."""

import argparse
import os
from pathlib import Path
from torch.utils.cpp_extension import load


def build_directory(reference):
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return (
        cache
        / "dev-scratch/tiny-llm/cuda-week2"
        / ("reference" if reference else "student")
    )


def load_extension(reference=True, verbose=False):
    root = Path(__file__).resolve().parent.parent / "cuda"
    build = build_directory(reference)
    build.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MAX_JOBS", "1")
    return load(
        name="tiny_llm_week2_cuda_ref" if reference else "tiny_llm_week2_cuda_student",
        sources=[
            str(root / "bindings.cpp"),
            str(root / ("reference.cu" if reference else "student.cu")),
        ],
        build_directory=str(build),
        extra_cflags=["-O2"],
        extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr"],
        verbose=verbose,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", action="store_true")
    args = parser.parse_args()
    load_extension(args.reference, verbose=True)
