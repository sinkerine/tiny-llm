"""Standalone wrapper for the supplied CUDA extension builder."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tiny_llm_cuda.build import main

if __name__ == "__main__":
    main()
