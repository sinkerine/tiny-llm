"""Week 1 uses PyTorch/CUDA; later course weeks retain their MLX CLI."""

import runpy
import sys


if __name__ == "__main__":
    loader = "week1"
    if "--loader" in sys.argv:
        index = sys.argv.index("--loader")
        if index + 1 < len(sys.argv):
            loader = sys.argv[index + 1]
            if loader in ("week1", "week2"):
                del sys.argv[index : index + 2]
    if loader in ("week1", "week2"):
        from scripts.week1 import main

        main(week=1 if loader == "week1" else 2)
    else:
        runpy.run_path("scripts/main_mlx.py", run_name="__main__")
