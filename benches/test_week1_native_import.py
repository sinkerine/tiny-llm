"""Week 1 CUDA imports must not compile or load the Week 2 extension."""

import os
from pathlib import Path
import subprocess
import sys


def test_python_course_imports_defer_native_loading():
    root = Path(__file__).resolve().parents[1]
    script = """
import sys
import torch.utils.cpp_extension
calls=[]
def unavailable(*a, **kw):
    calls.append(kw.get("name"))
    raise RuntimeError("native extension unavailable")
torch.utils.cpp_extension.load=unavailable
import tiny_llm
import tiny_llm_ref
assert not calls
assert not any(name == "mlx" or name.startswith("mlx.") for name in sys.modules)
from tiny_llm import cuda_ops
try:
    cuda_ops.extension()
except RuntimeError as exc:
    assert str(exc)=="native extension unavailable"
else:
    raise AssertionError("native call did not reach the deferred loader")
assert calls==["tiny_llm_week2_cuda_student"]
"""
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
