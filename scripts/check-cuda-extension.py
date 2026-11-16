"""Verify learner CUDA bindings load without executing unfinished kernels."""

from tiny_llm_cuda.build import load_extension

extension = load_extension(reference=False)
for name in ("w4", "rms_norm", "rope", "swiglu", "attention"):
    assert callable(getattr(extension, name))
print("CUDA extension bindings loaded: True (student kernels remain unfinished)")
