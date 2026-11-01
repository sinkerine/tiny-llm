import torch
import transformers

print(f"PyTorch: {torch.__version__}; Transformers: {transformers.__version__}")
if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA unavailable. Check the NVIDIA driver and CUDA-enabled PyTorch installation."
    )
x = torch.randn(32, 32, device="cuda")
y = x @ x.T
torch.cuda.synchronize()
assert torch.isfinite(y).all()
print(
    f"CUDA {torch.version.cuda}: {torch.cuda.get_device_name(0)} — matrix multiplication passed"
)
