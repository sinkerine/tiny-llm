import torch


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable to PyTorch")

    device = torch.device("cuda")
    left = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
    right = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device=device)
    result = left @ right
    torch.cuda.synchronize()

    expected = torch.tensor([[19.0, 22.0], [43.0, 50.0]], device=device)
    if not torch.equal(result, expected):
        raise SystemExit(f"CUDA matmul returned {result}")

    print(f"CUDA device: {torch.cuda.get_device_name(device)}")
    print(f"CUDA matmul: {result.cpu().tolist()}")


if __name__ == "__main__":
    main()
