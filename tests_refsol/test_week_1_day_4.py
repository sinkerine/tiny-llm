import pytest
import torch
import torch.nn.functional as F
from .tiny_llm_base import RMSNorm, silu, qwen3_week1
from .week1_utils import DEVICES, DTYPES, assert_close, tiny_model


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(7,), (2, 7), (2, 3, 7)])
def test_task_1_rms_norm(device, dtype, shape):
    x = torch.randn(*shape, device=device, dtype=dtype) * 1000
    weight = torch.randn(7, device=device, dtype=dtype)
    expected = F.rms_norm(x.float(), (7,), eps=1e-5).to(dtype) * weight
    assert_close(RMSNorm(7, weight)(x), expected)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_task_2_silu(device, dtype):
    x = torch.tensor(
        [-100.0, -12.0, -4.0, 0.0, 4.0, 12.0, 100.0], device=device, dtype=dtype
    )
    actual = silu(x)
    assert_close(actual, F.silu(x))
    assert actual[1] != 0


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_task_2_qwen_mlp(device, dtype):
    model = tiny_model(device, dtype)
    layer = model.model.layers[0].mlp
    x = torch.randn(2, 1, 3, 24, device=device, dtype=dtype)
    actual = qwen3_week1.Qwen3MLP(
        24, 48, layer.gate_proj.weight, layer.up_proj.weight, layer.down_proj.weight
    )(x)
    assert_close(actual, layer(x))
