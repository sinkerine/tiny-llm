import pytest
import torch
from .tiny_llm_base import RoPE
from .week1_utils import DEVICES, DTYPES, assert_close


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("traditional", [False, True])
@pytest.mark.parametrize("offset", [None, slice(3, 8), [slice(1, 6), slice(7, 12)]])
def test_task_1_and_2_rope(device, dtype, traditional, offset):
    x = torch.randn(2, 5, 3, 8, device=device, dtype=dtype)
    positions = torch.arange(5, device=device).expand(2, -1)
    if isinstance(offset, slice):
        positions = positions + offset.start
    elif isinstance(offset, list):
        positions = positions + torch.tensor([1, 7], device=device)[:, None]
    angles = positions.float()[..., None] / (
        10000 ** (torch.arange(0, 8, 2, device=device).float() / 8)
    )
    z = (
        x.float().reshape(2, 5, 3, 4, 2)
        if traditional
        else torch.stack(x.float().chunk(2, dim=-1), dim=-1)
    )
    complex_x = torch.view_as_complex(z.contiguous())
    rotated = torch.view_as_real(
        complex_x * torch.polar(torch.ones_like(angles), angles)[:, :, None]
    )
    expected = (
        rotated.flatten(-2) if traditional else torch.cat(rotated.unbind(-1), dim=-1)
    )
    actual = RoPE(8, 32, traditional=traditional)(x, offset)
    assert_close(actual, expected.to(dtype))
