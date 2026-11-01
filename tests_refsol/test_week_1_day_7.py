import pytest
import torch
from .tiny_llm_base import make_sampler
from .week1_utils import DEVICES


@pytest.mark.parametrize("device", DEVICES)
def test_task_1_greedy_preserves_input_and_rng(device):
    x = torch.tensor([[0.1, 0.6, 0.3]], device=device).log()
    before = x.clone()
    torch.manual_seed(2026)
    expected = torch.rand(8, device=device)
    torch.manual_seed(2026)
    result = make_sampler(0, 0.1, 1)(x)
    assert result.shape == (1,)
    assert result.dtype == torch.long
    assert result.item() == 1
    torch.testing.assert_close(x, before)
    torch.testing.assert_close(torch.rand(8, device=device), expected)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "temp,top_p,top_k,expected",
    [
        (0.5, None, None, [0.55**2, 0.25**2, 0.15**2, 0.05**2]),
        (1.0, None, 2, [0.55, 0.25, 0, 0]),
        (1.0, 0.7, None, [0.55, 0.25, 0, 0]),
        (0.5, 0.7, 3, [0.55**2, 0.25**2, 0, 0]),
        (1.0, 1.0, 4, [0.55, 0.25, 0.15, 0.05]),
        (1.0, 0.0, 0, [0.55, 0.25, 0.15, 0.05]),
    ],
)
def test_task_1_sampling_distribution(device, temp, top_p, top_k, expected):
    x = torch.tensor([[0.55, 0.25, 0.15, 0.05]], device=device).log().expand(4096, -1)
    before = x.clone()
    torch.manual_seed(2026)
    tokens = make_sampler(temp, top_p, top_k)(x)
    actual = tokens.bincount(minlength=4).float() / len(tokens)
    expected = torch.tensor(expected, device=device)
    expected /= expected.sum()
    torch.testing.assert_close(actual, expected, atol=0.04, rtol=0)
    assert (actual[expected == 0] == 0).all()
    torch.testing.assert_close(x, before)


@pytest.mark.parametrize("device", DEVICES)
def test_task_1_seed_and_invalid_top_k(device):
    x = torch.zeros(64, 4, device=device)
    sampler = make_sampler(1.0, None, None)
    torch.manual_seed(42)
    first = sampler(x)
    torch.manual_seed(42)
    torch.testing.assert_close(first, sampler(x))
    torch.manual_seed(43)
    assert not torch.equal(first, sampler(x))
    with pytest.raises(ValueError):
        make_sampler(1.0, None, 5)(x)
