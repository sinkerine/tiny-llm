import pytest
import torch
from .tiny_llm_base import simple_generate
from .week1_utils import DEVICES


class FakeTokenizer:
    eos_token_id = 0

    def encode(self, prompt, add_special_tokens=False):
        assert not add_special_tokens
        return [7, 0] if prompt else []

    def decode(self, tokens, **kwargs):
        return "".join({2: "A", 3: "B", 4: "C"}.get(int(t), "") for t in tokens)


class ScriptedModel:
    def __init__(self, tokens, device):
        self.tokens = iter(tokens)
        self.device = device
        self.inputs = []

    def __call__(self, tokens):
        self.inputs.append(tokens.clone())
        logits = torch.full((*tokens.shape, 8), -10000.0, device=self.device)
        logits[:, -1, next(self.tokens)] = 10000.0
        return logits


@pytest.mark.parametrize("device", DEVICES)
def test_task_1_greedy_prefill_decode_and_eos(device, capsys):
    model = ScriptedModel([2, 3, 0], device)
    assert simple_generate(model, FakeTokenizer(), "prompt", None) is None
    assert capsys.readouterr().out.strip() == "AB"
    assert model.inputs[0].tolist() == [[7, 0]]
    assert model.inputs[-1].tolist() == [[7, 0, 2, 3]]


@pytest.mark.parametrize("device", DEVICES)
def test_task_1_sampler_and_bound(device, capsys):
    model = ScriptedModel([2] * 4, device)

    def sampler(logprobs):
        assert torch.isfinite(logprobs).all()
        torch.testing.assert_close(logprobs.exp().sum(-1), torch.ones(1, device=device))
        return torch.tensor([4], device=device)

    simple_generate(model, FakeTokenizer(), "prompt", sampler, max_tokens=3)
    assert capsys.readouterr().out.strip() == "CCC"
    assert len(model.inputs) == 3


def test_task_1_default_bound(capsys):
    model = ScriptedModel([2] * 257, "cpu")
    simple_generate(model, FakeTokenizer(), "prompt", None)
    assert len(model.inputs) == 256
    assert capsys.readouterr().out.strip() == "A" * 256


@pytest.mark.parametrize("limit", [-1, True, 1.5])
def test_task_1_invalid_limit(limit):
    with pytest.raises(ValueError):
        simple_generate(
            ScriptedModel([], "cpu"), FakeTokenizer(), "prompt", None, max_tokens=limit
        )


def test_task_1_zero_and_empty(capsys):
    simple_generate(
        ScriptedModel([], "cpu"), FakeTokenizer(), "prompt", None, max_tokens=0
    )
    assert capsys.readouterr().out == ""
    with pytest.raises(ValueError):
        simple_generate(ScriptedModel([], "cpu"), FakeTokenizer(), "", None)
