import pytest
import torch
from transformers import AutoModelForCausalLM
from .tiny_llm_base import Qwen3ModelWeek1, Embedding, qwen3_week1
from .week1_utils import DEVICES, DTYPES, assert_close, tiny_model, block_kwargs


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_task_1_transformer_block(device, dtype):
    model = tiny_model(device, dtype)
    x = torch.randn(2, 5, 24, device=device, dtype=dtype)
    positions = torch.arange(5, device=device)[None].expand(2, -1)
    mask = torch.randn(2, 1, 5, 5, device=device, dtype=dtype)
    expected = model.model.layers[0](
        x, attention_mask=mask, position_embeddings=model.model.rotary_emb(x, positions)
    )
    actual = qwen3_week1.Qwen3TransformerBlock(**block_kwargs(model))(x, mask)
    assert_close(actual, expected)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(), (3,), (2, 3), (2, 1, 3)])
def test_task_2_embedding(device, dtype, shape):
    weight = torch.randn(17, 8, device=device, dtype=dtype)
    layer = Embedding(17, 8, weight)
    tokens = torch.randint(17, shape, device=device)
    assert_close(layer(tokens), torch.nn.functional.embedding(tokens, weight))
    x = torch.randn(*shape, 8, device=device, dtype=dtype)
    assert_close(layer.as_linear(x), torch.nn.functional.linear(x, weight))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("tied", [True, False])
@torch.inference_mode()
def test_task_3_qwen_without_download(device, dtype, tied):
    model = tiny_model(device, dtype, tied=tied)
    tokens = torch.randint(43, (2, 7), device=device)
    assert_close(Qwen3ModelWeek1(model)(tokens), model(tokens).logits)


@pytest.mark.parametrize(
    "model_name", ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B"]
)
@torch.inference_mode()
def test_task_3_downloaded_model(model_name):
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        path = snapshot_download(model_name, local_files_only=True)
    except LocalEntryNotFoundError:
        pytest.skip(f"Download with: hf download {model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = (
        AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, local_files_only=True
        )
        .to(device)
        .eval()
    )
    tokens = torch.arange(5, device=device)[None]
    assert_close(Qwen3ModelWeek1(model)(tokens), model(tokens).logits)
