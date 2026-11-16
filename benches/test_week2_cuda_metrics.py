"""CUDA integration checks for the upstream request-metric contract."""

import json
from types import SimpleNamespace
import pytest
from benches import bench_week2_cuda as bench
from benches.bench_request_metrics import PEAK_CUDA_SCOPE
from tests.week2_utils import CUDA, tiny_model


@CUDA
def test_cuda_request_reports_per_request_allocated_peaks(monkeypatch, capsys):
    hf = tiny_model()
    monkeypatch.setattr(
        bench.AutoModelForCausalLM, "from_pretrained", lambda *a, **kw: hf
    )
    args = SimpleNamespace(
        solution="tiny_llm_ref",
        model="tiny-offline-fixture",
        checkpoint=["capacity-cache"],
        prefill_tokens=3,
        decode_tokens=2,
        prefill_logits="all",
        repeats=2,
        profile=None,
    )
    monkeypatch.setattr(bench, "parse_args", lambda: args)
    bench.main()
    result = json.loads(capsys.readouterr().out)
    assert len(result["request_peak_cuda_allocated_bytes"]) == 2
    assert result["peak_cuda_allocated_bytes"] == max(
        result["request_peak_cuda_allocated_bytes"]
    )
    assert result["peak_cuda_allocated_bytes"] > 0
    assert result["peak_cuda_scope"] == PEAK_CUDA_SCOPE
    assert result["request_wall_ms"] > 0
    assert result["cache_bytes"]["physical_growth_copy_bytes"] == 0
    assert result["cache_bytes"]["slice_write_bytes"] > 0


def test_last_logit_mode_refuses_unmatched_week1_comparison(monkeypatch):
    monkeypatch.setattr(
        bench,
        "parse_args",
        lambda: SimpleNamespace(prefill_logits="last", checkpoint=["week1"]),
    )
    with pytest.raises(ValueError, match="all logits"):
        bench.main()
