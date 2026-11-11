"""Regression coverage for the copied decomposed GDN entry point."""

import pytest
import torch

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import DEFAULT_STANDARDS


def generate_data(bounds, dtype, with_state=True, noncontiguous=False, fixed_length=65):
    # Synthetic token counts with Qwen3-Next TP4 head dimensions (Hg=4/H=8,
    # K=V=128), following the existing chunk_o public tests.
    torch.manual_seed(42)
    tokens = fixed_length if bounds is None else bounds[-1]
    groups, heads, dim = 4, 8, 128
    q = torch.randn(1, tokens, groups, dim, dtype=dtype) * 0.05
    k = torch.randn_like(q) * 0.05
    v = torch.randn(1, tokens, heads, dim, dtype=dtype) * 0.1
    g = -torch.rand(1, tokens, heads, dtype=torch.float32) * 0.05
    beta = torch.rand(1, tokens, heads, dtype=dtype)
    data = dict(q=q, k=k, v=v, g=g, beta=beta)
    if noncontiguous:
        for name, tensor in data.items():
            backing = torch.empty(
                *tensor.shape[:-1], tensor.shape[-1] * 2, dtype=tensor.dtype
            )
            data[name] = backing[..., ::2].copy_(tensor)
    data["initial_state"] = (
        torch.randn(1 if bounds is None else len(bounds) - 1, heads, dim, dim) * 0.01
        if with_state
        else None
    )
    data["cu_seqlens"] = (
        None if bounds is None else torch.tensor(bounds, dtype=torch.int32)
    )
    return data


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "bounds,with_state,normalize,output_final_state,scale,noncontiguous",
    [
        pytest.param(None, True, False, True, None, False, id="fixed-chunk-tail"),
        pytest.param((0, 1), False, True, False, 0.25, False, id="one-token"),
        pytest.param((0, 64), True, True, True, None, False, id="full-chunk"),
        pytest.param(
            (0, 1, 66, 131),
            True,
            True,
            False,
            0.0,
            True,
            id="packed-grouped-strided-zero-scale",
        ),
        pytest.param(
            (0, 31, 100),
            False,
            False,
            True,
            None,
            False,
            id="packed-without-initial-state",
        ),
    ],
)
def test_chunk_gated_delta_rule_npu(
    dtype,
    bounds,
    with_state,
    normalize,
    output_final_state,
    scale,
    noncontiguous,
    monkeypatch,
):
    from sgl_kernel_npu.operators.attention.gdn import chunk_gated_delta_rule_npu

    monkeypatch.setenv("GDN_USE_MEGA_GDN", "0")
    monkeypatch.setenv("TRITON_F32_DEFAULT", "ieee")
    data = generate_data(bounds, dtype, with_state, noncontiguous)
    saved = {k: v.clone() for k, v in data.items() if isinstance(v, torch.Tensor)}
    options = dict(use_qk_l2norm_in_kernel=normalize, scale=scale)
    # Omit the flag in True cases to check the legacy True default.
    if not output_final_state:
        options["output_final_state"] = False
    # FP32 final states inherit the pipeline's input-dtype rounding error.
    standard = {"torch.float32": DEFAULT_STANDARDS[str(dtype)]}
    executor = check_accuracy(
        chunk_gated_delta_rule_npu,
        kwargs={**data, **options},
        standard=standard,
    )
    for name, original in saved.items():
        torch.testing.assert_close(data[name], original, atol=0, rtol=0)
    check_performance(executor, enable_perf=False)


def test_chunk_gated_delta_rule_npu_qwen3_next_tp4_prefill(monkeypatch):
    from sgl_kernel_npu.operators.attention.gdn import chunk_gated_delta_rule_npu

    # Qwen3-Next-80B-A3B-Instruct: 16 key heads, 32 value heads, K=V=128.
    # https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/raw/main/config.json
    # TP4 gives Hg=4/H=8; 2048 tokens represent a prefill workload.
    monkeypatch.setenv("GDN_USE_MEGA_GDN", "0")
    monkeypatch.setenv("TRITON_F32_DEFAULT", "ieee")
    data = generate_data(None, torch.bfloat16, with_state=False, fixed_length=2048)
    executor = check_accuracy(
        chunk_gated_delta_rule_npu,
        kwargs={**data, "use_qk_l2norm_in_kernel": True},
        standard={"torch.float32": DEFAULT_STANDARDS["torch.bfloat16"]},
    )
    check_performance(executor, enable_perf=True)


@pytest.mark.parametrize("output_final_state", [False, True])
def test_chunk_gated_delta_rule_npu_mega(output_final_state, monkeypatch):
    from sgl_kernel_npu.operators.attention.gdn import chunk_gated_delta_rule_npu

    if not hasattr(torch.ops.npu, "mega_chunk_gdn"):
        pytest.skip("The mega_chunk_gdn native operator is required")
    monkeypatch.setenv("GDN_USE_MEGA_GDN", "1")
    data = generate_data((0, 31, 100), torch.float16)
    options = dict(output_final_state=output_final_state)
    standard = {"torch.float32": DEFAULT_STANDARDS["torch.float16"]}
    executor = check_accuracy(
        chunk_gated_delta_rule_npu,
        kwargs={**data, **options},
        standard=standard,
    )
    check_performance(executor, enable_perf=False)
