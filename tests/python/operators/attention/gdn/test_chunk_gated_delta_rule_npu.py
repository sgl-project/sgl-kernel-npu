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


@pytest.mark.parametrize(
    "dtype,fixed_length,bounds,with_state,normalize,output_final_state,scale,noncontiguous",
    [
        pytest.param(
            torch.float16,
            65,
            None,
            True,
            False,
            True,
            None,
            False,
            id="fixed-chunk-tail-fp16",
        ),
        pytest.param(
            torch.float16,
            65,
            (0, 1),
            False,
            True,
            False,
            0.25,
            False,
            id="one-token-fp16",
        ),
        pytest.param(
            torch.float16,
            65,
            (0, 64),
            True,
            True,
            True,
            None,
            False,
            id="full-chunk-fp16",
        ),
        pytest.param(
            torch.float16,
            65,
            (0, 1, 66, 131),
            True,
            True,
            False,
            0.0,
            True,
            id="packed-grouped-strided-zero-scale-fp16",
        ),
        pytest.param(
            torch.float16,
            65,
            (0, 31, 100),
            False,
            False,
            True,
            None,
            False,
            id="packed-without-initial-state-fp16",
        ),
        pytest.param(
            torch.bfloat16,
            65,
            None,
            True,
            False,
            True,
            None,
            False,
            id="fixed-chunk-tail-bf16",
        ),
        pytest.param(
            torch.bfloat16,
            65,
            (0, 1),
            False,
            True,
            False,
            0.25,
            False,
            id="one-token-bf16",
        ),
        pytest.param(
            torch.bfloat16,
            65,
            (0, 64),
            True,
            True,
            True,
            None,
            False,
            id="full-chunk-bf16",
        ),
        pytest.param(
            torch.bfloat16,
            65,
            (0, 1, 66, 131),
            True,
            True,
            False,
            0.0,
            True,
            id="packed-grouped-strided-zero-scale-bf16",
        ),
        pytest.param(
            torch.bfloat16,
            65,
            (0, 31, 100),
            False,
            False,
            True,
            None,
            False,
            id="packed-without-initial-state-bf16",
        ),
        # Qwen3-Next-80B-A3B-Instruct: 16 key/32 value heads, K=V=128.
        # https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/raw/main/config.json
        # TP4 gives Hg=4/H=8; T=2048 represents a BF16 prefill workload.
        pytest.param(
            torch.bfloat16,
            2048,
            None,
            False,
            True,
            True,
            None,
            False,
            id="qwen3-next-80b-a3b-tp4-prefill-2048-bf16",
        ),
    ],
)
def test_chunk_gated_delta_rule_npu(
    dtype,
    fixed_length,
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
    data = generate_data(bounds, dtype, with_state, noncontiguous, fixed_length)
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
    check_performance(executor, enable_perf=fixed_length == 2048)


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
