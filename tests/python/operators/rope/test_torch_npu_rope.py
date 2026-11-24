"""Numerical checks for the three public TorchNPU RoPE variants."""

import pytest

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance

torch = pytest.importorskip("torch")


def rope_inv_freq(head_size, theta):
    return 1.0 / (
        theta ** (torch.arange(0, head_size, 2, dtype=torch.float32) / head_size)
    )


@pytest.mark.parametrize(
    "num_tokens",
    [
        pytest.param(16, id="deepseek-v3.2-dsa-prefill-shared-key-tp8"),
    ],
)
def test_interleave_rope(num_tokens):
    from sgl_kernel_npu.operators.rope import interleave_rope

    torch.npu.manual_seed(42)
    # Fused split QK normalization materializes a shared one-head rotary key.
    key = torch.randn((num_tokens, 1, 1, 64), dtype=torch.bfloat16)
    base_inv_freq = rope_inv_freq(head_size=64, theta=10000.0)
    # YaRN correction bounds floor(10.472), ceil(22.513), from original
    # context length 4096 and beta_fast/beta_slow 32/1. mscale ratio is 1.
    ramp = ((torch.arange(32, dtype=torch.float32) - 10) / (23 - 10)).clamp(0, 1)
    inv_freq = base_inv_freq * (1 - ramp) + base_inv_freq / 40 * ramp
    positions = torch.arange(num_tokens, dtype=torch.float32)
    angles = torch.outer(positions, inv_freq).repeat(1, 2).view(num_tokens, 1, 1, 64)
    cos, sin = angles.cos().to(key.dtype), angles.sin().to(key.dtype)
    executor = check_accuracy(interleave_rope, args=(key, cos, sin))
    check_performance(executor, enable_perf=True)


@pytest.mark.parametrize(
    "num_tokens, rotary_mode",
    [pytest.param(16, "half", id="gemma-3-1b-local-attn-query-tp1")],
)
def test_rotary_mul(num_tokens, rotary_mode):
    from sgl_kernel_npu.operators.rope import rotary_mul

    torch.npu.manual_seed(42)
    head_size = 256
    query = torch.randn((1, num_tokens, 4, head_size), dtype=torch.bfloat16)
    positions = torch.arange(num_tokens, dtype=torch.float32)
    angles = torch.outer(positions, rope_inv_freq(head_size, theta=10000.0))
    angles = angles.repeat(1, 2).view(1, num_tokens, 1, head_size)
    cos, sin = angles.cos().to(query.dtype), angles.sin().to(query.dtype)
    executor = check_accuracy(rotary_mul, args=(query, cos, sin, rotary_mode))
    check_performance(executor, enable_perf=True)


def generate_mrope_data(num_tokens, head_size):
    torch.npu.manual_seed(42)
    # Both TP configurations have 14 query and 2 KV heads per rank.
    # Splitting the fused projection preserves the full QKV row stride.
    qkv = torch.randn((num_tokens, 18 * head_size), dtype=torch.bfloat16)
    query, key, _ = qkv.split((14 * head_size, 2 * head_size, 2 * head_size), dim=-1)
    positions = torch.arange(num_tokens, dtype=torch.int64)
    cache_positions = torch.arange(32768, dtype=torch.float32)
    angles = torch.outer(cache_positions, rope_inv_freq(head_size, theta=1000000.0))
    cache = torch.cat((angles.cos(), angles.sin()), dim=-1).to(query.dtype)
    return positions, query, key, cache


@pytest.mark.parametrize(
    "head_size, section",
    [
        pytest.param(64, [0, 0, 0], id="qwen2-0.5b-text-prefill-tp1"),
        pytest.param(128, [16, 24, 24], id="qwen2.5-vl-7b-image-prefill-tp2"),
    ],
)
def test_mrope(head_size, section):
    from sgl_kernel_npu.operators.rope import mrope
    from sgl_kernel_npu.operators.utils.registry import get_impl

    assert get_impl(mrope, "auto") is get_impl(mrope, "ops-transformer")
    positions, query, key, cache = generate_mrope_data(16, head_size)
    if section[0]:
        # One 4x4 image grid after a 17-token text prefix, in (T, H, W) order.
        positions = 17 + torch.stack(
            (torch.zeros_like(positions), positions // 4, positions % 4)
        )
    executor = check_accuracy(
        mrope,
        args=(positions, query, key, cache, head_size),
        kwargs={
            "mrope_section": section,
            "rotary_mode": "half",
            "cache_mode": "default",
        },
    )
    check_performance(executor, enable_perf=True)
