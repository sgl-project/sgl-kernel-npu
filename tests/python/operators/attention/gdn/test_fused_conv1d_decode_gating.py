"""Tests for attention.gdn.fused_conv1d_decode_gating public API."""

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

# Production Qwen3.8-27B (NK=16, NV=48) and Qwen3.5-9B (NK=16, NV=32) decode shapes
perf_shape = [
    (1, 16, 48),
    (8, 16, 48),
    (32, 16, 48),
]


def generate_data(bs, nk, nv, dtype, with_pad_slot):
    torch.npu.manual_seed(42)
    d = 128
    qkv_dim = (2 * nk + nv) * d
    qkvz_dim = qkv_dim + nv * d
    qkvz = torch.randn((bs, qkvz_dim), dtype=dtype)
    ba = torch.randn((bs, 2 * nv), dtype=dtype)
    w_t = torch.randn((4, qkv_dim), dtype=dtype) * 0.1
    conv_state = torch.randn((max(16, bs + 4), 3, qkv_dim), dtype=dtype) * 0.1
    cache_idx = torch.arange(bs, dtype=torch.int32)
    if with_pad_slot and bs >= 2:
        cache_idx[-1] = -1
    A_log = torch.randn((nv,), dtype=torch.float32) * 0.2 - 1.0
    dt_bias = torch.randn((nv,), dtype=dtype) * 0.1
    return qkvz, ba, w_t, conv_state, cache_idx, A_log, dt_bias, nv


@pytest.mark.parametrize(
    "bs,nk,nv",
    [
        pytest.param(1, 16, 48, id="qwen38-27b-bs1"),
        pytest.param(4, 16, 48, id="qwen38-27b-bs4"),
        pytest.param(8, 16, 48, id="qwen38-27b-bs8"),
        pytest.param(32, 16, 48, id="qwen38-27b-bs32"),
        pytest.param(8, 16, 32, id="qwen35-9b-bs8"),
        pytest.param(2, 4, 4, id="edge-qkv512-multiple"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("with_pad_slot", [False, True])
def test_fused_conv1d_decode_gating(bs, nk, nv, dtype, with_pad_slot):
    from sgl_kernel_npu.operators.attention.gdn import fused_conv1d_decode_gating

    qkvz, ba, w_t, conv_state, cache_idx, A_log, dt_bias, num_v_heads = generate_data(
        bs, nk, nv, dtype, with_pad_slot
    )
    standard = {
        "torch.bfloat16": AccuracyStandard(atol=2e-2, rtol=2e-2),
        "torch.float16": AccuracyStandard(atol=1e-2, rtol=1e-2),
        "torch.float32": AccuracyStandard(atol=1e-4, rtol=1e-4),
    }
    executor = check_accuracy(
        fused_conv1d_decode_gating,
        kwargs={
            "projected_states_qkvz": qkvz,
            "projected_states_ba": ba,
            "conv_weights_t": w_t,
            "conv_state": conv_state,
            "cache_indices": cache_idx,
            "A_log": A_log,
            "dt_bias": dt_bias,
            "num_v_heads": num_v_heads,
        },
        mutable_kwargs=("conv_state",),
        standard=standard,
    )
    check_performance(
        executor,
        enable_perf=(
            (bs, nk, nv) in perf_shape and dtype == torch.bfloat16 and not with_pad_slot
        ),
    )
