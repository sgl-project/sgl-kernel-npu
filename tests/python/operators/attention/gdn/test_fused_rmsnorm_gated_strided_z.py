"""Tests for attention.gdn.fused_rmsnorm_gated_strided_z public API."""

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

# Production Qwen3.8-27B (NV=48, D=128) decode and prefill token counts
perf_shape = [
    (1, 16, 48, 128),
    (32, 16, 48, 128),
    (733, 16, 48, 128),
    (2932, 16, 48, 128),
]


def generate_data(num_tokens, nk, nv, d, dtype):
    torch.npu.manual_seed(42)
    qkv_dim = (2 * nk + nv) * d
    qkvz_dim = qkv_dim + nv * d
    core_attn_out = torch.randn((num_tokens, nv, d), dtype=dtype)
    qkvz = torch.randn((num_tokens, qkvz_dim), dtype=dtype)
    norm_weight = torch.randn((d,), dtype=dtype)
    return core_attn_out, qkvz, norm_weight, qkv_dim


@pytest.mark.parametrize(
    "num_tokens,nk,nv,d",
    [
        pytest.param(1, 16, 48, 128, id="qwen38-27b-decode-bs1"),
        pytest.param(8, 16, 48, 128, id="qwen38-27b-decode-bs8"),
        pytest.param(32, 16, 48, 128, id="qwen38-27b-decode-bs32"),
        pytest.param(733, 16, 48, 128, id="qwen38-27b-prefill-733"),
        pytest.param(2932, 16, 48, 128, id="qwen38-27b-prefill-2932"),
        pytest.param(512, 16, 32, 128, id="qwen35-9b-prefill-512"),
        pytest.param(3, 4, 6, 64, id="edge-nonpow2-heads"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("eps", [1e-6])
def test_fused_rmsnorm_gated_strided_z(num_tokens, nk, nv, d, dtype, eps):
    from sgl_kernel_npu.operators.attention.gdn import fused_rmsnorm_gated_strided_z

    core_attn_out, qkvz, norm_weight, z_offset = generate_data(
        num_tokens, nk, nv, d, dtype
    )
    standard = {
        "torch.bfloat16": AccuracyStandard(atol=2e-2, rtol=2e-2),
        "torch.float16": AccuracyStandard(atol=1e-2, rtol=1e-2),
    }
    executor = check_accuracy(
        fused_rmsnorm_gated_strided_z,
        args=(core_attn_out, qkvz, norm_weight, z_offset, eps),
        standard=standard,
    )
    check_performance(
        executor,
        enable_perf=(num_tokens, nk, nv, d) in perf_shape and dtype == torch.bfloat16,
    )
