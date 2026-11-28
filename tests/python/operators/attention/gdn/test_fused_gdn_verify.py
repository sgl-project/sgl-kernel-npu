"""Tests for attention.gdn.fused_gdn_verify public API."""

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

# Production Qwen3.8-27B (GQA=3, MTP K=4 / DSpark K=8) and Qwen3.5-9B (GQA=2, MTP K=4)
perf_shape = [
    (1, 4, 16, 48),
    (32, 4, 16, 48),
    (1, 8, 16, 48),
    (32, 8, 16, 48),
    (32, 4, 16, 32),
]


def generate_data(bs, k_ver, nk, nv, with_pad_slot):
    torch.npu.manual_seed(42)
    d = 128
    qkv_dim = (2 * nk + nv) * d
    mixed_qkv = torch.randn((bs, k_ver, qkv_dim), dtype=torch.bfloat16) * 0.4
    a = torch.randn((bs, k_ver, nv), dtype=torch.bfloat16) * 0.3
    b = torch.randn((bs, k_ver, nv), dtype=torch.bfloat16) * 0.5
    A_log = torch.randn((nv,), dtype=torch.float32) * 0.2 - 1.0
    dt_bias = torch.randn((nv,), dtype=torch.bfloat16) * 0.1
    num_slots = max(16, bs + 4)
    ssm_states = torch.randn((num_slots, nv, d, d), dtype=torch.bfloat16) * 0.1
    cache_indices = torch.arange(1, bs + 1, dtype=torch.int32)
    if with_pad_slot and bs >= 2:
        cache_indices[-1] = -1
    return mixed_qkv, a, b, A_log, dt_bias, ssm_states, cache_indices, nk, nv


@pytest.mark.parametrize(
    "bs,k_ver,nk,nv",
    [
        pytest.param(1, 4, 16, 48, id="qwen38-27b-mtp-bs1"),
        pytest.param(8, 4, 16, 48, id="qwen38-27b-mtp-bs8"),
        pytest.param(32, 4, 16, 48, id="qwen38-27b-mtp-bs32"),
        pytest.param(1, 8, 16, 48, id="qwen38-27b-dspark-bs1"),
        pytest.param(32, 8, 16, 48, id="qwen38-27b-dspark-bs32"),
        pytest.param(32, 4, 16, 32, id="qwen35-9b-gqa2-bs32"),
        pytest.param(13, 4, 16, 48, id="edge-odd-bs13-rpack"),
        pytest.param(17, 8, 16, 48, id="edge-odd-bs17-rpack"),
        pytest.param(4, 3, 16, 48, id="edge-nonpow2-k3"),
        pytest.param(10, 5, 16, 32, id="edge-nonpow2-k5"),
        pytest.param(4, 12, 8, 32, id="edge-gqa4-k12"),
        pytest.param(12, 16, 16, 16, id="edge-gqa1-k16"),
    ],
)
@pytest.mark.parametrize("with_pad_slot", [False, True])
def test_fused_gdn_verify(bs, k_ver, nk, nv, with_pad_slot):
    from sgl_kernel_npu.operators.attention.gdn import fused_gdn_verify

    (
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        ssm_states,
        cache_indices,
        num_k_heads,
        num_v_heads,
    ) = generate_data(bs, k_ver, nk, nv, with_pad_slot)
    standard = {
        "torch.bfloat16": AccuracyStandard(atol=3e-2, rtol=3e-2),
        "torch.float32": AccuracyStandard(atol=1e-3, rtol=1e-3),
    }
    executor = check_accuracy(
        fused_gdn_verify,
        args=(
            mixed_qkv,
            a,
            b,
            A_log,
            dt_bias,
            ssm_states,
            cache_indices,
            num_k_heads,
            num_v_heads,
        ),
        standard=standard,
    )
    check_performance(
        executor,
        enable_perf=((bs, k_ver, nk, nv) in perf_shape and not with_pad_slot),
    )


def test_fused_gdn_verify_workspace_reuse():
    from sgl_kernel_npu.operators.attention.gdn import (
        create_gdn_verify_workspace,
        fused_gdn_verify,
    )

    ws = create_gdn_verify_workspace()
    data = generate_data(16, 4, 16, 32, with_pad_slot=False)
    mixed_qkv = data[0]
    assert ws.ensure_allocated(
        num_layers=2,
        min_bs=16,
        k_ver=4,
        nk=16,
        nv=32,
        head_dim=128,
        device=mixed_qkv.device,
    )
    out_ws, rk_ws, rv_ws, rg_ws = fused_gdn_verify(
        *data, workspace=ws, linear_layer_idx=1, impl="auto"
    )
    assert rk_ws.data_ptr() == ws.rec_k_all[1].data_ptr()
    assert rv_ws.data_ptr() == ws.rec_v_all[1].data_ptr()
    assert rg_ws.data_ptr() == ws.rec_g_all[1].data_ptr()

    out_ref, rk_ref, rv_ref, rg_ref = fused_gdn_verify(*data, impl="torch_native")
    torch.testing.assert_close(out_ws, out_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(rk_ws, rk_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(rv_ws, rv_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(rg_ws, rg_ref, atol=1e-3, rtol=1e-3)
