"""Tests for attention.gdn.fused_conv1d_gdn_verify public API."""

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


def generate_data(bs, k_ver, nk, nv, pad_mode):
    torch.npu.manual_seed(77)
    d = 128
    qkv_dim = (2 * nk + nv) * d
    qkvz_dim = qkv_dim + nv * d
    win_len = k_ver + 2
    num_slots = max(16, bs + 4)

    qkvz = torch.randn((bs * k_ver, qkvz_dim), dtype=torch.bfloat16) * 0.4
    ba = torch.randn((bs * k_ver, 2 * nv), dtype=torch.bfloat16) * 0.4
    conv_w_t = torch.randn((4, qkv_dim), dtype=torch.bfloat16) * 0.1
    conv_states = torch.randn((num_slots, win_len, qkv_dim), dtype=torch.bfloat16) * 0.2
    A_log = torch.randn((nv,), dtype=torch.float32) * 0.2 - 1.0
    dt_bias = torch.randn((nv,), dtype=torch.bfloat16) * 0.1
    ssm_states = torch.randn((num_slots, nv, d, d), dtype=torch.bfloat16) * 0.1

    cache_indices = torch.arange(1, bs + 1, dtype=torch.int32)
    query_start_loc = None
    if pad_mode == "neg_slot" and bs >= 2:
        cache_indices[-1] = -1
    elif pad_mode == "aclgraph_qsl_pad" and bs >= 2:
        # Simulate padded ACLGraph row pointing to slot 0 with empty qsl span.
        cache_indices[-1] = 0
        qsl = [i * k_ver for i in range(bs)]
        qsl.append(qsl[-1])  # last row has 0 valid tokens
        query_start_loc = torch.tensor(qsl, dtype=torch.int32)

    return {
        "qkvz": qkvz,
        "ba": ba,
        "conv_weights_t": conv_w_t,
        "conv_states": conv_states,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "ssm_states": ssm_states,
        "cache_indices": cache_indices,
        "batch_size": bs,
        "k_ver": k_ver,
        "num_k_heads": nk,
        "num_v_heads": nv,
        "query_start_loc": query_start_loc,
    }


@pytest.mark.parametrize(
    "bs,k_ver,nk,nv",
    [
        pytest.param(1, 4, 16, 48, id="qwen38-27b-mtp-bs1"),
        pytest.param(32, 4, 16, 48, id="qwen38-27b-mtp-bs32"),
        pytest.param(1, 8, 16, 48, id="qwen38-27b-dspark-bs1"),
        pytest.param(32, 8, 16, 48, id="qwen38-27b-dspark-bs32"),
        pytest.param(32, 4, 16, 32, id="qwen35-9b-gqa2-bs32"),
        pytest.param(6, 3, 8, 32, id="edge-gqa4-nonpow2-k3"),
    ],
)
@pytest.mark.parametrize("pad_mode", ["normal", "neg_slot", "aclgraph_qsl_pad"])
def test_fused_conv1d_gdn_verify(bs, k_ver, nk, nv, pad_mode):
    from sgl_kernel_npu.operators.attention.gdn import fused_conv1d_gdn_verify

    kwargs = generate_data(bs, k_ver, nk, nv, pad_mode)
    slot0_before = kwargs["conv_states"][0].clone()
    standard = {
        "torch.bfloat16": AccuracyStandard(atol=3e-2, rtol=3e-2),
        "torch.float32": AccuracyStandard(atol=1e-3, rtol=1e-3),
    }
    executor = check_accuracy(
        fused_conv1d_gdn_verify,
        kwargs=kwargs,
        mutable_kwargs=("conv_states",),
        standard=standard,
    )
    if pad_mode == "aclgraph_qsl_pad" and bs >= 2:
        torch.testing.assert_close(
            kwargs["conv_states"][0], slot0_before, atol=0, rtol=0
        )
    check_performance(
        executor,
        enable_perf=((bs, k_ver, nk, nv) in perf_shape and pad_mode == "normal"),
    )
