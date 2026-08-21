"""Correctness tests for the decode-optimized GDN recurrent state update.

Compares ``fused_sigmoid_gating_delta_rule_update_decode_npu`` against a naive
fp32 torch reference and the generic fused kernel
(``fused_sigmoid_gating_delta_rule_update_npu``) on the decode (T=1) path,
using the Qwen3.5/Qwen3.6 GDN head layout.

In-repo Python sources are preferred over the installed package so the test
exercises the current tree; compiled ops still come from the installed
``sgl_kernel_npu`` package.
"""

import os
import sys

import pytest
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)

# Installed package: loads the compiled ops and provides shared utils.
import sgl_kernel_npu  # noqa: E402

_repo_pkg = os.path.join(REPO_ROOT, "python", "sgl_kernel_npu", "sgl_kernel_npu")
if os.path.isdir(_repo_pkg) and _repo_pkg not in sgl_kernel_npu.__path__:
    sgl_kernel_npu.__path__.insert(0, _repo_pkg)

from sgl_kernel_npu.fla.fused_sigmoid_gating_recurrent import (  # noqa: E402
    fused_sigmoid_gating_delta_rule_update_npu as generic_kernel,
)
from sgl_kernel_npu.fla.fused_sigmoid_gating_recurrent_decode_optimized import (  # noqa: E402
    fused_sigmoid_gating_delta_rule_update_decode_npu as decode_kernel,
)

# Fail loudly if the submodule did not come from the in-repo sources.
if os.path.isdir(_repo_pkg):
    _mod = sys.modules[
        "sgl_kernel_npu.fla.fused_sigmoid_gating_recurrent_decode_optimized"
    ]
    assert _mod.__file__.startswith(_repo_pkg), (
        f"expected in-repo module, got {_mod.__file__}"
    )

DEVICE = "npu:0"

# GDN head layout: H query/key heads, HV value heads, K/V head dims.
H = 16
HV = 32
K = 128
V = 128


def _make_inputs(bs: int, dtype: torch.dtype = torch.bfloat16):
    q = torch.randn(1, bs, H, K, dtype=dtype, device=DEVICE)
    k = torch.randn(1, bs, H, K, dtype=dtype, device=DEVICE)
    v = torch.randn(1, bs, HV, V, dtype=dtype, device=DEVICE)
    a = torch.randn(bs, HV, dtype=dtype, device=DEVICE)
    b = torch.randn(bs, HV, dtype=dtype, device=DEVICE)
    A_log = torch.randn(HV, dtype=torch.float32, device=DEVICE)
    dt_bias = torch.randn(HV, dtype=torch.float32, device=DEVICE)
    num_slots = max(bs, 64)
    ssm_states = torch.randn(num_slots, HV, K, V, dtype=torch.float32, device=DEVICE)
    cache_indices = torch.arange(bs, dtype=torch.int32, device=DEVICE)
    query_start_loc = torch.arange(bs + 1, dtype=torch.int32, device=DEVICE)
    scale = K**-0.5
    return q, k, v, a, b, A_log, dt_bias, ssm_states, cache_indices, query_start_loc, scale


def _reference_update(q, k, v, a, b, A_log, dt_bias, ssm_states, cache_indices, scale):
    """Naive fp32 reference for one decode step with L2-normalized q/k."""
    bs = q.shape[1]
    out = torch.zeros_like(v)
    for t in range(bs):
        for hv in range(HV):
            h = hv // (HV // H)
            q_t = q[0, t, h, :].float()
            k_t = k[0, t, h, :].float()
            v_t = v[0, t, hv, :].float()
            g_val = -(A_log[hv].exp()) * F.softplus(a[t, hv] + dt_bias[hv])
            beta_val = torch.sigmoid(b[t, hv])
            q_t = q_t / (q_t.pow(2).sum().sqrt() + 1e-6)
            k_t = k_t / (k_t.pow(2).sum().sqrt() + 1e-6)
            q_t = q_t * scale
            slot = cache_indices[t].item()
            h_state = ssm_states[slot, hv].clone()
            h_state *= g_val.exp()
            v_t = v_t - (h_state * k_t[:, None]).sum(dim=0)
            v_t = v_t * beta_val
            h_state += k_t[:, None] * v_t[None, :]
            ssm_states[slot, hv] = h_state
            out[0, t, hv, :] = (h_state * q_t[:, None]).sum(dim=0)
    return out


def _run(kernel, inputs, is_generic: bool):
    q, k, v, a, b, A_log, dt_bias, ssm, idx, loc, scale = inputs
    kwargs = dict(
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        q=q,
        k=k,
        v=v,
        b=b,
        initial_state_source=ssm,
        initial_state_indices=idx,
        scale=scale,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=loc,
    )
    if is_generic:
        kwargs["is_kda"] = False
    return kernel(**kwargs)


@pytest.mark.parametrize("bs", [1, 8, 32, 128])
def test_gdn_decode_update_correctness(bs: int):
    torch.manual_seed(42 + bs)
    q, k, v, a, b, A_log, dt_bias, ssm, idx, loc, scale = _make_inputs(bs)

    ssm_ref = ssm.clone()
    out_ref = _reference_update(
        q, k, v, a, b, A_log, dt_bias, ssm_ref, idx, scale
    )

    inputs = (q, k, v, a, b, A_log, dt_bias, ssm.clone(), idx, loc, scale)
    ssm_generic = inputs[7]
    out_generic = _run(generic_kernel, inputs, is_generic=True)

    inputs = (q, k, v, a, b, A_log, dt_bias, ssm.clone(), idx, loc, scale)
    ssm_decode = inputs[7]
    out_decode = _run(decode_kernel, inputs, is_generic=False)

    ref_f = out_ref.float()
    generic_out_err = (out_generic.float() - ref_f).abs().max().item()
    decode_out_err = (out_decode.float() - ref_f).abs().max().item()
    decode_state_err = (ssm_decode.float() - ssm_ref.float()).abs().max().item()

    # The generic kernel sets the achievable accuracy against the naive fp32
    # reference (different reduction order); the decode kernel must land
    # within 2x of it, and the updated state must match closely.
    assert decode_out_err < max(1e-3, generic_out_err * 2), (
        f"bs={bs}: output err {decode_out_err:.3e} vs generic {generic_out_err:.3e}"
    )
    assert decode_state_err < 1e-2, f"bs={bs}: state err {decode_state_err:.3e}"


if __name__ == "__main__":
    for bs in [1, 8, 32, 128]:
        test_gdn_decode_update_correctness(bs)
        print(f"bs={bs}: PASS")
    print("All GDN decode update tests passed.")
