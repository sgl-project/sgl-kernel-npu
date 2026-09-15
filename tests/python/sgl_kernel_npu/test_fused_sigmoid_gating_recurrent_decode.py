import pytest
import torch
import torch.nn.functional as F
import torch_npu
from sgl_kernel_npu.fla.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update_npu as generic_kernel,
)
from sgl_kernel_npu.fla.fused_sigmoid_gating_recurrent_decode_optimized import (
    fused_sigmoid_gating_delta_rule_update_decode_npu as decode_kernel,
)


def _has_npu() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


pytestmark = pytest.mark.skipif(not _has_npu(), reason="NPU is required")

device = "npu:0"

# Default Qwen3.5/Qwen3.6 GDN head layout: H query/key heads, HV value heads.
H = 16
HV = 32
K = 128
V = 128


def _make_inputs(
    bs: int,
    dtype: torch.dtype = torch.bfloat16,
    h: int = H,
    hv: int = HV,
    head_k: int = K,
    head_v: int = V,
    non_contiguous: bool = False,
):
    """Generate decode inputs.  Returns q/k/v in (1, bs, *, *) varlen layout."""
    if non_contiguous:
        # Produce logically (1, bs, h/hv, head_k/head_v) tensors that are
        # non-contiguous in memory.  The wrapper/kernel must normalize them.
        q = torch.randn(1, bs, h, head_k * 2, dtype=dtype, device=device)[..., ::2]
        k = torch.randn(1, bs, h, head_k * 2, dtype=dtype, device=device)[..., ::2]
        v = torch.randn(1, bs, hv, head_v * 2, dtype=dtype, device=device)[..., ::2]
        a = torch.randn(bs, hv * 2, dtype=dtype, device=device)[:, ::2]
        b = torch.randn(bs, hv * 2, dtype=dtype, device=device)[:, ::2]
    else:
        q = torch.randn(1, bs, h, head_k, dtype=dtype, device=device)
        k = torch.randn(1, bs, h, head_k, dtype=dtype, device=device)
        v = torch.randn(1, bs, hv, head_v, dtype=dtype, device=device)
        a = torch.randn(bs, hv, dtype=dtype, device=device)
        b = torch.randn(bs, hv, dtype=dtype, device=device)

    A_log = torch.randn(hv, dtype=torch.float32, device=device)
    dt_bias = torch.randn(hv, dtype=torch.float32, device=device)
    num_slots = max(bs, 64)
    ssm_states = torch.randn(
        num_slots, hv, head_k, head_v, dtype=torch.float32, device=device
    )
    cache_indices = torch.arange(bs, dtype=torch.int32, device=device)
    query_start_loc = torch.arange(bs + 1, dtype=torch.int32, device=device)
    scale = head_k**-0.5
    return (
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        ssm_states,
        cache_indices,
        query_start_loc,
        scale,
    )


def _make_inputs_non_varlen(
    bs: int,
    dtype: torch.dtype = torch.bfloat16,
    h: int = H,
    hv: int = HV,
    head_k: int = K,
    head_v: int = V,
):
    """Generate non-varlen decode inputs with q/k/v shape (bs, 1, h/hv, k/v)."""
    q = torch.randn(bs, 1, h, head_k, dtype=dtype, device=device)
    k = torch.randn(bs, 1, h, head_k, dtype=dtype, device=device)
    v = torch.randn(bs, 1, hv, head_v, dtype=dtype, device=device)
    a = torch.randn(bs, hv, dtype=dtype, device=device)
    b = torch.randn(bs, hv, dtype=dtype, device=device)
    A_log = torch.randn(hv, dtype=torch.float32, device=device)
    dt_bias = torch.randn(hv, dtype=torch.float32, device=device)
    num_slots = max(bs, 64)
    ssm_states = torch.randn(
        num_slots, hv, head_k, head_v, dtype=torch.float32, device=device
    )
    cache_indices = torch.arange(bs, dtype=torch.int32, device=device)
    scale = head_k**-0.5
    return q, k, v, a, b, A_log, dt_bias, ssm_states, cache_indices, None, scale


def _reference_update(
    q, k, v, a, b, A_log, dt_bias, ssm_states, cache_indices, scale, use_l2norm: bool
):
    """Naive fp32 reference for one decode step."""
    # Handle both varlen layout (1, bs, ...) and non-varlen layout (bs, 1, ...).
    if q.shape[0] == 1:
        bs = q.shape[1]
    else:
        bs = q.shape[0]
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

    hv = v.shape[2]
    h = q.shape[2]
    out = torch.zeros_like(v)
    for t in range(bs):
        for i_hv in range(hv):
            i_h = i_hv // (hv // h)
            q_t = q[0, t, i_h, :].float()
            k_t = k[0, t, i_h, :].float()
            v_t = v[0, t, i_hv, :].float()
            g_val = -(A_log[i_hv].exp()) * F.softplus(a[t, i_hv] + dt_bias[i_hv])
            beta_val = torch.sigmoid(b[t, i_hv])
            if use_l2norm:
                q_t = q_t / (q_t.pow(2).sum().sqrt() + 1e-6)
                k_t = k_t / (k_t.pow(2).sum().sqrt() + 1e-6)
            q_t = q_t * scale
            slot = cache_indices[t].item()
            h_state = ssm_states[slot, i_hv].clone()
            h_state *= g_val.exp()
            v_t = v_t - (h_state * k_t[:, None]).sum(dim=0)
            v_t = v_t * beta_val
            h_state += k_t[:, None] * v_t[None, :]
            ssm_states[slot, i_hv] = h_state
            out[0, t, i_hv, :] = (h_state * q_t[:, None]).sum(dim=0)
    return out


def _run(kernel, inputs, is_generic: bool, use_l2norm: bool = True):
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
        use_qk_l2norm_in_kernel=use_l2norm,
        cu_seqlens=loc,
    )
    if is_generic:
        kwargs["is_kda"] = False
    return kernel(**kwargs)


def _check(
    bs,
    inputs,
    use_l2norm: bool = True,
    rtol_out_vs_ref: float = 2.0,
    atol_out_vs_ref: float = 1e-3,
    atol_out_vs_generic: float = 1e-3,
    atol_state_vs_generic: float = 1e-2,
):
    q, k, v, a, b, A_log, dt_bias, ssm, idx, loc, scale = inputs

    ssm_ref = ssm.clone()
    out_ref = _reference_update(
        q, k, v, a, b, A_log, dt_bias, ssm_ref, idx, scale, use_l2norm
    )

    inputs_generic = (q, k, v, a, b, A_log, dt_bias, ssm.clone(), idx, loc, scale)
    out_generic = _run(
        generic_kernel, inputs_generic, is_generic=True, use_l2norm=use_l2norm
    )
    ssm_generic = inputs_generic[7]

    inputs_decode = (q, k, v, a, b, A_log, dt_bias, ssm.clone(), idx, loc, scale)
    out_decode = _run(
        decode_kernel, inputs_decode, is_generic=False, use_l2norm=use_l2norm
    )
    ssm_decode = inputs_decode[7]

    # Normalize all outputs to (bs, hv, v) before comparison.  The generic
    # kernel keeps an extra size-1 dimension for varlen/non-varlen layouts,
    # while the decode kernel always returns (N, hv, v).
    hv = v.shape[2]
    head_v = v.shape[-1]
    out_generic = out_generic.reshape(bs, hv, head_v)
    out_decode = out_decode.reshape(bs, hv, head_v)
    out_ref = out_ref.reshape(bs, hv, head_v)

    ref_f = out_ref.float()
    generic_out_err = (out_generic.float() - ref_f).abs().max().item()
    decode_out_err = (out_decode.float() - ref_f).abs().max().item()
    decode_vs_generic_out_err = (
        (out_decode.float() - out_generic.float()).abs().max().item()
    )
    decode_vs_generic_state_err = (
        (ssm_decode.float() - ssm_generic.float()).abs().max().item()
    )

    # Output should match the reference within the generic kernel's accuracy.
    assert decode_out_err < max(
        atol_out_vs_ref, generic_out_err * rtol_out_vs_ref
    ), f"bs={bs}: output err {decode_out_err:.3e} vs generic {generic_out_err:.3e}"
    # The decode kernel must agree with the generic kernel (same bf16 compute).
    assert (
        decode_vs_generic_out_err < atol_out_vs_generic
    ), f"bs={bs}: decode vs generic output err {decode_vs_generic_out_err:.3e}"
    assert (
        decode_vs_generic_state_err < atol_state_vs_generic
    ), f"bs={bs}: decode vs generic state err {decode_vs_generic_state_err:.3e}"


@pytest.mark.parametrize("bs", [1, 8, 32, 128])
def test_gdn_decode_update_correctness(bs: int):
    torch.manual_seed(42 + bs)
    inputs = _make_inputs(bs)
    _check(bs, inputs)


@pytest.mark.parametrize("bs", [1, 8, 32, 128])
def test_gdn_decode_update_non_varlen(bs: int):
    """Non-varlen layout (bs, 1, h/hv, k/v) with cu_seqlens=None."""
    torch.manual_seed(100 + bs)
    inputs = _make_inputs_non_varlen(bs)
    _check(bs, inputs)


@pytest.mark.parametrize("bs", [1, 8, 32, 128])
def test_gdn_decode_update_no_l2norm(bs: int):
    """Decode path without in-kernel q/k L2 normalization."""
    torch.manual_seed(200 + bs)
    inputs = _make_inputs(bs)
    _check(bs, inputs, use_l2norm=False)


@pytest.mark.parametrize("bs", [1, 8, 32, 128])
def test_gdn_decode_update_non_contiguous(bs: int):
    """Decode path with non-contiguous input tensors."""
    torch.manual_seed(300 + bs)
    inputs = _make_inputs(bs, non_contiguous=True)
    assert not inputs[0].is_contiguous()
    assert not inputs[3].is_contiguous()
    _check(bs, inputs)


@pytest.mark.parametrize(
    "h,hv,k,v",
    [
        (8, 16, 64, 128),  # smaller k heads, k != v
        (8, 16, 128, 64),  # k != v, smaller v dim
        (4, 8, 64, 64),  # smaller everything
    ],
)
def test_gdn_decode_update_custom_head_dims(h: int, hv: int, k: int, v: int):
    """Decode path with heterogeneous head dims and head ratios."""
    torch.manual_seed(400 + h + hv + k + v)
    bs = 32
    inputs = _make_inputs(bs, h=h, hv=hv, head_k=k, head_v=v)
    _check(bs, inputs, atol_state_vs_generic=2e-2)


if __name__ == "__main__":
    for bs in [1, 8, 32, 128]:
        test_gdn_decode_update_correctness(bs)
        test_gdn_decode_update_non_varlen(bs)
        test_gdn_decode_update_no_l2norm(bs)
        test_gdn_decode_update_non_contiguous(bs)
        print(f"bs={bs}: PASS")
    test_gdn_decode_update_custom_head_dims(8, 16, 64, 128)
    test_gdn_decode_update_custom_head_dims(8, 16, 128, 64)
    test_gdn_decode_update_custom_head_dims(4, 8, 64, 64)
    print("All GDN decode update tests passed.")
