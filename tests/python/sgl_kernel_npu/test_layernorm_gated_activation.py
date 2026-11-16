"""Gate-activation coverage for the NPU gated (layer|rms)norm kernels."""

import pytest
import torch
from sgl_kernel_npu.fla.layernorm_gated import (
    SUPPORTED_GATE_ACTIVATIONS,
    layer_norm_fwd_npu,
    layer_norm_fwd_npu_simd,
)


def _has_npu() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


pytestmark = pytest.mark.skipif(not _has_npu(), reason="NPU is required")

device = "npu:0"

_TOLS = {
    torch.float32: dict(rtol=1e-5, atol=1e-5),
    torch.float16: dict(rtol=1e-3, atol=1e-3),
    torch.bfloat16: dict(rtol=1.6e-2, atol=1.6e-2),
}

_IMPLS = (layer_norm_fwd_npu, layer_norm_fwd_npu_simd)

# layer_norm_fwd_npu tiles 48 rows at a time, so the per-group width has to stay
# small enough for the tile to fit in UB; 128 is the width used in practice.
GROUP_WIDTH = 128


def _gate(z: torch.Tensor, activation: str) -> torch.Tensor:
    if activation == "sigmoid":
        return torch.sigmoid(z)
    return z * torch.sigmoid(z)


def _reference(
    x, weight, bias, eps, z, group_size, norm_before_gate, is_rms_norm, activation
):
    M, N = x.shape
    group_size = group_size or N
    xf = x.float()
    zf = None if z is None else z.float()

    if zf is not None and not norm_before_gate:
        xf = xf * _gate(zf, activation)

    xg = xf.reshape(M, N // group_size, group_size)
    if is_rms_norm:
        rstd = torch.rsqrt(xg.pow(2).mean(-1, keepdim=True) + eps)
        xhat = xg * rstd
    else:
        mean = xg.mean(-1, keepdim=True)
        rstd = torch.rsqrt(xg.var(-1, keepdim=True, unbiased=False) + eps)
        xhat = (xg - mean) * rstd

    out = xhat.reshape(M, N) * weight.float()
    if bias is not None:
        out = out + bias.float()
    if zf is not None and norm_before_gate:
        out = out * _gate(zf, activation)
    return out.to(x.dtype)


def _run(impl, *, activation, **kwargs):
    # layer_norm_fwd_npu_simd gained its `activation` argument alongside
    # layer_norm_fwd_npu; both accept it.
    return impl(activation=activation, **kwargs)[0]


@pytest.mark.parametrize("impl", _IMPLS, ids=lambda f: f.__name__)
@pytest.mark.parametrize("activation", SUPPORTED_GATE_ACTIVATIONS)
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16, torch.float32))
@pytest.mark.parametrize("norm_before_gate", (True, False))
def test_gated_rmsnorm_activations(impl, activation, dtype, norm_before_gate):
    torch.manual_seed(0)
    M, N = 64, GROUP_WIDTH
    x = torch.randn(M, N, dtype=dtype, device=device)
    z = torch.randn(M, N, dtype=dtype, device=device)
    weight = torch.randn(N, dtype=dtype, device=device)
    eps = 1e-6

    out = _run(
        impl,
        x=x,
        weight=weight,
        bias=None,
        eps=eps,
        z=z,
        group_size=None,
        norm_before_gate=norm_before_gate,
        is_rms_norm=True,
        activation=activation,
    )
    ref = _reference(x, weight, None, eps, z, None, norm_before_gate, True, activation)
    torch.testing.assert_close(out.float(), ref.float(), **_TOLS[dtype])


@pytest.mark.parametrize("impl", _IMPLS, ids=lambda f: f.__name__)
@pytest.mark.parametrize("activation", SUPPORTED_GATE_ACTIVATIONS)
def test_gated_layernorm_with_bias_and_groups(impl, activation):
    torch.manual_seed(0)
    dtype = torch.float32
    M, N, group_size = 32, 2 * GROUP_WIDTH, GROUP_WIDTH
    x = torch.randn(M, N, dtype=dtype, device=device)
    z = torch.randn(M, N, dtype=dtype, device=device)
    weight = torch.randn(N, dtype=dtype, device=device)
    bias = torch.randn(N, dtype=dtype, device=device)
    eps = 1e-5

    out = _run(
        impl,
        x=x,
        weight=weight,
        bias=bias,
        eps=eps,
        z=z,
        group_size=group_size,
        norm_before_gate=True,
        is_rms_norm=False,
        activation=activation,
    )
    ref = _reference(x, weight, bias, eps, z, group_size, True, False, activation)
    torch.testing.assert_close(out.float(), ref.float(), **_TOLS[dtype])


@pytest.mark.parametrize("impl", _IMPLS, ids=lambda f: f.__name__)
def test_swish_and_silu_agree(impl):
    # The two names select the same gate; they must be bit-identical.
    torch.manual_seed(0)
    M, N = 48, GROUP_WIDTH
    args = dict(
        x=torch.randn(M, N, dtype=torch.bfloat16, device=device),
        weight=torch.randn(N, dtype=torch.bfloat16, device=device),
        bias=None,
        eps=1e-6,
        z=torch.randn(M, N, dtype=torch.bfloat16, device=device),
        group_size=None,
        norm_before_gate=True,
        is_rms_norm=True,
    )
    swish = _run(impl, activation="swish", **args)
    silu = _run(impl, activation="silu", **args)
    torch.testing.assert_close(swish.float(), silu.float(), rtol=0, atol=0)


@pytest.mark.parametrize("impl", _IMPLS, ids=lambda f: f.__name__)
def test_default_activation_is_swish(impl):
    # Callers that predate the `activation` argument must keep the old gate.
    torch.manual_seed(0)
    M, N = 48, GROUP_WIDTH
    args = dict(
        x=torch.randn(M, N, dtype=torch.bfloat16, device=device),
        weight=torch.randn(N, dtype=torch.bfloat16, device=device),
        bias=None,
        eps=1e-6,
        z=torch.randn(M, N, dtype=torch.bfloat16, device=device),
        group_size=None,
        norm_before_gate=True,
        is_rms_norm=True,
    )
    default = impl(**args)[0]
    swish = _run(impl, activation="swish", **args)
    torch.testing.assert_close(default.float(), swish.float(), rtol=0, atol=0)


@pytest.mark.parametrize("impl", _IMPLS, ids=lambda f: f.__name__)
def test_unsupported_activation_rejected(impl):
    with pytest.raises(ValueError, match="Unsupported gated RMSNorm activation"):
        _run(
            impl,
            x=torch.randn(8, GROUP_WIDTH, dtype=torch.bfloat16, device=device),
            weight=torch.randn(GROUP_WIDTH, dtype=torch.bfloat16, device=device),
            bias=None,
            eps=1e-6,
            z=torch.randn(8, GROUP_WIDTH, dtype=torch.bfloat16, device=device),
            group_size=None,
            norm_before_gate=True,
            is_rms_norm=True,
            activation="gelu",
        )
