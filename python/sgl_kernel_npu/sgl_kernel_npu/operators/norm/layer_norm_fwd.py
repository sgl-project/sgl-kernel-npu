"""Public gated normalization with shared checks and implementation-specific preparation."""

import torch

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("norm.layer_norm_fwd", "auto")
@register_impl("norm.layer_norm_fwd", "triton")
def _triton_impl(
    x,
    weight,
    bias,
    eps,
    z=None,
    out=None,
    group_size=None,
    norm_before_gate=True,
    is_rms_norm=False,
    activation=None,
):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.norm.layer_norm_fwd import layer_norm_fwd

        _triton_fn = layer_norm_fwd

    return _triton_fn(
        x,
        weight,
        bias,
        eps,
        z,
        out,
        group_size,
        norm_before_gate,
        is_rms_norm,
        activation,
    )


@register_impl("norm.layer_norm_fwd", "torch_native")
def _torch_native_impl(
    x,
    weight,
    bias,
    eps,
    z=None,
    out=None,
    group_size=None,
    norm_before_gate=True,
    is_rms_norm=False,
    activation=None,
):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.norm.layer_norm_fwd import layer_norm_fwd

        _torch_native_fn = layer_norm_fwd

    rows, hidden = x.shape
    size = hidden if group_size is None else group_size
    groups = hidden // size
    output = torch.empty_like(x) if out is None else out
    mean = (
        None
        if is_rms_norm
        else torch.empty((groups * rows,), device=x.device, dtype=torch.float32)
    )
    rstd = torch.empty((groups * rows,), device=x.device, dtype=torch.float32)
    # activation is accepted but ignored by the original kernel; gating is always SiLU.
    _torch_native_fn(
        x.reshape(rows, groups, size),
        weight.reshape(1, groups, size),
        None if bias is None else bias.reshape(1, groups, size),
        None if z is None else z.reshape(rows, groups, size),
        output.view(rows, groups, size),
        None if mean is None else mean.view(groups, rows).transpose(0, 1),
        rstd.view(groups, rows).transpose(0, 1),
        eps,
        norm_before_gate,
        is_rms_norm,
    )
    return output, mean, rstd


@register_operator("norm.layer_norm_fwd")
def layer_norm_fwd(
    x,
    weight,
    bias,
    eps,
    z=None,
    out=None,
    group_size=None,
    norm_before_gate=True,
    is_rms_norm=False,
    activation=None,
    *,
    impl="auto"
):
    """Grouped LayerNorm or RMSNorm with optional affine bias and SiLU gating.

    x is [rows, hidden]; group_size defaults to hidden and must divide it.
    When norm_before_gate is True, gate the affine-normalized output; otherwise
    gate x before computing normalization statistics. The legacy activation
    argument is accepted but ignored: the kernel always uses SiLU for z.

    Return (out, mean, rstd), with FP32 statistics flattened in group-major
    order. mean is None for RMSNorm. A supplied out is written in place and
    returned by identity. impl selects auto (Triton), triton, or torch_native."""
    rows, hidden = x.shape
    size = hidden if group_size is None else group_size
    assert hidden % size == 0
    assert x.stride(-1) == 1
    if z is not None:
        assert z.stride(-1) == 1
        assert z.shape == (rows, hidden)
    assert weight.shape == (hidden,)
    assert weight.stride(-1) == 1
    if bias is not None:
        assert bias.stride(-1) == 1
        assert bias.shape == (hidden,)
    if out is not None:
        assert out.shape == x.shape
        assert out.stride(-1) == 1
    maximum = 65536 // x.element_size()
    block = min(maximum, 1 << (size - 1).bit_length())
    if size > block:
        raise RuntimeError("Feature dim too large.")

    return get_impl_by_name("norm.layer_norm_fwd", impl)(
        x,
        weight,
        bias,
        eps,
        z,
        out,
        group_size,
        norm_before_gate,
        is_rms_norm,
        activation,
    )
