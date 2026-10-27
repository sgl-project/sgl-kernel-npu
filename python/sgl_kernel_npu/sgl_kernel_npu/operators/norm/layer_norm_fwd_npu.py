"""Public gated normalization with shared checks and implementation-specific preparation."""

import torch

from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl("norm.layer_norm_fwd_npu", "auto")
@register_impl("norm.layer_norm_fwd_npu", "triton")
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
    import triton

    from ...implementation.triton.norm.layernorm_gated import (
        _layer_norm_fwd_1pass_kernel_npu,
    )
    from ...utils.triton_utils import get_device_properties

    M, N = x.shape
    if group_size is None:
        group_size = N
    ngroups = N // group_size

    # allocate output
    if out is None:
        out = torch.empty_like(x)
    mean = (
        torch.empty((ngroups * M,), dtype=torch.float32, device=x.device)
        if not is_rms_norm
        else None
    )
    rstd = torch.empty((ngroups * M,), dtype=torch.float32, device=x.device)

    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(group_size))

    # Choose BLOCK_M: e.g., 16, 32, 64 — depends on NPU vector core capacity
    BLOCK_M = 48
    _, num_vectorcore = get_device_properties()
    num_blocks_m = min(triton.cdiv(M, BLOCK_M), num_vectorcore)

    # Now grid is (num blocks over M, num groups)
    grid = (num_blocks_m, ngroups)
    _layer_norm_fwd_1pass_kernel_npu[grid](
        x,
        out,
        weight,
        bias,
        z,
        mean,
        rstd,
        x.stride(0),
        out.stride(0),
        z.stride(0) if z is not None else 0,
        M,
        group_size,
        eps,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        NORM_BEFORE_GATE=norm_before_gate,
        IS_RMS_NORM=is_rms_norm,
        # Remove multibuffer if not needed
    )
    return out, mean, rstd


@register_impl("norm.layer_norm_fwd_npu", "torch_native")
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
    from ...implementation.torch_native.norm.layernorm_gated import layer_norm_fwd_npu

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
    layer_norm_fwd_npu(
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


@register_operator("norm.layer_norm_fwd_npu")
def layer_norm_fwd_npu(
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

    return get_impl_by_name("norm.layer_norm_fwd_npu", impl)(
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
