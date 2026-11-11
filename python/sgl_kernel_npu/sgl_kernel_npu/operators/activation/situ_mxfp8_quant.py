"""Public A5 SiTU + MXFP8 quantization API."""

import torch
from sgl_kernel_npu.operators.utils.generation import Generation
from sgl_kernel_npu.operators.utils.registry import (
    get_impl_by_name,
    register_impl,
    register_operator,
)

_IMPL_NAME = "activation.situ_mxfp8_quant"
_ONLY_A5 = {Generation.ARCH35}
_torch_native_fn = None


@register_impl(_IMPL_NAME, "auto", supported_generations=_ONLY_A5)
@register_impl(_IMPL_NAME, "ascendc", supported_generations=_ONLY_A5)
def _ascendc_impl(hidden_states, group_list, group_list_type, beta, linear_beta):
    return torch.ops.npu.situ_mxfp8_quant(
        hidden_states,
        group_list,
        group_list_type,
        float(beta),
        float(linear_beta),
    )


@register_impl(_IMPL_NAME, "torch_native", supported_generations=_ONLY_A5)
def _torch_native_impl(hidden_states, group_list, group_list_type, beta, linear_beta):
    global _torch_native_fn
    if _torch_native_fn is None:
        from sgl_kernel_npu.implementation.torch_native.activation.situ_mxfp8_quant import (
            situ_mxfp8_quant as reference,
        )

        _torch_native_fn = reference
    return _torch_native_fn(
        hidden_states, group_list, group_list_type, beta, linear_beta
    )


@register_operator(_IMPL_NAME)
def situ_mxfp8_quant(
    hidden_states,
    group_list,
    group_list_type=1,
    beta=4.0,
    linear_beta=25.0,
    *,
    impl="auto",
):
    """Fuse Kimi-K3 SiTU with per-32-element MXFP8 quantization.

    ``hidden_states`` is the BF16 GMM1 output ``[capacity, 6144]``: the first
    3072 columns are the gate branch and the last 3072 are the up branch. Only
    rows selected by ``group_list`` are written; padded rows are left
    uninitialized, matching the downstream grouped-matmul contract.

    Args:
        hidden_states: Contiguous BF16 tensor of shape ``[capacity, 6144]`` on
            an Ascend NPU.
        group_list: Non-empty 1D int32/int64 tensor on the same NPU. With
            ``group_list_type=0`` it is a cumulative token prefix per expert
            (the last element is the valid row count); with type 1 it holds
            per-expert token counts (their sum is the valid row count). Only
            this resulting row count is consumed; per-expert boundaries are
            not used by the kernel.
        group_list_type: 0 for the cumulative layout, 1 for the count layout.
        beta: Positive gate soft-saturation bound (Kimi-K3 default 4.0).
        linear_beta: Positive up soft-saturation bound (Kimi-K3 default 25.0).
        impl: ``auto``/``ascendc`` select the Ascend C operator; ``torch_native``
            selects the pure PyTorch reference. This operator only supports
            arch35 (Ascend950).

    Returns:
        ``(payload, scales)``. ``payload`` is ``[capacity, 3072]`` in
        ``torch.float8_e4m3fn``; ``scales`` is ``[capacity, 48, 2]`` in
        ``torch.float8_e8m0fnu``, i.e. the 96 E8M0 bytes of each row, one per
        32-element group of the 3072-column payload.
    """
    if hidden_states.ndim != 2 or hidden_states.shape[1] != 6144:
        raise ValueError(
            "situ_mxfp8_quant expects hidden_states shape [capacity, 6144], "
            f"got {tuple(hidden_states.shape)}"
        )
    if hidden_states.dtype != torch.bfloat16:
        raise ValueError(f"hidden_states must be bfloat16, got {hidden_states.dtype}")
    if not hidden_states.is_contiguous():
        raise RuntimeError("hidden_states must be contiguous")
    if hidden_states.shape[0] <= 0:
        raise ValueError("hidden_states must contain at least one row")
    if group_list.ndim != 1 or group_list.numel() == 0:
        raise ValueError("group_list must be a non-empty 1D tensor")
    if group_list.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"group_list must use int32 or int64, got {group_list.dtype}")
    if not group_list.is_contiguous():
        raise RuntimeError("group_list must be contiguous")
    if group_list.device != hidden_states.device:
        raise ValueError("hidden_states and group_list must be on the same NPU")
    if group_list_type not in (0, 1):
        raise ValueError(f"group_list_type must be 0 or 1, got {group_list_type}")
    if not (beta > 0.0) or not (linear_beta > 0.0):
        raise ValueError(
            "beta and linear_beta must be positive, got "
            f"beta={beta}, linear_beta={linear_beta}"
        )
    return get_impl_by_name(_IMPL_NAME, impl)(
        hidden_states, group_list, group_list_type, beta, linear_beta
    )
