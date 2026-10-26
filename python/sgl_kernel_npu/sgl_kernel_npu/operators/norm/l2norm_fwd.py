"""Public l2norm_fwd operator with parameter preparation in registered adapters."""

from typing import Optional

import torch

from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl("norm.l2norm_fwd", "auto")
@register_impl("norm.l2norm_fwd", "triton")
def _triton_impl(
    x: torch.Tensor, eps: float = 1e-6, output_dtype: Optional[torch.dtype] = None
):
    import triton

    from ...implementation.triton.norm.l2norm import (
        l2norm_fwd_kernel1,
        l2norm_fwd_kernel_opt,
    )
    from ...utils.triton_utils import get_device_properties
    from ._l2norm_inputs import prepare_input

    x = prepare_input(x)
    x_shape_og = x.shape
    x = x.view(-1, x.shape[-1])
    if output_dtype is None:
        y = torch.empty_like(x)
    else:
        y = torch.empty_like(x, dtype=output_dtype)
    assert y.stride(-1) == 1
    T, D = x.shape[0], x.shape[-1]
    # rstd = torch.empty((T,), dtype=torch.float32, device=x.device)
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()

    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))

    if D > BD:
        raise RuntimeError("This layer doesn't support feature dim >= 64KB.")

    if T == 0:
        return y.view(x_shape_og)

    if D <= 512:
        NB = triton.cdiv(T, 2048)

        # Keep the tile no larger than the existing 109 x 128 configuration.
        # Wider padded feature dimensions otherwise overflow UB with the FP32
        # intermediates and multi-buffering used by the optimized kernel.
        bt = min(109, (109 * 128) // BD)
        num_core = get_device_properties()[1]
        main_bs = triton.cdiv(T, num_core)
        grid = (num_core,)

        l2norm_fwd_kernel_opt[grid](
            x,
            y,
            eps,
            NB=NB,
            T=T,
            D=D,
            MBS=main_bs,
            BD=BD,
            BT=bt,
            num_warps=8,
            num_stages=3,
        )
    else:
        l2norm_fwd_kernel1[(T,)](
            x,
            y,
            eps=eps,
            D=D,
            BD=BD,
            num_warps=8,
            num_stages=3,
        )

    return y.view(x_shape_og)


@register_impl("norm.l2norm_fwd", "torch_native")
def _torch_native_impl(
    x: torch.Tensor, eps: float = 1e-06, output_dtype: Optional[torch.dtype] = None
):
    from ...implementation.torch_native.norm.l2norm import l2norm_fwd
    from ._l2norm_inputs import prepare_input

    x = prepare_input(x)
    shape = x.shape
    rows = x.numel() // x.shape[-1]
    flat = x.reshape(rows, x.shape[-1])
    output = torch.empty_like(
        flat, dtype=x.dtype if output_dtype is None else output_dtype
    )
    l2norm_fwd(flat, output, eps)
    return output.reshape(shape)


@register_operator("norm.l2norm_fwd")
def l2norm_fwd(
    x: torch.Tensor,
    eps: float = 1e-06,
    output_dtype: Optional[torch.dtype] = None,
    *,
    impl="auto"
):
    """Normalize the last dimension by sqrt(sum(x**2) + eps).

    Preserve the input shape. Output dtype defaults to x.dtype; output_dtype
    overrides it. impl selects auto (Triton), triton, or torch_native.
    The two original Triton kernels remain internal width-dependent variants."""
    return get_impl_by_name("norm.l2norm_fwd", impl)(x, eps, output_dtype)
