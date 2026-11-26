"""Unified normalization operators with independent, lazily loaded backends."""

from typing import Optional, Tuple

import torch

from ...implementation.triton.norm.norm_utils import (
    validate_split_qkv,
    validate_split_qkvgate,
)
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None


# Wrap the Triton-Ascend launch as an opaque torch.library.custom_op with a
# FakeTensor meta implementation so TorchDynamo / npugraph_ex can trace through
# it without graph-breaking inside Triton's Ascend driver.
@torch.library.custom_op(
    "sgl_kernel_npu::split_qkv_rmsnorm_rope_triton", mutates_args=()
)
def _split_qkv_rmsnorm_rope_triton_op(
    input: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor,
    q_hidden_size: int,
    kv_hidden_size: int,
    head_dim: int,
    eps: Optional[float] = None,
    q_weight: Optional[torch.Tensor] = None,
    k_weight: Optional[torch.Tensor] = None,
    q_bias: Optional[torch.Tensor] = None,
    k_bias: Optional[torch.Tensor] = None,
    is_neox_style: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.norm.split_qkv_rmsnorm_rope import (
            split_qkv_rmsnorm_rope,
        )

        _triton_fn = split_qkv_rmsnorm_rope
    return _triton_fn(
        input,
        sin,
        cos,
        q_hidden_size,
        kv_hidden_size,
        head_dim,
        eps,
        q_weight,
        k_weight,
        q_bias,
        k_bias,
        is_neox_style,
    )


@_split_qkv_rmsnorm_rope_triton_op.register_fake
def _split_qkv_rmsnorm_rope_triton_fake(
    input: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor,
    q_hidden_size: int,
    kv_hidden_size: int,
    head_dim: int,
    eps: Optional[float] = None,
    q_weight: Optional[torch.Tensor] = None,
    k_weight: Optional[torch.Tensor] = None,
    q_bias: Optional[torch.Tensor] = None,
    k_bias: Optional[torch.Tensor] = None,
    is_neox_style: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = input.shape[0]
    return (
        torch.empty(
            (batch_size, q_hidden_size), device=input.device, dtype=input.dtype
        ),
        torch.empty(
            (batch_size, kv_hidden_size), device=input.device, dtype=input.dtype
        ),
        torch.empty(
            (batch_size, kv_hidden_size), device=input.device, dtype=input.dtype
        ),
    )


@register_impl("norm.split_qkv_rmsnorm_rope", "auto")
@register_impl("norm.split_qkv_rmsnorm_rope", "triton")
def _triton_impl(
    input,
    sin,
    cos,
    q_hidden_size,
    kv_hidden_size,
    head_dim,
    eps=None,
    q_weight=None,
    k_weight=None,
    q_bias=None,
    k_bias=None,
    is_neox_style=True,
):
    # Validate before custom_op dispatch so invalid Python types/shapes keep
    # raising AssertionError instead of a dispatcher TypeError/RuntimeError.
    validate_split_qkv(
        input,
        sin,
        cos,
        q_hidden_size,
        kv_hidden_size,
        head_dim,
        eps,
        q_weight,
        k_weight,
        q_bias,
        k_bias,
        is_neox_style,
    )
    return _split_qkv_rmsnorm_rope_triton_op(
        input,
        sin,
        cos,
        q_hidden_size,
        kv_hidden_size,
        head_dim,
        None if eps is None else float(eps),
        q_weight,
        k_weight,
        q_bias,
        k_bias,
        is_neox_style,
    )


_torch_native_fn = None


@register_impl("norm.split_qkv_rmsnorm_rope", "torch_native")
def _torch_native_impl(
    input,
    sin,
    cos,
    q_hidden_size,
    kv_hidden_size,
    head_dim,
    eps=None,
    q_weight=None,
    k_weight=None,
    q_bias=None,
    k_bias=None,
    is_neox_style=True,
):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.norm.split_qkv_rmsnorm_rope import (
            split_qkv_rmsnorm_rope,
        )

        _torch_native_fn = split_qkv_rmsnorm_rope
    return _torch_native_fn(
        input,
        sin,
        cos,
        q_hidden_size,
        kv_hidden_size,
        head_dim,
        eps,
        q_weight,
        k_weight,
        q_bias,
        k_bias,
        is_neox_style,
    )


@register_operator("norm.split_qkv_rmsnorm_rope")
def split_qkv_rmsnorm_rope(
    input,
    sin,
    cos,
    q_hidden_size,
    kv_hidden_size,
    head_dim,
    eps=None,
    q_weight=None,
    k_weight=None,
    q_bias=None,
    k_bias=None,
    is_neox_style=True,
    *,
    impl="auto"
):
    """Split [N, Q+2K], optionally normalize Q/K per head, and apply RoPE.

    auto selects Triton. Q/K are q_hidden_size/kv_hidden_size; head_dim is D.
    sin/cos contain N*R values with last dimension R (even, at most D).
    eps=None skips normalization; otherwise q_weight/k_weight contain at
    least D floats. Optional q_bias/k_bias must be supplied together.
    is_neox_style=True pairs halves; False pairs adjacent features.
    Returns new Q/K/V tensors [N, Q], [N, K], [N, K], in input dtype.

    impl is keyword-only: "auto", "triton", or "torch_native".
    """
    impl_fn = get_impl_by_name("norm.split_qkv_rmsnorm_rope", impl)
    return impl_fn(
        input,
        sin,
        cos,
        q_hidden_size,
        kv_hidden_size,
        head_dim,
        eps,
        q_weight,
        k_weight,
        q_bias,
        k_bias,
        is_neox_style,
    )


_gate_triton_fn = None


@torch.library.custom_op(
    "sgl_kernel_npu::split_qkvgate_gemma_rmsnorm_rope_triton", mutates_args=()
)
def _split_qkvgate_gemma_rmsnorm_rope_triton_op(
    input: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor,
    q_hidden_size: int,
    kv_hidden_size: int,
    head_dim: int,
    rope_dim: int,
    eps: float,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    global _gate_triton_fn
    if _gate_triton_fn is None:
        from ...implementation.triton.norm.split_qkv_rmsnorm_rope import (
            split_qkvgate_gemma_rmsnorm_rope,
        )

        _gate_triton_fn = split_qkvgate_gemma_rmsnorm_rope
    return _gate_triton_fn(
        input,
        sin,
        cos,
        q_hidden_size,
        kv_hidden_size,
        head_dim,
        rope_dim,
        eps,
        q_weight,
        k_weight,
    )


@_split_qkvgate_gemma_rmsnorm_rope_triton_op.register_fake
def _split_qkvgate_gemma_rmsnorm_rope_triton_fake(
    input: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor,
    q_hidden_size: int,
    kv_hidden_size: int,
    head_dim: int,
    rope_dim: int,
    eps: float,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = input.shape[0]
    return (
        torch.empty(
            (batch_size, q_hidden_size), device=input.device, dtype=input.dtype
        ),
        torch.empty(
            (batch_size, kv_hidden_size), device=input.device, dtype=input.dtype
        ),
        torch.empty(
            (batch_size, kv_hidden_size), device=input.device, dtype=input.dtype
        ),
        torch.empty(
            (batch_size, q_hidden_size), device=input.device, dtype=input.dtype
        ),
    )


@register_impl("norm.split_qkvgate_gemma_rmsnorm_rope", "auto")
@register_impl("norm.split_qkvgate_gemma_rmsnorm_rope", "triton")
def _gate_triton_impl(
    input,
    sin,
    cos,
    q_hidden_size,
    kv_hidden_size,
    head_dim,
    rope_dim,
    eps,
    q_weight,
    k_weight,
):
    validate_split_qkvgate(
        input,
        sin,
        cos,
        q_hidden_size,
        kv_hidden_size,
        head_dim,
        rope_dim,
        eps,
        q_weight,
        k_weight,
    )
    return _split_qkvgate_gemma_rmsnorm_rope_triton_op(
        input,
        sin,
        cos,
        q_hidden_size,
        kv_hidden_size,
        head_dim,
        rope_dim,
        float(eps),
        q_weight,
        k_weight,
    )


_gate_torch_native_fn = None


@register_impl("norm.split_qkvgate_gemma_rmsnorm_rope", "torch_native")
def _gate_torch_native_impl(
    input,
    sin,
    cos,
    q_hidden_size,
    kv_hidden_size,
    head_dim,
    rope_dim,
    eps,
    q_weight,
    k_weight,
):
    global _gate_torch_native_fn
    if _gate_torch_native_fn is None:
        from ...implementation.torch_native.norm.split_qkv_rmsnorm_rope import (
            split_qkvgate_gemma_rmsnorm_rope,
        )

        _gate_torch_native_fn = split_qkvgate_gemma_rmsnorm_rope
    return _gate_torch_native_fn(
        input,
        sin,
        cos,
        q_hidden_size,
        kv_hidden_size,
        head_dim,
        rope_dim,
        eps,
        q_weight,
        k_weight,
    )


@register_operator("norm.split_qkvgate_gemma_rmsnorm_rope")
def split_qkvgate_gemma_rmsnorm_rope(
    input,
    sin,
    cos,
    q_hidden_size,
    kv_hidden_size,
    head_dim,
    rope_dim,
    eps,
    q_weight,
    k_weight,
    *,
    impl="auto"
):
    """Deinterleave Q/gate, apply Gemma RMSNorm to Q/K, then partial NeoX RoPE.

    auto selects Triton. input is [N, 2Q+2K], with each Q head immediately
    followed by its gate head. sin/cos contain N*rope_dim floats; q_weight
    and k_weight have at least head_dim floats. Positive eps controls RMSNorm
    with multiplier 1+weight. Returns new Q/K/V/gate tensors of shapes
    [N, Q], [N, K], [N, K], [N, Q], preserving input dtype and device.

    impl is keyword-only: "auto", "triton", or "torch_native".
    """
    impl_fn = get_impl_by_name("norm.split_qkvgate_gemma_rmsnorm_rope", impl)
    return impl_fn(
        input,
        sin,
        cos,
        q_hidden_size,
        kv_hidden_size,
        head_dim,
        rope_dim,
        eps,
        q_weight,
        k_weight,
    )
