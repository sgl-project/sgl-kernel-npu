"""Stateful experimental ops-transformer compressor for CANN 9.2.0 / arch22."""

import torch

from ..utils.cann import detect_cann_version
from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_cann_fn = None
_native_fn = None


def _binding():
    global _cann_fn
    if _cann_fn is None:
        from ..._third_party.sgl_ops_transformer_experimental.ops.ds41 import compressor

        _cann_fn = compressor
    return _cann_fn


@register_impl(
    "attention.compressor_v2", "auto", supported_generations={Generation.ARCH22}
)
@register_impl(
    "attention.compressor_v2",
    "cann_ops_transformers_cann920",
    supported_generations={Generation.ARCH22},
)
def _cann_impl(*args, **kwargs):
    if detect_cann_version() != "9.2.0":
        raise RuntimeError("compressor_v2 requires CANN 9.2.0")
    return _binding()(*args, **kwargs)


@register_impl("attention.compressor_v2", "torch_native")
def _native_impl(*args, **kwargs):
    global _native_fn
    if _native_fn is None:
        from ...implementation.torch_native.attention.compressor_v2 import compressor_v2

        _native_fn = compressor_v2
    return _native_fn(*args, **kwargs)


@register_operator("attention.compressor_v2")
def compressor_v2(
    x,
    wkv,
    wgate,
    state_cache,
    cmp_ratio=4,
    *,
    state_block_table=None,
    cu_seqlens=None,
    seqused=None,
    start_pos=None,
    impl="auto",
):
    """Compress completed token groups and update state_cache in place.

    The default auto implementation uses bundled CANN 9.2.0 operators on A3.

    Args:
        x (Tensor): Contiguous fp16/bf16 [B,S,H] or packed [T,H] activations.
        wkv (Tensor): Projection weights [D,H], with the dtype of x; D=128/512.
        wgate (Tensor): Gate weights with the shape and dtype of wkv.
        state_cache (Tensor): FP32 circular history [P,C,2*D], updated in place.
        cmp_ratio (int): Tokens per compressed group, a power of two in 2..128.
        state_block_table (Tensor): Required int32 [B] physical cache block IDs.
        cu_seqlens (Tensor | None): Int32 [B+1] cumulative TH boundaries.
        seqused (Tensor | None): Int32 [B] valid lengths; None uses all tokens.
        start_pos (Tensor | None): Int32 [B] absolute positions; None means zero.
        impl (str): auto, cann_ops_transformers_cann920, or torch_native.

    All tensors share x.device. Returns x.dtype compressed rows [B,ceil(S/r),D]
    or [min(T,T//r+B),D], where r=cmp_ratio. Only completed groups are valid;
    unused output padding is unspecified. The Torch reference requires C>=r.
    """
    if x.ndim not in (2, 3):
        raise ValueError("x must be TH or BSH")
    if cmp_ratio not in (2, 4, 8, 16, 32, 64, 128):
        raise ValueError("cmp_ratio must be a power of two in [2, 128]")
    if (
        x.dtype not in (torch.float16, torch.bfloat16)
        or wkv.dtype != x.dtype
        or wgate.dtype != x.dtype
    ):
        raise TypeError("x, wkv and wgate require matching fp16/bf16")
    hidden = x.shape[-1]
    if not 1024 <= hidden <= 10240 or hidden % 512:
        raise ValueError("H must be 512-aligned in [1024, 10240]")
    if (
        wkv.ndim != 2
        or wkv.shape[0] not in (128, 512)
        or wkv.shape[1] != hidden
        or wgate.shape != wkv.shape
    ):
        raise ValueError("wkv/wgate must have shape [D,H], D=128/512")
    dim = wkv.shape[0]
    if state_cache.dtype != torch.float32:
        raise TypeError("state_cache must be fp32")
    if (
        state_cache.ndim != 3
        or state_cache.shape[-1] != 2 * dim
        or state_cache.shape[0] == 0
        or not 1 <= state_cache.shape[1] <= 1024
    ):
        raise ValueError("state_cache must have shape [P,C,2*D], P>0, C=1..1024")
    if (
        state_cache.stride(2) != 1
        or state_cache.stride(1) != 2 * dim
        or state_cache.stride(0) < state_cache.shape[1] * 2 * dim
    ):
        raise ValueError("Only axis 0 of state_cache may be noncontiguous")
    for tensor in (x, wkv, wgate):
        if not tensor.is_contiguous():
            raise ValueError("x and weights must be contiguous")
    for tensor in (wkv, wgate, state_cache):
        if tensor.device != x.device:
            raise ValueError("x, weights and cache must share a device")
    if x.ndim == 2:
        if cu_seqlens is None or cu_seqlens.ndim != 1 or cu_seqlens.numel() < 1:
            raise ValueError("TH x requires int32 cu_seqlens[B+1]")
        batch = cu_seqlens.numel() - 1
    else:
        if cu_seqlens is not None:
            raise ValueError("BSH x does not accept cu_seqlens")
        batch = x.shape[0]
    if state_cache.shape[0] < batch:
        raise ValueError("state_cache requires at least B physical blocks")
    if state_block_table is None:
        raise ValueError("arch22 requires state_block_table[B]")
    for name, tensor in (
        ("state_block_table", state_block_table),
        ("cu_seqlens", cu_seqlens),
        ("seqused", seqused),
        ("start_pos", start_pos),
    ):
        if tensor is None:
            continue
        if tensor.dtype != torch.int32 or tensor.device != x.device:
            raise ValueError(name + " must be int32 on x.device")
        expected = batch + 1 if name == "cu_seqlens" else batch
        if tensor.shape != (expected,) or not tensor.is_contiguous():
            raise ValueError(name + " has an invalid shape or stride")
    # Tensor contents (unique physical block IDs, lengths, positions and valid
    # history) are caller obligations; do not synchronize the launch path.
    return get_impl_by_name("attention.compressor_v2", impl)(
        x,
        wkv,
        wgate,
        state_cache,
        cmp_ratio,
        state_block_table=state_block_table,
        cu_seqlens=cu_seqlens,
        seqused=seqused,
        start_pos=start_pos,
    )
