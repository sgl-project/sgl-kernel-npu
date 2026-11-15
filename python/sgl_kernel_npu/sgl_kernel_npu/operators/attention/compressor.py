"""Stateful KV compression with the legacy Ascend C computation."""

import torch

from ..utils.cann import detect_cann_version
from ..utils.generation import Generation, get_hardware
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_torch_native_fn = None


@register_impl("attention.compressor", "auto")
def _auto_impl(*args):
    version = detect_cann_version()
    if version.startswith("9.1."):
        # The fused repository kernel preserves old ring history until reads
        # finish. The 9.1 experimental provider writes overlapping slots first.
        backend = "ascendc"
    elif version.startswith("9.2."):
        backend = "cann_ops_transformers_cann920"
    else:
        raise RuntimeError(
            f"Unsupported CANN release {version}; expected CANN 9.1.x or 9.2.x"
        )
    return get_impl_by_name("attention.compressor", backend)(*args)


@register_impl("attention.compressor", "cann_ops_transformers_cann920")
def _cann920_impl(*args):
    version = detect_cann_version()
    if not version.startswith("9.2."):
        raise RuntimeError(f"This implementation requires CANN 9.2.x, got {version}")
    return _ascendc_impl(*args)


@register_impl("attention.compressor", "ascendc")
def _ascendc_impl(*args):
    return torch.ops.npu.compressor(*args)


@register_impl("attention.compressor", "torch_native")
def _torch_native_impl(*args):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.attention.compressor import compressor

        _torch_native_fn = compressor
    return _torch_native_fn(
        *args, arch35=get_hardware().generation == Generation.ARCH35
    )


@register_operator("attention.compressor")
def compressor(
    x,
    wkv,
    wgate,
    state_cache,
    ape,
    norm_weight,
    rope_sin,
    rope_cos,
    state_block_table=None,
    cu_seqlens=None,
    seqused=None,
    start_pos=None,
    rope_head_dim=64,
    cmp_ratio=4,
    coff=1,
    norm_eps=1e-6,
    rotary_mode=2,
    cache_mode=1,
    state_cache_stride_dim0=0,
    *,
    impl="auto",
):
    """Compress KV groups, apply RMSNorm/RoPE, and update state_cache in place.

    ``auto`` uses the repository's fused Ascend C operator on A3 and A5:
    ``ascendc`` on CANN 9.1, its ``cann_ops_transformers_cann920`` alias on 9.2.
    All tensors are contiguous on the same NPU.
    Args:
        x: FP16/BF16 input [B, S, H] or packed [T, H].
        wkv: Same dtype as x, projection [coff*D, H].
        wgate: Same shape/dtype as wkv, score projection.
        state_cache: FP32 mutable cache [P, L, 2*coff*D], KV then scores.
        ape: FP32 position bias [cmp_ratio, coff*D].
        norm_weight: FP32 RMSNorm scale [D], D=128 or 512.
        rope_sin: FP32 sine [B, ceil(S/r), 64] or [min(T,T//r+B), 64].
        rope_cos: FP32 cosine, same shape as rope_sin.
        state_block_table: Required INT32 cache locations; paged [B, M],
            or explicit physical state rows [B, coff*r+capacity] on both
            architectures. Legacy arch22 mode-1 single-page [B] is also accepted.
        cu_seqlens: INT32 cumulative token offsets [B+1], required only for TH.
        seqused: Optional INT32 used lengths [B]; None uses input capacities.
        start_pos: Optional INT32 absolute starting positions [B]; None means 0.
        rope_head_dim: Integer rotary width, must be 64 (default 64).
        cmp_ratio: Integer group size r in {2,4,8,16,32,64,128} (default 4).
        coff: Integer overlap factor 1 or 2 (default 1).
        norm_eps: Float RMSNorm epsilon (default 1e-6).
        rotary_mode: Integer RoPE mode, default 2 (interleaved). Only 2 has
            a compiled kernel; other values are rejected before launch.
        cache_mode: Integer 1 for paged cache or 2 for explicit state rows.
        state_cache_stride_dim0: Integer cache stride; 0 derives L*2*coff*D.
        impl: Backend name, default "auto"; also "ascendc",
            "cann_ops_transformers_cann920" (CANN 9.2), or "torch_native".
    Returns:
        Same dtype as x, [B, ceil(S/r), D] or [min(T,T//r+B), D]. Only
        completed groups are written; remaining output rows are unspecified.
    """
    # Mode 2 indexes physical rows. Keep the arch22 mode-1 single-page shorthand.
    if state_block_table is not None:
        if cache_mode == 2 and (
            state_block_table.ndim != 2
            or state_block_table.shape[1] <= coff * cmp_ratio
        ):
            raise ValueError(
                "compressor: mode 2 requires a 2D state_block_table with "
                "at least coff*cmp_ratio+1 columns"
            )
        elif (
            cache_mode == 1
            and state_block_table.ndim == 1
            and get_hardware().generation == Generation.ARCH22
        ):
            state_block_table = state_block_table.view(-1, 1)
    return get_impl_by_name("attention.compressor", impl)(
        x,
        wkv,
        wgate,
        state_cache,
        ape,
        norm_weight,
        rope_sin,
        rope_cos,
        state_block_table,
        cu_seqlens,
        seqused,
        start_pos,
        rope_head_dim,
        cmp_ratio,
        coff,
        norm_eps,
        rotary_mode,
        cache_mode,
        state_cache_stride_dim0,
    )
