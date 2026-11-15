"""Non-official CANN 9.2.0 Flash MLA operators backed by ops-transformer."""

from ..utils.cann import detect_cann_version
from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_cann_fn = None
_metadata_fn = None
_native_fn = None


@register_impl("attention.flash_mla", "auto", supported_generations={Generation.ARCH35})
def _auto_impl(*args, **kwargs):
    version = detect_cann_version()
    if version != "9.2.0":
        raise RuntimeError(
            "flash_mla auto supports CANN 9.2.0 only; detected CANN {}".format(version)
        )
    return get_impl_by_name("attention.flash_mla", "cann_ops_transformers_cann920")(
        *args, **kwargs
    )


@register_impl(
    "attention.flash_mla",
    "cann_ops_transformers_cann920",
    supported_generations={Generation.ARCH35},
)
def _cann_impl(*args, **kwargs):
    global _cann_fn
    if _cann_fn is None:
        from ..._third_party.sgl_ops_transformer.ops import flash_mla

        _cann_fn = flash_mla
    return _cann_fn(*args, **kwargs)


@register_impl("attention.flash_mla", "torch_native")
def _native_impl(*args, **kwargs):
    global _native_fn
    if _native_fn is None:
        from ...implementation.torch_native.attention.flash_mla import flash_mla

        _native_fn = flash_mla
    return _native_fn(*args, **kwargs)


@register_operator("attention.flash_mla")
def flash_mla(
    q,
    k_cache,
    block_table=None,
    cache_seqlens=None,
    cu_seqlens_q=None,
    seqused_q=None,
    sinks=None,
    attn_mask=None,
    metadata=None,
    head_dim_v=512,
    softmax_scale=1.0,
    mask_mode=0,
    win_left=-1,
    win_right=-1,
    max_seqlen_q=-1,
    max_seqlen_kv=-1,
    layout_q="BSND",
    layout_kv="PA_BBND",
    layout_out="BSND",
    return_softmax_lse=False,
    *,
    impl="auto",
):
    """Compute paged MLA attention.

    auto selects cann_ops_transformers_cann920 on arch35 with CANN 9.2.0.

    This is a non-official operator introduced in CANN 9.2.0.
    auto raises on other CANN releases; torch_native is not version-gated.

    q (Tensor): [T, N, 576] (TND), [B, S, N, 576] (BSND), or
        [B, N, S, 576] (BNSD), fp16/bf16.
    k_cache (Tensor): [P, 128, 1, 576] (PA_BBND), or
        [P, 1, 36, 128, 16] (PA_NZ); values are its first head_dim_v channels.
    block_table (Tensor): int32 [B, max_pages], logical-to-physical page IDs.
    cache_seqlens (Tensor): int32 [B], valid KV lengths.
    cu_seqlens_q (Tensor | None): int32 [B+1], packed query boundaries.
    seqused_q (Tensor | None): int32 [B], valid query lengths.
    sinks (Tensor | None): float32 [N], optional per-head softmax sink logits.
    attn_mask (Tensor | None): upstream attention mask for the selected mode.
    metadata (Tensor | None): opaque int32 scheduling data from flash_mla_metadata.
    head_dim_v (int): value width, 512 for the CANN kernel.
    softmax_scale (float): multiplier applied to QK scores.
    mask_mode (int): 0 unmasked, 3 right-aligned causal, 4 sliding window.
    win_left/win_right (int): window bounds, -1 for upstream defaults.
    max_seqlen_q/max_seqlen_kv (int): upstream length attributes; use -1.
    layout_q/layout_kv/layout_out (str): input and output layouts.
    return_softmax_lse (bool): return FP32 log-sum-exp alongside the output.
    impl (str): auto, cann_ops_transformers_cann920, or torch_native.

    Returns (output, lse); disabled lse is an empty FP32 tensor. torch_native
    supports TND/PA_BBND, TND or NTD output, mask_mode 0/3, fully used packed
    query segments, no sinks and default windows/maximum lengths. Mode 3
    requires the standard compressed causal mask. Metadata is opaque.
    """
    return get_impl_by_name("attention.flash_mla", impl)(
        q,
        k_cache,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        seqused_q=seqused_q,
        sinks=sinks,
        attn_mask=attn_mask,
        metadata=metadata,
        head_dim_v=head_dim_v,
        softmax_scale=softmax_scale,
        mask_mode=mask_mode,
        win_left=win_left,
        win_right=win_right,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        layout_q=layout_q,
        layout_kv=layout_kv,
        layout_out=layout_out,
        return_softmax_lse=return_softmax_lse,
    )


@register_impl(
    "attention.flash_mla_metadata", "auto", supported_generations={Generation.ARCH35}
)
def _auto_metadata(*args, **kwargs):
    version = detect_cann_version()
    if version != "9.2.0":
        raise RuntimeError(
            "flash_mla_metadata auto supports CANN 9.2.0 only; detected CANN {}".format(
                version
            )
        )
    return get_impl_by_name(
        "attention.flash_mla_metadata", "cann_ops_transformers_cann920"
    )(*args, **kwargs)


@register_impl(
    "attention.flash_mla_metadata",
    "cann_ops_transformers_cann920",
    supported_generations={Generation.ARCH35},
)
def _cann_metadata(*args, **kwargs):
    global _metadata_fn
    if _metadata_fn is None:
        from ..._third_party.sgl_ops_transformer.ops import flash_mla_metadata

        _metadata_fn = flash_mla_metadata
    return _metadata_fn(*args, **kwargs)


@register_operator("attention.flash_mla_metadata")
def flash_mla_metadata(
    cache_seqlens,
    num_heads_q,
    num_heads_kv,
    cu_seqlens_q=None,
    seqused_q=None,
    max_seqlen_q=-1,
    max_seqlen_kv=-1,
    head_dim_qk=576,
    head_dim_v=512,
    mask_mode=0,
    win_left=-1,
    win_right=-1,
    layout_q="BSND",
    *,
    impl="auto",
):
    """Build scheduling metadata.

    auto selects cann_ops_transformers_cann920 on arch35 with CANN 9.2.0.

    This is a non-official operator introduced in CANN 9.2.0.
    auto raises on other CANN releases.

    cache_seqlens (Tensor): int32 [B], KV lengths.
    num_heads_q/num_heads_kv (int): query/KV head counts.
    cu_seqlens_q (Tensor | None): int32 [B+1], packed query boundaries.
    seqused_q (Tensor | None): int32 [B], valid query lengths.
    max_seqlen_q/max_seqlen_kv (int): use -1 for upstream defaults.
    head_dim_qk/head_dim_v (int): QK/value widths.
    mask_mode (int): attention mask mode; layout_q (str): query layout.
    win_left/win_right (int): window bounds, -1 for upstream defaults.
    impl (str): auto or cann_ops_transformers_cann920.
    Returns an opaque one-dimensional int32 tensor sized by batch/core counts.
    """
    return get_impl_by_name("attention.flash_mla_metadata", impl)(
        cache_seqlens,
        num_heads_q,
        num_heads_kv,
        cu_seqlens_q=cu_seqlens_q,
        seqused_q=seqused_q,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        head_dim_qk=head_dim_qk,
        head_dim_v=head_dim_v,
        mask_mode=mask_mode,
        win_left=win_left,
        win_right=win_right,
        layout_q=layout_q,
    )
