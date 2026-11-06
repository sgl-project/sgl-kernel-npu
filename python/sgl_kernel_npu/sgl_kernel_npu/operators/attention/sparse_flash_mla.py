"""Sparse Flash MLA operators backed by the CANN ops-transformer repository."""

from ..utils.cann import detect_cann_version
from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_CANN_FNS = None
_NATIVE_FN = None


def _load_cann_fns():
    """Import the wheel-bundled ops-transformer bindings once."""
    global _CANN_FNS
    if _CANN_FNS is None:
        try:
            from ..._third_party.sgl_ops_transformer.ops import sparse_flash_mla as op
            from ..._third_party.sgl_ops_transformer.ops import (
                sparse_flash_mla_metadata as metadata_op,
            )
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "sparse_flash_mla requires the sgl_ops_transformer dependency "
                "bundled in the wheel; build it with the third-party profile"
            ) from error
        _CANN_FNS = (op, metadata_op)
    return _CANN_FNS


@register_impl(
    "attention.sparse_flash_mla", "auto", supported_generations={Generation.ARCH22}
)
def _auto_impl(*args, **kwargs):
    version = detect_cann_version()
    if version != "9.2.0":
        raise RuntimeError(
            "sparse_flash_mla auto supports CANN 9.2.0 only; detected CANN {}".format(
                version
            )
        )
    return get_impl_by_name(
        "attention.sparse_flash_mla", "cann_ops_transformers_cann920"
    )(*args, **kwargs)


@register_impl(
    "attention.sparse_flash_mla",
    "cann_ops_transformers_cann920",
    supported_generations={Generation.ARCH22},
)
def _cann_impl(*args, **kwargs):
    return _load_cann_fns()[0](*args, **kwargs)


@register_impl("attention.sparse_flash_mla", "torch_native")
def _native_impl(*args, **kwargs):
    global _NATIVE_FN
    if _NATIVE_FN is None:
        from ...implementation.torch_native.attention.sparse_flash_mla import (
            sparse_flash_mla as reference,
        )

        _NATIVE_FN = reference
    return _NATIVE_FN(*args, **kwargs)


@register_operator("attention.sparse_flash_mla")
def sparse_flash_mla(
    q,
    *,
    ori_kv=None,
    cmp_kv=None,
    ori_sparse_indices=None,
    cmp_sparse_indices=None,
    ori_block_table=None,
    cmp_block_table=None,
    cu_seqlens_q=None,
    cu_seqlens_ori_kv=None,
    cu_seqlens_cmp_kv=None,
    seqused_q=None,
    seqused_ori_kv=None,
    seqused_cmp_kv=None,
    cmp_residual_kv=None,
    ori_topk_length=None,
    cmp_topk_length=None,
    sinks=None,
    metadata=None,
    softmax_scale=1.0,
    cmp_ratio=1,
    ori_mask_mode=0,
    cmp_mask_mode=0,
    ori_win_left=-1,
    ori_win_right=-1,
    layout_q="BSND",
    layout_kv="BSND",
    topk_value_mode=1,
    return_softmax_lse=False,
    impl="auto",
):
    """Compute sparse MLA attention over shared original and compressed KV.

    ``auto`` selects ``cann_ops_transformers_cann920`` on arch22 with
    CANN 9.2.0. This is a non-official operator introduced in CANN 9.2.0.
    ``auto`` raises on other CANN releases; ``torch_native`` is not
    version-gated.

    The operator serves three scenarios: SWA uses only ``ori_kv``; HCA adds a
    contiguous ``cmp_kv`` window; CSA adds the compressed tokens picked by
    ``cmp_sparse_indices``. ``cmp_ratio`` is 1 for SWA, 1/2/4 for CSA and 128
    for HCA on Atlas A2/A3. ``metadata`` must come from
    :func:`sparse_flash_mla_metadata` with identical scheduling arguments.

    q (Tensor): [T, N, 512] (TND) or [B, S, N, 512] (BSND), fp16/bf16.
    ori_kv (Tensor): uncompressed shared K/V in the ``layout_kv`` layout.
    cmp_kv (Tensor | None): compressed shared K/V; None for SWA.
    ori_sparse_indices (Tensor | None): unsupported on Atlas A2/A3.
    cmp_sparse_indices (Tensor | None): int32 CSA token ids, -1 terminates.
    ori_block_table/cmp_block_table (Tensor | None): int32 page tables.
    cu_seqlens_q/cu_seqlens_ori_kv/cu_seqlens_cmp_kv (Tensor | None):
        int32 [B+1] packed boundaries for the TND layout.
    seqused_q/seqused_ori_kv/seqused_cmp_kv (Tensor | None): int32 [B] lengths.
    cmp_residual_kv (Tensor | None): int32 [B] compression remainders;
        required when cmp_mask_mode=3 and cmp_ratio!=1, rejected otherwise.
    ori_topk_length/cmp_topk_length (Tensor | None): int32 valid index counts.
    sinks (Tensor | None): float32 [N] per-head softmax sinks; the Atlas
        A2/A3 kernel requires a nonempty tensor and rejects None.
    metadata (Tensor): int32 [2048] scheduling data from the metadata operator.
    softmax_scale (float): multiplier applied to QK scores.
    cmp_ratio (int): compression ratio; 1/2/4 for CSA on Atlas A2/A3.
    ori_mask_mode (int): 0 unmasked, 3 causal, 4 sliding window.
    cmp_mask_mode (int): 0 unmasked, 3 causal.
    ori_win_left/ori_win_right (int): window bounds; Atlas A3 requires 127/0.
    layout_q/layout_kv (str): input layouts; PA_BBND needs a block table.
    topk_value_mode (int): index convention; 1 only on Atlas A2/A3.
    return_softmax_lse (bool): return FP32 log-sum-exp alongside the output.
    impl (str): auto, cann_ops_transformers_cann920, or torch_native.

    Returns (attention_out, softmax_lse); a disabled lse is an empty FP32
    tensor. torch_native computes the same math in FP32, requires full prompt
    batches, and ignores ``metadata`` as an opaque scheduling artifact.
    """
    return get_impl_by_name("attention.sparse_flash_mla", impl)(
        q,
        ori_kv=ori_kv,
        cmp_kv=cmp_kv,
        ori_sparse_indices=ori_sparse_indices,
        cmp_sparse_indices=cmp_sparse_indices,
        ori_block_table=ori_block_table,
        cmp_block_table=cmp_block_table,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_ori_kv=cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
        seqused_q=seqused_q,
        seqused_ori_kv=seqused_ori_kv,
        seqused_cmp_kv=seqused_cmp_kv,
        cmp_residual_kv=cmp_residual_kv,
        ori_topk_length=ori_topk_length,
        cmp_topk_length=cmp_topk_length,
        sinks=sinks,
        metadata=metadata,
        softmax_scale=softmax_scale,
        cmp_ratio=cmp_ratio,
        ori_mask_mode=ori_mask_mode,
        cmp_mask_mode=cmp_mask_mode,
        ori_win_left=ori_win_left,
        ori_win_right=ori_win_right,
        layout_q=layout_q,
        layout_kv=layout_kv,
        topk_value_mode=topk_value_mode,
        return_softmax_lse=return_softmax_lse,
    )


@register_impl(
    "attention.sparse_flash_mla_metadata",
    "auto",
    supported_generations={Generation.ARCH22},
)
def _auto_metadata(*args, **kwargs):
    version = detect_cann_version()
    if version != "9.2.0":
        raise RuntimeError(
            "sparse_flash_mla_metadata auto supports CANN 9.2.0 only; "
            "detected CANN {}".format(version)
        )
    return get_impl_by_name(
        "attention.sparse_flash_mla_metadata", "cann_ops_transformers_cann920"
    )(*args, **kwargs)


@register_impl(
    "attention.sparse_flash_mla_metadata",
    "cann_ops_transformers_cann920",
    supported_generations={Generation.ARCH22},
)
def _cann_metadata(*args, **kwargs):
    return _load_cann_fns()[1](*args, **kwargs)


@register_operator("attention.sparse_flash_mla_metadata")
def sparse_flash_mla_metadata(
    num_heads_q,
    num_heads_kv,
    head_dim,
    *,
    cu_seqlens_q=None,
    cu_seqlens_ori_kv=None,
    cu_seqlens_cmp_kv=None,
    seqused_q=None,
    seqused_ori_kv=None,
    seqused_cmp_kv=None,
    cmp_residual_kv=None,
    ori_topk_length=None,
    cmp_topk_length=None,
    batch_size=None,
    max_seqlen_q=None,
    max_seqlen_ori_kv=None,
    max_seqlen_cmp_kv=None,
    ori_topk=None,
    cmp_topk=None,
    cmp_ratio=None,
    ori_mask_mode=None,
    cmp_mask_mode=None,
    ori_win_left=None,
    ori_win_right=None,
    layout_q=None,
    layout_kv=None,
    has_ori_kv=None,
    has_cmp_kv=None,
    impl="auto",
):
    """Build the scheduling metadata consumed by :func:`sparse_flash_mla`.

    ``auto`` selects ``cann_ops_transformers_cann920`` on arch22 with
    CANN 9.2.0 and raises on other releases.

    Every scheduling argument must match the later ``sparse_flash_mla`` call;
    ``cmp_ratio``, ``ori_mask_mode``/``cmp_mask_mode`` and the window bounds
    are explicit here because this operator defaults them to None.

    Returns an opaque one-dimensional int32 tensor with 2048 elements.
    """
    return get_impl_by_name("attention.sparse_flash_mla_metadata", impl)(
        num_heads_q,
        num_heads_kv,
        head_dim,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_ori_kv=cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
        seqused_q=seqused_q,
        seqused_ori_kv=seqused_ori_kv,
        seqused_cmp_kv=seqused_cmp_kv,
        cmp_residual_kv=cmp_residual_kv,
        ori_topk_length=ori_topk_length,
        cmp_topk_length=cmp_topk_length,
        batch_size=batch_size,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_ori_kv=max_seqlen_ori_kv,
        max_seqlen_cmp_kv=max_seqlen_cmp_kv,
        ori_topk=ori_topk,
        cmp_topk=cmp_topk,
        cmp_ratio=cmp_ratio,
        ori_mask_mode=ori_mask_mode,
        cmp_mask_mode=cmp_mask_mode,
        ori_win_left=ori_win_left,
        ori_win_right=ori_win_right,
        layout_q=layout_q,
        layout_kv=layout_kv,
        has_ori_kv=has_ori_kv,
        has_cmp_kv=has_cmp_kv,
    )
