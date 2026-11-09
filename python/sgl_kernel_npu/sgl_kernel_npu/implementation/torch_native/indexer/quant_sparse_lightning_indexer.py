"""Independent Torch quantized indexer reference; accepts CPU or NPU inputs."""

from .indexer_utils import indexer_reference


def quant_sparse_lightning_indexer(
    q,
    k,
    w,
    descale_q,
    candidate_block_indices,
    candidate_block_length,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    block_table=None,
    output_idx_offset=None,
    metadata=None,
    *,
    descale_k=None,
    topk,
    candidate_block_size,
    quant_mode,
    max_seqlen_q=-1,
    mask_mode=0,
    cmp_ratio=1,
    layout_q="TND",
    layout_k="TND",
    return_value=False
):
    return indexer_reference(**locals())
