"""Independent Torch quantized indexer reference; accepts CPU or NPU inputs."""

from .indexer_utils import indexer_reference


def quant_lightning_indexer_dsl(
    q,
    k,
    w,
    descale_q,
    descale_k,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    block_table=None,
    output_idx_offset=None,
    metadata=None,
    *,
    topk,
    quant_mode,
    max_seqlen_q=-1,
    mask_mode=0,
    cmp_ratio=1,
    layout_q="TND",
    layout_k="TND",
    return_value=False,
    candidate_topk_blocks=-1,
    candidate_block_size=-1
):
    return indexer_reference(**locals())
