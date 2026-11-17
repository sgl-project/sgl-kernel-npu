"""PyTorch implementation of fused decode-indexer input preparation."""

import torch


def kpool_indexer_prepare_decode_input(
    x,
    weights_proj_weight,
    seqlens_32,
    block_tables,
    pool_size=4,
    n_heads_pow=32**-0.5,
    softmax_scale=128**-0.5,
):
    """Prepare weights, pooled lengths and pooled page tables with PyTorch."""
    w = ((x.float() @ weights_proj_weight.T) * (n_heads_pow * softmax_scale)).to(
        torch.bfloat16
    )
    pool_seqlens = torch.div(seqlens_32, pool_size, rounding_mode="floor").to(
        torch.int32
    )

    batch_size, num_blocks = block_tables.shape
    padded_blocks = -(-num_blocks // pool_size) * pool_size or pool_size
    dummy = torch.zeros(
        (batch_size, padded_blocks - num_blocks),
        dtype=block_tables.dtype,
        device=block_tables.device,
    )
    padded = torch.cat((block_tables, dummy), dim=-1)
    pool_block_tables = padded.view(batch_size, padded_blocks // pool_size, pool_size)[
        :, :, 0
    ].contiguous()
    return w, pool_seqlens, pool_block_tables
