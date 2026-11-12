"""Independent PyTorch reference for packed MLA sparse attention."""

import torch

from .sparse_flash_attention import sparse_flash_attention


def kv_quant_sparse_flash_attention(
    query,
    query_rope,
    packed_kv,
    sparse_indices,
    block_table,
    actual_seq_lengths_query,
    actual_seq_lengths_kv,
    scale_value,
):
    """Decode the packed MLA record and use the independent FP32 attention reference."""
    packed = packed_kv.detach().cpu()
    latent_bytes = 512
    rope_end = latent_bytes + 64 * 2
    latent = packed[..., :latent_bytes].contiguous().view(torch.float8_e4m3fn)
    rope = packed[..., latent_bytes:rope_end].contiguous().view(torch.bfloat16)
    scales = packed[..., rope_end:].contiguous().view(torch.float32)
    # Each FP32 scale applies to a contiguous group of 128 latent values.
    latent = latent.float().unflatten(-1, (4, 128)) * scales.unsqueeze(-1)
    latent = latent.flatten(-2).to(torch.bfloat16)
    out, maximum, total = sparse_flash_attention(
        query,
        latent,
        latent,
        sparse_indices,
        scale_value,
        block_table=block_table,
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        query_rope=query_rope,
        key_rope=rope,
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=0,
        return_softmax_lse=True,
    )
    return out, maximum.masked_fill(total == 0, 0), total
