"""Independent PyTorch reference for packed MLA sparse attention."""

import torch

from ..utils.sparse_attention import sparse_flash_attention


def kv_quant_sparse_flash_attention(
    query,
    query_rope,
    packed_kv,
    sparse_indices,
    block_table,
    actual_seq_lengths_query,
    actual_seq_lengths_kv,
    scale_value,
    *,
    return_softmax_lse=True,
):
    """Decode the packed MLA record and round the weights before multiplying by V."""
    packed = packed_kv.detach().cpu()
    latent_bytes = 512
    rope_end = latent_bytes + query_rope.shape[-1] * 2
    latent = packed[..., :latent_bytes].contiguous()
    if packed.dtype == torch.uint8:
        latent = latent.view(torch.float8_e4m3fn)
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
        return_softmax_lse=return_softmax_lse,
        round_weights_to_bf16=True,
    )
    return out, maximum.masked_fill(total == 0, 0), total
