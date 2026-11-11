"""Pure PyTorch reference for the AdaBlockSparseAttention operator."""

import torch


def _expand_block_mask(sparse_mask, sparse_size, seq_q, seq_kv):
    """Expand [B, N, S1b, S2b] block flags to [B, N, S1, S2] element flags."""
    blocks = sparse_mask[
        :,
        :,
        : (seq_q + sparse_size - 1) // sparse_size,
        : (seq_kv + sparse_size - 1) // sparse_size,
    ]
    expanded = blocks.bool()
    expanded = expanded.repeat_interleave(sparse_size, dim=2)[:, :, :seq_q, :]
    expanded = expanded.repeat_interleave(sparse_size, dim=3)[:, :, :, :seq_kv]
    return expanded


def ada_block_sparse_attention(
    query,
    key,
    value,
    sparse_mask,
    sparse_count_table,
    *,
    input_layout="BNSD",
    sparse_size=128,
    num_heads=None,
    num_key_value_heads=None,
    scale_value=1.0,
    causal=True,
    inner_precise=1,
    pre_tokens=214748647,
    next_tokens=0,
    actual_seq_lengths=None,
    actual_seq_lengths_kv=None,
):
    """Compute block-sparse attention for BNSD tensors.

    Every enabled sparse block contributes all of its KV positions; causal only
    trims the diagonal block, matching the Kernel. Rows without any enabled
    position produce zeros, matching the Kernel's empty-row branch.
    """
    if input_layout != "BNSD":
        raise ValueError("ada_block_sparse_attention torch_native supports BNSD only")
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        raise ValueError("query/key/value must be 4D [B, N, S, D]")
    if sparse_size <= 0:
        raise ValueError("sparse_size must be positive")
    batch, heads, seq_q, head_dim = query.shape
    kv_heads, seq_kv, value_dim = key.shape[1], key.shape[2], value.shape[3]
    if key.shape[0] != batch or value.shape[0] != batch:
        raise ValueError("query/key/value must share the batch dimension")
    if key.shape[3] != head_dim or value.shape[2] != seq_kv:
        raise ValueError("key/value shapes must be [B, Nkv, S, D]")
    if value_dim != head_dim:
        raise NotImplementedError("torch_native requires equal value head width")
    if heads % kv_heads != 0:
        raise ValueError("query heads must be a multiple of key heads")
    if num_heads is not None and num_heads != heads:
        raise ValueError("num_heads must match query.size(1)")
    if num_key_value_heads is not None and num_key_value_heads not in (0, kv_heads):
        raise ValueError("num_key_value_heads must match key.size(1)")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("query/key/value must be float16 or bfloat16")
    if sparse_count_table.dtype != torch.int32:
        raise ValueError("sparse_count_table must be int32")
    if sparse_mask.dim() != 4 or sparse_count_table.dim() != 3:
        raise ValueError("sparse_mask must be 4D and sparse_count_table 3D")
    if actual_seq_lengths is not None or actual_seq_lengths_kv is not None:
        raise NotImplementedError("torch_native does not support varlen sequences")
    # next_tokens == 0 means unrestricted on the next side (legacy default).
    unrestricted_window = pre_tokens >= seq_kv and (
        next_tokens >= seq_kv or next_tokens == 0
    )
    if not unrestricted_window:
        raise NotImplementedError("torch_native supports the unrestricted window only")

    group = heads // kv_heads
    q = query.float()
    k = key.float().repeat_interleave(group, dim=1)
    v = value.float().repeat_interleave(group, dim=1)
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale_value

    allowed = _expand_block_mask(sparse_mask, sparse_size, seq_q, seq_kv)
    if causal:
        # Match the Kernel: causal trims only the diagonal block when it is the last enabled block.
        blocks_q = (seq_q + sparse_size - 1) // sparse_size
        blocks_kv = (seq_kv + sparse_size - 1) // sparse_size
        blocks = sparse_mask[:, :, :blocks_q, :blocks_kv].bool()
        has_block = blocks.any(dim=-1)
        last_block = (blocks_kv - 1) - blocks.flip(-1).to(torch.int32).argmax(dim=-1)
        last_block = torch.where(has_block, last_block, torch.full_like(last_block, -1))
        diag_is_last = has_block & (
            last_block == torch.arange(blocks_q, device=query.device)
        )
        diag_is_last = diag_is_last.repeat_interleave(sparse_size, dim=2)[:, :, :seq_q]
        positions = torch.arange(seq_q, device=query.device).view(-1, 1)
        kv_positions = torch.arange(seq_kv, device=query.device).view(1, -1)
        diagonal = (positions // sparse_size) == (kv_positions // sparse_size)
        trim = diag_is_last.unsqueeze(-1) & diagonal
        allowed = allowed & (~trim | (kv_positions <= positions))
    has_any = allowed.any(dim=-1, keepdim=True)
    scores = scores.masked_fill(~allowed, float("-inf"))
    scores = torch.where(has_any, scores, torch.zeros_like(scores))
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v)
    out = torch.where(has_any, out, torch.zeros_like(out))
    return out.to(query.dtype)
