"""Small-shape mathematical MLA reference, not a kernel rounding emulator."""

import torch


def flash_mla(
    q,
    k_cache,
    block_table=None,
    cache_seqlens=None,
    cu_seqlens_q=None,
    seqused_q=None,
    attn_mask=None,
    metadata=None,
    head_dim_v=512,
    softmax_scale=1.0,
    mask_mode=0,
    max_seqlen_q=-1,
    max_seqlen_kv=-1,
    layout_q="BSND",
    layout_kv="PA_BBND",
    layout_out="BSND",
    return_softmax_lse=False,
):
    """Compute softmax(scale * QK^T)V with FP32 arithmetic on the input device.

    Supports packed TND queries, single-head PA_BBND cache and no mask.
    Metadata is a backend scheduling artifact and does not affect the math.
    Host reads of lengths/page IDs make this a reference, not a graph-safe kernel.
    """
    if (
        layout_q != "TND"
        or layout_kv != "PA_BBND"
        or layout_out not in ("TND", "NTD")
        or mask_mode != 0
        or seqused_q is not None
        or attn_mask is not None
        or max_seqlen_q != -1
        or max_seqlen_kv != -1
    ):
        raise NotImplementedError(
            "torch_native Flash MLA supports unmasked TND/PA_BBND only"
        )
    if any(t is None for t in (block_table, cache_seqlens, cu_seqlens_q)):
        raise ValueError("block_table, cache_seqlens and cu_seqlens_q are required")
    if q.ndim != 3 or k_cache.ndim != 4 or k_cache.shape[2] != 1:
        raise ValueError("Expected q [T,N,D] and k_cache [P,page_size,1,D]")
    if q.shape[-1] != k_cache.shape[-1] or not 0 < head_dim_v <= q.shape[-1]:
        raise ValueError("Invalid QK/value dimensions")
    if (
        q.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or k_cache.dtype != q.dtype
    ):
        raise ValueError("Query/cache must have the same floating-point dtype")
    tensors = (k_cache, block_table, cache_seqlens, cu_seqlens_q)
    if any(t.device != q.device for t in tensors):
        raise ValueError("All attention inputs must be on the same device")
    if any(t.dtype not in (torch.int32, torch.int64) for t in tensors[1:]):
        raise ValueError("Page IDs and lengths must be integer tensors")
    if cache_seqlens.ndim != 1 or cu_seqlens_q.ndim != 1:
        raise ValueError("Sequence lengths must be one-dimensional")
    lengths = cache_seqlens.tolist()
    boundaries = cu_seqlens_q.tolist()
    if (
        not lengths
        or len(boundaries) != len(lengths) + 1
        or boundaries[0] != 0
        or boundaries[-1] != q.shape[0]
        or any(a >= b for a, b in zip(boundaries, boundaries[1:]))
    ):
        raise ValueError("Query boundaries must partition nonempty packed sequences")
    if block_table.ndim != 2 or block_table.shape[0] != len(lengths):
        raise ValueError("block_table batch size mismatch")
    page_size = k_cache.shape[1]
    if page_size <= 0:
        raise ValueError("Cache page size must be positive")
    outputs, lses = [], []
    for batch, length in enumerate(lengths):
        count = (length + page_size - 1) // page_size
        if length <= 0 or count > block_table.shape[1]:
            raise ValueError("Invalid KV length or insufficient page table")
        ids = block_table[batch, :count].to(torch.int64)
        if any(i < 0 or i >= k_cache.shape[0] for i in ids.tolist()):
            raise ValueError("Page ID is outside the cache")
        kv = k_cache.index_select(0, ids).reshape(-1, q.shape[-1])[:length].float()
        query = q[boundaries[batch] : boundaries[batch + 1]].transpose(0, 1).float()
        scores = torch.matmul(query, kv.T) * softmax_scale
        outputs.append(
            torch.matmul(torch.softmax(scores, dim=-1), kv[:, :head_dim_v]).transpose(
                0, 1
            )
        )
        lses.append(torch.logsumexp(scores, dim=-1))
    output = torch.cat(outputs, dim=0).to(q.dtype)
    if layout_out == "NTD":
        output = output.transpose(0, 1).contiguous()
    lse = (
        torch.cat(lses, dim=1)
        if return_softmax_lse
        else torch.empty(0, dtype=torch.float32, device=q.device)
    )
    return output, lse
