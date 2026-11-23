"""Independent reference for TorchNPU per-tile quantized MLA caches."""

import torch

from ..utils.sparse_attention import sparse_flash_attention


def _unpack(cache, rope_dtype, tile_size, rope_dim):
    rows = cache.detach().view(torch.uint8).cpu()
    latent_size = 512
    scale_offset = latent_size + rope_dim * 2
    if rows.size(-1) != scale_offset + 4 * (latent_size // tile_size):
        raise ValueError("Invalid packed KV width for the given tile size")
    data = rows[..., :latent_size].contiguous().view(cache.dtype).float()
    rope = rows[..., latent_size:scale_offset].contiguous().view(rope_dtype)
    scales = rows[..., scale_offset:].contiguous().view(torch.float32)
    values = data * scales.repeat_interleave(tile_size, dim=-1)
    return values.to(rope_dtype), rope


def kv_quant_sparse_flash_attention_torch_npu(
    query,
    key,
    value,
    sparse_indices,
    scale_value,
    key_quant_mode,
    value_quant_mode,
    *,
    key_dequant_scale=None,
    value_dequant_scale=None,
    block_table=None,
    actual_seq_lengths_query=None,
    actual_seq_lengths_kv=None,
    sparse_block_size=1,
    layout_query="BSND",
    layout_kv="PA_BSND",
    sparse_mode=3,
    pre_tokens=2**63 - 1,
    next_tokens=2**63 - 1,
    attention_mode=2,
    quant_scale_repo_mode=1,
    tile_size=128,
    rope_head_dim=64,
):
    if (
        (key_quant_mode, value_quant_mode, attention_mode, quant_scale_repo_mode)
        != (2, 2, 2, 1)
        or key_dequant_scale is not None
        or value_dequant_scale is not None
        or (pre_tokens, next_tokens) != (2**63 - 1, 2**63 - 1)
        or layout_kv != "PA_BSND"
        or layout_query not in ("TND", "BSND")
        or tile_size not in (64, 128)
        or rope_head_dim != 64
        or key.dtype not in (torch.int8, torch.float8_e4m3fn)
        or key.dtype != value.dtype
        or query.dtype not in (torch.float16, torch.bfloat16)
        or key.size(-2) != 1
    ):
        raise NotImplementedError("torch_native requires packed INT8/FP8 paged MLA KV")
    keys, key_rope = _unpack(key, query.dtype, tile_size, rope_head_dim)
    values = (
        keys
        if value is key
        else _unpack(value, query.dtype, tile_size, rope_head_dim)[0]
    )
    return sparse_flash_attention(
        query,
        keys,
        values,
        sparse_indices,
        scale_value,
        block_table=block_table,
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        key_rope=key_rope,
        sparse_block_size=sparse_block_size,
        layout_query=layout_query,
        layout_kv=layout_kv,
        sparse_mode=sparse_mode,
        attention_mode=attention_mode,
    )[0]
