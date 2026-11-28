"""Independent PyTorch backend for kpool_decode_update_and_maybe_write_cache."""

from typing import Optional

import torch

from ..utils.hadamard import hadamard128


@torch.no_grad()
def kpool_decode_update_and_maybe_write_cache(
    buf: torch.Tensor,
    tail_k: torch.Tensor,
    tail_score: torch.Tensor,
    key: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
    block_tables: torch.Tensor,
    req_pool_indices: torch.Tensor,
    positions: torch.Tensor,
    seq_lens: torch.Tensor,
    out_cache_loc: torch.Tensor,
    *,
    pool_size: int,
    slots_per_page: int,
    head_dim: int = 128,
    scale_cache: Optional[torch.Tensor] = None,
):
    """Pure PyTorch reference with the BF16/FP8/INT8 kernel semantics."""
    assert (scale_cache is not None) == (buf.dtype in (torch.int8, torch.float8_e4m3fn))
    if scale_cache is not None:
        assert scale_cache.dtype == (
            torch.float16 if buf.dtype == torch.int8 else torch.float32
        )
    if buf.dtype == torch.float8_e4m3fn:
        assert head_dim == 128, "FP8 decode requires head_dim=128"
    batch = key.shape[0]
    if batch == 0:
        return
    tail_size = tail_k.shape[1]
    for row in range(batch):
        req_raw = req_pool_indices[row].item()
        pos = positions[row].item()
        seq_len = seq_lens[row].item()
        cache_loc = out_cache_loc[row].item()
        req_valid = req_raw >= 0 and req_raw < tail_k.shape[0]
        pos_valid = req_valid and cache_loc != 0 and (pos >= 0) and (pos < seq_len)
        req = max(0, min(req_raw, tail_k.shape[0] - 1))
        slot = pos % pool_size
        phys_slot = pos % tail_size
        cur_key = key[row].float()
        cur_score = slot_score[row].float()
        if pos_valid and slot == pool_size - 1:
            pool_logical_start = pos - slot
            scores_list = []
            keys_list = []
            for pool_slot in range(pool_size):
                is_current = pool_slot == slot
                phys = (pool_logical_start + pool_slot) % tail_size
                if is_current:
                    score = cur_score
                    current_key = cur_key
                else:
                    score = tail_score[req, phys].float()
                    current_key = tail_k[req, phys].float()
                scores_list.append(score + ape[pool_slot].float())
                keys_list.append(current_key)
            all_scores = torch.stack(scores_list)
            all_keys = torch.stack(keys_list)
            max_score = all_scores.max(dim=0).values
            probs = torch.exp(all_scores - max_score.unsqueeze(0))
            denom = probs.sum(dim=0)
            acc = (all_keys * probs).sum(dim=0)
            compressed = acc / denom
            pool_id = pos // pool_size
            pool_page_group = pool_id // slots_per_page
            token_page_row = pool_page_group * pool_size
            token_page_row = max(0, min(token_page_row, block_tables.shape[1] - 1))
            packed_page = block_tables[row, token_page_row].item()
            loc_token_offset = pool_id % slots_per_page
            out_offset = (packed_page // pool_size) * buf.stride(
                0
            ) + loc_token_offset * head_dim
            if scale_cache is None:
                buf.view(-1)[out_offset : out_offset + head_dim] = compressed.to(
                    torch.bfloat16
                )
            elif buf.dtype == torch.float8_e4m3fn:
                compressed = compressed.to(torch.bfloat16).float()
                compressed = hadamard128(compressed).to(torch.bfloat16).float()
                scale = compressed.abs().amax().clamp_min(1e-4) / 448.0
                payload = (compressed / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
                buf.view(-1)[out_offset : out_offset + head_dim] = payload
                scale_cache.view(-1)[out_offset // head_dim] = scale
            else:
                amax = compressed.abs().amax()
                scale = amax / 127.0 if amax > 0 else torch.ones_like(amax)
                scaled = compressed / scale
                rounded = scaled + torch.where(scaled >= 0, 0.5, -0.5)
                quantized = rounded.trunc().clamp(-127, 127).to(torch.int8)
                buf.view(-1)[out_offset : out_offset + head_dim] = quantized
                scale_cache.view(-1)[out_offset // head_dim] = scale.to(torch.float16)
        if pos_valid:
            tail_k[req, phys_slot] = key[row]
            tail_score[req, phys_slot] = slot_score[row]
