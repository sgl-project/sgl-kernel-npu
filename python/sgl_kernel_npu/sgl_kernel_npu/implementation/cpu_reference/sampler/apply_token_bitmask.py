"""Bit-by-bit CPU reference migrated from the legacy bitmask test."""

import torch


def apply_token_bitmask(logits, bitmask, indices=None):
    """Compute on CPU, then preserve the input logits' device and in-place alias."""
    if logits.dim() != 2 or bitmask.dim() != 2:
        raise RuntimeError("logits and bitmask must be 2D")
    if logits.shape[0] != bitmask.shape[0]:
        raise RuntimeError("logits and bitmask batch size mismatch")
    if not logits.is_contiguous() or not bitmask.is_contiguous():
        raise RuntimeError("logits and bitmask must be contiguous")
    if logits.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise RuntimeError("logits must be float32, float16 or bfloat16")
    if bitmask.dtype != torch.int32 or bitmask.device != logits.device:
        raise RuntimeError("bitmask must be int32 on the same device as logits")
    vocab_size = logits.shape[1]
    if vocab_size <= 0:
        raise RuntimeError("vocab_size must be > 0")

    # Match the public contract: None and empty indices both select all rows.
    if indices is not None and indices.numel() > 0:
        if indices.dim() != 1:
            raise RuntimeError("indices must be 1D")
        if indices.device.type != "cpu" and indices.device != logits.device:
            raise RuntimeError("indices must be on CPU or the same device as logits")
        rows = indices.to(device="cpu", dtype=torch.int64).tolist()
    else:
        rows = range(logits.shape[0])

    logits_cpu = logits.cpu()
    bitmask_cpu = bitmask.cpu()
    for row in rows:
        bm = bitmask_cpu[row]
        for bm_i in range((vocab_size + 31) // 32):
            # Missing mask words are zero, as in the public AscendC interface.
            bm_val = bm[bm_i].item() if bm_i < bm.shape[0] else 0
            start_col = bm_i * 32
            end_col = min(start_col + 32, vocab_size)
            for bit in range(end_col - start_col):
                if not ((bm_val >> bit) & 1):
                    logits_cpu[row, start_col + bit] = float("-inf")

    if logits.device.type != "cpu":
        logits.copy_(logits_cpu)
    return logits
