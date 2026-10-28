"""Independent PyTorch reference; no custom native operator calls."""

import torch


def apply_token_bitmask(logits, bitmask, indices=None):
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
    batch, vocab = logits.shape
    if vocab <= 0:
        raise RuntimeError("vocab_size must be > 0")
    has_indices = indices is not None and indices.numel() > 0
    if has_indices:
        if indices.dim() != 1:
            raise RuntimeError("indices must be 1D")
        if indices.device.type != "cpu" and indices.device != logits.device:
            raise RuntimeError("indices must be on CPU or the same device as logits")
        rows = indices.to(device=logits.device, dtype=torch.long).contiguous()
    else:
        rows = None
    if batch == 0 and not has_indices:
        return logits
    values = logits[rows] if has_indices else logits
    words = bitmask[rows] if has_indices else bitmask
    width = (vocab + 31) // 32
    if words.shape[1] < width:
        padding = words.new_zeros((values.shape[0], width - words.shape[1]))
        words = torch.cat((words, padding), dim=1)
    columns = torch.arange(vocab, device=logits.device)
    selected = torch.index_select(words, 1, columns // 32).to(torch.int64)
    allowed = ((selected >> (columns % 32)) & 1).bool()
    values.masked_fill_(~allowed, float("-inf"))
    if has_indices:
        logits[rows] = values
    return logits
