"""Independent PyTorch reference for sgemmv_expand; no native operator calls."""

import torch

_SUPPORTED_RANKS = (0, 8, 16, 32, 64)


def _check_int32_vector(tensor, name, device):
    if tensor.device != device:
        raise RuntimeError("{} must be on the same device as y".format(name))
    if tensor.dtype != torch.int32:
        raise RuntimeError("{} must be int32".format(name))
    if tensor.dim() != 1:
        raise RuntimeError("{} must be 1D".format(name))
    if not tensor.is_contiguous():
        raise RuntimeError("{} must be contiguous".format(name))


def sgemmv_expand(x, weight, lora_indices, seq_len, lora_ranks, slice_offsets, y):
    if x.dim() != 2:
        raise RuntimeError("x must be 2D")
    if weight.dim() not in (3, 4):
        raise RuntimeError("weight must be 3D or 4D")
    if y.dim() != 2:
        raise RuntimeError("y must be 2D")
    if x.dtype != torch.float32:
        raise RuntimeError("x must be float32")
    if y.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError("y must be float16 or bfloat16")
    if weight.dtype != y.dtype:
        raise RuntimeError("weight and y must have the same dtype")
    if x.device != y.device or weight.device != y.device:
        raise RuntimeError("x, weight and y must be on the same device")
    if not (x.is_contiguous() and weight.is_contiguous() and y.is_contiguous()):
        raise RuntimeError("x, weight and y must be contiguous")
    for tensor, name in (
        (lora_indices, "lora_indices"),
        (seq_len, "seq_len"),
        (lora_ranks, "lora_ranks"),
        (slice_offsets, "slice_offsets"),
    ):
        _check_int32_vector(tensor, name, y.device)
    if x.shape[0] == 0:
        raise RuntimeError("batch size must be > 0")
    if slice_offsets.numel() < 2:
        raise RuntimeError("slice_offsets must contain at least two entries")
    if weight.dim() == 4:
        if weight.shape[1] != 1:
            raise RuntimeError(
                "4D weight must have shape [num_loras, 1, hidden_out, hidden_in]"
            )
        weight = weight.squeeze(1)

    slice_count = slice_offsets.numel() - 1
    token_offset = 0
    for block in range(seq_len.numel()):
        length = int(seq_len[block].item())
        if length == 0:
            continue
        lora_index = int(lora_indices[block].item())
        if lora_index < 0:
            token_offset += length
            continue
        rank = int(lora_ranks[lora_index].item())
        if rank == 0:
            token_offset += length
            continue
        if rank not in _SUPPORTED_RANKS:
            raise RuntimeError("unsupported LoRA rank: {}".format(rank))

        rows = slice(token_offset, token_offset + length)
        for index in range(slice_count):
            out_start = int(slice_offsets[index].item())
            out_end = int(slice_offsets[index + 1].item())
            # Each slice reads its own packed rank-sized segment of the row.
            x_slice = x[rows, index * rank : (index + 1) * rank]
            w_slice = weight[lora_index, out_start:out_end, :rank]
            # The Kernel accumulates in float32 and rounds once when writing y.
            delta = torch.matmul(x_slice.float(), w_slice.float().t())
            y[rows, out_start:out_end] = (
                y[rows, out_start:out_end].float() + delta
            ).to(y.dtype)
        token_offset += length

    return y
