"""Independent PyTorch reference for sgemmc_expand; no native operator calls."""

import torch

_SUPPORTED_RANKS = (0, 8, 16, 32, 64)


def _check_vector(tensor, name, device):
    if tensor.device != device:
        raise RuntimeError("{} must be on the same device as y".format(name))
    if tensor.dtype != torch.int32:
        raise RuntimeError("{} must be int32".format(name))
    if tensor.dim() != 1:
        raise RuntimeError("{} must be 1D".format(name))
    if not tensor.is_contiguous():
        raise RuntimeError("{} must be contiguous".format(name))


def _check_common(x, weight, y):
    if x.dim() != 2:
        raise RuntimeError("x must be 2D")
    if weight.dim() not in (3, 4):
        raise RuntimeError("weight must be 3D or 4D")
    if y.dim() != 2:
        raise RuntimeError("y must be 2D")
    if y.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError("y must be float16 or bfloat16")
    if x.dtype != y.dtype:
        raise RuntimeError("x and y must have the same dtype")
    if weight.dtype != y.dtype:
        raise RuntimeError("weight and y must have the same dtype")
    if x.device != y.device or weight.device != y.device:
        raise RuntimeError("x, weight and y must be on the same device")
    if not (x.is_contiguous() and weight.is_contiguous() and y.is_contiguous()):
        raise RuntimeError("x, weight and y must be contiguous")
    if x.shape[0] == 0:
        raise RuntimeError("batch size must be > 0")


def _squeeze_weight(weight):
    if weight.dim() == 4:
        if weight.shape[1] != 1:
            raise RuntimeError(
                "4D weight must have shape [num_loras, 1, hidden_out, hidden_in]"
            )
        weight = weight.squeeze(1)
    return weight


def sgemmc_expand(x, weight, lora_indices, seq_len, lora_ranks, slice_offsets, y):
    _check_common(x, weight, y)
    for tensor, name in (
        (lora_indices, "lora_indices"),
        (seq_len, "seq_len"),
        (lora_ranks, "lora_ranks"),
        (slice_offsets, "slice_offsets"),
    ):
        _check_vector(tensor, name, y.device)
    if slice_offsets.numel() < 2:
        raise RuntimeError("slice_offsets must contain at least two entries")
    weight = _squeeze_weight(weight)

    slice_count = slice_offsets.numel() - 1
    token_offset = 0
    for block in range(seq_len.numel()):
        length = int(seq_len[block].item())
        lora_index = int(lora_indices[block].item())
        rank = int(lora_ranks[lora_index].item()) if lora_index >= 0 else 0
        if lora_index >= 0 and rank not in _SUPPORTED_RANKS:
            raise RuntimeError("unsupported LoRA rank: {}".format(rank))
        if length > 0 and lora_index >= 0 and rank != 0:
            rows = slice(token_offset, token_offset + length)
            x_rows = x[rows].float()
            for index in range(slice_count):
                out_start = int(slice_offsets[index].item())
                out_end = int(slice_offsets[index + 1].item())
                # Slice index selects a rank-sized segment of the packed input
                # and the matching block of output rows of this adapter.
                x_slice = x_rows[:, index * rank : (index + 1) * rank]
                w_rows = weight[lora_index, out_start:out_end, :rank].float()
                delta = torch.matmul(x_slice, w_rows.t())
                # The Kernel adds in float32 and rounds once when storing y.
                y[rows, out_start:out_end] = (
                    y[rows, out_start:out_end].float() + delta
                ).to(y.dtype)
        token_offset += length
    return y
