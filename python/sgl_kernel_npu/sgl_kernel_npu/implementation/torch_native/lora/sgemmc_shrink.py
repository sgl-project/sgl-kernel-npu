"""Independent PyTorch reference for sgemmc_shrink; no native operator calls."""

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


def _check_common(x, weight, y, slice_count):
    if x.dim() != 2:
        raise RuntimeError("x must be 2D")
    if weight.dim() not in (3, 4):
        raise RuntimeError("weight must be 3D or 4D")
    if y.dim() != 2:
        raise RuntimeError("y must be 2D")
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError("x must be float16 or bfloat16")
    if x.dtype != y.dtype:
        raise RuntimeError("x and y must have the same dtype")
    if weight.dtype != y.dtype:
        raise RuntimeError("weight and y must have the same dtype")
    if x.device != y.device or weight.device != y.device:
        raise RuntimeError("x, weight and y must be on the same device")
    if not (x.is_contiguous() and weight.is_contiguous() and y.is_contiguous()):
        raise RuntimeError("x, weight and y must be contiguous")
    if x.shape[1] <= y.shape[1]:
        raise RuntimeError("hidden in should be greater than hidden out")
    if x.shape[0] == 0:
        raise RuntimeError("batch size must be > 0")
    if slice_count < 1:
        raise RuntimeError("slice_count must be >= 1")


def _squeeze_weight(weight):
    if weight.dim() == 4:
        if weight.shape[1] != 1:
            raise RuntimeError(
                "4D weight must have shape [num_loras, 1, hidden_out, hidden_in]"
            )
        weight = weight.squeeze(1)
    return weight


def sgemmc_shrink(
    x, weight, lora_indices, seq_len, lora_ranks, lora_scales, y, slice_count
):
    _check_common(x, weight, y, slice_count)
    for tensor, name in (
        (lora_indices, "lora_indices"),
        (seq_len, "seq_len"),
        (lora_ranks, "lora_ranks"),
    ):
        _check_vector(tensor, name, y.device)
    if lora_scales.device != y.device:
        raise RuntimeError("lora_scales must be on the same device as y")
    if lora_scales.dtype != torch.float16:
        raise RuntimeError("lora_scales must be float16")
    if lora_scales.dim() != 1:
        raise RuntimeError("lora_scales must be 1D")
    weight = _squeeze_weight(weight)

    token_offset = 0
    for block in range(seq_len.numel()):
        length = int(seq_len[block].item())
        lora_index = int(lora_indices[block].item())
        rank = int(lora_ranks[lora_index].item()) if lora_index >= 0 else 0
        if lora_index >= 0 and rank not in _SUPPORTED_RANKS:
            raise RuntimeError("unsupported LoRA rank: {}".format(rank))
        if length > 0 and lora_index >= 0 and rank != 0:
            scale = float(lora_scales[lora_index].item())
            rows = slice(token_offset, token_offset + length)
            x_rows = x[rows].float()
            for index in range(slice_count):
                # Slice index selects a rank-sized weight segment for this
                # adapter and the matching rank-sized output columns.
                w_rows = weight[
                    lora_index, index * rank : (index + 1) * rank, :
                ].float()
                scaled = torch.matmul(x_rows, w_rows.t()) * scale
                # The Kernel scales in float32 and rounds once when storing y.
                y[rows, index * rank : index * rank + rank] = scaled.to(y.dtype)
        token_offset += length
    return y
