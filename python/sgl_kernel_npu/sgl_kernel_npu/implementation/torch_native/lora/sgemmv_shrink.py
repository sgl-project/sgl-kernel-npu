"""Independent PyTorch reference for sgemmv_shrink; no native operator calls."""

import torch

_SUPPORTED_RANKS = (0, 8, 16, 32, 64)


def _check_vector(tensor, name, device, dtype):
    if tensor.device != device:
        raise RuntimeError("{} must be on the same device as y".format(name))
    if tensor.dtype != dtype:
        raise RuntimeError("{} must be {}".format(name, dtype))
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
    if y.dtype != torch.float32:
        raise RuntimeError("y must be float32")
    if weight.dtype != x.dtype:
        raise RuntimeError("weight and x must have the same dtype")
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


def sgemmv_shrink(
    x, weight, lora_indices, seq_len, lora_ranks, lora_scales, y, slice_count=1
):
    _check_common(x, weight, y, slice_count)
    for tensor, name in (
        (lora_indices, "lora_indices"),
        (seq_len, "seq_len"),
        (lora_ranks, "lora_ranks"),
    ):
        _check_vector(tensor, name, y.device, torch.int32)
    _check_vector(lora_scales, "lora_scales", y.device, torch.float16)
    weight = _squeeze_weight(weight)

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
        scale = float(lora_scales[lora_index].item())

        rows = slice(token_offset, token_offset + length)
        x_rows = x[rows].float()
        for index in range(slice_count):
            # Each slice owns a rank-sized segment of the row, both in x and in
            # the per-adapter weight block.
            w_rows = weight[lora_index, index * rank : (index + 1) * rank, :].float()
            # Dot products and scaling stay in float32; y is float32 as well.
            y[rows, index * rank : index * rank + rank] = (
                torch.matmul(x_rows, w_rows.t()) * scale
            )
        token_offset += length
    return y
