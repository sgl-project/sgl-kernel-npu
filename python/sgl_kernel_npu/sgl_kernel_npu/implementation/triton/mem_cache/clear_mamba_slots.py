import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["num_idx", "layer_stride", "slot_stride", "feat_len"])
def _clear_mamba_slots_1d_kernel(
    BUF_ptr,
    INDICES_ptr,
    num_idx,
    layer_stride,
    slot_stride,
    feat_len,
    NUM_LAYERS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    n_progs = tl.num_programs(0)
    n_blocks = (feat_len + BLOCK - 1) // BLOCK
    total_tasks = num_idx * NUM_LAYERS * n_blocks
    zeros = tl.zeros((BLOCK,), dtype=BUF_ptr.dtype.element_ty)
    offs = tl.arange(0, BLOCK).to(tl.int64)
    ls = layer_stride.to(tl.int64)
    ss = slot_stride.to(tl.int64)
    fl = feat_len.to(tl.int64)
    for task in range(pid, total_tasks, n_progs):
        blk = task % n_blocks
        rem = task // n_blocks
        lid = (rem % NUM_LAYERS).to(tl.int64)
        iid = rem // NUM_LAYERS
        slot = tl.load(INDICES_ptr + iid).to(tl.int64)
        cols = (blk * BLOCK).to(tl.int64) + offs
        base = BUF_ptr + lid * ls + slot * ss
        tl.store(base + cols, zeros, mask=cols < fl)


def clear_mamba_slots(buf: torch.Tensor, indices: torch.Tensor) -> None:
    """Zero dim-1 slots across all layers on Ascend NPU using 64-bit addressing."""
    if buf.ndim < 2:
        raise ValueError(f"buf must have at least 2 dimensions, got {buf.ndim}")
    if indices.ndim != 1:
        raise ValueError(f"indices must be 1D, got {indices.ndim}D")
    if indices.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"indices must be int32 or int64, got {indices.dtype}")
    if buf.device != indices.device:
        raise ValueError(
            f"buf and indices must be on the same device, got {buf.device} and {indices.device}"
        )
    if not indices.is_contiguous():
        raise ValueError("indices must be contiguous for the Triton backend")
    num_idx = int(indices.numel())
    if num_idx == 0 or buf.numel() == 0:
        return
    if not buf[0, 0].is_contiguous():
        raise ValueError(
            "per-slot slice buf[0, 0] must be contiguous for the Triton backend"
        )
    num_layers = int(buf.shape[0])
    feat_len = int(buf[0, 0].numel())
    block = 8192
    n_blocks = (feat_len + block - 1) // block
    grid = (min(56, max(1, num_idx * num_layers * n_blocks)),)
    _clear_mamba_slots_1d_kernel[grid](
        buf,
        indices,
        num_idx,
        int(buf.stride(0)),
        int(buf.stride(1)),
        feat_len,
        NUM_LAYERS=num_layers,
        BLOCK=block,
    )
