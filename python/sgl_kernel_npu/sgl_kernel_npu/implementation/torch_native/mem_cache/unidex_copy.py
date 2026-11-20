import torch


def unidex_copy_inplace(
    src,
    dst,
    src_index,
    dst_index,
    valid_mask,
    src_rows,
    dst_rows,
    block_bytes,
    block_dim,
    src_ptr,
    dst_ptr,
):
    if src_ptr is not None or dst_ptr is not None:
        raise NotImplementedError(
            "raw address overrides require the ascendc implementation"
        )
    for tensor, name in (
        (src, "src"),
        (dst, "dst"),
        (src_index, "src_index"),
        (dst_index, "dst_index"),
        (valid_mask, "valid_mask"),
    ):
        if tensor.device != src.device:
            raise RuntimeError(f"{name} must be on the same device as src")
        if not tensor.is_contiguous():
            raise RuntimeError(f"{name} must be contiguous")
    for tensor, name in (
        (src_index, "src_index"),
        (dst_index, "dst_index"),
        (valid_mask, "valid_mask"),
    ):
        if tensor.ndim != 1:
            raise RuntimeError(f"{name} must be 1-D")
    if src_index.dtype != torch.int64 or dst_index.dtype != torch.int64:
        name = "src_index" if src_index.dtype != torch.int64 else "dst_index"
        raise RuntimeError(f"{name} must be int64")
    if valid_mask.dtype not in (torch.bool, torch.uint8):
        raise RuntimeError("valid_mask must be bool or uint8")
    if not 0 < block_bytes <= 32 * 1024:
        raise RuntimeError("block_bytes must be positive and at most 32 KiB")
    if min(src_rows, dst_rows, block_dim) <= 0:
        raise RuntimeError("row counts and block_dim must be positive")
    if (
        max(
            src_rows * block_bytes, dst_rows * block_bytes, src_index.numel(), block_dim
        )
        >= 2**32
    ):
        raise RuntimeError("copy geometry exceeds the uint32 range")
    if src_index.numel() == 0:
        return dst

    valid = (
        (valid_mask != 0)
        & (src_index >= 0)
        & (src_index < src_rows)
        & (dst_index >= 0)
        & (dst_index < dst_rows)
    )
    selected = valid.nonzero(as_tuple=True)[0]
    if selected.numel():
        dst.view(dst_rows, -1)[dst_index[selected]] = src.view(src_rows, -1)[
            src_index[selected]
        ]
    return dst
