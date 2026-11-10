"""Adler-32 over indexed, strided device buffers using Triton-Ascend.

The byte stream is ordered by descriptor, then by the supplied item indices.
Only the reduced checksum is copied to the host, not the KV cache itself.
"""

import torch
import triton
import triton.language as tl

_MOD_ADLER = 65521
# 255 * BLOCK * (BLOCK + 1) // 2 fits in a signed int32.
_BLOCK_SIZE = 4096


@triton.jit(do_not_specialize=["remaining_mod"])
def _adler32_partial_kernel(
    base,
    indices,
    partials,
    stride,
    tiles_per_item,
    remaining_mod,
    MOD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    item = pid // tiles_per_item
    start = (pid % tiles_per_item).to(tl.int64) * BLOCK
    row = tl.load(indices + item).to(tl.int64)
    ptr = base.to(tl.pointer_type(tl.uint8))
    lane = tl.arange(0, BLOCK)
    valid_bytes = tl.minimum(stride - start, BLOCK).to(tl.int32)
    tile = ptr + row * stride + start
    values = tl.load(tile + lane, mask=lane < valid_bytes, other=0)
    values = values.to(tl.int32)

    # Split the stream weight (N - j) into a tile constant and a local
    # weight (BLOCK - lane). Masked tail bytes contribute zero to both sums.
    # Only the reduced scalars need modular arithmetic.
    a = (tl.sum(values, 0) % MOD).to(tl.uint32)
    b = (tl.sum(values * (BLOCK - lane), 0) % MOD).to(tl.uint32)
    logical_start = item.to(tl.int64) * stride + start
    weight = (remaining_mod - logical_start % MOD - BLOCK + 2 * MOD) % MOD
    # (MOD - 1)**2 + MOD - 1 fits in uint32.
    b = (b + a * weight.to(tl.uint32)) % MOD
    output = partials + pid.to(tl.int64) * 2
    tl.store(output, a)
    tl.store(output + 1, b)


def adler32_strided_checksum(
    data_ptrs: list[int],
    strides: list[int],
    indices: list[torch.Tensor],
) -> int:
    """Match the CUDA strided checksum's ordered raw-byte semantics.

    The KV pool owns the pointer lifetimes and index bounds.
    Returning the Python integer synchronizes the device result.
    """
    if not (len(data_ptrs) == len(strides) == len(indices)):
        raise ValueError("data_ptrs, strides and indices must have the same length")
    if not indices:
        return 1
    device = indices[0].device
    if device.type != "npu":
        raise ValueError("NPU checksum requires NPU index tensors")

    plans = []
    total_bytes = 0
    total_tiles = 0
    for base, stride, idx in zip(data_ptrs, strides, indices):
        if idx.device != device or idx.ndim != 1 or not idx.is_contiguous():
            raise ValueError("indices must be contiguous 1D tensors on the same NPU")
        if idx.dtype not in (torch.int32, torch.int64):
            raise ValueError("indices must have int32 or int64 dtype")
        if stride < 0 or (stride > 0 and idx.numel() > 0 and base <= 0):
            raise ValueError("nonempty descriptors require a valid pointer and stride")
        if stride and idx.numel():
            tiles_per_item = triton.cdiv(stride, _BLOCK_SIZE)
            tiles = idx.numel() * tiles_per_item
            plans.append(
                (base, stride, idx, total_bytes, total_tiles, tiles_per_item, tiles)
            )
            total_bytes += idx.numel() * stride
            total_tiles += tiles

    if total_bytes == 0:
        return 1
    partials = torch.empty((total_tiles, 2), dtype=torch.int32, device=device)
    for base, stride, idx, byte_offset, tile_offset, tiles_per_item, tiles in plans:
        _adler32_partial_kernel[(tiles,)](
            base,
            idx,
            partials[tile_offset:],
            stride,
            tiles_per_item,
            (total_bytes - byte_offset) % _MOD_ADLER,
            MOD=_MOD_ADLER,
            BLOCK=_BLOCK_SIZE,
        )

    sums = partials.sum(dim=0, dtype=torch.int64).cpu().tolist()
    a = (1 + sums[0]) % _MOD_ADLER
    b = (total_bytes + sums[1]) % _MOD_ADLER
    return (b << 16) | a
