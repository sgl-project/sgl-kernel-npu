"""Pack FP8 values, BF16 RoPE bytes and per-group scales on the input device."""

import struct

import torch


def kv_compress_epilog(
    cache, x, slots, quant_group_size, quant_mode, round_scale_flag, layout
):
    cache = cache.squeeze(2) if cache.ndim == 4 else cache
    device = x.device
    n, d = x.shape
    qcols = d - 64
    group = quant_group_size if quant_mode == 2 else 128
    groups = (qcols + group - 1) // group
    max_value = 448.0 if cache.dtype == torch.float8_e4m3fn else 57344.0
    if quant_mode == 2:
        coeff = torch.tensor(1.0 / max_value, device=device, dtype=torch.float32)
    else:
        bits = 0x3B124925 if max_value == 448.0 else 0x37924925
        coeff = struct.unpack("<f", struct.pack("<I", bits))[0]
    xf = x.float()
    scales = torch.empty((n, groups), dtype=torch.float32, device=device)
    quant = torch.empty((n, qcols), dtype=torch.uint8, device=device)
    for j in range(groups):
        block = xf[:, j * group : min((j + 1) * group, qcols)]
        maximum = block.abs().amax(dim=1)
        if quant_mode == 2:
            scale = maximum.clamp(min=1e-4) * coeff
            exponent = (scale.view(torch.int32) >> 23) & 255
            if round_scale_flag:
                exponent = exponent + (exponent != 0).to(torch.int32)
            scale = (exponent << 23).view(torch.float32)
        else:
            scale = maximum * coeff
        scales[:, j] = scale
        # A zero FP8 group keeps zero values while storing its zero scale.
        normalized = torch.where(scale[:, None] == 0, block, block / scale[:, None])
        quant[:, j * group : min((j + 1) * group, qcols)] = (
            normalized.clamp(-max_value, max_value).to(cache.dtype).view(torch.uint8)
        )
    rope = x[:, -64:].contiguous().view(torch.uint8)
    scale_bytes = (
        ((scales.view(torch.int32) >> 23) & 255).to(torch.uint8)
        if quant_mode == 2
        else scales.view(torch.uint8)
    )
    valid = slots != -1
    indices = slots[valid].long()
    if layout == 2:
        block_size = cache.shape[1]
        value_width = qcols + 128
        scale_width = (groups + 7) // 8 * 8
        values = torch.cat((quant, rope), dim=1)
        scale_rows = torch.zeros((n, scale_width), dtype=torch.uint8, device=device)
        scale_rows[:, :groups] = scale_bytes
        byte = cache.view(torch.uint8).as_strided(
            (cache.shape[0], block_size * (value_width + scale_width)),
            (cache.stride(0), 1),
        )
        block_ids, row_ids = indices // block_size, indices % block_size
        byte[
            block_ids[:, None],
            row_ids[:, None] * value_width + torch.arange(value_width, device=device),
        ] = values[valid]
        byte[
            block_ids[:, None],
            block_size * value_width
            + row_ids[:, None] * scale_width
            + torch.arange(scale_width, device=device),
        ] = scale_rows[valid]
    else:
        used = qcols + 128 + scale_bytes.shape[1]
        width = (used + 127) // 128 * 128
        packed = torch.zeros((n, width), dtype=torch.uint8, device=device)
        if quant_mode == 2:
            packed[:, :128] = rope
            packed[:, 128 : 128 + qcols] = quant
        else:
            packed[:, :qcols] = quant
            packed[:, qcols : qcols + 128] = rope
        packed[:, qcols + 128 : used] = scale_bytes
        stride = cache.stride(0) if layout == 1 else width
        byte = cache.view(torch.uint8).as_strided((cache.shape[0], width), (stride, 1))
        byte[indices] = packed[valid]
