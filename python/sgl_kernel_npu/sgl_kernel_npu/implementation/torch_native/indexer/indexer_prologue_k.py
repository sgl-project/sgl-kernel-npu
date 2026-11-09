# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE.cannbot-dsl under implementation/cannbot-dsl for the full text of the License.

"""Independent CPU Torch K projection, RoPE, MXFP4 encoding and cache update."""

import torch


def _pack_mxfp4(key):
    """Return signed-E2M1 packed data and UE8M0 scale bytes."""
    t, d = key.shape
    groups = key.float().view(t, d // 32, 32)
    amax = groups.abs().amax(dim=-1).clamp_min(6.0 * (2.0**-126))
    scale = torch.exp2(torch.ceil(torch.log2(amax / 6.0)))
    magnitude = (groups.abs() / scale.unsqueeze(-1)).clamp_max(6.0)
    code = torch.full_like(magnitude, 7, dtype=torch.int64)
    for threshold, lower_code in (
        (5.0, 6),
        (3.5, 5),
        (2.5, 4),
        (1.75, 3),
        (1.25, 2),
        (0.75, 1),
        (0.25, 0),
    ):
        code = torch.where(magnitude <= threshold, lower_code, code)
    code = code + (groups < 0).to(torch.int64) * 8
    packed = code[..., 0::2] | (code[..., 1::2] << 4)
    packed_data = packed.flatten(1).to(torch.uint8)
    packed_scale = (torch.log2(scale).to(torch.int64) + 127).to(torch.uint8)
    return packed_data, packed_scale


def _golden(
    latent,
    wk,
    norm_weight,
    norm_eps,
    rope_sin,
    rope_cos,
    k_cache,
    cache_index,
    k_scale_cache,
    storage_mode,
    combined_block_size,
):
    # Match the projection's FP32 accumulation and BF16 output boundary
    # without relying on the host CPU's BF16 GEMM implementation.
    key_fp32 = torch.matmul(latent.float(), wk.float().T).to(torch.bfloat16).float()
    inverse_rms = torch.rsqrt(key_fp32.square().mean(dim=-1, keepdim=True) + norm_eps)
    key = (key_fp32 * inverse_rms * norm_weight).to(torch.bfloat16)

    dr = rope_sin.shape[-1]
    tail = key[:, -dr:].float()
    even = tail[:, 0::2]
    odd = tail[:, 1::2]
    rotated = torch.empty_like(tail)
    rotated[:, 0::2] = even * rope_cos[:, 0::2] - odd * rope_sin[:, 0::2]
    rotated[:, 1::2] = odd * rope_cos[:, 1::2] + even * rope_sin[:, 1::2]
    key[:, -dr:] = rotated.to(torch.bfloat16)

    packed_data, packed_scale = _pack_mxfp4(key)
    expected_cache = k_cache.clone()
    expected_scale = None if k_scale_cache is None else k_scale_cache.clone()
    if storage_mode == 0:
        block_size = expected_cache.shape[1]
        for row, slot in enumerate(cache_index.tolist()):
            if slot != -1:
                expected_cache[slot // block_size, slot % block_size, 0] = packed_data[
                    row
                ]
                if expected_scale is not None:
                    expected_scale[slot // block_size, slot % block_size].reshape(
                        -1
                    ).copy_(packed_scale[row])
    else:
        block_size = expected_cache.shape[1] * combined_block_size
        value_bytes = packed_data.shape[-1]
        scale_bytes = packed_scale.shape[-1]
        for row, flat_slot in enumerate(cache_index.tolist()):
            if flat_slot == -1:
                continue
            block_id = flat_slot // block_size
            slot = flat_slot % block_size
            group_id = slot // combined_block_size
            sub_slot = slot % combined_block_size
            data_start = sub_slot * value_bytes
            scale_start = combined_block_size * value_bytes + sub_slot * scale_bytes
            expected_cache[
                block_id, group_id, 0, data_start : data_start + value_bytes
            ] = packed_data[row]
            expected_cache[
                block_id, group_id, 0, scale_start : scale_start + scale_bytes
            ] = packed_scale[row]
    return expected_cache, expected_scale


def indexer_prologue_k(
    latent,
    wk,
    norm_weight,
    rope_sin,
    rope_cos,
    k_cache,
    k_scale_cache=None,
    *,
    cache_index,
    storage_mode,
    norm_eps,
    combined_block_size=-1
):
    if storage_mode not in (0, 1):
        raise ValueError("storage_mode must be 0 or 1")
    if storage_mode == 1 and combined_block_size <= 0:
        raise ValueError("combined_block_size must be positive for combined storage")
    tensors = [latent, wk, norm_weight, rope_sin, rope_cos, k_cache, cache_index]
    latent_cpu, wk_cpu, norm, sin, cos, cache, indices = [
        x.detach().cpu() for x in tensors
    ]
    scale = None if k_scale_cache is None else k_scale_cache.detach().cpu()
    with torch.device("cpu"):
        result, scale_result = _golden(
            latent_cpu,
            wk_cpu,
            norm,
            norm_eps,
            sin,
            cos,
            cache,
            indices,
            scale,
            storage_mode,
            combined_block_size,
        )
    k_cache.copy_(result.to(k_cache.device))
    if scale_result is not None:
        k_scale_cache.copy_(scale_result.to(k_scale_cache.device))
    return k_cache
