"""
Complete the following functions:
    Fully fused gather-scatter with built-in masking for mamba state updates.

    This function fuses the following operations into a single kernel:
    1. valid_mask = step_indices_raw >= 0
    2. valid_indices = valid_mask.nonzero()
    3. dst_indices = dst_indices_raw[valid_indices]  (index_select)
    4. step_indices = step_indices_raw[valid_indices]  (index_select)
    5. for each valid i: dst[:, dst_indices[i], :] = src[:, i, step_indices[i], :]

follow gpu kernel: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/mamba/mamba_state_scatter_triton.py
"""

import triton
import triton.language as tl

from ..implementation.triton.mamba.mamba_state_update import (
    _conv_state_rollback_kernel as _conv_state_rollback_kernel,
)
from ..implementation.triton.mamba.mamba_state_update import (
    conv_state_rollback as conv_state_rollback,
)
from ..implementation.triton.mamba.mamba_state_update import (
    move_cache_dynamic_last_kernel_h_block_kda as move_cache_dynamic_last_kernel_h_block_kda,
)
from ..implementation.triton.mamba.mamba_state_update import (
    move_intermediate_cache_kda as move_intermediate_cache_kda,
)


@triton.jit
def move_cache_dynamic_last_kernel_h_block(
    dst_cache_ptr,
    src_cache_ptr,
    dst_indices_ptr,
    src_indices_ptr,
    last_steps_ptr,
    layer_stride,
    size_stride,
    draft_stride,
    dst_layer_stride,
    dst_size_stride,
    h_dim,
    dim_v,
    dim_k,
    num_layers,
    H_BLOCK_SIZE: tl.constexpr,
    BLOCK_V: tl.constexpr,  # Block size for dim_v
    BLOCK_K: tl.constexpr,  # Block size for dim_k
):
    valid_id = tl.program_id(0)

    # Load actual indices
    dst_idx_val = tl.load(dst_indices_ptr + valid_id)
    src_idx_val = tl.load(src_indices_ptr + valid_id)
    last_step_val = tl.load(last_steps_ptr + valid_id)
    if last_step_val < 0:
        return
    h_offsets = tl.arange(0, H_BLOCK_SIZE)
    v_offsets = tl.arange(0, BLOCK_V)
    k_offsets = tl.arange(0, BLOCK_K)

    # Process each layer
    for l in range(num_layers):
        src_base_addr = (
            src_cache_ptr
            + tl.cast(l, tl.int64) * layer_stride
            + tl.cast(src_idx_val, tl.int64) * size_stride
        )
        dst_base_addr = (
            dst_cache_ptr
            + tl.cast(l, tl.int64) * dst_layer_stride
            + tl.cast(dst_idx_val, tl.int64) * dst_size_stride
        )
        src_addr = src_base_addr + tl.cast(last_step_val, tl.int64) * draft_stride

        # Process h dimension in blocks
        for h_start in range(0, h_dim, H_BLOCK_SIZE):
            h_real = h_start + h_offsets
            h_mask = h_real < h_dim

            v_mask = v_offsets < dim_v
            k_mask = k_offsets < dim_k

            mask = h_mask[:, None, None] & v_mask[None, :, None] & k_mask[None, None, :]

            linear_offset = (
                h_real[:, None, None] * dim_v * dim_k
                + v_offsets[None, :, None] * dim_k
                + k_offsets[None, None, :]
            )

            src_block = tl.load(src_addr + linear_offset, mask=mask, other=0)
            tl.store(dst_base_addr + linear_offset, src_block, mask=mask)


def move_intermediate_cache(
    ssm_states,
    intermediate_state_cache,
    dst_indices_tensor,
    src_indices_tensor,
    last_steps_tensor,
    h_block_size=2,
):
    """
    Move intermediate cache to SSM states using Triton kernel.

    Args:
        ssm_states: Destination SSM states tensor
        intermediate_state_cache: Source intermediate state cache
        dst_indices_tensor: Valid destination indices tensor
        src_indices_tensor: Valid source indices tensor
        last_steps_tensor: Last steps tensor
        h_block_size: Block size for h dimension processing
    """
    L, S, D, H, V, K = intermediate_state_cache.shape

    strides = intermediate_state_cache.stride()
    layer_stride, size_stride, draft_stride = (
        int(strides[0]),
        int(strides[1]),
        int(strides[2]),
    )
    dst_layer_stride, dst_size_stride = int(ssm_states.stride()[0]), int(
        ssm_states.stride()[1]
    )
    assert len(dst_indices_tensor) == len(
        last_steps_tensor
    ), "Destination indices lengths must match"
    assert len(src_indices_tensor) == len(
        last_steps_tensor
    ), "Source indices lengths must match"

    # Grid: one thread per valid index
    grid = (len(dst_indices_tensor),)

    move_cache_dynamic_last_kernel_h_block[grid](
        dst_cache_ptr=ssm_states,
        src_cache_ptr=intermediate_state_cache,
        dst_indices_ptr=dst_indices_tensor,
        src_indices_ptr=src_indices_tensor,
        last_steps_ptr=last_steps_tensor,
        layer_stride=layer_stride,
        size_stride=size_stride,
        draft_stride=draft_stride,
        dst_layer_stride=dst_layer_stride,
        dst_size_stride=dst_size_stride,
        h_dim=H,
        dim_v=V,
        dim_k=K,
        num_layers=L,
        H_BLOCK_SIZE=h_block_size,  # Process 2 h elements per block
        BLOCK_V=triton.next_power_of_2(V),  # Block size for dim_v
        BLOCK_K=triton.next_power_of_2(K),  # Block size for dim_k
    )

    return ssm_states
