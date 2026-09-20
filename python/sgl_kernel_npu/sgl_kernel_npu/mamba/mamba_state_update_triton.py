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

import torch
import triton
import triton.language as tl


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
    elem_per_entry,
    num_layers,
    BLOCK_SIZE: tl.constexpr,
):
    """Flat 1D tiling variant that copies H*V*K elements in BLOCK_SIZE chunks.

    This avoids materializing a large 3D [H, V, K] tile in the NPU Unified
    Buffer, which overflows for models with large head_dim * state_size (e.g.
    Qwen3.6-35B-A3B with V=128, K=128).
    """
    valid_id = tl.program_id(0)

    # Load actual indices
    dst_idx_val = tl.load(dst_indices_ptr + valid_id)
    src_idx_val = tl.load(src_indices_ptr + valid_id)
    last_step_val = tl.load(last_steps_ptr + valid_id)
    if last_step_val < 0:
        return

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

        # Flat 1D copy over H*V*K elements
        for offset in range(0, elem_per_entry, BLOCK_SIZE):
            offsets = offset + tl.arange(0, BLOCK_SIZE)
            mask = offsets < elem_per_entry
            data = tl.load(src_addr + offsets, mask=mask, other=0)
            tl.store(dst_base_addr + offsets, data, mask=mask)


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
        h_block_size: Unused, kept for API compatibility
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

    elem_per_entry = H * V * K

    # Grid: one thread per valid index
    grid = (len(dst_indices_tensor),)

    BLOCK_SIZE = 1024

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
        elem_per_entry=elem_per_entry,
        num_layers=L,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return ssm_states


@triton.jit
def move_cache_dynamic_last_kernel_h_block_kda(
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
    dst_h_stride,
    dst_v_stride,
    dst_k_stride,
    h_dim,
    dim_v,
    dim_k,
    num_layers,
    BLOCK_SIZE: tl.constexpr,
):
    """KDA-specific mover with flat 1D tiling for NPU UB safety.

    On NPU the temporal state (dst) is transposed (-1, -2), so its
    (H, V, K) layout differs from the source. This kernel uses flat 1D
    element indices decoded into (h, v, k) coordinates, keeping the
    on-chip tile to BLOCK_SIZE elements to avoid UB overflow.
    """
    valid_id = tl.program_id(0)

    dst_idx_val = tl.load(dst_indices_ptr + valid_id)
    src_idx_val = tl.load(src_indices_ptr + valid_id)
    last_step_val = tl.load(last_steps_ptr + valid_id)
    if last_step_val < 0:
        return

    elem_per_entry = h_dim * dim_v * dim_k
    vk = dim_v * dim_k

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

        # Flat 1D iteration; decode each element index into (h, v, k)
        for offset in range(0, elem_per_entry, BLOCK_SIZE):
            flat = offset + tl.arange(0, BLOCK_SIZE)
            mask = flat < elem_per_entry

            h = flat // vk
            rem = flat % vk
            v = rem // dim_k
            k = rem % dim_k

            # src is contiguous in (H, V, K)
            src_off = h * dim_v * dim_k + v * dim_k + k
            # dst uses real per-element strides
            dst_off = h * dst_h_stride + v * dst_v_stride + k * dst_k_stride

            data = tl.load(src_addr + src_off, mask=mask, other=0)
            tl.store(dst_base_addr + dst_off, data, mask=mask)


def move_intermediate_cache_kda(
    ssm_states,
    intermediate_state_cache,
    dst_indices_tensor,
    src_indices_tensor,
    last_steps_tensor,
    h_block_size=1,
):
    """Move intermediate cache to SSM states (KDA-transposed-dst aware).

    Compared with ``move_intermediate_cache``, this variant preserves the
    destination (H, V, K) per-element strides and tiles dim_v in 64-wide
    chunks. Required when the SSM temporal state is not contiguous in the
    (V, K) plane -- e.g. Kimi-K3 on Ascend where the cache is transposed
    (-1, -2).
    """
    L, S, D, H, V, K = intermediate_state_cache.shape

    strides = intermediate_state_cache.stride()
    layer_stride, size_stride, draft_stride = (
        int(strides[0]),
        int(strides[1]),
        int(strides[2]),
    )
    dst_strides = ssm_states.stride()
    dst_layer_stride, dst_size_stride = int(dst_strides[0]), int(dst_strides[1])
    dst_h_stride, dst_v_stride, dst_k_stride = (
        int(dst_strides[2]),
        int(dst_strides[3]),
        int(dst_strides[4]),
    )
    assert len(dst_indices_tensor) == len(
        last_steps_tensor
    ), "Destination indices lengths must match"
    assert len(src_indices_tensor) == len(
        last_steps_tensor
    ), "Source indices lengths must match"

    if len(dst_indices_tensor) == 0:
        return ssm_states

    grid = (len(dst_indices_tensor),)

    BLOCK_SIZE = 1024

    move_cache_dynamic_last_kernel_h_block_kda[grid](
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
        dst_h_stride=dst_h_stride,
        dst_v_stride=dst_v_stride,
        dst_k_stride=dst_k_stride,
        h_dim=H,
        dim_v=V,
        dim_k=K,
        num_layers=L,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return ssm_states


@triton.jit
def _conv_state_rollback_kernel(
    conv_states_ptr,
    state_indices_ptr,
    step_indices_ptr,
    draft_token_num,
    num_layers,
    num_dims,
    conv_window_size: tl.constexpr,
    layer_stride,
    req_stride,
    window_stride,
    dim_stride,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for rolling back conv states after MTP verification.

    Uses flat 1D tiling over the dim axis to avoid allocating a large
    tl.arange(0, num_dims) tile in the NPU Unified Buffer.

    Args:
        conv_states_ptr: Pointer to conv states tensor [num_layers, pool_size, conv_window_size, num_dims]
        state_indices_ptr: Pointer to state indices [num_requests]
        step_indices_ptr: Pointer to step indices (accepted steps) [num_requests]
        draft_token_num: Number of draft tokens
        num_layers: Number of layers
        num_dims: Number of dimensions
        conv_window_size: Convolution window size
        layer_stride: Stride for layer dimension
        req_stride: Stride for request dimension
        window_stride: Stride for window dimension
        dim_stride: Stride for dimension dimension
        BLOCK_SIZE: Tile width for dim iteration
    """
    pid_req = tl.program_id(0)

    # Load state and step indices
    state_idx = tl.load(state_indices_ptr + pid_req).to(tl.int64)
    step_idx = tl.load(step_indices_ptr + pid_req).to(tl.int64)

    if step_idx < 0:
        return

    # Calculate rollback shift
    shift = (draft_token_num - 1) - step_idx

    # Early exit if no rollback needed
    if shift <= 0:
        return

    # Process each layer
    for layer in range(num_layers):
        # Calculate base offset for this request and layer
        base_offset = state_idx * req_stride + layer * layer_stride

        # Process each window position that needs to be moved
        # Move data from [0, conv_window_size-shift) to [shift, conv_window_size)
        for window_idx1 in range(0, conv_window_size - shift):
            window_idx = conv_window_size - shift - 1 - window_idx1

            src_base = base_offset + window_idx * window_stride
            dst_base = base_offset + (window_idx + shift) * window_stride

            # Tile over dim axis in BLOCK_SIZE chunks
            for d_off in range(0, num_dims, BLOCK_SIZE):
                dim_offsets = d_off + tl.arange(0, BLOCK_SIZE)
                mask = dim_offsets < num_dims

                src_ptr = conv_states_ptr + src_base + dim_offsets * dim_stride
                dst_ptr = conv_states_ptr + dst_base + dim_offsets * dim_stride

                data = tl.load(src_ptr, mask=mask, other=0)
                tl.store(dst_ptr, data, mask=mask)


def conv_state_rollback(
    conv_states: torch.Tensor,  # [num_layers, pool_size, conv_window_size, num_dims]
    state_indices: torch.Tensor,  # [num_requests]
    step_indices: torch.Tensor,  # [num_requests]
    draft_token_num: int,
):
    """
    Roll back conv states after MTP verification using Triton kernel.

    Args:
        conv_states: Conv states tensor [num_layers, pool_size, conv_window_size, num_dims]
        state_indices: State indices for each request [num_requests]
        step_indices: Accepted steps for each request [num_requests]
        draft_token_num: Number of draft tokens
    """
    num_requests = state_indices.shape[0]
    if num_requests == 0:
        return

    if conv_states.ndim != 4:
        raise ValueError(f"conv_states must be 4D, got {conv_states.ndim}D")
    if state_indices.ndim != 1 or step_indices.ndim != 1:
        raise ValueError("state_indices and step_indices must be 1D")
    if state_indices.shape[0] != step_indices.shape[0]:
        raise ValueError("state_indices and step_indices must have the same length")

    num_layers = conv_states.shape[0]
    conv_window_size = conv_states.shape[2]
    num_dims = conv_states.shape[3]

    # Get strides (in elements, not bytes)
    layer_stride = conv_states.stride(0)
    req_stride = conv_states.stride(1)
    window_stride = conv_states.stride(2)
    dim_stride = conv_states.stride(3)

    # Ensure indices are int32 and contiguous
    state_indices = state_indices.to(torch.int32).contiguous()
    step_indices = step_indices.to(torch.int32).contiguous()

    # Grid over all requests
    grid = (num_requests,)

    BLOCK_SIZE = 1024

    _conv_state_rollback_kernel[grid](
        conv_states,
        state_indices,
        step_indices,
        draft_token_num,
        num_layers,
        num_dims,
        conv_window_size,
        layer_stride,
        req_stride,
        window_stride,
        dim_stride,
        BLOCK_SIZE,
    )

    return conv_states
