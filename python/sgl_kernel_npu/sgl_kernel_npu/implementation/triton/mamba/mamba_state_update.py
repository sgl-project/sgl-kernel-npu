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
    H_BLOCK_SIZE: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """KDA-specific mover that respects non-contiguous destination strides.

    On NPU the temporal state (dst) is transposed (-1, -2), so its
    (H, V, K) layout differs from the source. This kernel indexes the
    destination through its real per-element strides and splits dim_v
    into BLOCK_V-sized chunks to keep the on-chip tile within budget.
    """
    valid_id = tl.program_id(0)
    pid_layer = tl.program_id(1)

    dst_idx_val = tl.load(dst_indices_ptr + valid_id)
    src_idx_val = tl.load(src_indices_ptr + valid_id)
    last_step_val = tl.load(last_steps_ptr + valid_id)
    if last_step_val < 0:
        return
    h_offsets = tl.arange(0, H_BLOCK_SIZE)
    k_offsets = tl.arange(0, BLOCK_K)

    src_base_addr = (
        src_cache_ptr
        + tl.cast(pid_layer, tl.int64) * layer_stride
        + tl.cast(src_idx_val, tl.int64) * size_stride
    )
    dst_base_addr = (
        dst_cache_ptr
        + tl.cast(pid_layer, tl.int64) * dst_layer_stride
        + tl.cast(dst_idx_val, tl.int64) * dst_size_stride
    )
    src_addr = src_base_addr + tl.cast(last_step_val, tl.int64) * draft_stride

    for h_start in range(0, h_dim, H_BLOCK_SIZE):
        h_real = h_start + h_offsets
        h_mask = h_real < h_dim
        k_mask = k_offsets < dim_k

        for v_start in range(0, dim_v, BLOCK_V):
            v_offsets = v_start + tl.arange(0, BLOCK_V)
            v_mask = v_offsets < dim_v

            mask = h_mask[:, None, None] & v_mask[None, :, None] & k_mask[None, None, :]

            # src is contiguous in (H, V, K) -> flat offset.
            src_linear_offset = (
                h_real[:, None, None] * dim_v * dim_k
                + v_offsets[None, :, None] * dim_k
                + k_offsets[None, None, :]
            )
            # dst uses its real per-element strides.
            dst_linear_offset = (
                h_real[:, None, None] * dst_h_stride
                + v_offsets[None, :, None] * dst_v_stride
                + k_offsets[None, None, :] * dst_k_stride
            )

            src_block = tl.load(src_addr + src_linear_offset, mask=mask, other=0)
            tl.store(dst_base_addr + dst_linear_offset, src_block, mask=mask)


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

    # Each program copies one request's state for one layer.
    grid = (len(dst_indices_tensor), L)

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
        H_BLOCK_SIZE=h_block_size,
        BLOCK_V=64,
        BLOCK_K=triton.next_power_of_2(K),
    )

    return ssm_states


@triton.jit
def _conv_state_rollback_kernel(
    conv_states_ptr,
    state_indices_ptr,
    step_indices_ptr,
    draft_token_num,
    num_dims: tl.constexpr,
    conv_window_size: tl.constexpr,
    layer_stride,
    req_stride,
    window_stride,
    dim_stride,
    active_tail: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Strided Triton kernel for rolling back conv states after speculative verification."""
    pid_req = tl.program_id(0)
    pid_layer = tl.program_id(1)

    state_idx = tl.load(state_indices_ptr + pid_req).to(tl.int64)
    step_idx = tl.load(step_indices_ptr + pid_req).to(tl.int64)

    if state_idx < 0 or step_idx < 0:
        return

    shift = (draft_token_num - 1) - step_idx
    if shift <= 0 or shift >= conv_window_size:
        return

    base_offset = state_idx * tl.cast(req_stride, tl.int64) + tl.cast(
        pid_layer, tl.int64
    ) * tl.cast(layer_stride, tl.int64)
    win_stride_i64 = tl.cast(window_stride, tl.int64)
    dim_stride_i64 = tl.cast(dim_stride, tl.int64)
    offs_d = tl.arange(0, BLOCK_D)

    max_src = conv_window_size - shift
    if active_tail > 0:
        start_src = tl.maximum(max_src - active_tail, tl.cast(0, tl.int64))
    else:
        start_src = tl.cast(0, tl.int64)
    n_rows = (max_src - start_src).to(tl.int32)

    for d0 in range(0, num_dims, BLOCK_D):
        d_idx = d0 + offs_d
        d_mask = d_idx < num_dims
        d_off = d_idx.to(tl.int64) * dim_stride_i64
        for step_back in range(0, n_rows):
            window_idx = max_src - 1 - step_back
            src_ptr = (
                conv_states_ptr + base_offset + window_idx * win_stride_i64 + d_off
            )
            dst_ptr = (
                conv_states_ptr
                + base_offset
                + (window_idx + shift) * win_stride_i64
                + d_off
            )
            data = tl.load(src_ptr, mask=d_mask)
            tl.store(dst_ptr, data, mask=d_mask)


@triton.jit(do_not_specialize=["B", "NUM_LAYERS", "DRAFT_NUM"])
def _conv_state_rollback_persistent_kernel(
    CONV_ptr,
    STATE_IDX_ptr,
    STEP_IDX_ptr,
    B,
    NUM_LAYERS,
    DRAFT_NUM,
    layer_stride,
    slot_stride,
    win_stride,
    WIN_LEN: tl.constexpr,
    NUM_DIMS: tl.constexpr,
    ACTIVE_TAIL: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Persistent 56-AIV tiled rollback for contiguous 4D conv_states.

    Supports both full-window rollback (ACTIVE_TAIL == 0) and active-tail-only
    rollback (e.g. ACTIVE_TAIL == 3 for kernel_size=4 depthwise causal conv1d).
    """
    pid = tl.program_id(0)
    n_progs = tl.num_programs(0)
    n_d = (NUM_DIMS + BLOCK_D - 1) // BLOCK_D
    total_tasks = NUM_LAYERS * B * n_d
    offs_d = tl.arange(0, BLOCK_D)

    for task in range(pid, total_tasks, n_progs):
        i_l = task // (B * n_d)
        rem = task - i_l * (B * n_d)
        i_b = rem // n_d
        i_d = rem - i_b * n_d

        step_idx = tl.load(STEP_IDX_ptr + i_b).to(tl.int64)
        shift = (DRAFT_NUM - 1) - step_idx
        if step_idx >= 0 and shift > 0 and shift < WIN_LEN:
            slot = tl.load(STATE_IDX_ptr + i_b).to(tl.int64)
            if slot >= 0:
                d_idx = i_d * BLOCK_D + offs_d
                d_mask = d_idx < NUM_DIMS
                base = (
                    tl.cast(i_l, tl.int64) * layer_stride
                    + slot * slot_stride
                    + d_idx.to(tl.int64)
                )
                max_src = WIN_LEN - shift
                if ACTIVE_TAIL == 3 and max_src >= 3:
                    src0 = max_src - 3
                    r0 = tl.load(CONV_ptr + base + src0 * win_stride, mask=d_mask)
                    r1 = tl.load(CONV_ptr + base + (src0 + 1) * win_stride, mask=d_mask)
                    r2 = tl.load(CONV_ptr + base + (src0 + 2) * win_stride, mask=d_mask)
                    tl.store(
                        CONV_ptr + base + (WIN_LEN - 3) * win_stride, r0, mask=d_mask
                    )
                    tl.store(
                        CONV_ptr + base + (WIN_LEN - 2) * win_stride, r1, mask=d_mask
                    )
                    tl.store(
                        CONV_ptr + base + (WIN_LEN - 1) * win_stride, r2, mask=d_mask
                    )
                else:
                    if ACTIVE_TAIL > 0:
                        start_src = tl.maximum(
                            max_src - ACTIVE_TAIL, tl.cast(0, tl.int64)
                        )
                    else:
                        start_src = tl.cast(0, tl.int64)
                    n_rows = (max_src - start_src).to(tl.int32)
                    for step_back in range(0, n_rows):
                        src_row = max_src - 1 - step_back
                        val = tl.load(
                            CONV_ptr + base + src_row * win_stride, mask=d_mask
                        )
                        tl.store(
                            CONV_ptr + base + (src_row + shift) * win_stride,
                            val,
                            mask=d_mask,
                        )


def conv_state_rollback(
    conv_states: torch.Tensor,  # [num_layers, pool_size, conv_window_size, num_dims]
    state_indices: torch.Tensor,  # [num_requests]
    step_indices: torch.Tensor,  # [num_requests]
    draft_token_num: int,
    active_tail: int | None = None,
):
    """Roll back conv states after speculative verification using Triton kernel.

    Args:
        conv_states: Conv states tensor [num_layers, pool_size, conv_window_size, num_dims]
        state_indices: State indices for each request [num_requests]
        step_indices: Accepted steps for each request [num_requests]
        draft_token_num: Number of draft tokens
        active_tail: Optional number of trailing active window rows to restore
            (e.g. ``conv_kernel_size - 1 == 3``). ``None`` rolls back the full window.
    """
    num_requests = state_indices.shape[0]
    if num_requests == 0:
        return None

    if conv_states.ndim != 4:
        raise ValueError(f"conv_states must be 4D, got {conv_states.ndim}D")
    if state_indices.ndim != 1 or step_indices.ndim != 1:
        raise ValueError("state_indices and step_indices must be 1D")
    if state_indices.shape[0] != step_indices.shape[0]:
        raise ValueError("state_indices and step_indices must have the same length")
    if active_tail is not None and (
        isinstance(active_tail, bool)
        or not isinstance(active_tail, int)
        or active_tail < 1
    ):
        raise ValueError(
            f"active_tail must be a positive int or None, got {active_tail!r}"
        )

    num_layers = conv_states.shape[0]
    conv_window_size = conv_states.shape[2]
    num_dims = conv_states.shape[3]
    tail_val = 0 if active_tail is None else min(active_tail, conv_window_size)

    state_indices = state_indices.to(torch.int32).contiguous()
    step_indices = step_indices.to(torch.int32).contiguous()

    if conv_states.is_contiguous():
        block_d = min(8192, triton.next_power_of_2(max(1, num_dims)))
        n_d = (num_dims + block_d - 1) // block_d
        grid = (min(56, num_layers * num_requests * n_d),)
        _conv_state_rollback_persistent_kernel[grid](
            conv_states,
            state_indices,
            step_indices,
            num_requests,
            num_layers,
            draft_token_num,
            conv_states.stride(0),
            conv_states.stride(1),
            conv_states.stride(2),
            WIN_LEN=conv_window_size,
            NUM_DIMS=num_dims,
            ACTIVE_TAIL=tail_val,
            BLOCK_D=block_d,
        )
        return conv_states

    block_d = min(4096, triton.next_power_of_2(max(1, num_dims)))
    grid = (num_requests, num_layers)
    _conv_state_rollback_kernel[grid](
        conv_states,
        state_indices,
        step_indices,
        draft_token_num,
        num_dims,
        conv_window_size,
        conv_states.stride(0),
        conv_states.stride(1),
        conv_states.stride(2),
        conv_states.stride(3),
        active_tail=tail_val,
        BLOCK_D=block_d,
    )

    return conv_states


@triton.jit(do_not_specialize=["NUM_LAYERS", "B", "ROW_ELEMS"])
def _copy_tracked_conv_states_kernel(
    CONV_ptr,
    SRC_SLOTS_ptr,
    DST_SLOTS_ptr,
    TRACK_STEPS_ptr,
    NUM_LAYERS,
    B,
    ROW_ELEMS,
    layer_stride,
    slot_stride,
    BLOCK_E: tl.constexpr,
):
    """Persistent 56-AIV cross-layer conv_state snapshot copy for tracked rows only."""
    pid = tl.program_id(0)
    n_progs = tl.num_programs(0)
    total_tasks = NUM_LAYERS * B
    offs = tl.arange(0, BLOCK_E)

    for task in range(pid, total_tasks, n_progs):
        i_l = task // B
        i_b = task - i_l * B
        step = tl.load(TRACK_STEPS_ptr + i_b).to(tl.int64)
        if step >= 0:
            src = tl.load(SRC_SLOTS_ptr + i_b).to(tl.int64)
            dst = tl.load(DST_SLOTS_ptr + i_b).to(tl.int64)
            if src >= 0 and dst >= 0 and src != dst:
                base = tl.cast(i_l, tl.int64) * layer_stride
                src_base = base + src * slot_stride
                dst_base = base + dst * slot_stride
                for off0 in range(0, ROW_ELEMS, BLOCK_E):
                    idx = off0 + offs
                    mask = idx < ROW_ELEMS
                    val = tl.load(CONV_ptr + src_base + idx, mask=mask)
                    tl.store(CONV_ptr + dst_base + idx, val, mask=mask)


def copy_tracked_conv_states(
    conv_states: torch.Tensor,
    src_slots: torch.Tensor,
    dst_slots: torch.Tensor,
    track_steps: torch.Tensor,
) -> torch.Tensor:
    """Copy ``conv_states[:, src_slots[b]]`` to ``conv_states[:, dst_slots[b]]`` where ``track_steps[b] >= 0``.

    Skips rows with ``track_steps[b] < 0``, negative slot indices, or ``src == dst``,
    avoiding full-batch self-copy overhead across all layers during speculative Radix tracking.
    """
    if conv_states.ndim < 2:
        raise ValueError(f"conv_states must be at least 2D, got {conv_states.ndim}D")
    if src_slots.ndim != 1 or dst_slots.ndim != 1 or track_steps.ndim != 1:
        raise ValueError("src_slots, dst_slots, and track_steps must be 1D")
    B = dst_slots.shape[0]
    if src_slots.shape[0] != B or track_steps.shape[0] != B:
        raise ValueError("src_slots, dst_slots, and track_steps must have equal length")
    if B == 0:
        return conv_states
    if not conv_states.is_contiguous():
        valid = (
            (track_steps >= 0)
            & (src_slots >= 0)
            & (dst_slots >= 0)
            & (src_slots != dst_slots)
        )
        if valid.any():
            src_v = src_slots[valid].to(torch.int64)
            dst_v = dst_slots[valid].to(torch.int64)
            conv_states[:, dst_v] = conv_states[:, src_v].clone()
        return conv_states

    num_layers = conv_states.shape[0]
    row_elems = conv_states.stride(1)
    src_c = src_slots.contiguous()
    dst_c = dst_slots.contiguous()
    steps_c = track_steps.contiguous()
    grid = (min(56, num_layers * B),)
    _copy_tracked_conv_states_kernel[grid](
        conv_states,
        src_c,
        dst_c,
        steps_c,
        num_layers,
        B,
        row_elems,
        conv_states.stride(0),
        conv_states.stride(1),
        BLOCK_E=4096,
    )
    return conv_states
