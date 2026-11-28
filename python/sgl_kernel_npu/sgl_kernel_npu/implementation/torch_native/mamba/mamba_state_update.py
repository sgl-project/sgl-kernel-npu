import torch


def move_intermediate_cache(
    ssm_states,
    intermediate_state_cache,
    dst_indices_tensor,
    src_indices_tensor,
    last_steps_tensor,
    h_block_size=2,
):
    assert len(dst_indices_tensor) == len(
        last_steps_tensor
    ), "Destination indices lengths must match"
    assert len(src_indices_tensor) == len(
        last_steps_tensor
    ), "Source indices lengths must match"
    valid = last_steps_tensor >= 0
    dst = dst_indices_tensor[valid].long()
    src = src_indices_tensor[valid].long()
    steps = last_steps_tensor[valid].long()
    layers, pool, heads, vdim, kdim = ssm_states.shape
    physical = torch.as_strided(
        ssm_states,
        ssm_states.shape,
        (ssm_states.stride(0), ssm_states.stride(1), vdim * kdim, kdim, 1),
    )
    physical[:, dst] = intermediate_state_cache[:, src, steps]
    return ssm_states


def move_intermediate_cache_kda(
    ssm_states,
    intermediate_state_cache,
    dst_indices_tensor,
    src_indices_tensor,
    last_steps_tensor,
    h_block_size=1,
):
    assert len(dst_indices_tensor) == len(
        last_steps_tensor
    ), "Destination indices lengths must match"
    assert len(src_indices_tensor) == len(
        last_steps_tensor
    ), "Source indices lengths must match"
    valid = last_steps_tensor >= 0
    ssm_states[:, dst_indices_tensor[valid].long()] = intermediate_state_cache[
        :, src_indices_tensor[valid].long(), last_steps_tensor[valid].long()
    ]
    return ssm_states


def conv_state_rollback(
    conv_states, state_indices, step_indices, draft_token_num, active_tail=None
):
    if state_indices.shape[0] == 0:
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

    win_len = conv_states.shape[2]
    for idx, step in zip(
        state_indices.to(torch.int32).tolist(), step_indices.to(torch.int32).tolist()
    ):
        if idx < 0 or step < 0:
            continue
        shift = draft_token_num - 1 - step
        if shift <= 0 or shift >= win_len:
            continue
        state = conv_states[:, idx]
        max_src = win_len - shift
        if active_tail is None:
            state[:, shift:].copy_(state[:, :max_src].clone())
        else:
            start_src = max(0, max_src - active_tail)
            dst_start = start_src + shift
            state[:, dst_start:].copy_(state[:, start_src:max_src].clone())
    return conv_states


def copy_tracked_conv_states(conv_states, src_slots, dst_slots, track_steps):
    if conv_states.ndim < 2:
        raise ValueError(f"conv_states must be at least 2D, got {conv_states.ndim}D")
    if src_slots.ndim != 1 or dst_slots.ndim != 1 or track_steps.ndim != 1:
        raise ValueError("src_slots, dst_slots, and track_steps must be 1D")
    B = dst_slots.shape[0]
    if src_slots.shape[0] != B or track_steps.shape[0] != B:
        raise ValueError("src_slots, dst_slots, and track_steps must have equal length")
    if B == 0:
        return conv_states
    for src, dst, step in zip(
        src_slots.to(torch.int64).tolist(),
        dst_slots.to(torch.int64).tolist(),
        track_steps.to(torch.int64).tolist(),
    ):
        if step >= 0 and src >= 0 and dst >= 0 and src != dst:
            conv_states[:, dst].copy_(conv_states[:, src].clone())
    return conv_states
