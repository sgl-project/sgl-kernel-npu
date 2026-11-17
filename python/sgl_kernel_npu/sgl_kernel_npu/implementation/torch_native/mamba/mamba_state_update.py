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


def conv_state_rollback(conv_states, state_indices, step_indices, draft_token_num):
    if state_indices.shape[0] == 0:
        return
    if conv_states.ndim != 4:
        raise ValueError(f"conv_states must be 4D, got {conv_states.ndim}D")
    if state_indices.ndim != 1 or step_indices.ndim != 1:
        raise ValueError("state_indices and step_indices must be 1D")
    if state_indices.shape[0] != step_indices.shape[0]:
        raise ValueError("state_indices and step_indices must have the same length")
    for idx, step in zip(
        state_indices.to(torch.int32).tolist(), step_indices.to(torch.int32).tolist()
    ):
        if step < 0:
            continue
        shift = draft_token_num - 1 - step
        if shift > 0:
            state = conv_states[:, idx]
            state[:, shift:].copy_(state[:, :-shift].clone())
    return conv_states
