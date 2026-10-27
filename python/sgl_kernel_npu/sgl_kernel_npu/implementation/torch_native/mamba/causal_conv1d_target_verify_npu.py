"""Independent PyTorch backend for causal_conv1d_target_verify_npu."""

import torch


def _evaluate(
    x,
    conv_state,
    weight,
    bias,
    activation,
    conv_state_indices,
    intermediate_conv_window,
    intermediate_state_indices,
    retrieve_next_token=None,
    retrieve_next_sibling=None,
    retrieve_parent_token=None,
    pad_slot_id=-1,
):
    """Compute outputs and per-token causal windows from original inputs only."""
    device = x.device
    dtype = x.dtype
    x_ref = x.detach().cpu()
    state_ref = conv_state.detach().cpu()
    weight_ref = weight.detach().cpu().float()
    bias_ref = None if bias is None else bias.detach().cpu().float()
    state_indices = conv_state_indices.detach().cpu().tolist()
    scratch_indices = intermediate_state_indices.detach().cpu().tolist()
    expected_scratch = intermediate_conv_window.detach().cpu().clone()
    expected_parent = (
        None
        if retrieve_parent_token is None
        else retrieve_parent_token.detach().cpu().clone()
    )
    has_tree = retrieve_next_token is not None
    next_tokens = retrieve_next_token.detach().cpu().tolist() if has_tree else None
    next_siblings = retrieve_next_sibling.detach().cpu().tolist() if has_tree else None

    batch, dim, steps = x_ref.shape
    width = weight_ref.shape[1]
    expected_out = x_ref.float().clone()

    for batch_idx in range(batch):
        if state_indices[batch_idx] == pad_slot_id:
            continue
        history = state_ref[state_indices[batch_idx], :, : width - 1].float()
        scratch_idx = scratch_indices[batch_idx]
        parents = [0] * steps if has_tree else None

        for token_idx in range(steps):
            acc = (
                bias_ref.clone()
                if bias_ref is not None
                else torch.zeros_like(weight_ref[:, 0])
            )

            if not has_tree:
                current = x_ref[batch_idx, :, token_idx].float()
                for tap in range(width - 1):
                    acc += history[:, tap] * weight_ref[:, tap]
                acc += current * weight_ref[:, width - 1]

                history = torch.cat((history[:, 1:], current.unsqueeze(1)), dim=1)
                expected_scratch[scratch_idx, token_idx, :, : width - 1] = history.to(
                    dtype
                )
            else:
                child_idx = next_tokens[batch_idx][token_idx]
                if child_idx != -1:
                    parents[child_idx] = token_idx
                sibling_idx = next_siblings[batch_idx][token_idx]
                if sibling_idx != -1:
                    parents[sibling_idx] = parents[token_idx]

                ancestor_idx = token_idx
                value = x_ref[batch_idx, :, token_idx].float()
                for tap in range(width):
                    acc += value * weight_ref[:, width - tap - 1]
                    if tap == width - 1:
                        continue

                    expected_scratch[scratch_idx, token_idx, :, width - tap - 2] = (
                        value.to(dtype)
                    )
                    if ancestor_idx > 0:
                        ancestor_idx = parents[ancestor_idx]
                        value = x_ref[batch_idx, :, ancestor_idx].float()
                    else:
                        history_idx = width - 2 + ancestor_idx
                        value = history[:, history_idx]
                        ancestor_idx -= 1

            if activation in ("silu", "swish"):
                acc = torch.nn.functional.silu(acc)
            expected_out[batch_idx, :, token_idx] = acc

        if has_tree:
            expected_parent[batch_idx, :steps] = expected_parent.new_tensor(parents)

    expected_out = expected_out.to(dtype).to(device)
    expected_scratch = expected_scratch.to(device)
    if expected_parent is not None:
        expected_parent = expected_parent.to(device)
    return expected_out, expected_scratch, expected_parent


@torch.no_grad()
def causal_conv1d_target_verify_npu(
    x,
    conv_state,
    weight,
    bias=None,
    activation=None,
    conv_state_indices=None,
    intermediate_conv_window=None,
    intermediate_state_indices=None,
    retrieve_next_token=None,
    retrieve_next_sibling=None,
    retrieve_parent_token=None,
    pad_slot_id=-1,
    validate_data=False,
):
    """Independent Torch convolution; linear writes x, tree preserves x."""
    if conv_state_indices is None or intermediate_state_indices is None:
        raise ValueError(
            "target verify requires conv_state_indices and intermediate_state_indices"
        )
    if intermediate_conv_window is None:
        raise ValueError("target verify requires intermediate_conv_window")

    if isinstance(activation, bool):
        activation = "silu" if activation else None
    elif activation not in (None, "silu", "swish"):
        raise ValueError("activation must be None, silu, or swish")

    if x.dim() not in (2, 3):
        raise ValueError(f"x must be rank 2 or 3, got shape {tuple(x.shape)}")
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)

    batch, dim, seqlen = x.shape
    if batch == 0 or seqlen == 0:
        raise ValueError("target verify requires non-empty batch and token dimensions")
    if weight.dim() != 2:
        raise ValueError("weight must have shape [dim, width] or [width, dim]")
    if weight.shape[0] == dim:
        weight_channel_last = False
        width = weight.shape[1]
        stride_weight_dim, stride_weight_width = weight.stride()
    elif weight.shape[1] == dim:
        weight_channel_last = True
        width = weight.shape[0]
        stride_weight_width, stride_weight_dim = weight.stride()
    else:
        raise ValueError("weight must have shape [dim, width] or [width, dim]")
    if width < 2 or width > 4:
        raise ValueError(f"target-verify NPU kernel supports width 2..4, got {width}")
    if conv_state.dim() != 3 or (
        conv_state.shape[1] != dim or conv_state.shape[2] < width - 1
    ):
        raise ValueError(
            "conv_state must have shape [cache_lines, dim, state_len >= width - 1]"
        )
    if bias is not None and (bias.dim() != 1 or bias.numel() != dim):
        raise ValueError("bias must have shape [dim]")
    if conv_state_indices.dim() != 1 or intermediate_state_indices.dim() != 1:
        raise ValueError("state index tensors must be one-dimensional")
    if conv_state_indices.numel() < batch or intermediate_state_indices.numel() < batch:
        raise ValueError(
            "state index tensors must contain at least one entry per batch"
        )
    if intermediate_conv_window.dim() != 4 or (
        intermediate_conv_window.shape[1] < seqlen
        or intermediate_conv_window.shape[2] != dim
        or intermediate_conv_window.shape[3] < width - 1
    ):
        raise ValueError(
            "intermediate_conv_window must have shape "
            "[scratch_size, steps >= seqlen, dim, window >= width - 1]"
        )

    has_tree = retrieve_next_token is not None
    if has_tree:
        if retrieve_next_sibling is None or retrieve_parent_token is None:
            raise ValueError(
                "tree verify requires next-token, next-sibling, and parent tensors"
            )
        tree_tensors = (
            retrieve_next_token,
            retrieve_next_sibling,
            retrieve_parent_token,
        )
        if any(
            tensor.dim() != 2 or tensor.shape[0] < batch or tensor.shape[1] < seqlen
            for tensor in tree_tensors
        ):
            raise ValueError(
                "tree mapping tensors must have shape [rows >= batch, steps >= seqlen]"
            )
    elif retrieve_next_sibling is not None or retrieve_parent_token is not None:
        raise ValueError(
            "tree mapping tensors must either all be provided or all be None"
        )

    if validate_data:
        if x.stride(1) != 1:
            raise ValueError("x must be contiguous along the channel dimension")
        if conv_state.stride(1) != 1:
            raise ValueError(
                "conv_state must be contiguous along the channel dimension"
            )
        if weight_channel_last:
            if stride_weight_dim != 1:
                raise ValueError(
                    "[width, dim] weight must be contiguous along the channel dimension"
                )
        elif stride_weight_width != 1:
            raise ValueError(
                "[dim, width] weight must be contiguous along the width dimension"
            )

    weight_ref = weight.transpose(0, 1) if weight_channel_last else weight
    result, scratch, parent = _evaluate(
        x,
        conv_state,
        weight_ref,
        bias,
        activation,
        conv_state_indices,
        intermediate_conv_window,
        intermediate_state_indices,
        retrieve_next_token,
        retrieve_next_sibling,
        retrieve_parent_token,
        pad_slot_id,
    )
    intermediate_conv_window.copy_(scratch)
    if parent is not None:
        retrieve_parent_token.copy_(parent)
    if not has_tree:
        x.copy_(result)
        result = x
    return result.squeeze(-1) if unsqueeze else result
