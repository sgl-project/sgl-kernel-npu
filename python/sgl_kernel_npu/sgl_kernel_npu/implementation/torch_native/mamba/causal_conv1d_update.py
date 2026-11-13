from typing import Optional, Union

import torch
import torch.nn.functional as F

PAD_SLOT_ID = -1


def causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Union[bool, str, None] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    conv_state_indices: Optional[torch.Tensor] = None,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    intermediate_conv_window: Optional[torch.Tensor] = None,
    pad_slot_id: int = PAD_SLOT_ID,
    metadata=None,
    validate_data=False,
):
    """
    x: (batch, dim) or (batch, dim, seqlen)
        [shape=2: single token prediction]
        [shape=3: single or multiple tokens prediction]
    conv_state: (..., dim, state_len), where state_len >= width - 1
    weight: (dim, width)
    bias: (dim,)
    cache_seqlens: (batch,), dtype int32.
        If not None, the conv_state is treated as a circular buffer.
        The conv_state will be updated by copying x to the conv_state
        starting at the index
        @cache_seqlens % state_len.
    conv_state_indices: (batch,), dtype int32
        If not None, the conv_state is a larger tensor along the batch dim,
        and we are selecting the batch coords specified by conv_state_indices.
        Useful for a continuous batching scenario.
    pad_slot_id: int
            if cache_indices is passed, lets the kernel identify padded
            entries that will not be processed,
            for example: cache_indices = [pad_slot_id, 1 ,20 ,pad_slot_id]
            in this case, the kernel will not process entries at
            indices 0 and 3
    out: (batch, dim) or (batch, dim, seqlen)
    """
    if validate_data:
        assert cache_seqlens is None  # not implemented yet - ok for vLLM
        assert pad_slot_id is not None
        assert x.stride(1) == 1
    if isinstance(activation, bool):
        activation = "silu" if activation is True else None
    elif activation is not None:
        assert activation in ["silu", "swish"]
    unsqueeze = x.dim() == 2
    if unsqueeze:
        # make it (batch, dim, seqlen) with seqlen == 1
        x = x.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    _, width = weight.shape
    # conv_state: (..., dim, state_len), where state_len >= width - 1
    num_cache_lines, _, state_len = conv_state.size()

    if validate_data:
        assert dim == weight.size(0)
        assert (
            conv_state.stride(-2) == 1
        ), f"ERROR: expect contiguous along feat-dim of conv_state (currently stride={conv_state.stride()})"
        assert state_len >= width - 1
        # when above happens, we don't shift-left to keep any records in conv_state
        assert dim == conv_state.size(1)
        if conv_state_indices is None:
            assert conv_state.size(0) >= batch
        else:
            assert (batch,) == conv_state_indices.shape

        assert num_cache_lines >= batch
        assert weight.stride(1) == 1  # Need this
        assert cache_seqlens is None  # not needed for vLLM - circular buffer

    if cache_seqlens is None and num_accepted_tokens is None:
        selected = conv_state[conv_state_indices]
        signal = torch.cat((selected, x), -1).to(weight.dtype)
        conv_state[conv_state_indices] = signal[:, :, -selected.shape[-1] :]
        out = F.conv1d(
            signal.float(),
            weight.float().unsqueeze(1),
            bias.float() if bias is not None else None,
            groups=dim,
        ).to(x.dtype)
        if activation in ("silu", "swish"):
            out = F.silu(out)
        return out.squeeze(-1) if unsqueeze else out
    original = x.clone()
    state_len = width - 1 + seqlen - 1
    for request in range(batch):
        slot = (
            int(conv_state_indices[request])
            if conv_state_indices is not None
            else request
        )
        if pad_slot_id is not None and slot == pad_slot_id:
            continue
        offset = (
            int(num_accepted_tokens[request]) - 1
            if num_accepted_tokens is not None
            else 0
        )
        history = conv_state[slot, :, offset : offset + width - 1].clone()
        shifted = torch.cat(
            (
                conv_state[slot, :, offset + 1 : offset + width - 1].clone(),
                original[request],
            ),
            -1,
        )
        conv_state[slot, :, :state_len].copy_(shifted)
        for token in range(seqlen):
            window = torch.cat((history, original[request, :, token : token + 1]), -1)
            acc = (
                torch.zeros(dim, dtype=torch.float32, device=x.device)
                if bias is None
                else bias.float().clone()
            )
            for column in range(width):
                acc = acc + (window[:, column] * weight[:, column]).float()
            if activation in ("silu", "swish"):
                acc = acc / (1 + torch.exp(-acc))
            x[request, :, token] = acc.to(x.dtype)
            history = window[:, 1:]
            if intermediate_conv_window is not None:
                intermediate_conv_window[slot, token].copy_(history)
    return x.squeeze(-1) if unsqueeze else x
