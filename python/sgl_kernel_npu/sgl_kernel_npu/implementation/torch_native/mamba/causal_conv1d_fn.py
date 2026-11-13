# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/ops/causal_conv1d.py
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2024, Tri Dao.
# Adapted from https://github.com/Dao-AILab/causal-conv1d/blob/main/causal_conv1d/causal_conv1d_interface.py
# and https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/ops/causal_conv1d.py

from typing import Optional, Union

import torch
import torch.nn.functional as F

PAD_SLOT_ID = -1


def causal_conv1d_fn_native(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    seqlens: Optional[torch.Tensor] = None,
    has_initial_state: Optional[torch.Tensor] = None,
    initial_states: Optional[torch.Tensor] = None,
    return_final_states: bool = False,
    activation: Optional[str] = "silu",
):
    """
    x: (batch, dim, seqlen)
    weight: (dim, width)
    bias: (dim,)
    seqlens: (batch,)
    has_initial_state: (batch,)
    initial_states: (batch, dim, width - 1)
    out: (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    x = x.to(weight.dtype)
    seqlen = x.shape[-1]
    dim, width = weight.shape

    if initial_states is None:
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)
    else:
        if x.ndim == 2:
            x = x.unsqueeze(0)
        x = torch.cat([initial_states, x], dim=-1)
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=0, groups=dim)

    out = out[..., :seqlen]
    if return_final_states:
        base = seqlens - (width - 1) * (~has_initial_state)
        positions = base.unsqueeze(1) + torch.arange(
            width - 1, device=seqlens.device
        ).unsqueeze(0)
        indices = positions.unsqueeze(1).expand(-1, x.shape[-2], -1)
        final_states = (
            x.gather(2, indices.clamp(min=0)).masked_fill(indices < 0, 0).to(dtype_in)
        )  # (batch, dim, width - 1)
    out = (out if activation is None else F.silu(out)).to(dtype=dtype_in)
    return (out, None) if not return_final_states else (out, final_states)


def prepare_data(
    x: torch.Tensor,
    weight: torch.Tensor,
    query_start_loc: Optional[torch.Tensor] = None,
    cache_indices: Optional[torch.Tensor] = None,
    has_initial_state: Optional[torch.Tensor] = None,
    conv_states: Optional[torch.Tensor] = None,
):
    initial_states = (
        torch.index_select(conv_states, 0, cache_indices)
        * has_initial_state[:, None, None]
        if has_initial_state is not None and has_initial_state.any()
        else None
    )

    seqlens = query_start_loc[1:] - query_start_loc[:-1]

    if x.ndim == 3:
        return x, initial_states, seqlens, None

    dtype, device = weight.dtype, weight.device
    batch_size = seqlens.size(0)
    max_T = seqlens.max()
    dim, cu_seq_len = x.size(0), query_start_loc[-1]

    x_flat = torch.zeros(size=(dim, batch_size * max_T), dtype=dtype, device=device)

    base_idx = torch.arange(batch_size, device=device, dtype=torch.int32) * max_T
    base_t = torch.arange(max_T, device=device, dtype=torch.int32).unsqueeze(0)
    mask = base_t < seqlens.unsqueeze(1)
    indices = (base_idx.unsqueeze(1) + base_t)[mask]

    x_flat.index_copy_(1, indices, x[..., :cu_seq_len])
    x_pad = x_flat.view(dim, batch_size, max_T).transpose(0, 1).contiguous()

    return x_pad, initial_states, seqlens, indices


def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    query_start_loc: Optional[torch.Tensor] = None,
    cache_indices: Optional[torch.Tensor] = None,
    has_initial_state: Optional[torch.Tensor] = None,
    conv_states: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
    **kwargs,
):
    """
    x: (batch, dim, seqlen) or (dim,cu_seq_len) for varlen
        sequences are concatenated from left to right for varlen
    weight: (dim, width)
    bias: (dim,)
    query_start_loc: (batch + 1) int32
        The cumulative sequence lengths of the sequences in
        the batch, used to index into sequence. prepended by 0.
        for example: query_start_loc = torch.Tensor([0,10,16,17]),
        x.shape=(dim,17)
    cache_indices: (batch)  int32
        indicates the corresponding state index,
        like so: conv_state = conv_states[cache_indices[batch_id]]
    has_initial_state: (batch) bool
        indicates whether should the kernel take the current state as initial
        state for the calculations
    conv_states: (...,dim,width - 1) itype
        updated inplace if provided
    activation: either None or "silu" or "swish"
    pad_slot_id: int
            if cache_indices is passed, lets the kernel identify padded
            entries that will not be processed,
            for example: cache_indices = [pad_slot_id, 1, 20, pad_slot_id]
            in this case, the kernel will not process entries at
            indices 0 and 3


    out: (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    if x.stride(-1) != 1:
        x = x.contiguous()
    bias = bias.contiguous() if bias is not None else None

    assert query_start_loc[-1] <= x.shape[-1], f"{query_start_loc=}, {x.shape=}"

    x_pad, initial_state_pad, seqlens, indices = prepare_data(
        x, weight, query_start_loc, cache_indices, has_initial_state, conv_states
    )

    out, final_states_out = causal_conv1d_fn_native(
        x_pad,
        weight,
        bias,
        seqlens=seqlens,
        has_initial_state=has_initial_state,
        initial_states=initial_state_pad,
        activation=activation,
        return_final_states=True,
    )
    conv_states.index_copy_(0, cache_indices, final_states_out)

    if x.ndim == 3:
        return out  # [batch_size, dim, seq_len]

    out = out.transpose(1, 2).contiguous().view(out.size(0) * out.size(2), out.size(1))
    out_final = torch.index_select(out, 0, indices).transpose(0, 1).contiguous()

    pad_seq_len = x.size(-1) - out_final.size(-1)
    if pad_seq_len > 0:
        out_final = F.pad(out_final, (0, pad_seq_len))

    return out_final  # [dim, cu_seq_len]


def torch_causal_conv1d_update_npu(
    hidden_state: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    conv_state_update: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    activation: Union[bool, str, None] = None,
):
    bsz, hidden_size, seq_len = hidden_state.shape
    state_len = conv_state.shape[-1]
    hidden_states_new = torch.cat([conv_state, hidden_state], dim=-1).to(weight.dtype)
    if conv_state_update is not None:
        for i in range(seq_len):
            end = i - seq_len + 1
            start = end - state_len
            slice_range = slice(start, end if end != 0 else None)
            conv_state_update[:, i] = hidden_states_new[:, :, slice_range]
    else:
        conv_state_update = hidden_states_new[:, :, -state_len:]

    out = torch.sum(
        hidden_states_new.to(torch.float32) * weight.to(torch.float32),
        dim=-1,
        keepdim=True,
    )
    if bias is not None:
        out = out + bias.to(torch.float32)[None, :, None]
    out = out.to(hidden_state.dtype)
    if activation in ["silu", "swish"]:
        out = F.silu(out)
    out = out.to(hidden_state.dtype)
    return out, conv_state_update
