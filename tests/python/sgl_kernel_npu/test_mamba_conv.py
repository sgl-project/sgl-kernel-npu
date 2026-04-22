# -*- coding: utf-8 -*-
import random
from typing import List, Optional, Union

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from sgl_kernel_npu.mamba.causal_conv1d import PAD_SLOT_ID, causal_conv1d_update_npu, causal_conv1d_update_mtp_npu

device = "npu"


def get_abs_err(x, y):
    return (x.detach() - y.detach()).flatten().abs().max().item()


def get_err_ratio(x, y):
    err = (x.detach() - y.detach()).flatten().square().mean().sqrt().item()
    base = (x.detach()).flatten().square().mean().sqrt().item()
    return err / (base + 1e-8)


def assert_close(prefix, ref, tri, ratio, warning=False, err_atol=1e-6):
    abs_atol = get_abs_err(ref, tri)
    msg = f"{prefix:>16} diff: {abs_atol:.6f} ratio: {get_err_ratio(ref, tri):.6f}"
    error_rate = get_err_ratio(ref, tri)
    if abs_atol <= err_atol:
        return
    else:
        assert error_rate < ratio, msg


def _causal_conv1d_update_ref(
    x,
    conv_state,
    weight,
    bias=None,
    activation=None,
    cache_seqlens=None,
    conv_state_indices=None,
):
    """
    x: (batch, dim) or (batch, dim, seqlen)
    conv_state: (batch, dim, state_len), where state_len >= width - 1
    weight: (dim, width)
    bias: (dim,)
    cache_seqlens: (batch,), dtype int32.
        If not None, the conv_state is treated as a circular buffer.
        The conv_state will be updated by copying x to the
        conv_state starting at the index
        @cache_seqlens % state_len before performing the convolution.

    out: (batch, dim) or (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    width = weight.shape[1]
    state_len = conv_state.shape[-1]
    assert weight.shape == (dim, width)

    global FORMA_INDICES

    # skip padding
    real_bs = 0
    for i in range(batch):
        if conv_state_indices[i] != -1:
            real_bs += 1
        else:
            break
    real_x = x[:real_bs]
    real_conv_state_indices = conv_state_indices[:real_bs]
    out = x.clone()

    if cache_seqlens is None:
        x_new = torch.cat([conv_state[real_conv_state_indices], real_x], dim=-1).to(
            weight.dtype
        )  # (batch, dim, state_len + seqlen)
        to_copy = x_new[:, :, -state_len:]
        conv_state[real_conv_state_indices] = to_copy
    else:
        width_idx = torch.arange(
            -(width - 1), 0, dtype=torch.long, device=x.device
        ).unsqueeze(0) + cache_seqlens.unsqueeze(1)
        width_idx = (
            torch.remainder(width_idx, state_len).unsqueeze(1).expand(-1, dim, -1)
        )
        x_new = torch.cat([conv_state.gather(2, width_idx), x], dim=-1).to(weight.dtype)
        copy_idx = torch.arange(seqlen, dtype=torch.long, device=x.device).unsqueeze(
            0
        ) + cache_seqlens.unsqueeze(1)
        copy_idx = torch.remainder(copy_idx, state_len).unsqueeze(1).expand(-1, dim, -1)
        conv_state.scatter_(2, copy_idx, x)

    real_out = F.conv1d(x_new, weight.unsqueeze(1), bias, padding=0, groups=dim)[
        :, :, -seqlen:
    ]
    out[:real_bs] = real_out
    if unsqueeze:
        out = out.squeeze(-1)
    return (out if activation is None else F.silu(out)).to(dtype=dtype_in)


def causal_conv1d_update_ref(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    conv_state_indices: Optional[torch.Tensor] = None,
    pad_slot_id: int = PAD_SLOT_ID,
    use_triton: bool = False,
):
    """
    x: (batch, dim) or (batch, dim, seqlen)
    conv_state: (batch, dim, state_len), where state_len >= width - 1
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
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError(
            f"activation must be None, silu, or swish, actual: {activation}"
        )
    activation_val = activation in ["silu", "swish"]
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)

    x = _causal_conv1d_update_ref(
        x,
        conv_state,
        weight,
        bias,
        activation=activation,
        conv_state_indices=conv_state_indices,
    )
    if unsqueeze:
        x = x.squeeze(-1)
    return x


@pytest.mark.parametrize(
    ("N", "T", "D", "W", "activation", "has_bias", "has_residual", "dtype"),
    [
        pytest.param(
            *test,
            id="N{0}_T{1}_D{2}_W{3}_activation{4}_has_bias{5}_has_residual{6}_{7}".format(
                *test
            ),
        )
        for test in [
            (16, 16, 1024, 3, "swish", True, False, torch.float16),
            (16, 32, 1024, 3, "swish", False, False, torch.float16),
            (32, 32, 2048, 3, "swish", True, False, torch.float16),
            (32, 64, 2048, 3, "swish", False, False, torch.float16),
            (16, 16, 1024, 4, "swish", True, False, torch.float16),
            (16, 32, 1024, 4, "swish", False, False, torch.float16),
            (32, 32, 2048, 4, "swish", True, False, torch.float16),
            (32, 64, 2048, 4, "swish", False, False, torch.float16),
        ]
    ],
)
@torch.no_grad
def test_conv_varlen_update(
    N: int,
    T: int,
    D: int,
    W: int,
    activation: str,
    has_bias: bool,
    has_residual: bool,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    assert T % N == 0
    seq_len = T // N
    cu_seqlens = torch.arange(0, T + 1, seq_len).to(device).long()
    state_indexs = list(range(T))
    random.shuffle(state_indexs)
    state_indexs = torch.Tensor(state_indexs).to(device).long()

    x = torch.randn(T, D).to(device, dtype)
    weight = torch.randn(D, W).to(device, dtype) * 0
    bias = torch.randn(D).to(device, dtype) if has_bias else None
    residual = x.clone() if has_residual else None
    conv_states_ref = torch.randn(T + 1, D, W - 1).to(device, dtype)
    conv_states_tri = conv_states_ref.clone()

    ref = causal_conv1d_update_ref(
        x=x,
        conv_state=conv_states_ref,
        weight=weight,
        bias=bias,
        activation=activation,
        conv_state_indices=state_indexs,
    )
    if has_residual:
        ref += residual

    tri = causal_conv1d_update_npu(
        x=x,
        conv_state=conv_states_tri,
        weight=weight,
        bias=bias,
        activation=activation,
        conv_state_indices=state_indexs,
    )
    if has_residual:
        tri += residual

    assert_close("    y", ref, tri, 1e-3)
    assert_close("cache", conv_states_ref, conv_states_tri, 1e-3)


def native_causal_conv1d_update_mtp(
    hidden_state: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    silu_activation: bool = True,
):
    bsz, hidden_size, seq_len = hidden_state.shape
    kernel_size = weight.shape[-1]
    target_state_len = (kernel_size - 1) + (seq_len - 1)
    full_context = torch.cat([conv_state, hidden_state], dim=-1).to(weight.dtype)
    computation_input = full_context[:, :, -(kernel_size - 1 + seq_len) :]
    windows = computation_input.unfold(-1, kernel_size, 1)

    out = (windows * weight[None, :, None, :]).sum(dim=-1)
    if bias is not None:
        out = out + bias[None, :, None]
    if silu_activation:
        out = F.silu(out)
    out = out.to(hidden_state.dtype)

    if target_state_len > 0:
        new_conv_state = full_context[:, :, -target_state_len:]
    else:
        new_conv_state = torch.empty(
            bsz, hidden_size, 0, device=hidden_state.device, dtype=hidden_state.dtype
        )

    return out, new_conv_state

def test_npu_causal_conv1d_update_mtp():
    device = "npu"
    itype = torch.bfloat16
    rtol, atol = (5e-2, 5e-2) if itype == torch.float32 else (5e-2, 5e-2)
    if itype == torch.bfloat16:
        rtol, atol = 1e-2, 5e-2

    batch_size = 5
    draft_token_num = 4
    dim = 4096
    kernel_size = 4  # width
    num_states = 929
    state_len = kernel_size - 1 + draft_token_num - 1

    slen = batch_size * draft_token_num

    # 1. mixed_qkv: [slen, dim]
    mixed_qkv = torch.randn(slen, dim, dtype=itype, device=device)

    # 2. Conv states: [num_states, state_len, dim] 
    conv_states = torch.randn(num_states, state_len, dim, dtype=itype, device=device)
    conv_states_clone = conv_states.detach().clone()

    # 3. Weight: [dim, kernel_size] 
    conv_weights = torch.randn(dim, kernel_size, dtype=itype, device=device)

    # 4. Bias: [dim]
    bias = torch.randn(dim, dtype=itype, device=device)

    # 5. Cache indices: [batch_size]
    cache_indices = torch.randint(0, num_states, (batch_size,), dtype=torch.int32, device=device)

    num_accepted_tokens = torch.full(
        (batch_size,),
        draft_token_num,
        dtype=torch.int32,
        device=device,
    )
    
    activation = "silu"
    pad_slot_id = -1
    validate_data = False

    mixed_qkv_reshaped = mixed_qkv.view(batch_size, draft_token_num, dim).clone()

    # ---------------- 1. NPU Kernel ----------------
    out_npu = causal_conv1d_update_mtp_npu(
        x=mixed_qkv.view(batch_size, draft_token_num, -1).contiguous(),
        conv_state=conv_states.contiguous(),
        weight=conv_weights.transpose(0, 1).contiguous(),  # [kernel_size, dim]
        bias=bias,
        activation=activation,
        conv_state_indices=cache_indices,
        num_accepted_tokens=num_accepted_tokens,
        pad_slot_id=pad_slot_id,
        validate_data=validate_data,
    ).view(slen, -1)

    # ---------------- 2. ref ----------------
    conv_states_to_use = conv_states_clone[cache_indices]  # [batch_size, state_len, dim]

    mixed_qkv_processed_ref, new_conv_state_ref = native_causal_conv1d_update_mtp(
        hidden_state=mixed_qkv_reshaped.transpose(1, 2).contiguous(),  # [batch, dim, draft_token_num]
        weight=conv_weights,  # [dim, kernel_size]
        conv_state=conv_states_to_use.transpose(1, 2).contiguous(),  # [batch, dim, state_len]
        bias=bias,
        silu_activation=True,
    )

    out_ref = (
        mixed_qkv_processed_ref.transpose(1, 2).contiguous().view(slen, -1)
    )

    conv_states_clone[cache_indices] = new_conv_state_ref.transpose(1, 2).contiguous()

    assert out_npu.shape == out_ref.shape, f"Output shape mismatch: {out_npu.shape} vs {out_ref.shape}"
    assert torch.allclose(out_npu, out_ref, rtol=rtol, atol=atol), f"Output value mismatch,{(out_npu-out_ref).max()=}"

    assert torch.allclose(conv_states, conv_states_clone, rtol=rtol, atol=atol), "Conv state update mismatch"

    print("✅  Test passed!")


