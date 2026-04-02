# -*- coding: utf-8 -*-
import random

import pytest
import torch
from sgl_kernel_npu.mamba.mamba_state_update_triton import (
    conv_state_rollback,
    move_intermediate_cache,
)

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


def conv_state_rollback_ref(
    conv_states, valid_state_indices, last_steps, draft_token_num
):
    cpu_valid_indices = valid_state_indices.cpu().numpy()
    cpu_last_steps = last_steps.cpu().numpy()

    for idx, step in zip(cpu_valid_indices, cpu_last_steps):
        # Calculate rollback steps
        shift = (draft_token_num - 1) - step
        if shift > 0:
            # Select states for this request across all layers and dims
            req_conv_state = conv_states[:, idx, :, :]

            # Perform right shift (Rollback)
            req_conv_state[:, shift:, :].copy_(req_conv_state[:, :-shift, :])

    return conv_states


@pytest.mark.parametrize(
    ("num_layers", "pool_size", "num_dims", "draft_token_num", "num_requests", "dtype"),
    [
        pytest.param(
            *test,
            id="layers{0}_pool{1}_dims{2}_draft{3}_req{4}_{5}".format(*test),
        )
        for test in [
            (32, 32, 2048, 3, 3, torch.bfloat16),
            (16, 16, 1024, 3, 2, torch.bfloat16),
            (32, 32, 2048, 4, 4, torch.bfloat16),
            (16, 16, 1024, 4, 1, torch.bfloat16),
        ]
    ],
)
@torch.no_grad
def test_conv_state_rollback(
    num_layers: int,
    pool_size: int,
    num_dims: int,
    draft_token_num: int,
    num_requests: int,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    conv_window_size = 3 + draft_token_num - 1

    conv_states = torch.randn(
        num_layers, pool_size, conv_window_size, num_dims, device=device, dtype=dtype
    )

    valid_state_indices = torch.randint(
        0, pool_size, (num_requests,), device=device, dtype=torch.int32
    )
    last_steps = torch.randint(
        -1, draft_token_num, (num_requests,), device=device, dtype=torch.int32
    )
    original_states = conv_states.clone()

    gt_states = conv_state_rollback_ref(
        original_states, valid_state_indices, last_steps, draft_token_num
    )

    result_states = conv_state_rollback(
        conv_states,
        valid_state_indices,
        last_steps,
        draft_token_num,
    )
    assert_close("conv_state", gt_states, result_states, 1e-3)


@pytest.mark.parametrize(
    ("L", "S", "D", "H", "V", "K", "num_valid", "dtype"),
    [
        pytest.param(
            *test,
            id="L{0}_S{1}_D{2}_H{3}_V{4}_K{5}_valid{6}_{7}".format(*test),
        )
        for test in [
            (36, 229, 4, 8, 128, 128, 180, torch.bfloat16),
            (18, 100, 4, 4, 64, 64, 50, torch.bfloat16),
            (36, 229, 4, 8, 128, 128, 229, torch.bfloat16),
            (18, 100, 4, 4, 64, 64, 100, torch.bfloat16),
        ]
    ],
)
@torch.no_grad
def test_move_intermediate_cache(
    L: int,
    S: int,
    D: int,
    H: int,
    V: int,
    K: int,
    num_valid: int,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    # prepare input data
    dst_cache = torch.randn(L, S, H, V, K, device=device, dtype=dtype)
    dst_cache_clone = dst_cache.clone()
    src_cache = torch.randn(L, S, D, H, V, K, device=device, dtype=dtype)

    # prepare input data
    population = range(S)
    valid_indices = random.sample(population, num_valid)
    last_step_pos = [random.randint(0, D - 1) for _ in range(num_valid)]
    dst_indices_tensor = torch.tensor(valid_indices, device=device, dtype=torch.int32)
    src_indices_tensor = torch.arange(dst_indices_tensor.shape[0], device=device, dtype=torch.int32)
    last_steps_tensor = torch.tensor(last_step_pos, device=device, dtype=torch.int32)

    valid_mask = last_steps_tensor >= 0
    dst_state_indices = dst_indices_tensor[valid_mask].to(torch.int64)
    src_state_indices = src_indices_tensor[valid_mask].to(torch.int64)
    valid_last_steps = last_steps_tensor[valid_mask].to(torch.int64)
    # prepare output verify
    dst_cache[:, dst_state_indices, :] = src_cache[
        :, src_state_indices, valid_last_steps
    ]

    move_intermediate_cache(dst_cache_clone, src_cache, dst_indices_tensor, src_indices_tensor, last_steps_tensor)

    assert_close("move_cache", dst_cache, dst_cache_clone, 1e-3)
