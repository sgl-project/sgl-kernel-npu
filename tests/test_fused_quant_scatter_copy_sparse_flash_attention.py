# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Minimal functional test for FusedQuantScatterCopySparseFlashAttention."""

from __future__ import annotations

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")


from fused_quant_scatter_copy_sparse_flash_attention_golden import (  # noqa: E402
    MAX_CACHE_TOKENS,
    NOPE_DIM,
    TOPK,
    make_test_data,
)

_dram_addresses = {}


@pytest.fixture(scope="module", autouse=True)
def load_a5_custom_operator():
    import acl
    import sgl_kernel_npu  # noqa: F401
    from sgl_kernel_npu.sparsity_driven_kv_offload import free_shm

    if acl.rt.get_device_info(0, 601) != (3510, 0):
        pytest.skip("C8 numerical kernels require Ascend950 (A5)")
    assert hasattr(torch.ops.npu, "fused_quant_scatter_copy_sparse_flash_attention")
    yield
    torch.npu.synchronize()
    _dram_addresses.clear()
    free_shm()


def swapped_from_cpu(cpu_tensor, device):
    from sgl_kernel_npu.sparsity_driven_kv_offload import create_shm_tensor

    owner, _, pointer = create_shm_tensor(
        cpu_tensor.shape, torch.int8, device_id=device.index or 0
    )
    owner.copy_(cpu_tensor)
    _dram_addresses[owner.data_ptr()] = pointer
    return owner


def run_operator(query, hbm, dram, *args):
    return torch.ops.npu.fused_quant_scatter_copy_sparse_flash_attention(
        query, hbm, dram, _dram_addresses[dram.data_ptr()], *args
    )


def check_attention(actual: torch.Tensor, golden: torch.Tensor) -> None:
    actual = actual.detach().cpu().float()
    golden = golden.float()
    if actual.shape != golden.shape or not bool(torch.isfinite(actual).all()):
        raise AssertionError(
            "attention output has an invalid shape or non-finite value"
        )
    relative_l2 = (actual - golden).flatten(1).norm(dim=1) / golden.flatten(1).norm(
        dim=1
    ).clamp_min(1e-7)
    if (
        not torch.allclose(actual, golden, atol=0.008, rtol=0.05)
        or float(relative_l2.max()) > 0.08
    ):
        raise AssertionError(
            f"attention output differs from CPU golden, max_rel_l2={float(relative_l2.max()):.6f}"
        )


def run_test_data(data) -> None:
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    hbm_kv_bytes = data.initial_hbm.to(device)
    dram_kv_bytes = swapped_from_cpu(data.dram, device)
    inputs = [
        data.query,
        data.hbm_block_table,
        data.dram_block_table,
        data.topk_dst_slots,
        data.topk_src_ids,
        data.topk_miss_counts,
        data.actual_seq_lengths_query,
        data.actual_seq_lengths_kv,
        data.num_cache_tokens,
        data.miss_src_ids,
        data.miss_dst_slots,
        data.miss_counts,
    ]
    device_inputs = [tensor.to(device) for tensor in inputs]
    attention_out = torch.empty(
        (*data.query.shape[:-1], NOPE_DIM), dtype=data.query.dtype, device=device
    )
    run_operator(
        device_inputs[0],
        hbm_kv_bytes,
        dram_kv_bytes,
        *device_inputs[1:],
        float(data.scale_value),
        attention_out,
    )
    torch.npu.synchronize()
    if not torch.equal(hbm_kv_bytes.cpu(), data.expected_hbm):
        raise AssertionError("hbm_kv_bytes was not updated in place")
    check_attention(attention_out, data.attention_golden)


@pytest.mark.parametrize(
    ("miss_count", "cache_tokens"),
    [
        (0, TOPK),
        (8, TOPK),
        (512, TOPK),
        (513, TOPK),
        (8, MAX_CACHE_TOKENS),
        (512, MAX_CACHE_TOKENS),
        (513, MAX_CACHE_TOKENS),
    ],
    ids=[
        "non_offload",
        "fused_miss",
        "fused_miss_boundary",
        "union_copy_fallback",
        "high_slot_fused_miss",
        "high_slot_fused_miss_boundary",
        "high_slot_union_copy_fallback",
    ],
)
def test_fused_quant_scatter_copy_sparse_flash_attention(
    miss_count: int, cache_tokens: int
) -> None:
    data = make_test_data(miss_count, cache_tokens)
    run_test_data(data)


@pytest.mark.parametrize(
    ("query_dtype", "query_heads"),
    [
        (torch.float16, 8),
        (torch.bfloat16, 16),
        (torch.bfloat16, 32),
        (torch.bfloat16, 64),
        (torch.bfloat16, 128),
    ],
)
def test_fused_quant_scatter_copy_sparse_flash_attention_dispatch(
    query_dtype: torch.dtype, query_heads: int
) -> None:
    data = make_test_data(
        8, query_dtype=query_dtype, query_heads=query_heads, seed=20261032
    )
    run_test_data(data)
