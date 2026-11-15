# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Minimal functional test for FusedQuantScatterCopySparseFlashAttention."""

from __future__ import annotations

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")


from fused_quant_scatter_copy_sparse_flash_attention_golden import (  # noqa: E402
    BLOCK_SIZE,
    MAX_CACHE_TOKENS,
    NOPE_DIM,
    PACKED_DIM,
    POISON,
    TOPK,
    make_packed_bytes,
    make_test_data,
    physical_rows,
    unpack_rows,
)

_dram_addresses = {}


@pytest.fixture(scope="module", autouse=True)
def load_a5_custom_operator():
    import acl
    import sgl_kernel_npu  # noqa: F401
    from sgl_kernel_npu.sparsity_driven_kv_offload.fused_shm import free_fused_shm

    if acl.rt.get_device_info(0, 601) != (3510, 0):
        pytest.skip("C8 numerical kernels require Ascend950 (A5)")
    assert hasattr(torch.ops.npu, "fused_quant_scatter_copy_sparse_flash_attention")
    yield
    torch.npu.synchronize()
    _dram_addresses.clear()
    free_fused_shm()


def swapped_from_cpu(cpu_tensor, device):
    from sgl_kernel_npu.sparsity_driven_kv_offload.fused_shm import create_fused_shm_tensor

    owner, _, pointer = create_fused_shm_tensor(
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


@pytest.mark.parametrize("query_dtype", [torch.float16, torch.bfloat16])
def test_v2_split_query_matches_legacy(query_dtype):
    data = make_test_data(8, 8448, query_dtype=query_dtype)
    device = torch.device("npu:0")
    hbm = data.initial_hbm.to(device)
    dram = swapped_from_cpu(data.dram, device)
    query = data.query.to(device)
    metadata = [
        value.to(device)
        for value in (
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
        )
    ]
    out = torch.empty((1, 8, NOPE_DIM), dtype=query_dtype, device=device)
    torch.ops.npu.fused_quant_scatter_copy_sparse_flash_attention_v2(
        query[..., :512].contiguous(),
        query[..., 512:].contiguous(),
        hbm,
        dram,
        _dram_addresses[dram.data_ptr()],
        *metadata,
        data.scale_value,
        out,
        8192,
    )
    torch.npu.synchronize()
    assert torch.equal(hbm.cpu(), data.expected_hbm)
    check_attention(out, data.attention_golden)


@pytest.mark.parametrize("first,width", [(8325, 1), (8325, 4), (8446, 4)])
@pytest.mark.parametrize("active", [True, False])
def test_v2_ring_restoration_matches_prestaged_legacy(first, width, active):
    """Check rotated pages, missing history, current overwrites and pad writes."""
    capacity = 8192
    prefix = first // BLOCK_SIZE * BLOCK_SIZE
    end = first + width
    generator = torch.Generator().manual_seed(20261113 + first + width)
    host_pages = (end + 127) // 128
    host_table = torch.randperm(host_pages, generator=generator).int()[None]
    hbm_table = torch.randperm(capacity // 128 + 2, generator=generator).int()[None]
    if prefix // 128 % 2:
        hbm_table[:, -2:] = hbm_table[:, -2:].flip(1)
    host = make_packed_bytes(host_pages, generator)
    current = make_packed_bytes(1, generator).view(-1, PACKED_DIM)[:width].clone()
    hbm = torch.full((66, 128, 1, PACKED_DIM), POISON, dtype=torch.int8)
    resident = torch.arange(capacity)
    hbm.view(-1, PACKED_DIM)[physical_rows(hbm_table, resident)] = host.view(
        -1, PACKED_DIM
    )[physical_rows(host_table, resident)]
    # The state covers only part of this prefix. Remaining history must be
    # restored from Host, while current rows differ from stale Host bytes.
    old_end = prefix + 2
    covered = torch.arange(prefix, old_end)
    hbm.view(-1, PACKED_DIM)[physical_rows(hbm_table, capacity + covered - prefix)] = (
        host.view(-1, PACKED_DIM)[physical_rows(host_table, covered)]
    )
    expected = hbm.clone()
    if active:
        history = torch.arange(prefix, first)
        expected.view(-1, PACKED_DIM)[
            physical_rows(hbm_table, capacity + history - prefix)
        ] = host.view(-1, PACKED_DIM)[physical_rows(host_table, history)]
        expected.view(-1, PACKED_DIM)[
            physical_rows(hbm_table, capacity + torch.arange(first, end) - prefix)
        ] = current
    query = torch.randn((width, 8, 576), generator=generator).bfloat16()
    sources = torch.full((width, 1, TOPK), -1, dtype=torch.int32)
    slots = torch.full_like(sources, -1)
    golden = []
    for row in range(width):
        if not active:
            continue
        tail = torch.arange(prefix, first + row + 1)
        selected = torch.cat((torch.arange(TOPK - tail.numel()), tail))
        destinations = torch.where(
            selected < prefix, selected, capacity + selected - prefix
        )
        sources[row, 0] = selected.int()
        slots[row, 0] = destinations.int()
        key, value = unpack_rows(
            expected.view(-1, PACKED_DIM)[physical_rows(hbm_table, destinations)]
        )
        golden.append(torch.softmax(query[row].float() @ key.T / 24, -1) @ value)
    empty_misses = torch.full((1, 1, 32768), -1, dtype=torch.int32)
    counts = torch.tensor([capacity + end - prefix], dtype=torch.int32)
    metadata = [
        hbm_table,
        host_table,
        slots,
        sources,
        torch.zeros(width, dtype=torch.int32),
        torch.tensor([width], dtype=torch.int32),
        torch.tensor([end], dtype=torch.int32),
        counts,
        empty_misses,
        empty_misses.clone(),
        torch.zeros(1, dtype=torch.int32),
    ]
    starts = torch.tensor([-7, -7, prefix, -7], dtype=torch.int32)
    ends = torch.tensor([-9, -9, old_end, -9], dtype=torch.int32)
    device = torch.device("npu:0")
    dram = swapped_from_cpu(host, device)
    hbm_device = hbm.to(device)
    query_device = query.to(device)
    metadata_device = [tensor.to(device) for tensor in metadata]
    starts_device, ends_device = starts.to(device), ends.to(device)
    output = torch.empty((width, 8, 512), dtype=query.dtype, device=device)
    torch.ops.npu.fused_quant_scatter_copy_sparse_flash_attention_v2(
        query_device[..., :512].contiguous(),
        query_device[..., 512:].contiguous(),
        hbm_device,
        dram,
        _dram_addresses[dram.data_ptr()],
        *metadata_device,
        1 / 24,
        output,
        capacity,
        current[:, None].to(device),
        torch.arange(first, end, dtype=torch.int64, device=device),
        torch.full((width,), active, dtype=torch.bool, device=device),
        torch.tensor([2 if active else 0], dtype=torch.int32, device=device),
        starts_device,
        ends_device,
    )
    torch.npu.synchronize()
    assert torch.equal(hbm_device.cpu(), expected)
    assert torch.equal(metadata_device[7].cpu(), counts), "input cache lengths changed"
    if active:
        starts[2], ends[2] = prefix, end
        legacy_output = torch.empty_like(output)
        run_operator(
            query_device,
            expected.to(device),
            dram,
            *metadata_device,
            1 / 24,
            legacy_output,
        )
        torch.npu.synchronize()
        check_attention(output, torch.stack(golden))
        check_attention(output, legacy_output.cpu())
    assert torch.equal(starts_device.cpu(), starts)
    assert torch.equal(ends_device.cpu(), ends)
