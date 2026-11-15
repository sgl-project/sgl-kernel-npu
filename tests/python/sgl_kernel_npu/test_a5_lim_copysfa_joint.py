"""Actual A5 original-LIM output consumed by native CopySFA, eager and graph.

Cold, warm and request-reuse phases keep captured addresses fixed. Precision
and whole-HBM payload checks use the real native LIM selection, not frozen plans.
"""

import math
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("torch_npu")


@pytest.mark.parametrize("width", [1, 4])
@pytest.mark.parametrize("graph_mode", [False, True])
def test_real_lim_to_copysfa(width, graph_mode):
    import acl
    from sgl_kernel_npu.sparsity_driven_kv_offload.fused_shm import (
        create_fused_shm_tensor as create_shm_tensor,
    )
    from sgl_kernel_npu.sparsity_driven_kv_offload.fused_shm import (
        free_fused_shm as free_shm,
    )

    if acl.rt.get_device_info(0, 601) != (3510, 0):
        pytest.skip("Joint quantized attention requires Ascend950")
    torch.npu.set_device(0)
    operator = torch.ops.npu.fused_quant_lightning_indexer_manage.default
    kernel_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(kernel_root / "tests"))
    from fused_quant_scatter_copy_sparse_flash_attention_golden import (
        make_packed_bytes,
        unpack_rows,
    )

    prefix, capacity = 8320, 8192
    final_length = prefix + (129 if width == 4 else 127)
    source_capacity = (final_length + 127) // 128 * 128
    pages = source_capacity // 128
    device = "npu"
    query = (
        torch.ones(width, 32, 128, dtype=torch.float32)
        .to(torch.float8_e4m3fn)
        .to(device)
    )
    key_cpu = torch.zeros(source_capacity, 128)
    # Tail scores strictly exceed every prefix score, guaranteeing selection.
    key_cpu[prefix:final_length] = 1
    key = key_cpu.to(torch.float8_e4m3fn).view(pages, 128, 1, 128).to(device)
    weights = torch.ones(width, 32, dtype=torch.bfloat16, device=device)
    query_scale = torch.ones(width, 32, dtype=torch.float32, device=device)
    key_scale = torch.ones(pages, 128, 1, dtype=torch.float32, device=device)
    table = torch.arange(pages, dtype=torch.int32, device=device).view(1, pages)
    ends = torch.tensor([width], dtype=torch.int32, device=device)
    lengths = torch.tensor([final_length], dtype=torch.int32, device=device)
    request = torch.tensor([1], dtype=torch.int32, device=device)
    pool = torch.full((3, source_capacity), -1, dtype=torch.int32, device=device)
    sources = torch.full((width, 1, 2048), -1, dtype=torch.int32, device=device)
    destinations = torch.full_like(sources, -1)
    route_misses = torch.empty(width, dtype=torch.int32, device=device)
    miss_src = torch.empty(1, 32768, dtype=torch.int32, device=device)
    miss_dst = torch.empty_like(miss_src)
    misses = torch.empty(1, dtype=torch.int32, device=device)

    generator = torch.Generator().manual_seed(20261102)
    dram_cpu = make_packed_bytes(pages, generator)
    owner, _, pointer = create_shm_tensor(dram_cpu.shape, torch.int8, device_id=0)
    owner.copy_(dram_cpu)
    hbm_pages = capacity // 128 + 2
    hbm_table_cpu = torch.arange(hbm_pages, dtype=torch.int32).view(1, -1)
    # LIM uses C+source-prefix tail slots; rotate physical tail pages with prefix.
    hbm_table_cpu[0, -2:] = torch.tensor(
        [
            capacity // 128 + (prefix // 128) % 2,
            capacity // 128 + (prefix // 128 + 1) % 2,
        ]
    )
    hbm_table_cpu += hbm_pages
    hbm_table = hbm_table_cpu.to(device)
    hbm = torch.empty(3 * hbm_pages, 128, 1, 656, dtype=torch.int8, device=device)
    attention_query = (
        torch.randn(width, 8, 576, generator=generator).to(torch.bfloat16).to(device)
    )
    output = torch.empty(width, 8, 512, dtype=torch.bfloat16, device=device)
    cache_lengths = torch.tensor(
        [capacity + final_length - prefix], dtype=torch.int32, device=device
    )
    copy = torch.ops.npu.fused_quant_scatter_copy_sparse_flash_attention
    graph = None
    from sgl_kernel_npu.sparsity_driven_kv_offload.c8_metadata import (
        fused_lim_copy_sfa_metadata,
        fused_prepare_lim_metadata,
    )

    last_prefix = torch.full((4,), -1, dtype=torch.int32, device=device)
    lim_table = torch.empty_like(table)
    copy_table = torch.empty_like(table)
    copy_lengths = torch.empty_like(lengths)
    pre = {
        name: torch.empty(1, dtype=dtype, device=device)
        for name, dtype in (
            ("safe_slots", torch.int32),
            ("lengths", torch.int32),
            ("prefixes", torch.int32),
            ("sparse_rows", torch.bool),
            ("valid_rows", torch.bool),
            ("request_states", torch.int32),
            ("next_prefixes", torch.int32),
            ("cache_sizes", torch.int32),
        )
    }

    def enqueue():
        fused_prepare_lim_metadata(
            seq_lens=lengths,
            request_slots=request,
            last_prefix=last_prefix,
            block_table=table,
            out_block_table=lim_table,
            capacity=capacity,
            request_rows=3,
            full_visible=False,
            query_width=width,
            **pre,
        )
        operator(
            weights,
            query_scale,
            query,
            key_scale,
            key,
            lim_table,
            ends,
            pre["lengths"],
            pre["prefixes"],
            pre["cache_sizes"],
            pre["request_states"],
            pre["safe_slots"],
            pool,
            sources,
            destinations,
            route_misses,
            miss_src,
            miss_dst,
            misses,
        )
        last_prefix.index_copy_(0, pre["safe_slots"].long(), pre["next_prefixes"])
        fused_lim_copy_sfa_metadata(
            source_ids=sources[:, 0],
            topk_misses=route_misses,
            miss_counts=misses,
            valid_rows=pre["valid_rows"],
            sparse_rows=pre["sparse_rows"],
            request_slots=pre["safe_slots"],
            actual_seq_lengths_kv=copy_lengths,
            num_cache_tokens=cache_lengths,
            hbm_block_table=hbm_table,
            dram_block_table=copy_table,
            capacity=capacity,
            host_context_len=source_capacity,
            request_rows=3,
            seq_lens=pre["lengths"],
            query_width=width,
            host_block_table=lim_table,
            packed_c8=True,
            prefix_lens=pre["prefixes"],
        )
        copy(
            attention_query,
            hbm,
            owner,
            pointer,
            hbm_table,
            copy_table,
            destinations,
            sources,
            route_misses,
            ends,
            copy_lengths,
            cache_lengths,
            miss_src.view(1, 1, -1),
            miss_dst.view(1, 1, -1),
            misses,
            1 / math.sqrt(576),
            output,
        )

    def reset_cold():
        pool.fill_(-1)
        last_prefix.fill_(-1)
        initial = torch.full(tuple(hbm.shape), -91, dtype=torch.int8)
        for source in range(prefix, final_length):
            slot = capacity + source - prefix
            physical = int(hbm_table_cpu[0, slot // 128]) * 128 + slot % 128
            initial.view(-1, 656)[physical] = dram_cpu.view(-1, 656)[source]
        hbm.copy_(initial.to(device))
        return initial

    try:
        expected = reset_cold()
        enqueue()  # Compile ACLNN before graph capture.
        torch.npu.synchronize()
        if graph_mode:
            reset_cold()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                enqueue()
        for phase in ("cold", "warm", "request_reuse"):
            if phase != "warm":
                expected = reset_cold()
            attention_query.copy_(
                torch.randn(width, 8, 576, generator=generator)
                .to(torch.bfloat16)
                .to(device)
            )
            output.fill_(float("nan"))
            graph.replay() if graph_mode else enqueue()
            torch.npu.synchronize()
            source_ids, dst_slots = sources.cpu().long(), destinations.cpu().long()
            assert int(pre["request_states"].cpu()[0]) == (
                -1 if phase == "warm" else -2
            )
            assert int(pre["prefixes"].cpu()[0]) == prefix
            torch.testing.assert_close(hbm_table.cpu(), hbm_table_cpu, rtol=0, atol=0)
            torch.testing.assert_close(copy_table.cpu(), table.cpu(), rtol=0, atol=0)
            count = int(misses.cpu()[0])
            assert 0 <= count <= capacity
            history_src = miss_src.cpu()[0, :count].long()
            history_dst = miss_dst.cpu()[0, :count].long()
            assert bool(((history_src >= 0) & (history_src < prefix)).all())
            assert bool(((history_dst >= 0) & (history_dst < capacity)).all())
            physical = (
                hbm_table_cpu[0, history_dst // 128].long() * 128 + history_dst % 128
            )
            expected.view(-1, 656)[physical] = dram_cpu.view(-1, 656)[history_src]
            torch.testing.assert_close(hbm.cpu(), expected, rtol=0, atol=0)
            torch.testing.assert_close(owner, dram_cpu, rtol=0, atol=0)
            q = attention_query.cpu().float()
            golden = torch.empty_like(output.cpu(), dtype=torch.float32)
            for row in range(width):
                ids = source_ids[row, 0]
                slots = dst_slots[row, 0]
                visible = final_length - (width - 1 - row)
                assert bool(((ids >= 0) & (ids < visible)).all())
                tail = ids >= prefix
                assert torch.equal(slots[tail], capacity + ids[tail] - prefix)
                physical = hbm_table_cpu[0, slots // 128].long() * 128 + slots % 128
                torch.testing.assert_close(
                    expected.view(-1, 656)[physical],
                    dram_cpu.view(-1, 656)[ids],
                    rtol=0,
                    atol=0,
                )
                k, v = unpack_rows(dram_cpu.view(-1, 656)[ids])
                golden[row] = torch.softmax(q[row] @ k.T / math.sqrt(576), dim=-1) @ v
            actual = output.cpu().float()
            torch.testing.assert_close(actual, golden, atol=0.008, rtol=0.05)
            relative = (actual - golden).flatten(1).norm(dim=1) / golden.flatten(
                1
            ).norm(dim=1).clamp_min(1e-7)
            assert float(relative.max()) <= 0.08, (phase, float(relative.max()))
    finally:
        torch.npu.synchronize()
        free_shm()
