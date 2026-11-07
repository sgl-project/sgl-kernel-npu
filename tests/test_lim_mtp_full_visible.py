"""Acceptance gates for full-visible Q4 LIM and exact-K CopySFA on A3.

Run on one allocated die: pytest -q tests/test_lim_mtp_full_visible.py
The exact-K CopySFA case requires the proposed opt-in operator argument.
"""

import json
import math

import pytest
import torch

pytest.importorskip("torch_npu")
pytest.importorskip("sgl_kernel_npu")

from test_copy_sfa_dynamic_tnd import _DynamicTndCase

CAPACITY = 12288
TOPK = 2048
ROUTES = 4
LENGTH = 16389


def _lim_case(length=LENGTH):
    device = torch.device("npu", torch.npu.current_device())
    source_capacity = (length + 127) // 128 * 128
    query = torch.zeros(ROUTES, 32, 128, dtype=torch.bfloat16, device=device)
    query[:, :, 0] = 1
    weights = torch.full((ROUTES, 32), 1 / 32, dtype=query.dtype, device=device)
    key = torch.zeros(
        source_capacity // 128, 128, 1, 128, dtype=query.dtype, device=device
    )
    key_flat = key.view(-1, 128)
    key_flat[:, 0] = -8
    key_flat[:TOPK, 0] = 4
    # Each query must select its newest visible token, including the partial
    # page that the stable-prefix MTP policy omits. Future rows score highest.
    key_flat[length - ROUTES : length, 0] = 16

    def vector(value):
        return torch.tensor([value], dtype=torch.int32, device=device)

    ids = torch.empty((ROUTES, 1, TOPK), dtype=torch.int32, device=device)
    misses = torch.empty((1, 32768), dtype=torch.int32, device=device)
    return [
        weights,
        torch.ones_like(weights, dtype=torch.float32),
        query,
        torch.ones(key.shape[:3], dtype=torch.float32, device=device),
        key,
        torch.arange(source_capacity // 128, dtype=torch.int32, device=device)[None],
        vector(ROUTES),
        vector(length),
        vector(length),
        vector(CAPACITY),
        vector(-2),
        vector(0),
        torch.full((1, source_capacity), -1, dtype=torch.int32, device=device),
        ids,
        torch.empty_like(ids),
        torch.empty(ROUTES, dtype=torch.int32, device=device),
        misses,
        torch.empty_like(misses),
        vector(0),
    ]


def _check_lim(args, resident, length=LENGTH):
    torch.npu.synchronize()
    sources, slots = args[13].cpu()[:, 0], args[14].cpu()[:, 0]
    miss_sources, miss_slots, count = (
        args[16].cpu()[0],
        args[17].cpu()[0],
        int(args[18].item()),
    )
    assert 0 < count <= CAPACITY
    assert torch.all((miss_slots[:count] >= 0) & (miss_slots[:count] < CAPACITY))
    assert miss_slots[:count].unique().numel() == count
    resident[miss_slots[:count].long()] = miss_sources[:count]
    valid_slots = (slots >= 0) & (slots < CAPACITY)
    mismatches = int(
        (
            (resident[slots.clamp(0, CAPACITY - 1).long()] != sources) | ~valid_slots
        ).sum()
    )
    future = sum(
        int((sources[r] >= length - ROUTES + r + 1).sum()) for r in range(ROUTES)
    )
    print(
        "LIM_METRICS "
        + json.dumps(
            {
                "length": length,
                "queries": ROUTES,
                "selected": ROUTES * TOPK,
                "future_violations": future,
                "slot_payload_mismatches": mismatches,
                "miss_count": count,
            }
        )
    )
    for route in range(ROUTES):
        visible = length - ROUTES + route + 1
        assert sources[route].unique().numel() == TOPK
        assert torch.all((sources[route] >= 0) & (sources[route] < visible))
        # The score oracle has 2048 keys at 4 and each visible tail key at
        # 16; every other key scores -8. Equal-score ties within the prefix
        # are interchangeable, but including any -8 key is incorrect.
        assert torch.all((sources[route] < TOPK) | (sources[route] >= length - ROUTES))
        assert set(range(length - ROUTES, visible)).issubset(
            set(sources[route].tolist())
        )
        assert torch.all((slots[route] >= 0) & (slots[route] < CAPACITY))
        torch.testing.assert_close(
            resident[slots[route].long()], sources[route], rtol=0, atol=0
        )


def test_baseline_q4_c12288_existing_abi_rejects_full_visible():
    """Record the legacy ABI sentinel outcome without the opt-in mode."""
    args = _lim_case()
    torch.ops.npu.fused_lightning_indexer_manage(*args)
    torch.npu.synchronize()
    assert torch.all(args[13].cpu() == -1)
    assert int(args[18].item()) == 0


@pytest.mark.parametrize("length", [16384, 16385, 16389])
@pytest.mark.parametrize("graph", [False, True])
def test_lim_q4_c12288_full_visible_union_and_causal(graph, length):
    args = _lim_case(length)
    resident = torch.full((CAPACITY,), -1, dtype=torch.int32)
    op = torch.ops.npu.fused_lightning_indexer_manage
    if graph:
        op(*args, full_visible=True)
        torch.npu.synchronize()
        capture = torch.npu.NPUGraph()
        with torch.npu.graph(capture):
            op(*args, full_visible=True)
        args[10].fill_(-2)
        capture.replay()
    else:
        op(*args, full_visible=True)
    _check_lim(args, resident, length)
    # Reusing the request after rejection must force a cold refill.
    args[10].fill_(-2)
    resident.fill_(-1)
    if graph:
        capture.replay()
    else:
        op(*args, full_visible=True)
    _check_lim(args, resident, length)


@pytest.mark.parametrize("graph", [False, True])
def test_lim_q4_c12288_full_visible_steady_residency(graph):
    args = _lim_case()
    resident = torch.full((CAPACITY,), -1, dtype=torch.int32)
    op = torch.ops.npu.fused_lightning_indexer_manage
    op(*args, full_visible=True)
    _check_lim(args, resident)
    cold_pool = args[12].clone()
    keys = args[4].view(-1, 128)
    keys[:, 0] = -8
    keys[14000 : 14000 + TOPK, 0] = 4
    keys[LENGTH - ROUTES : LENGTH, 0] = 16
    args[10].fill_(-1)
    if graph:
        capture = torch.npu.NPUGraph()
        with torch.npu.graph(capture):
            op(*args, full_visible=True)
        args[12].copy_(cold_pool)
        capture.replay()
    else:
        op(*args, full_visible=True)
    torch.npu.synchronize()
    sources = args[13].cpu()[:, 0]
    slots = args[14].cpu()[:, 0]
    count = int(args[18].item())
    assert 0 < count <= CAPACITY
    miss_sources, miss_slots = args[16].cpu()[0, :count], args[17].cpu()[0, :count]
    assert torch.all((miss_slots >= 0) & (miss_slots < CAPACITY))
    resident[miss_slots.long()] = miss_sources
    for route in range(ROUTES):
        visible = LENGTH - ROUTES + route + 1
        assert torch.all((sources[route] >= 0) & (sources[route] < visible))
        assert torch.all(
            ((sources[route] >= 14000) & (sources[route] < 14000 + TOPK))
            | (sources[route] >= LENGTH - ROUTES)
        )
        assert torch.all((slots[route] >= 0) & (slots[route] < CAPACITY))
        torch.testing.assert_close(
            resident[slots[route].long()], sources[route], rtol=0, atol=0
        )


@pytest.mark.parametrize("graph", [False, True])
def test_lim_q4_partial_chunk_preserves_live_batch_slot_ownership(graph):
    """A warm partial chunk must not admit an invalid slot-zero victim."""
    batch = 16
    length = 66817
    source_capacity = (length + 127) // 128 * 128
    cases = [_lim_case(length) for _ in range(batch)]
    args = [None] * 19
    for index in (0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18):
        args[index] = torch.cat([case[index] for case in cases], dim=0)
    device = args[0].device
    pages = source_capacity // 128
    args[5] = torch.arange(batch * pages, dtype=torch.int32, device=device).view(
        batch, pages
    )
    args[6] = torch.arange(4, 4 * batch + 1, 4, dtype=torch.int32, device=device)
    args[11].copy_(torch.arange(1, batch + 1, dtype=torch.int32, device=device))
    args[12] = torch.full(
        (batch + 1, source_capacity), -1, dtype=torch.int32, device=device
    )
    keys = args[4].view(batch, source_capacity, 128)
    op = torch.ops.npu.fused_lightning_indexer_manage

    def configure(actual, state, tail_high):
        keys.zero_()
        if tail_high:
            keys[:, :, 0] = -8
            keys[:, :TOPK, 0] = 4
            keys[:, actual - ROUTES : actual, 0] = 16
        else:
            keys[:, :, 0] = 1
            keys[:, 14000 : 14000 + TOPK, 0] = 4
            keys[:, actual - ROUTES : actual, 0] = -8
        args[7].fill_(actual)
        args[8].fill_(actual)
        args[10].fill_(-2 if state == "cold" else -1)
        if state == "mixed":
            args[10][::2] = -2

    if graph:
        configure(65536, "cold", True)
        saved = args[12].clone()
        op(*args, full_visible=True)
        torch.npu.synchronize()
        args[12].copy_(saved)
        capture = torch.npu.NPUGraph()
        with torch.npu.graph(capture):
            op(*args, full_visible=True)
        torch.npu.synchronize()
        args[12].copy_(saved)

    steps = (
        (65536, "cold", True),
        (65536, "warm", True),
        (65537, "warm", True),
        (65537, "warm", False),
        (65537, "mixed", True),
        (65537, "warm", False),
    )
    for step, (actual, state, tail_high) in enumerate(steps):
        configure(actual, state, tail_high)
        before = args[12].cpu()
        if step:
            for row in range(1, batch + 1):
                resident = before[row][before[row] >= 0]
                assert resident.numel() == CAPACITY
                assert resident.unique().numel() == CAPACITY
        if graph:
            capture.replay()
        else:
            op(*args, full_visible=True)
        torch.npu.synchronize()
        mapping = args[12].cpu()
        sources = args[13].cpu().reshape(batch, ROUTES, TOPK)
        slots = args[14].cpu().reshape(batch, ROUTES, TOPK)
        miss_sources, miss_slots = args[16].cpu(), args[17].cpu()
        counts = args[18].cpu().tolist()
        assert torch.equal(mapping[0], before[0])
        evicted_tail = retained_refresh = 0
        for row in range(batch):
            pool_row = row + 1
            resident = mapping[pool_row][mapping[pool_row] >= 0]
            assert resident.numel() == CAPACITY, (step, pool_row)
            assert resident.unique().numel() == CAPACITY, (step, pool_row)
            assert bool((resident < CAPACITY).all())
            count = counts[row]
            assert 0 <= count <= CAPACITY
            plan_sources = miss_sources[row, :count]
            plan_slots = miss_slots[row, :count]
            assert plan_sources.unique().numel() == count
            assert plan_slots.unique().numel() == count
            assert bool(((plan_sources >= 0) & (plan_sources < actual)).all())
            assert bool(((plan_slots >= 0) & (plan_slots < CAPACITY)).all())
            assert torch.equal(mapping[pool_row, plan_sources.long()], plan_slots)
            for route in range(ROUTES):
                selected = sources[row, route]
                visible = actual - ROUTES + route + 1
                assert selected.unique().numel() == TOPK
                assert bool(((selected >= 0) & (selected < visible)).all())
                assert bool(
                    ((slots[row, route] >= 0) & (slots[row, route] < CAPACITY)).all()
                )
                assert torch.equal(
                    mapping[pool_row, selected.long()], slots[row, route]
                )
                if tail_high:
                    assert bool(
                        ((selected < TOPK) | (selected >= actual - ROUTES)).all()
                    )
                    assert set(range(actual - ROUTES, visible)).issubset(
                        selected.tolist()
                    )
                else:
                    assert bool(((selected >= 14000) & (selected < 14000 + TOPK)).all())
            plan = set(zip(plan_sources.tolist(), plan_slots.tolist()))
            for source in range(actual - ROUTES, actual):
                old_slot = int(before[pool_row, source])
                new_slot = int(mapping[pool_row, source])
                if old_slot >= 0 and new_slot < 0:
                    evicted_tail += any(
                        s != source and slot == old_slot for s, slot in plan
                    )
                if (
                    old_slot >= 0
                    and new_slot == old_slot
                    and (source, old_slot) in plan
                ):
                    retained_refresh += 1
        if not tail_high:
            assert evicted_tail > 0, (step, "tail eviction was not exercised")
        if step == 1:
            assert retained_refresh > 0, "resident refresh was not exercised"


def test_lim_q4_steady_hits_include_rewritten_residents():
    args = _lim_case()
    resident = torch.full((CAPACITY,), -1, dtype=torch.int32)
    op = torch.ops.npu.fused_lightning_indexer_manage
    op(*args, full_visible=True)
    _check_lim(args, resident)
    args[10].fill_(-1)
    op(*args, full_visible=True)
    torch.npu.synchronize()
    count = int(args[18].item())
    assert count == ROUTES
    sources = args[16].cpu()[0, :count]
    slots = args[17].cpu()[0, :count]
    assert sorted(sources.tolist()) == list(range(LENGTH - ROUTES, LENGTH))
    assert slots.unique().numel() == ROUTES
    torch.testing.assert_close(resident[slots.long()], sources, rtol=0, atol=0)


@pytest.mark.parametrize("refresh_tail", [False, True])
@pytest.mark.parametrize("graph", [False, True])
def test_copy_sfa_full_visible_refreshes_resident_hit(graph, refresh_tail):
    case = _DynamicTndCase(
        query_counts=[ROUTES],
        heads=8,
        cache_budgets=[CAPACITY],
        first_fill=True,
        tail_tokens=4,
    )
    try:

        def run():
            torch.ops.npu.fused_scatter_copy_sparse_flash_attention(
                case.query,
                case.query_rope,
                case.hbm,
                case.actual_q,
                case.actual_kv,
                case.cache_tokens,
                case.topk_src,
                case.sparse_slots,
                case.topk_miss_counts,
                case.miss_src,
                case.miss_dst,
                case.miss_counts,
                case.hbm_block_table,
                case.dram_block_table,
                case.dram_dev_ptr,
                case.output,
                case.scale,
                full_visible=True,
                resident_refresh_tail=refresh_tail,
            )

        run()
        torch.npu.synchronize()
        case.topk_src.copy_(case.sparse_slots)
        case.topk_miss_counts.zero_()  # all routes hit; refresh still must copy
        case.miss_src[0, 0] = 0
        case.miss_dst[0, 0] = 0
        case.miss_counts.fill_(1)
        if graph:
            capture = torch.npu.NPUGraph()
            with torch.npu.graph(capture):
                run()
        # Simulate a rejected draft whose absolute position is rewritten.
        torch.npu.synchronize()
        case.dram[0, 0].fill_(3)
        if graph:
            capture.replay()
        else:
            run()
        torch.npu.synchronize()
        torch.testing.assert_close(
            case.hbm[0, 0, 0].cpu(), case.dram[0, 0], rtol=0, atol=0
        )
        selected = case.hbm.cpu().view(-1, 576)[:TOPK].float()
        q, rope = case.query.cpu().float(), case.query_rope.cpu().float()
        scores = (q @ selected[:, :512].T + rope @ selected[:, 512:].T) / math.sqrt(576)
        expected = torch.softmax(scores, dim=-1) @ selected[:, :512]
        error = case.output.cpu().float() - expected
        print(
            "COPY_SFA_REFRESH_METRICS "
            + json.dumps(
                {
                    "graph": graph,
                    "max_abs": error.abs().max().item(),
                    "rms": error.square().mean().sqrt().item(),
                }
            )
        )
        torch.testing.assert_close(
            case.output.cpu().float(), expected, rtol=0.01, atol=0.002
        )
    finally:
        case.close()


@pytest.mark.parametrize("refresh_tail", [False, True])
def test_copy_sfa_q4_c12288_exact_k_no_tail(refresh_tail):
    case = _DynamicTndCase(
        query_counts=[ROUTES], heads=8, cache_budgets=[CAPACITY], tail_tokens=4
    )
    try:
        # The reference is exactly the chosen K resident slots; extra tail
        # values would perturb both softmax and output for these random inputs.
        selected = case.hbm.cpu().view(-1, 576)[:TOPK].float()
        q, rope = case.query.cpu().float(), case.query_rope.cpu().float()
        scores = (q @ selected[:, :512].T + rope @ selected[:, 512:].T) / math.sqrt(576)
        expected = torch.softmax(scores, dim=-1) @ selected[:, :512]
        torch.ops.npu.fused_scatter_copy_sparse_flash_attention(
            case.query,
            case.query_rope,
            case.hbm,
            case.actual_q,
            case.actual_kv,
            case.cache_tokens,
            case.topk_src,
            case.sparse_slots,
            case.topk_miss_counts,
            case.miss_src,
            case.miss_dst,
            case.miss_counts,
            case.hbm_block_table,
            case.dram_block_table,
            case.dram_dev_ptr,
            case.output,
            case.scale,
            full_visible=True,
            resident_refresh_tail=refresh_tail,
        )
        torch.npu.synchronize()
        error = case.output.cpu().float() - expected
        print(
            "COPY_SFA_METRICS "
            + json.dumps(
                {
                    "max_abs": error.abs().max().item(),
                    "rms": error.square().mean().sqrt().item(),
                    "values": error.numel(),
                }
            )
        )
        torch.testing.assert_close(
            case.output.cpu().float(), expected, rtol=0.01, atol=0.002
        )
        # A padded replay must not consume stale C or the previous output.
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            torch.ops.npu.fused_scatter_copy_sparse_flash_attention(
                case.query,
                case.query_rope,
                case.hbm,
                case.actual_q,
                case.actual_kv,
                case.cache_tokens,
                case.topk_src,
                case.sparse_slots,
                case.topk_miss_counts,
                case.miss_src,
                case.miss_dst,
                case.miss_counts,
                case.hbm_block_table,
                case.dram_block_table,
                case.dram_dev_ptr,
                case.output,
                case.scale,
                full_visible=True,
                resident_refresh_tail=refresh_tail,
            )
        case.actual_kv.zero_()
        case.output.fill_(123)
        graph.replay()
        torch.npu.synchronize()
        padding_max = case.output.cpu().abs().max().item()
        print("COPY_SFA_PADDING_METRICS " + json.dumps({"max_abs": padding_max}))
        torch.testing.assert_close(
            case.output.cpu(), torch.zeros_like(case.output.cpu()), rtol=0, atol=0
        )
    finally:
        case.close()
