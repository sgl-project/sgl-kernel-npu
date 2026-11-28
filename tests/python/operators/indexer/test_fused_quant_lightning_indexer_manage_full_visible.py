import pytest
import torch
from sgl_kernel_npu.operators.indexer.fused_quant_lightning_indexer_manage_full_visible import (
    fused_quant_lightning_indexer_manage_full_visible,
)

from tests.python.utils.common import require_test_impl


@pytest.mark.parametrize("width", [1, 4])
def test_torch_native_full_visible_residency(width):
    require_test_impl(fused_quant_lightning_indexer_manage_full_visible, "torch_native")
    page = 128
    top_k = 2048
    capacity = width * top_k
    length = capacity + width + 5
    page_count = (length + page - 1) // page
    ints = dict(dtype=torch.int32)
    query = torch.ones((width, 32, 128)).to(torch.float8_e4m3fn)
    key = torch.ones((page_count, page, 1, 128)).to(torch.float8_e4m3fn)
    weights = torch.ones((width, 32), dtype=torch.bfloat16)
    scales = torch.ones((page_count, page, 1))
    scales.view(-1)[length - width : length] = 2
    pages = torch.arange(page_count, **ints)[None, :]
    pool = torch.full((3, page_count * page), -1, **ints)
    selected = torch.empty((width, top_k), **ints)
    slots = torch.empty_like(selected)
    route_counts = torch.empty(width, **ints)
    copy_sources = torch.empty((1, 32768), **ints)
    copy_slots = torch.empty_like(copy_sources)
    copy_count = torch.empty(1, **ints)
    state = torch.tensor([-2], **ints)

    def run():
        fused_quant_lightning_indexer_manage_full_visible(
            query,
            weights,
            key,
            torch.ones_like(weights, dtype=torch.float32),
            scales,
            pages,
            torch.tensor([width], **ints),
            torch.tensor([length], **ints),
            torch.tensor([capacity], **ints),
            state,
            torch.tensor([1], **ints),
            pool,
            selected,
            slots,
            route_counts,
            copy_sources,
            copy_slots,
            copy_count,
            impl="torch_native",
        )
        if state.item() != -3:
            torch.testing.assert_close(pool[1, selected.long()], slots)
        for route, row in enumerate(selected.tolist()):
            assert max(row) == length - width + route
            assert len(set(row)) == top_k

    run()
    assert copy_count.item() == capacity
    assert copy_slots[0, :capacity].unique().numel() == capacity
    assert route_counts.tolist() == [top_k] * width

    state.fill_(-1)
    run()
    assert route_counts.count_nonzero() == 0
    assert copy_count.item() == width
    assert copy_sources[0, :width].tolist() == list(range(length - width, length))

    incoming = int((pool[1, :length] < 0).nonzero()[0].item())
    scales.view(-1)[incoming] = 4
    run()
    assert incoming in selected[0, : route_counts[0]].tolist()
    assert copy_count.item() > width

    state.fill_(-3)
    run()
    torch.testing.assert_close(selected, slots)
    assert copy_count.item() == 0


def _a5_li_score_keys(query, key, weights, query_scale, key_scale):
    scores = (key.float() @ query.float().T).relu()
    scores = (scores @ (weights.float() * query_scale)) * key_scale
    # A5 uses CAST_ROUND when forming BF16 sort keys. These scores are
    # nonnegative, so adding the half-ULP before truncation rounds ties up.
    return (scores.contiguous().view(torch.int32) + 0x8000) >> 16


def _assert_only_cutoff_ties(selected, native, score_keys, label):
    cutoff = min(score_keys[source] for source in native)
    wrong = sorted(
        source for source in selected ^ native if score_keys[source] != cutoff
    )
    assert not wrong, (
        f"{label}: {len(wrong)} non-tie TopK differences, "
        f"cutoff={cutoff:#06x}, "
        f"fused_only={sorted(selected - native)[:8]}, "
        f"native_only={sorted(native - selected)[:8]}, "
        f"non_tie_keys={[(source, hex(score_keys[source])) for source in wrong[:8]]}"
    )


def test_a5_topk_score_key_rounds_bf16_midpoints_up():
    scores = _a5_li_score_keys(
        torch.ones((1, 1)),
        torch.ones((3, 1)),
        torch.ones(1),
        torch.ones(1),
        torch.tensor([1.0, 1.0 + 1 / 256, 1.0 + 1 / 128]),
    )
    assert scores.tolist() == [0x3F80, 0x3F81, 0x3F81]


def test_a5_topk_parity_only_accepts_cutoff_ties():
    native = {0, 1, 2}
    scores = {0: 5, 1: 5, 2: 3, 3: 3, 4: 2, 5: 5}
    _assert_only_cutoff_ties({0, 1, 3}, native, scores, "cutoff tie")
    with pytest.raises(AssertionError, match="non-tie TopK differences"):
        _assert_only_cutoff_ties({0, 1, 4}, native, scores, "below cutoff")
    with pytest.raises(AssertionError, match="non-tie TopK differences"):
        _assert_only_cutoff_ties({0, 2, 5}, native, scores, "above cutoff")


@pytest.mark.parametrize("width", [1, 4])
@pytest.mark.parametrize(
    # At 140242 tokens, the steady miss scan starts at the final 978 entries.
    # The former 64-element padding requested 92 bytes in one DMA instruction.
    "long_length,graph",
    [(8199, False), (140242, False), (140242, True)],
)
def test_a5_fused_lim_matches_torch_native_topk_and_resident_plan(
    width, long_length, graph
):
    """Check AscendC TopK, cold fill, graph replay and resident plans."""
    pytest.importorskip("torch_npu")
    import acl

    if not torch.npu.is_available() or acl.rt.get_device_info(0, 601) != (3510, 0):
        pytest.skip("Ascend950 required")
    device = "npu:0"
    dtype = torch.int32
    capacity, page, top_k = 8192, 128, 2048
    short_length = 16 if long_length < 131072 else 4097
    lengths_cpu = [short_length, long_length]
    src_pages = (max(lengths_cpu) + page - 1) // page
    total_tokens = 2 * src_pages * page
    rng = torch.Generator().manual_seed(7409)

    def index_values(*shape):
        # Exact FP8 values with FP32-exact products and accumulation for the
        # score oracle, independent of the device's reduction order.
        return torch.randint(-8, 9, shape, generator=rng, dtype=torch.int8).float() / 32

    def scales(*shape):
        return torch.randint(1, 4, shape, generator=rng).float() / 2

    lengths = torch.tensor(lengths_cpu, dtype=dtype, device=device)
    ends = torch.tensor([width, width * 2], dtype=dtype, device=device)
    dram_pages = torch.arange(2 * src_pages, dtype=dtype, device=device).view(2, -1)
    li_query = index_values(2 * width, 32, 128)
    li_query[..., 0] = 0.75
    li_key = index_values(total_tokens, 128)
    # Each route must see its own latest token. Give those tokens clear scores
    # so cutoff ties cannot obscure the causal boundary and high source IDs.
    li_key[
        src_pages * page + long_length - width : src_pages * page + long_length, 0
    ] = 8
    li_key_cpu = li_key.to(torch.float8_e4m3fn)
    del li_key
    li_key = li_key_cpu.view(-1, page, 1, 128).to(device)
    weights_cpu = scales(2 * width, 32).to(torch.bfloat16)
    query_scale_cpu = scales(2 * width, 32)
    key_scale_cpu = scales(total_tokens)
    weights = weights_cpu.to(device)
    query_scale = query_scale_cpu.to(device)
    key_scale = key_scale_cpu.view(-1, page, 1).to(device)
    source_map = torch.full((4, src_pages * page), -1, dtype=dtype, device=device)
    sources = torch.empty((2 * width, top_k), dtype=dtype, device=device)
    destinations = torch.empty_like(sources)
    per_query = torch.empty(2 * width, dtype=dtype, device=device)
    miss_sources = torch.empty((2, 32768), dtype=dtype, device=device)
    miss_slots = torch.empty_like(miss_sources)
    miss_counts = torch.empty(2, dtype=dtype, device=device)
    cache_sizes = torch.tensor([0, capacity], dtype=dtype, device=device)
    pool_entries = torch.tensor([1, 2], dtype=dtype, device=device)
    q_index_cpu = li_query.to(torch.float8_e4m3fn)
    q_index = q_index_cpu.to(device)
    state = torch.tensor([-3, -2], dtype=dtype, device=device)
    op = fused_quant_lightning_indexer_manage_full_visible
    args = (
        q_index,
        weights,
        li_key,
        query_scale,
        key_scale,
        dram_pages,
        ends,
        lengths,
        cache_sizes,
        state,
        pool_entries,
        source_map,
        sources,
        destinations,
        per_query,
        miss_sources,
        miss_slots,
        miss_counts,
    )

    if graph:
        op(*args)
        torch.npu.synchronize()
        capture = torch.npu.NPUGraph()
        with torch.npu.graph(capture):
            op(*args)
    resident_by_slot = {}
    for step in range(4):
        if step == 1:
            state[1] = -1
        if step == 3:
            q_index_cpu = (-li_query).to(torch.float8_e4m3fn)
            q_index.copy_(q_index_cpu.to(device))
        if step in (0, 3):
            reference_outputs = [
                torch.empty_like(item, device="cpu") for item in args[12:]
            ]
            op(
                q_index_cpu,
                weights_cpu,
                li_key_cpu.view(-1, page, 1, 128),
                query_scale_cpu,
                key_scale_cpu.view(-1, page, 1),
                dram_pages.cpu(),
                ends.cpu(),
                lengths.cpu(),
                cache_sizes.cpu(),
                torch.tensor([-3, -2], dtype=dtype),
                pool_entries.cpu(),
                torch.full(source_map.shape, -1, dtype=dtype),
                *reference_outputs,
                impl="torch_native",
            )
            reference_topk_cpu = reference_outputs[0].tolist()
        if graph:
            capture.replay()
        else:
            op(*args)
        short_count, count = miss_counts.tolist()
        assert short_count == 0
        assert 0 <= count <= capacity
        if step == 0:
            assert count == capacity
        elif step == 2:
            assert count == width
            assert per_query[width:].eq(0).all()
        misses = miss_sources[1, :count].tolist()
        slots = miss_slots[1, :count].tolist()
        assert len(set(misses)) == len(set(slots)) == count
        assert all(0 <= slot < capacity for slot in slots)
        assert all(0 <= source < long_length for source in misses)
        previous_by_source = {source: slot for slot, source in resident_by_slot.items()}
        previous = set(previous_by_source)
        resident_by_slot.update(zip(slots, misses))
        assert len(set(resident_by_slot.values())) == capacity
        assert source_map[[0, 1, 3]].eq(-1).all()
        assert (source_map[2] >= 0).sum().item() == capacity

        source_cpu = sources.tolist()
        slot_cpu = destinations.tolist()
        route_misses = per_query.tolist()
        selected_union = set()
        for request, length in enumerate(lengths_cpu):
            for route in range(width):
                row = request * width + route
                visible = length - width + route + 1
                k = min(top_k, visible)
                selected = source_cpu[row][:k]
                assert len(set(selected)) == k
                assert min(selected) >= 0 and max(selected) < visible
                assert all(x == -1 for x in source_cpu[row][k:])
                label = f"Q{width} step={step} request={request} route={route} visible={visible} graph={graph}"
                native_ids = [
                    source for source in reference_topk_cpu[row] if source >= 0
                ]
                native = set(native_ids)
                assert len(native_ids) == len(native) == k and max(native) < visible, (
                    f"{label}: expected {k} unique causal native IDs, "
                    f"got {len(native)} unique of {len(native_ids)}"
                )
                fused = set(selected)
                if fused != native:
                    candidates = sorted(fused | native)
                    # The fixture maps each request to consecutive physical
                    # pages starting at request * src_pages.
                    physical = torch.tensor(candidates) + request * src_pages * page
                    keys = li_key_cpu.view(torch.int8)[physical].view(
                        torch.float8_e4m3fn
                    )
                    scores = _a5_li_score_keys(
                        q_index_cpu[row],
                        keys,
                        weights_cpu[row],
                        query_scale_cpu[row],
                        key_scale_cpu[physical],
                    )
                    _assert_only_cutoff_ties(
                        fused, native, dict(zip(candidates, scores.tolist())), label
                    )
                if request == 0:
                    assert slot_cpu[row] == source_cpu[row]
                else:
                    selected_union.update(selected)
                    if step < 3:
                        assert visible - 1 in selected
                    assert all(0 <= slot < capacity for slot in slot_cpu[row][:k])
                    assert [
                        resident_by_slot[slot] for slot in slot_cpu[row][:k]
                    ] == selected
                    assert route_misses[row] == len(set(selected) - previous)
        if step:
            rewritten = [
                source
                for source in range(long_length - width, long_length)
                if source in previous
                and resident_by_slot[previous_by_source[source]] == source
            ]
            ordinary_count = count - len(rewritten)
            assert ordinary_count >= 0
            if step == 3 and long_length > 131072:
                assert ordinary_count > 0
            assert misses[ordinary_count:] == rewritten
            assert set(misses[:ordinary_count]) == selected_union - previous
