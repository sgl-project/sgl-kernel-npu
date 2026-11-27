import importlib.util
from pathlib import Path

import pytest
import torch

path = (
    Path(__file__).resolve().parents[1]
    / "python/sgl_kernel_npu/sgl_kernel_npu/sparsity_driven_kv_offload/full_visible_planner.py"
)
spec = importlib.util.spec_from_file_location("full_visible_planner", path)
planner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(planner)


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
@pytest.mark.parametrize("device", ["cpu", "npu"])
def test_fully_visible_plan_tracks_one_union_across_query_routes(width, device):
    if device == "npu":
        pytest.importorskip("torch_npu")
        if not torch.npu.is_available():
            pytest.skip("No NPU")
    top_k, context = 2048, 32768
    capacity = width * top_k
    batch = 3
    dtype = torch.int32
    request_rows = torch.tensor([1, 2, 3], dtype=dtype, device=device)
    sparse = torch.tensor([True, False, False], device=device)
    source_map = torch.full((4, context + 1), -1, dtype=dtype, device=device)
    inverse = torch.full((4, capacity + 1), -1, dtype=dtype, device=device)
    sources = torch.empty((batch * width, top_k), dtype=dtype, device=device)
    destinations = torch.empty_like(sources)
    row_misses = torch.empty(batch * width, dtype=dtype, device=device)
    miss_sources = torch.empty((batch, 32768), dtype=dtype, device=device)
    miss_slots = torch.empty_like(miss_sources)
    counts = torch.empty(batch, dtype=dtype, device=device)
    pages = {}
    evicted = False

    for step, warm in enumerate((False, True, True, False)):
        length = capacity + 17 + step * (top_k // 2)
        topk = torch.full((batch, width, top_k), -1, dtype=dtype)
        for route in range(width):
            start = step * (top_k // 2) + route * (top_k - 205)
            topk[0, route] = torch.arange(start, start + top_k, dtype=dtype)
            topk[0, route, -2:] = torch.tensor(
                [length - width + route, length]
            )  # Latest visible row and an invisible row.
            topk[1, route, :2] = torch.tensor([0, route + 1])
            topk[2, route, 0] = 0  # Graph padding must not select a real row.
        previous = set(pages.values()) if warm else set()
        planner.plan_full_visible_cache(
            topk.to(device),
            torch.tensor([length, width + 2, 0], dtype=dtype, device=device),
            request_rows,
            sparse,
            torch.tensor([warm, False, False], device=device),
            source_map,
            inverse,
            sources,
            destinations,
            row_misses,
            miss_sources,
            miss_slots,
            counts,
        )
        union = set(topk[0, :, :-1].flatten().tolist())
        count, short_count, padding_count = counts.tolist()
        misses = miss_sources[0, :count].tolist()
        slots = miss_slots[0, :count].tolist()
        assert len(set(slots)) == count and all(0 <= slot < capacity for slot in slots)
        assert len(set(misses)) == count and set(misses) == union - previous
        assert short_count == padding_count == 0
        if not warm:
            pages.clear()
        evicted |= any(slot in pages for slot in slots)
        pages.update(zip(slots, misses))
        for route in range(width):
            assert sources[route, :-1].tolist() == topk[0, route, :-1].tolist()
            assert sources[route, -1].item() == -1
            assert [pages[slot] for slot in destinations[route, :-1].tolist()] == topk[
                0, route, :-1
            ].tolist()
            assert row_misses[route].item() == len(
                set(topk[0, route, :-1].tolist()) - previous
            )
            assert destinations[width + route, :3].tolist() == [0, route + 1, -1]
        assert destinations[2 * width :].eq(-1).all()
        assert inverse[0].eq(-1).all()
        assert inverse[3].eq(-1).all()
    assert evicted


@pytest.mark.parametrize("width", [1, 4])
@pytest.mark.parametrize(
    # At 140242 tokens, the steady miss scan starts at the final 978 entries.
    # The former 64-element padding requested 92 bytes in one DMA instruction.
    "long_length,graph",
    [(8199, False), (140242, False), (140242, True)],
)
def test_a5_fused_lim_full_visible_copysfa_matches_native_attention(
    width, long_length, graph
):
    """Check native LI TopK, cold fill, graph replay and C8 attention."""
    torch_npu = pytest.importorskip("torch_npu")
    import acl
    import sgl_kernel_npu  # noqa: F401

    if not torch.npu.is_available() or acl.rt.get_device_info(0, 601) != (3510, 0):
        pytest.skip("Ascend950 required")
    device = "npu:0"
    dtype = torch.int32
    capacity, page, top_k = 8192, 128, 2048
    short_length = 16 if long_length < 131072 else 4097
    lengths_cpu = [short_length, long_length]
    src_pages, dst_pages = (max(lengths_cpu) + page - 1) // page, capacity // page
    total_tokens = 2 * src_pages * page
    rng = torch.Generator().manual_seed(7409)

    def noise(*shape):
        return torch.randn(shape, generator=rng).mul_(0.125)

    def index_values(*shape):
        # Exact FP8 values with FP32-exact products and accumulation for the
        # score oracle, independent of the device's reduction order.
        return torch.randint(-8, 9, shape, generator=rng, dtype=torch.int8).float() / 32

    def scales(*shape):
        return torch.randint(1, 4, shape, generator=rng).float() / 2

    lengths = torch.tensor(lengths_cpu, dtype=dtype, device=device)
    ends = torch.tensor([width, width * 2], dtype=dtype, device=device)
    dram_pages = torch.arange(2 * src_pages, dtype=dtype, device=device).view(2, -1)
    hbm_pages = torch.arange(2 * dst_pages, dtype=dtype, device=device).view(2, -1)
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
    packed = (
        torch.cat(
            (
                noise(total_tokens, 512).to(torch.float8_e4m3fn).view(torch.int8),
                noise(total_tokens, 64).to(torch.bfloat16).view(torch.int8),
                torch.ones((total_tokens, 4), dtype=torch.float32).view(torch.int8),
            ),
            dim=-1,
        )
        .view(-1, page, 1, 656)
        .to(device)
    )
    hbm = torch.zeros((2 * dst_pages, page, 1, 656), dtype=torch.int8, device=device)
    flat_kv = packed.view(total_tokens, 656)
    short_current = slice(short_length - width, short_length)
    long_base = src_pages * page
    long_current = slice(long_base + long_length - width, long_base + long_length)
    current = torch.cat((flat_kv[short_current], flat_kv[long_current]))
    positions = torch.tensor(
        [
            *range(short_length - width, short_length),
            *range(long_length - width, long_length),
        ],
        device=device,
        dtype=torch.int64,
    )
    current_valid = torch.ones(2 * width, dtype=torch.bool, device=device)
    req_to_token = torch.full((4, long_base), -1, dtype=dtype, device=device)
    req_to_token[1:3].copy_(
        torch.arange(total_tokens, dtype=dtype, device=device).view(2, -1)
    )
    resident_end = torch.tensor([-1, -1, 16, -1], dtype=dtype, device=device)
    query = noise(2 * width, 8, 576).to(device=device, dtype=torch.bfloat16)
    query_nope = query[..., :512].contiguous()
    query_rope = query[..., 512:].contiguous()
    output = query.new_empty((2 * width, 8, 512))
    cache_sizes = torch.tensor([0, capacity], dtype=dtype, device=device)
    pool_entries = torch.tensor([1, 2], dtype=dtype, device=device)
    hbm_lengths = torch.tensor(
        [short_length, capacity + width - 1], dtype=dtype, device=device
    )
    cache = packed.view(torch.float8_e4m3fn)
    q_index_cpu = li_query.to(torch.float8_e4m3fn)
    q_index = q_index_cpu.to(device)
    state = torch.tensor([-3, -2], dtype=dtype, device=device)
    op = torch.ops.npu.fused_quant_lightning_indexer_manage_full_visible
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

    def copy_sfa(native_lim_plan=True):
        torch.ops.npu.fused_quant_scatter_copy_sparse_flash_attention_v2(
            query_nope,
            query_rope,
            hbm,
            packed,
            packed.data_ptr(),
            hbm_pages,
            dram_pages,
            destinations.unsqueeze(1),
            sources.unsqueeze(1),
            per_query,
            ends,
            hbm_lengths,
            cache_sizes,
            miss_sources,
            miss_slots,
            miss_counts,
            576**-0.5,
            output,
            current=current,
            positions=positions,
            current_valid=current_valid,
            request_slots=pool_entries,
            req_to_token=req_to_token,
            resident_end=resident_end,
            native_lim_plan=native_lim_plan,
        )

    if graph:
        op(*args)
        copy_sfa()
        torch.npu.synchronize()
        capture = torch.npu.NPUGraph()
        with torch.npu.graph(capture):
            op(*args)
        copy_graph = torch.npu.NPUGraph()
        with torch.npu.graph(copy_graph):
            copy_sfa()
        hbm.zero_()
        resident_end.fill_(-1)
        resident_end[2] = 16

    resident_by_slot = {}
    for step in range(4):
        if step == 1:
            state[1] = -1
        if step == 3:
            q_index_cpu = (-li_query).to(torch.float8_e4m3fn)
            q_index.copy_(q_index_cpu.to(device))
        if step in (0, 3):
            native_topk = torch.ops.npu.npu_quant_lightning_indexer(
                query=q_index,
                key=li_key,
                weights=weights,
                query_dequant_scale=query_scale,
                key_dequant_scale=key_scale,
                actual_seq_lengths_query=ends,
                actual_seq_lengths_key=lengths,
                block_table=dram_pages,
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=top_k,
                sparse_mode=3,
                query_quant_mode=0,
                key_quant_mode=0,
            )
            native_topk_cpu = native_topk.reshape_as(sources).cpu().tolist()
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
                native_ids = [source for source in native_topk_cpu[row] if source >= 0]
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

        if step == 2:
            # Leave DRAM with the original payload until after CopySFA, so
            # the test detects a stale read in the current-KV refresh path.
            current[:, :512].zero_()
        if graph:
            copy_graph.replay()
        else:
            copy_sfa()
        assert resident_end.tolist() == [-1, short_length, -1, -1]
        if step == 2:
            flat_kv[short_current].copy_(current[:width])
            flat_kv[long_current].copy_(current[width:])
            hbm_rows = hbm.view(-1, 656)
            torch.testing.assert_close(hbm_rows[short_current], current[:width])
            for row in range(width):
                slot = previous_by_source[long_length - width + row]
                torch.testing.assert_close(
                    hbm_rows[capacity + slot], current[width + row]
                )
        reference = torch_npu.npu_kv_quant_sparse_flash_attention(
            query=query,
            key=cache,
            value=cache,
            sparse_indices=sources.unsqueeze(1),
            scale_value=576**-0.5,
            key_quant_mode=2,
            value_quant_mode=2,
            key_dequant_scale=None,
            value_dequant_scale=None,
            actual_seq_lengths_query=ends,
            actual_seq_lengths_kv=lengths,
            block_table=dram_pages,
            sparse_block_size=1,
            layout_query="TND",
            layout_kv="PA_BSND",
            sparse_mode=3,
            attention_mode=2,
            quant_scale_repo_mode=1,
            tile_size=128,
            rope_head_dim=64,
        )
        if isinstance(reference, tuple):
            reference = reference[0]
        torch.testing.assert_close(
            output.cpu().float(), reference.cpu().float(), atol=0.008, rtol=0.05
        )

    # Replay with the first request padded. The framework relies on CopySFA
    # to zero these output rows even when the capture only had live requests.
    hbm_lengths[0] = 0
    sources[:width].fill_(-1)
    destinations[:width].fill_(-1)
    current_valid[:width].fill_(False)
    pool_entries[0] = 0
    output.fill_(float("nan"))
    if graph:
        copy_graph.replay()
    else:
        copy_sfa()
    assert torch.count_nonzero(output[:width]).item() == 0
    torch.testing.assert_close(
        output[width:].cpu().float(),
        reference[width:].cpu().float(),
        atol=0.008,
        rtol=0.05,
    )

    # Force a steady native miss prefix, shared across all routes. The newest
    # ordinary miss has stale DRAM. Q4 also carries a retained refresh after
    # the ordinary union. Compare every persisted byte against complete copy.
    first = long_length - width
    ordinary = [*range(7000, 7008), first]
    retained = [first + 1] if width > 1 else []
    planned = ordinary + retained
    ids = torch.arange(2048, 2048 + top_k, device=device, dtype=dtype)
    slot_ids = ids.clone()
    ids[: len(ordinary)] = torch.tensor(ordinary, device=device, dtype=dtype)
    slot_ids[: len(ordinary)] = torch.arange(len(ordinary), device=device, dtype=dtype)
    sources[width:].copy_(ids.expand(width, -1))
    destinations[width:].copy_(slot_ids.expand(width, -1))
    if retained:
        # Query zero cannot see first + 1. Later queries attend its updated slot.
        sources[width + 1 :, -1] = retained[0]
        destinations[width + 1 :, -1] = len(ordinary)
    per_query[width:].fill_(len(ordinary))
    miss_sources[1, : len(planned)] = torch.tensor(planned, device=device, dtype=dtype)
    miss_slots[1, : len(planned)] = torch.arange(
        len(planned), device=device, dtype=dtype
    )
    miss_counts[1] = len(planned)
    flat_kv[long_current, :512].zero_()
    current[width:, :512].fill_(64)
    before = hbm.clone()
    copy_sfa(native_lim_plan=False)
    expected_cache = hbm.clone()
    expected_output = output.clone()
    hbm.copy_(before)
    output.fill_(float("nan"))
    if graph:
        copy_graph.replay()
    else:
        copy_sfa()
    assert torch.equal(hbm, expected_cache)
    assert torch.equal(hbm.view(-1, 656)[capacity + len(ordinary) - 1], current[width])
    torch.testing.assert_close(output, expected_output, atol=0.008, rtol=0.05)


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("width,prefix", [(1, 0), (4, 126)], ids=["q1", "q4"])
def test_a5_short_staging_matches_preloaded_attention(graph, width, prefix):
    """Compare a single token and a Q4 request accepting one row across a page."""
    pytest.importorskip("torch_npu")
    import acl
    import sgl_kernel_npu  # noqa: F401

    if not torch.npu.is_available() or acl.rt.get_device_info(0, 601) != (3510, 0):
        pytest.skip("Ascend950 required")
    device = "npu:0"
    page = 128
    rng = torch.Generator().manual_seed(197)

    def packed_rows(rows):
        latent = torch.randn(rows, 512, generator=rng).to(torch.float8_e4m3fn)
        rope = torch.randn(rows, 64, generator=rng).to(torch.bfloat16)
        scales = torch.ones(rows, 4)
        return torch.cat([x.view(torch.int8) for x in (latent, rope, scales)], dim=1)

    def ids(values):
        return torch.tensor(values, dtype=torch.int32, device=device)

    logical = packed_rows(2 * page).to(device).view(2, page, 1, 656)
    host = logical.flip([0]).contiguous()
    hbm = torch.zeros_like(host)
    golden = logical.clone()
    query = torch.randn(width, 8, 576, generator=rng).to(device, torch.bfloat16)
    nope, rope = query[..., :512].contiguous(), query[..., 512:].contiguous()
    result = torch.empty_like(nope)
    expected = torch.empty_like(nope)
    topk = torch.full((width, 1, 2048), -1, dtype=torch.int32, device=device)
    ends, length = ids([width]), ids([prefix + width])
    hbm_pages, host_pages = ids([[0, 1]]), ids([[1, 0]])
    zero, row_misses = ids([0]), ids([0] * width)
    empty_union = torch.full((1, 32768), -1, dtype=torch.int32, device=device)
    current = torch.zeros((width, 656), dtype=torch.int8, device=device)
    position = torch.arange(prefix, prefix + width, device=device)
    extent = ids([-1, -1])
    req_to_token = torch.stack((ids([-1] * 256), ids([*range(128, 256), *range(128)])))
    resident = dict(
        current=current,
        positions=position,
        current_valid=torch.ones(width, device=device, dtype=torch.bool),
        request_slots=ids([1]),
        req_to_token=req_to_token,
        resident_end=extent,
        native_lim_plan=True,
    )

    def attention(cache, output, **kwargs):
        torch.ops.npu.fused_quant_scatter_copy_sparse_flash_attention_v2(
            nope,
            rope,
            cache,
            host,
            host.data_ptr(),
            hbm_pages,
            host_pages,
            topk,
            topk,
            row_misses,
            ends,
            length,
            zero,
            empty_union,
            empty_union,
            zero,
            576**-0.5,
            output,
            **kwargs,
        )

    def update(first):
        length.fill_(first + width)
        position.copy_(torch.arange(first, first + width, device=device))
        current.copy_(packed_rows(width).to(device))
        for row in range(width):
            visible = first + row + 1
            topk[row, 0, :visible] = torch.arange(visible, device=device)

    update(prefix)
    if graph:
        attention(hbm, result, **resident)
        torch.npu.synchronize()
        capture = torch.npu.NPUGraph()
        with torch.npu.graph(capture):
            attention(hbm, result, **resident)
        hbm.zero_()
        extent.fill_(-1)

    for first in (prefix, prefix + 1):
        update(first)
        stop = first + width
        golden.view(-1, 656)[first:stop].copy_(current)
        attention(golden, expected)
        if graph:
            capture.replay()
        else:
            attention(hbm, result, **resident)
        assert extent.tolist() == [-1, stop]
        assert torch.equal(hbm.view(-1, 656)[:stop], golden.view(-1, 656)[:stop])
        assert expected.isfinite().all() and expected.abs().max() > 0
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
