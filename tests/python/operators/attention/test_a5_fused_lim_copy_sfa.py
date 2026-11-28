import pytest
import torch
from sgl_kernel_npu.operators.attention.fused_quant_scatter_copy_sparse_flash_attention_v2 import (
    fused_quant_scatter_copy_sparse_flash_attention_v2,
)
from sgl_kernel_npu.operators.indexer.fused_quant_lightning_indexer_manage_full_visible import (
    fused_quant_lightning_indexer_manage_full_visible,
)

from tests.python.operators.indexer.test_fused_quant_lightning_indexer_manage_full_visible import (
    _a5_li_score_keys,
    _assert_only_cutoff_ties,
)


@pytest.mark.parametrize("width", [1, 4])
@pytest.mark.parametrize(
    # At 140242 tokens, the steady miss scan starts at the final 978 entries.
    # The former 64-element padding requested 92 bytes in one DMA instruction.
    "long_length,graph",
    [(8199, False), (140242, False), (140242, True)],
)
def test_a5_fused_lim_full_visible_copysfa_matches_torch_native_attention(
    width, long_length, graph
):
    """Check native LI TopK, cold fill, graph replay and C8 attention."""
    pytest.importorskip("torch_npu")
    import acl

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

    def copy_sfa(
        native_lim_plan=True,
        *,
        cache=hbm,
        result=output,
        extents=resident_end,
        impl="auto",
    ):
        fused_quant_scatter_copy_sparse_flash_attention_v2(
            query_nope,
            query_rope,
            cache,
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
            result,
            current=current,
            positions=positions,
            current_valid=current_valid,
            request_slots=pool_entries,
            req_to_token=req_to_token,
            resident_end=extents,
            native_lim_plan=native_lim_plan,
            impl=impl,
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

        if step == 2:
            # Leave DRAM with the original payload until after CopySFA, so
            # the test detects a stale read in the current-KV refresh path.
            current[:, :512].zero_()
        reference = torch.empty_like(output)
        copy_sfa(
            cache=hbm.clone(),
            result=reference,
            extents=resident_end.clone(),
            impl="torch_native",
        )
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
