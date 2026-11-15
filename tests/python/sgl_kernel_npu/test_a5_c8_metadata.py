"""A3/A5 Q1/Q4 metadata and opaque C8 staging, without native FP8 attention."""

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")


@pytest.fixture(scope="module", autouse=True)
def require_npu():
    assert torch.npu.is_available(), "Metadata/transport checks need an NPU"
    torch.npu.set_device(0)


@pytest.mark.parametrize("width", [1, 4])
@pytest.mark.parametrize("graph_replay", [False, True])
def test_a5_pre_post_metadata(width, graph_replay):
    from sgl_kernel_npu.sparsity_driven_kv_offload.c8_metadata import (
        fused_lim_copy_sfa_metadata,
        fused_prepare_lim_metadata,
    )

    batch, request_rows, capacity, context = 3, 4, 8192, 8704
    device = "npu"
    requests = torch.tensor([1, 2, 0], dtype=torch.int32, device=device)
    seq = torch.tensor(
        [8445 + width, 8200 + width, 0], dtype=torch.int32, device=device
    )
    last = torch.full((request_rows + 1,), -1, dtype=torch.int32, device=device)
    src_table = torch.arange(batch * 68, dtype=torch.int32, device=device).view(
        batch, 68
    )
    # Different requests deliberately share physical prefix pages.
    src_table[1, :60].copy_(src_table[0, :60])
    table = torch.empty_like(src_table)
    pre = {
        name: torch.empty(batch, dtype=dtype, device=device)
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
    sources = torch.full((batch * width, 2048), 17, dtype=torch.int32, device=device)
    misses = torch.full((batch * width,), 123, dtype=torch.int32, device=device)
    counts = torch.tensor([512, 513, 777], dtype=torch.int32, device=device)
    out_seq = torch.empty(batch, dtype=torch.int32, device=device)
    out_cache = torch.empty_like(out_seq)
    hbm = torch.empty(batch, 66, dtype=torch.int32, device=device)
    dram = torch.empty_like(table)

    def launch():
        fused_prepare_lim_metadata(
            seq_lens=seq,
            request_slots=requests,
            last_prefix=last,
            block_table=src_table,
            out_block_table=table,
            capacity=capacity,
            request_rows=request_rows,
            full_visible=False,
            query_width=width,
            **pre,
        )
        fused_lim_copy_sfa_metadata(
            source_ids=sources,
            topk_misses=misses,
            miss_counts=counts,
            valid_rows=pre["valid_rows"],
            sparse_rows=pre["sparse_rows"],
            request_slots=pre["safe_slots"],
            actual_seq_lengths_kv=out_seq,
            num_cache_tokens=out_cache,
            hbm_block_table=hbm,
            dram_block_table=dram,
            capacity=capacity,
            host_context_len=context,
            request_rows=request_rows,
            seq_lens=pre["lengths"],
            query_width=width,
            host_block_table=table,
            packed_c8=True,
            prefix_lens=pre["prefixes"],
        )
        # The real bridge commits this state after LIM. Replay must see the
        # previous prefix, including transitions back to a colder prefix.
        last.index_copy_(0, pre["safe_slots"].long(), pre["next_prefixes"])

    launch()
    torch.npu.synchronize()
    graph = None
    if graph_replay:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            launch()
    # Change device inputs while retaining every captured allocation.
    for lengths, expected_states in (
        ([8445 + width, 8200 + width, 0], [-1, -1, -3]),
        ([8310 + width, 8200 + width, 0], [-2, -1, -3]),
        ([8445 + width, 0, 0], [-1, -3, -3]),
    ):
        seq.copy_(torch.tensor(lengths, dtype=torch.int32, device=device))
        graph.replay() if graph is not None else launch()
        torch.npu.synchronize()
        assert pre["request_states"].cpu().tolist() == expected_states
        assert out_seq.cpu().tolist() == lengths
        prefix = ((torch.tensor(lengths) - width).clamp_min(0) // 128) * 128
        expected_cache = torch.where(
            torch.tensor(lengths) > 0, capacity + torch.tensor(lengths) - prefix, 0
        )
        torch.testing.assert_close(out_cache.cpu(), expected_cache.to(torch.int32))
        for row, request in enumerate([1, 2, 0]):
            if not lengths[row]:
                assert bool((sources[row * width : (row + 1) * width] == -1).all())
                assert counts[row].item() == 0
                continue
            cp = capacity // 128
            expected_tail = [
                request * 66 + cp + (int(prefix[row]) // 128 + page) % 2
                for page in range(2)
            ]
            assert hbm[row, -2:].cpu().tolist() == expected_tail
            torch.testing.assert_close(dram[row].cpu(), src_table[row].cpu())


@pytest.mark.parametrize("width", [1, 4])
@pytest.mark.parametrize("graph_replay", [False, True])
def test_a5_c8_tail_payload(width, graph_replay):
    from sgl_kernel_npu.sparsity_driven_kv_offload.c8_metadata import (
        stage_mtp_tail,
    )
    from sgl_kernel_npu.sparsity_driven_kv_offload.fused_shm import (
        create_fused_shm_tensor as create_shm_tensor,
    )
    from sgl_kernel_npu.sparsity_driven_kv_offload.fused_shm import (
        free_fused_shm as free_shm,
    )

    capacity, context, first = 8192, 8704, 8445
    host, _, pointer = create_shm_tensor((1, context, 1, 656), torch.int8)
    try:
        generator = torch.Generator().manual_seed(481)
        host.copy_(
            torch.randint(-128, 128, host.shape, dtype=torch.int8, generator=generator)
        )
        cache = torch.full(
            (3, capacity + 256, 1, 656), -17, dtype=torch.int8, device="npu"
        )
        current_cpu = torch.randint(
            -128, 128, (width, 656), dtype=torch.int8, generator=generator
        )
        current = current_cpu.to("npu")
        mapping_cpu = torch.arange(context).roll(256)
        mapping = mapping_cpu.expand(3, -1).clone().to("npu")
        requests = torch.tensor([1], device="npu")
        positions = torch.arange(first, first + width, device="npu")
        starts = torch.full((4,), -1, dtype=torch.int32, device="npu")
        ends = torch.full_like(starts, -1)
        hs = torch.empty(128, dtype=torch.int64, device="npu")
        hd = torch.empty_like(hs)
        hv = torch.empty(128, dtype=torch.bool, device="npu")
        cs = torch.arange(4, device="npu")
        cd = torch.empty_like(cs)
        cv = torch.empty(4, dtype=torch.bool, device="npu")

        def launch():
            stage_mtp_tail(
                current=current,
                cache=cache,
                host=host,
                host_address=pointer,
                req_to_token=mapping,
                requests=requests,
                positions=positions,
                starts=starts,
                ends=ends,
                workspace=(hs, hd, hv, cs, cd, cv),
                capacity=capacity,
                context_len=context,
                query_width=width,
            )

        launch()
        torch.npu.synchronize()
        starts.fill_(-1)
        ends.fill_(-1)
        if graph_replay:
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                launch()
            starts.fill_(-1)
            ends.fill_(-1)
            graph.replay()
        else:
            launch()
        torch.npu.synchronize()
        result = cache.cpu()
        prefix = first // 128 * 128
        for position in range(prefix, first):
            assert torch.equal(
                result[1, capacity + position % 256, 0],
                host[0, mapping_cpu[position], 0],
            )
        for route in range(width):
            assert torch.equal(
                result[1, capacity + (first + route) % 256, 0], current_cpu[route]
            )
        assert bool((result[0] == -17).all())
        assert bool((result[2] == -17).all())
    finally:
        torch.npu.synchronize()
        free_shm()
