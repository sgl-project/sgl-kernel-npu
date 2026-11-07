"""Native-LIM resident-refresh overlap with heterogeneous and padded TND rows."""

import math

import pytest
import torch

pytest.importorskip("torch_npu")
pytest.importorskip("sgl_kernel_npu")

from test_copy_sfa_dynamic_tnd import _DynamicTndCase


def _check_refresh_plan(graph, resident_count, genuine_misses, *, suffix_contract=True):
    counts, capacity, topk = [4, 1, 4], 12288, 2048
    case = _DynamicTndCase(
        query_counts=counts,
        heads=8,
        cache_budgets=[capacity] * 3,
        steady_miss_requests={0, 1},
        tail_tokens=4,
    )
    try:
        sources = case.sparse_slots_cpu.clone()
        request_src = torch.full_like(case.miss_src_cpu, -1)
        request_dst = torch.full_like(case.miss_dst_cpu, -1)
        request_counts = torch.zeros_like(case.miss_counts_cpu)
        route_misses = torch.zeros_like(case.topk_miss_counts_cpu)
        begin = 0
        for request, width in enumerate(counts):
            if request == 2:
                # Graph padding retains a positive TND width and zero KV counts.
                begin += width
                continue
            route_misses[begin : begin + width] = genuine_misses
            refresh = min(resident_count, width)
            missing = torch.arange(capacity - genuine_misses, capacity).int()
            request_src[request, :genuine_misses] = missing
            request_dst[request, :genuine_misses] = torch.arange(genuine_misses)
            sources[begin : begin + width, 0, :genuine_misses] = missing
            if refresh:
                resident = torch.arange(topk - refresh, topk).int()
                request_src[request, genuine_misses : genuine_misses + refresh] = (
                    resident
                )
                request_dst[request, genuine_misses : genuine_misses + refresh] = (
                    resident
                )
            request_counts[request] = genuine_misses + refresh
            if not suffix_contract and refresh:
                # Legacy callers may place rewritten residents before genuine misses.
                # Their omitted opt-in flag must retain the complete-copy behavior.
                order = torch.cat(
                    (
                        torch.arange(genuine_misses, genuine_misses + refresh),
                        torch.arange(genuine_misses),
                    )
                )
                request_src[request, : genuine_misses + refresh] = request_src[
                    request, order
                ].clone()
                request_dst[request, : genuine_misses + refresh] = request_dst[
                    request, order
                ].clone()
            begin += width
        case.topk_src.copy_(sources)
        case.topk_miss_counts.copy_(route_misses)
        case.miss_src.copy_(request_src)
        case.miss_dst.copy_(request_dst)
        case.miss_counts.copy_(request_counts)
        case.cache_tokens[-1] = 0
        case.actual_kv[-1] = 0

        def run():
            options = {"full_visible": True}
            if suffix_contract:
                options["resident_refresh_tail"] = True
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
                **options,
            )

        run()
        torch.npu.synchronize()
        if graph:
            capture = torch.npu.NPUGraph()
            with torch.npu.graph(capture):
                run()
        for step in range(2):
            # Rewrite resident payloads after capture, as a rejected draft does.
            for request, width in enumerate(counts):
                refresh = min(resident_count, width)
                for slot in range(topk - refresh, topk):
                    block = int(case.dram_block_table_cpu[request, slot // 128])
                    case.dram[block, slot % 128].fill_(1.25 + step + request)
            case.hbm.copy_(case.initial_hbm)
            expected_cache = case.initial_hbm.cpu().reshape(-1, 128, 576).clone()
            for request, count in enumerate(request_counts.tolist()):
                for entry in range(count):
                    source = int(request_src[request, entry])
                    dest = int(request_dst[request, entry])
                    src_block = int(case.dram_block_table_cpu[request, source // 128])
                    dst_block = int(case.hbm_block_table_cpu[request, dest // 128])
                    expected_cache[dst_block, dest % 128] = case.dram[
                        src_block, source % 128
                    ]
            if graph:
                capture.replay()
            else:
                run()
            torch.npu.synchronize()
            torch.testing.assert_close(
                case.hbm.cpu().reshape_as(expected_cache),
                expected_cache,
                rtol=0,
                atol=0,
            )
            # Build a mathematical oracle for every active request/query.
            query = case.query.cpu().float()
            rope = case.query_rope.cpu().float()
            begin = 0
            for request, width in enumerate(counts):
                if request == 2:
                    continue
                slots = torch.arange(topk)
                blocks = case.hbm_block_table_cpu[request, slots // 128].long()
                selected = expected_cache[blocks, slots % 128].float()
                q = query[begin : begin + width]
                qr = rope[begin : begin + width]
                scores = (
                    q @ selected[:, :512].T + qr @ selected[:, 512:].T
                ) / math.sqrt(576)
                expected = torch.softmax(scores, dim=-1) @ selected[:, :512]
                torch.testing.assert_close(
                    case.output[begin : begin + width].cpu().float(),
                    expected,
                    rtol=0.02,
                    atol=0.003,
                )
                begin += width
            assert torch.count_nonzero(case.output[begin:]).item() == 0
    finally:
        case.close()


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("resident_count", range(5))
@pytest.mark.parametrize("genuine_misses", [0, 17])
def test_copy_sfa_refresh_suffix_preserves_hits_and_padding(
    graph, resident_count, genuine_misses
):
    _check_refresh_plan(graph, resident_count, genuine_misses)


@pytest.mark.parametrize("graph", [False, True])
def test_copy_sfa_legacy_default_accepts_resident_first_plans(graph):
    _check_refresh_plan(
        graph, resident_count=4, genuine_misses=17, suffix_contract=False
    )
