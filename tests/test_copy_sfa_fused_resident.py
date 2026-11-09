"""End-to-end resident staging and attention using an independent CPU oracle."""

import math

import pytest
import torch
from test_copy_sfa_dynamic_tnd import _DynamicTndCase, _require_npu


@pytest.mark.parametrize("width", [1, 2, 3, 4, 5, 8, 14])
@pytest.mark.parametrize("old_end", [-1, 124, 127])
@pytest.mark.parametrize("graph", [False, True])
def test_short_resident_current_and_history(width, old_end, graph, split_host=False):
    ops = _require_npu()
    device_id = torch.npu.current_device()
    device = torch.device("npu", device_id)
    capacity = context = host_rows = 2048
    first = 129 if split_host else 127
    length = first + width
    generator = torch.Generator().manual_seed(20261103 + width)
    dtype = torch.bfloat16
    host_values = torch.randn(host_rows, 576, generator=generator, dtype=dtype)
    mapping = (torch.arange(context, dtype=torch.int32) * 13) % host_rows
    logical = host_values[mapping[:length].long()].clone()
    current_cpu = torch.randn(width + 2, 1, 576, generator=generator, dtype=dtype)
    logical[first:] = current_cpu[:width, 0]
    # The current rows must come from the device tensor, not the host mapping.
    host_values[mapping[first:length].long()] = 37
    try:
        host, _, combined_ptr = ops.create_fused_shm_tensor(
            (host_rows, 1, 576),
            dtype,
            device_id=device_id,
            name="copy_sfa_fused_resident",
        )
        host[:, 0].copy_(host_values)
        host_ptr, rope_ptr = combined_ptr, 0
        if split_host:
            host_k, _, host_ptr = ops.create_fused_shm_tensor(
                (host_rows, 512), dtype, device_id=device_id, name="copy_sfa_short_k"
            )
            host_rope, _, rope_ptr = ops.create_fused_shm_tensor(
                (host_rows, 64), dtype, device_id=device_id, name="copy_sfa_short_rope"
            )
            host_k.copy_(host_values[:, :512])
            host_rope.copy_(host_values[:, 512:])

        def npu(value):
            return value.to(device)

        hbm = torch.full(
            (2 * capacity // 128, 128, 1, 576), -91, dtype=dtype, device=device
        )
        physical_pages = torch.arange(31, 15, -1, dtype=torch.int32)
        if old_end >= 0:
            for pos in range(old_end):
                hbm[physical_pages[pos // 128], pos % 128, 0].copy_(
                    logical[pos].to(device)
                )
        initial_hbm = hbm.clone()
        req_table = torch.zeros((2, context), dtype=torch.int32)
        req_table[1] = mapping
        req_table = npu(req_table)
        positions = npu(
            torch.tensor(list(range(first, length)) + [-1, -1], dtype=torch.int64)
        )
        valid = npu(torch.tensor([True] * width + [False, False]))
        slots = npu(torch.tensor([1], dtype=torch.int32))
        begin = npu(torch.tensor([-1, 0 if old_end >= 0 else -1], dtype=torch.int32))
        end = npu(torch.tensor([-1, old_end], dtype=torch.int32))
        current = npu(current_cpu)
        q = npu(torch.randn(width, 8, 512, generator=generator, dtype=dtype))
        qr = npu(torch.randn(width, 8, 64, generator=generator, dtype=dtype))
        output = torch.empty_like(q)
        qends = npu(torch.tensor([width], dtype=torch.int32))
        lengths = npu(torch.tensor([length], dtype=torch.int32))
        zero = npu(torch.zeros(1, dtype=torch.int32))
        sources = npu(torch.full((width, 1, 2048), -1, dtype=torch.int32))
        indices_cpu = torch.full((width, 1, 2048), -1, dtype=torch.int32)
        indices_cpu[:, 0, :length] = torch.arange(length, dtype=torch.int32)
        indices = npu(indices_cpu)
        topk_misses = npu(torch.zeros(width, dtype=torch.int32))
        misses = npu(torch.full((1, 32768), -1, dtype=torch.int32))
        hbm_table = npu(physical_pages.view(1, -1))
        dram_table = npu(torch.zeros((1, 16), dtype=torch.int32))
        scale = 1 / math.sqrt(576)

        def run():
            torch.ops.npu.fused_scatter_copy_sparse_flash_attention(
                q,
                qr,
                hbm,
                qends,
                lengths,
                zero,
                sources,
                indices,
                topk_misses,
                misses,
                misses,
                zero,
                hbm_table,
                dram_table,
                host_ptr,
                output,
                scale,
                full_visible=True,
                current=current,
                positions=positions,
                valid=valid,
                request_slots=slots,
                req_to_token=req_table,
                resident_begin=begin,
                resident_end=end,
                host_token_capacity=host_rows,
                **({"dram_rope_dev_ptr": rope_ptr} if split_host else {}),
            )

        if graph:
            run()
            torch.npu.synchronize()
            hbm.copy_(initial_hbm)
            begin[1] = 0 if old_end >= 0 else -1
            end[1] = old_end
            capture = torch.npu.NPUGraph()
            with torch.npu.graph(capture):
                run()
            hbm.copy_(initial_hbm)
            begin[1] = 0 if old_end >= 0 else -1
            end[1] = old_end
            capture.replay()
        else:
            run()
        torch.npu.synchronize()
        actual_cache = hbm[physical_pages.to(device).long(), :, 0].reshape(
            capacity, 576
        )
        torch.testing.assert_close(actual_cache[:length].cpu(), logical, atol=0, rtol=0)
        expected_hbm = initial_hbm.cpu().clone()
        for pos in range(length):
            expected_hbm[physical_pages[pos // 128], pos % 128, 0] = logical[pos]
        torch.testing.assert_close(hbm.cpu(), expected_hbm, atol=0, rtol=0)
        assert begin.cpu().tolist() == [-1, 0]
        assert end.cpu().tolist() == [-1, length]
        fused_output = output.clone()
        old_output = torch.empty_like(output)
        torch.ops.npu.fused_scatter_copy_sparse_flash_attention(
            q,
            qr,
            hbm,
            qends,
            lengths,
            zero,
            sources,
            indices,
            topk_misses,
            misses,
            misses,
            zero,
            hbm_table,
            dram_table,
            combined_ptr,
            old_output,
            scale,
            full_visible=True,
        )
        torch.npu.synchronize()
        torch.testing.assert_close(fused_output, old_output, atol=0, rtol=0)
        actual = output.cpu().float()
        q_cpu, qr_cpu = q.cpu().float(), qr.cpu().float()
        errors = []
        expected_rows = []
        for route in range(width):
            visible = logical[: first + route + 1].float()
            scores = (
                q_cpu[route] @ visible[:, :512].T + qr_cpu[route] @ visible[:, 512:].T
            ) * scale
            expected = torch.softmax(scores, -1) @ visible[:, :512]
            error = actual[route] - expected
            errors.append(error.flatten())
            expected_rows.append(expected)
        errors = torch.cat(errors)
        print(
            f"width={width} old_end={old_end} graph={graph} "
            f"max_abs={errors.abs().max().item():.8g} rms={errors.square().mean().sqrt().item():.8g}"
        )
        torch.testing.assert_close(
            actual, torch.stack(expected_rows), atol=0.003, rtol=0.02
        )
    finally:
        torch.npu.synchronize()
        ops.free_fused_shm(device_id)


@pytest.mark.parametrize("graph", [False, True])
def test_split_host_short_resident_crosses_page(graph):
    # Restore across the page boundary through a permuted physical mapping.
    test_short_resident_current_and_history(4, -1, graph, split_host=True)


def _check_long_first_fill(width, graph, missing_current_page=False):
    case = _DynamicTndCase(
        query_counts=[width, 1],
        heads=8,
        cache_budgets=[2048, 2048],
        first_fill=True,
        first_fill_requests={0},
        tail_tokens=0,
    )
    try:
        first = 2048 - width
        current_cpu = torch.randn(width + 1, 1, 576, dtype=torch.bfloat16)
        host_rows = case.dram.view(-1, 576)
        expected = host_rows[:2048].clone()
        expected[first:2048] = current_cpu[:width, 0]
        host_rows[first:2048].fill_(41)
        if missing_current_page:
            # No history source refers to page 15. All destinations in that
            # range select the first current row, so DRAM page 15 is unnecessary.
            case.miss_src[0, 1920:2048] = first
            case.dram_block_table[0, 15] = -1
            expected[1920:2048] = current_cpu[0, 0]
        case.cache_tokens[-1] = 0
        case.actual_kv[-1] = 0
        current = current_cpu.to(case.device)
        positions = torch.tensor(
            list(range(first, 2048)) + [-1], dtype=torch.int64, device=case.device
        )
        valid = torch.tensor(
            [True] * width + [False], dtype=torch.bool, device=case.device
        )
        slots = torch.tensor([1, 0], dtype=torch.int32, device=case.device)
        table = torch.zeros((3, 2048), dtype=torch.int32, device=case.device)
        begin = torch.full((3,), -1, dtype=torch.int32, device=case.device)
        end = begin.clone()
        initial = case.hbm.clone()

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
                current=current,
                positions=positions,
                valid=valid,
                request_slots=slots,
                req_to_token=table,
                resident_begin=begin,
                resident_end=end,
                host_token_capacity=case.dram.numel() // 576,
            )

        if graph:
            run()
            torch.npu.synchronize()
            case.hbm.copy_(initial)
            begin.fill_(-1)
            end.fill_(-1)
            capture = torch.npu.NPUGraph()
            with torch.npu.graph(capture):
                run()
            case.hbm.copy_(initial)
            begin.fill_(-1)
            end.fill_(-1)
            capture.replay()
        else:
            run()
        torch.npu.synchronize()
        cache = case.hbm.cpu().view(-1, 128, 576)
        pages = case.hbm_block_table_cpu[0, :16].long()
        torch.testing.assert_close(
            cache[pages].reshape(2048, 576), expected, atol=0, rtol=0
        )
        expected_hbm = initial.cpu().view(-1, 128, 576).clone()
        expected_hbm[pages] = expected.reshape(16, 128, 576)
        torch.testing.assert_close(cache, expected_hbm, atol=0, rtol=0)
        torch.testing.assert_close(
            case.output[width:].cpu(),
            torch.zeros_like(case.output[width:].cpu()),
            atol=0,
            rtol=0,
        )
        assert begin.cpu().tolist() == [-1, -1, -1]
        assert end.cpu().tolist() == [-1, -1, -1]
    finally:
        case.close()


@pytest.mark.parametrize("width", [1, 4, 14])
@pytest.mark.parametrize("graph", [False, True])
def test_long_first_fill_uses_current_for_overlapping_sources(width, graph):
    _check_long_first_fill(width, graph)


def test_first_fill_current_does_not_require_dram_page():
    _check_long_first_fill(4, False, missing_current_page=True)


@pytest.mark.parametrize("graph", [False, True])
def test_long_source_aware_current_ignores_poisoned_host(graph):
    width = 4
    case = _DynamicTndCase(
        query_counts=[width, 1],
        heads=8,
        cache_budgets=[4096, 4096],
        steady_miss_requests={0},
        tail_tokens=0,
    )
    try:
        first = 4096 - width
        current_cpu = torch.randn(width + 1, 1, 576, dtype=torch.bfloat16)
        case.dram.view(-1, 576)[first].fill_(49)
        case.topk_src[:width, 0, 0] = first
        case.miss_src[0, 0] = first
        case.cache_tokens[-1] = 0
        case.actual_kv[-1] = 0
        case.actual_kv[0] = 4096 + width - 1
        current = current_cpu.to(case.device)
        positions = torch.tensor(
            list(range(first, 4096)) + [-1], dtype=torch.int64, device=case.device
        )
        valid = torch.tensor(
            [True] * width + [False], dtype=torch.bool, device=case.device
        )
        slots = torch.tensor([1, 0], dtype=torch.int32, device=case.device)
        table = torch.zeros((3, 4096), dtype=torch.int32, device=case.device)
        begin = torch.full((3,), -1, dtype=torch.int32, device=case.device)
        end = begin.clone()
        initial = case.hbm.clone()

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
                resident_refresh_tail=True,
                current=current,
                positions=positions,
                valid=valid,
                request_slots=slots,
                req_to_token=table,
                resident_begin=begin,
                resident_end=end,
                host_token_capacity=case.dram.numel() // 576,
            )

        if graph:
            run()
            torch.npu.synchronize()
            case.hbm.copy_(initial)
            begin.fill_(-1)
            end.fill_(-1)
            capture = torch.npu.NPUGraph()
            with torch.npu.graph(capture):
                run()
            case.hbm.copy_(initial)
            begin.fill_(-1)
            end.fill_(-1)
            capture.replay()
        else:
            run()
        torch.npu.synchronize()
        block = int(case.hbm_block_table_cpu[0, 0])
        torch.testing.assert_close(
            case.hbm[block, 0, 0].cpu(), current_cpu[0, 0], atol=0, rtol=0
        )
        expected_hbm = initial.cpu().clone()
        expected_hbm[block, 0, 0] = current_cpu[0, 0]
        torch.testing.assert_close(case.hbm.cpu(), expected_hbm, atol=0, rtol=0)
        torch.testing.assert_close(
            case.output[width:].cpu(),
            torch.zeros_like(case.output[width:].cpu()),
            atol=0,
            rtol=0,
        )
        assert begin.cpu().tolist() == [-1, -1, -1]
        assert end.cpu().tolist() == [-1, -1, -1]
        fused = case.output.clone()
        old_output = torch.empty_like(case.output)
        zero_route = torch.zeros_like(case.topk_miss_counts)
        zero_request = torch.zeros_like(case.miss_counts)
        torch.ops.npu.fused_scatter_copy_sparse_flash_attention(
            case.query,
            case.query_rope,
            case.hbm,
            case.actual_q,
            case.actual_kv,
            case.cache_tokens,
            case.topk_src,
            case.sparse_slots,
            zero_route,
            case.miss_src,
            case.miss_dst,
            zero_request,
            case.hbm_block_table,
            case.dram_block_table,
            case.dram_dev_ptr,
            old_output,
            case.scale,
            full_visible=True,
        )
        torch.npu.synchronize()
        torch.testing.assert_close(fused[:width], old_output[:width], atol=0, rtol=0)
    finally:
        case.close()


@pytest.mark.parametrize("graph", [False, True])
def test_mixed_short_first_fill_and_padding(graph):
    width, capacity, first = 4, 2048, 127
    case = _DynamicTndCase(
        query_counts=[width, width, width],
        heads=8,
        cache_budgets=[capacity] * 3,
        first_fill=True,
        first_fill_requests={1},
        tail_tokens=0,
    )
    try:
        current_cpu = torch.randn(3 * width, 1, 576, dtype=torch.bfloat16)
        host = case.dram.view(-1, 576)
        short_expected = host[: first + width].clone()
        short_expected[first:] = current_cpu[:width, 0]
        host[first : first + width].fill_(45)
        long_first = capacity - width
        long_host = capacity + long_first
        host[long_host : long_host + width].fill_(46)
        long_expected = host[capacity : 2 * capacity].clone()
        long_expected[long_first:] = current_cpu[width : 2 * width, 0]
        case.actual_kv.copy_(
            torch.tensor([first + width, capacity + width - 1, 0], device=case.device)
        )
        case.cache_tokens.copy_(torch.tensor([0, capacity, 0], device=case.device))
        case.sparse_slots[:width, :, first + width :] = -1
        case.topk_src[:width].fill_(-1)
        case.topk_miss_counts[:width] = 0
        case.miss_counts[0] = 0
        case.miss_counts[2] = 0
        positions = torch.tensor(
            list(range(first, first + width))
            + list(range(long_first, capacity))
            + [-1] * width,
            dtype=torch.int64,
            device=case.device,
        )
        valid = torch.tensor([True] * (2 * width) + [False] * width, device=case.device)
        slots = torch.tensor([1, 2, 0], dtype=torch.int32, device=case.device)
        table_cpu = torch.zeros((3, capacity), dtype=torch.int32)
        table_cpu[1] = torch.arange(capacity, dtype=torch.int32)
        table = table_cpu.to(case.device)
        begin = torch.full((3,), -1, dtype=torch.int32, device=case.device)
        end = begin.clone()
        current = current_cpu.to(case.device)
        initial = case.hbm.clone()

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
                current=current,
                positions=positions,
                valid=valid,
                request_slots=slots,
                req_to_token=table,
                resident_begin=begin,
                resident_end=end,
                host_token_capacity=case.dram.numel() // 576,
            )

        if graph:
            run()
            torch.npu.synchronize()
            case.hbm.copy_(initial)
            begin.fill_(-1)
            end.fill_(-1)
            capture = torch.npu.NPUGraph()
            with torch.npu.graph(capture):
                run()
            case.hbm.copy_(initial)
            begin.fill_(-1)
            end.fill_(-1)
            capture.replay()
        else:
            run()
        torch.npu.synchronize()
        cache = case.hbm.cpu().view(-1, 128, 576)
        short_pages = case.hbm_block_table_cpu[0, :16].long()
        long_pages = case.hbm_block_table_cpu[1, :16].long()
        torch.testing.assert_close(
            cache[short_pages].reshape(capacity, 576)[: first + width],
            short_expected,
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            cache[long_pages].reshape(capacity, 576)[long_first:capacity],
            current_cpu[width : 2 * width, 0],
            atol=0,
            rtol=0,
        )
        expected_hbm = initial.cpu().view(-1, 128, 576).clone()
        for pos in range(first + width):
            expected_hbm[short_pages[pos // 128], pos % 128] = short_expected[pos]
        expected_hbm[long_pages] = long_expected.reshape(16, 128, 576)
        torch.testing.assert_close(cache, expected_hbm, atol=0, rtol=0)
        torch.testing.assert_close(
            case.output[2 * width :].cpu(),
            torch.zeros_like(case.output[2 * width :].cpu()),
            atol=0,
            rtol=0,
        )
        assert begin.cpu().tolist() == [-1, 0, -1]
        assert end.cpu().tolist() == [-1, first + width, -1]
        q, qr, actual = (
            case.query.cpu().float(),
            case.query_rope.cpu().float(),
            case.output.cpu().float(),
        )
        for route in range(width):
            visible = short_expected[: first + route + 1].float()
            scores = (
                q[route] @ visible[:, :512].T + qr[route] @ visible[:, 512:].T
            ) * case.scale
            expected = torch.softmax(scores, -1) @ visible[:, :512]
            torch.testing.assert_close(actual[route], expected, atol=0.003, rtol=0.02)
    finally:
        case.close()


@pytest.mark.parametrize("graph", [False, True])
def test_split_host_long_misses_match_combined_across_pages(graph):
    case = _DynamicTndCase(
        query_counts=[1],
        heads=8,
        cache_budgets=[4096],
        steady_miss_requests={0},
        tail_tokens=0,
    )
    try:
        ops = _require_npu()
        rows = case.dram.view(-1, 576)
        host_k, _, k_ptr = ops.create_fused_shm_tensor(
            (rows.shape[0], 512),
            case.dtype,
            device_id=case.device_id,
            name="copy_sfa_long_k",
        )
        host_rope, _, rope_ptr = ops.create_fused_shm_tensor(
            (rows.shape[0], 64),
            case.dtype,
            device_id=case.device_id,
            name="copy_sfa_long_rope",
        )
        host_k.copy_(rows[:, :512])
        host_rope.copy_(rows[:, 512:])
        # Adjacent logical tokens straddle a page and map to distant physical
        # pages, exposing row-stride and page-table mistakes independently.
        pages = torch.arange(case.dram_blocks - 1, -1, -1, dtype=torch.int32)
        case.dram_block_table.copy_(pages.view_as(case.dram_block_table))
        case.topk_src[0, 0, :2] = torch.tensor([127, 128], device=case.device)
        case.topk_miss_counts[0] = 2
        case.miss_src[0, :2] = torch.tensor([127, 128], device=case.device)
        case.miss_dst[0, :2] = torch.tensor([0, 1], device=case.device)
        case.miss_counts[0] = 2
        initial = case.hbm.clone()
        expected = torch.stack(
            (rows[int(pages[0]) * 128 + 127], rows[int(pages[1]) * 128])
        )

        def run(kv_ptr, split_ptr):
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
                kv_ptr,
                case.output,
                case.scale,
                dram_rope_dev_ptr=split_ptr,
            )

        run(case.dram_dev_ptr, 0)
        torch.npu.synchronize()
        combined_hbm, combined_output = case.hbm.clone(), case.output.clone()
        page = int(case.hbm_block_table_cpu[0, 0])
        torch.testing.assert_close(
            combined_hbm[page, :2, 0].cpu(), expected, atol=0, rtol=0
        )
        case.hbm.copy_(initial)
        if graph:
            capture = torch.npu.NPUGraph()
            with torch.npu.graph(capture):
                run(k_ptr, rope_ptr)
            case.hbm.copy_(initial)
            capture.replay()
        else:
            run(k_ptr, rope_ptr)
        torch.npu.synchronize()
        torch.testing.assert_close(case.hbm, combined_hbm, atol=0, rtol=0)
        torch.testing.assert_close(case.output, combined_output, atol=0, rtol=0)
    finally:
        case.close()
