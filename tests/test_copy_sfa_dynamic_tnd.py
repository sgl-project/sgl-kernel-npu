# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Correctness coverage for combined-cache Copy-SFA with dynamic TND queries.

The upstream matrix is adapted from vllm-ascend PR #16640 at
3326f350c0de30a5170b86ecfce851322e3216e6. These tests require a built native
extension and an available NPU. Each case owns one registered host allocation
and releases it after synchronizing the device.
"""

from __future__ import annotations

import importlib
import math

import pytest
import torch

BLOCK_SIZE = 128
TOPK = 2048
CKV_DIM = 512
KPE_DIM = 64
TOKEN_STRIDE = CKV_DIM + KPE_DIM
MISS_CAPACITY = 32768


def _require_npu():
    try:
        importlib.import_module("torch_npu")
    except ImportError:
        pytest.skip("torch_npu is not installed")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("NPU is required for native Copy-SFA integration tests")
    importlib.import_module("sgl_kernel_npu")  # register native operators
    return importlib.import_module("sgl_kernel_npu.sparsity_driven_kv_offload")


def _prefix_sum(values: list[int]) -> list[int]:
    total = 0
    result: list[int] = []
    for value in values:
        total += value
        result.append(total)
    return result


def _logical_rows(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    request: int,
    logical_slots: torch.Tensor,
) -> torch.Tensor:
    physical_blocks = block_table[request, logical_slots // BLOCK_SIZE].to(torch.int64)
    offsets = logical_slots % BLOCK_SIZE
    return cache[physical_blocks, offsets]


class _DynamicTndCase:
    def __init__(
        self,
        query_counts: list[int],
        heads: int,
        *,
        first_fill: bool = False,
        cache_budgets: list[int] | None = None,
        dtype: torch.dtype = torch.bfloat16,
        first_fill_requests: set[int] | None = None,
        steady_miss_requests: set[int] | None = None,
        tail_tokens: int = 2,
        poisoned_future_tail: bool = False,
    ) -> None:
        ops = _require_npu()
        self.free_shm = ops.free_shm
        self.device_id = torch.npu.current_device()
        self.device = torch.device("npu", self.device_id)
        self.query_counts = query_counts
        self.heads = heads
        self.first_fill_requests = (
            first_fill_requests
            if first_fill_requests is not None
            else (set(range(len(query_counts))) if first_fill else set())
        )
        self.steady_miss_requests = steady_miss_requests or set()
        self.first_fill_all_queries = first_fill and not self.steady_miss_requests
        self.poisoned_future_tail = poisoned_future_tail
        self.dtype = dtype

        torch.manual_seed(20260901 + heads + sum(query_counts))
        batch_size = len(query_counts)
        total_query_tokens = sum(query_counts)
        if cache_budgets is None:
            cache_budgets = [TOPK] * batch_size
        if len(cache_budgets) != batch_size:
            raise ValueError("cache budgets must match batch size")
        if any(budget < TOPK or budget % BLOCK_SIZE for budget in cache_budgets):
            raise ValueError("cache budgets must be block aligned and >= TopK")
        self.cache_budgets = cache_budgets

        # Leave two request-local tail rows before the MTP query rows. The
        # kernel must hide future query rows from earlier routes.
        max_logical_tokens = max(
            budget + tail_tokens + count
            for budget, count in zip(cache_budgets, query_counts)
        )
        blocks_per_request = math.ceil(max_logical_tokens / BLOCK_SIZE)
        self.hbm_block_table_cpu = torch.arange(
            batch_size * blocks_per_request, dtype=torch.int32
        ).view(batch_size, blocks_per_request)
        self.hbm_blocks = batch_size * blocks_per_request
        initial_hbm_kpe_cpu = torch.randn(
            self.hbm_blocks, BLOCK_SIZE, KPE_DIM, dtype=dtype
        )
        initial_hbm_ckv_cpu = torch.randn(
            self.hbm_blocks, BLOCK_SIZE, CKV_DIM, dtype=dtype
        )
        query_cpu = torch.randn(total_query_tokens, heads, CKV_DIM, dtype=dtype)
        query_rope_cpu = torch.randn(total_query_tokens, heads, KPE_DIM, dtype=dtype)
        initial_hbm_cpu = torch.cat((initial_hbm_ckv_cpu, initial_hbm_kpe_cpu), dim=-1)
        if poisoned_future_tail:
            # Poison the two future speculative rows for request 0. Route 0
            # must not see either row; route 1 may see the first, and route 2
            # may see both. This makes a missing query-causal tail mask visible.
            request = 0
            start = cache_budgets[request] + 3
            slots = torch.arange(start, start + 2, dtype=torch.int64)
            blocks = self.hbm_block_table_cpu[request, slots // BLOCK_SIZE].to(
                torch.int64
            )
            offsets = slots % BLOCK_SIZE
            initial_hbm_cpu[blocks, offsets, CKV_DIM:] = 1000.0
        self.expected_hbm_cpu = initial_hbm_cpu.clone()

        self.sparse_slots_cpu = (
            torch.arange(TOPK, dtype=torch.int32)
            .view(1, 1, TOPK)
            .expand(total_query_tokens, 1, TOPK)
            .contiguous()
        )
        self.topk_src_cpu = self.sparse_slots_cpu.clone()
        self.topk_miss_counts_cpu = torch.zeros(total_query_tokens, dtype=torch.int32)
        if self.first_fill_all_queries:
            # Match the upstream fixture: sentinel TopK metadata spans all
            # query rows even though the request-level first-fill plan below
            # intentionally populates request 0 only.
            self.topk_src_cpu.fill_(-777)
            self.topk_miss_counts_cpu.fill_(TOPK)
        self.actual_q_cpu = torch.tensor(_prefix_sum(query_counts), dtype=torch.int32)
        self.actual_kv_cpu = torch.tensor(
            [
                budget + tail_tokens + count
                for budget, count in zip(cache_budgets, query_counts)
            ],
            dtype=torch.int32,
        )
        self.cache_tokens_cpu = torch.tensor(cache_budgets, dtype=torch.int32)

        # The operator reads host KV through this registered, device-mapped
        # allocation. Keeping both K/V portions in each 576-wide token matches
        # the production combined-cache layout.
        dram_blocks_per_request = (
            max(1, math.ceil(max(cache_budgets) / BLOCK_SIZE))
            if first_fill or self.steady_miss_requests
            else 1
        )
        self.dram_block_table_cpu = torch.arange(
            batch_size * dram_blocks_per_request, dtype=torch.int32
        ).view(batch_size, dram_blocks_per_request)
        self.dram_blocks = batch_size * dram_blocks_per_request
        dram_kpe_cpu = torch.randn(self.dram_blocks, BLOCK_SIZE, KPE_DIM, dtype=dtype)
        dram_ckv_cpu = torch.randn(self.dram_blocks, BLOCK_SIZE, CKV_DIM, dtype=dtype)
        dram_cpu = torch.cat((dram_ckv_cpu, dram_kpe_cpu), dim=-1)
        self.dram, _, self.dram_dev_ptr = ops.create_shm_tensor(
            (self.dram_blocks, BLOCK_SIZE, TOKEN_STRIDE),
            dtype,
            device_id=self.device_id,
            name="copy_sfa_dynamic_tnd_dram",
        )
        self.dram.copy_(dram_cpu)

        self.miss_src_cpu = torch.full(
            (batch_size, MISS_CAPACITY), -1, dtype=torch.int32
        )
        self.miss_dst_cpu = torch.full_like(self.miss_src_cpu, -1)
        self.miss_counts_cpu = torch.zeros(batch_size, dtype=torch.int32)
        query_starts = _prefix_sum([0] + query_counts[:-1])
        for request in self.first_fill_requests:
            budget = cache_budgets[request]
            start = query_starts[request]
            stop = start + query_counts[request]
            if not self.first_fill_all_queries:
                self.topk_src_cpu[start:stop].fill_(-777)
                self.topk_miss_counts_cpu[start:stop] = TOPK
            self.miss_src_cpu[request, :budget] = torch.arange(
                budget, dtype=torch.int32
            )
            self.miss_dst_cpu[request, :budget] = torch.arange(
                budget, dtype=torch.int32
            )
            self.miss_counts_cpu[request] = budget
            slots = torch.arange(budget, dtype=torch.int64)
            source_blocks = self.dram_block_table_cpu[request, slots // BLOCK_SIZE].to(
                torch.int64
            )
            source_offsets = slots % BLOCK_SIZE
            destination_blocks = self.hbm_block_table_cpu[
                request, slots // BLOCK_SIZE
            ].to(torch.int64)
            destination_offsets = slots % BLOCK_SIZE
            self.expected_hbm_cpu[destination_blocks, destination_offsets] = dram_cpu[
                source_blocks, source_offsets
            ]

        if self.steady_miss_requests:
            for request in self.steady_miss_requests:
                if request in self.first_fill_requests:
                    raise ValueError(
                        "a request cannot be both first-fill and steady-miss"
                    )
                # All MTP routes share this one missing source and slot. The
                # request plan contains the union once; each query carries the
                # same one-entry miss prefix, which exercises repeated writes
                # to the same destination with identical host payload.
                source_id = cache_budgets[request] - 1
                if source_id < TOPK:
                    raise ValueError("steady miss source must lie beyond TopK")
                start = query_starts[request]
                stop = start + query_counts[request]
                self.topk_src_cpu[start:stop, 0, 0] = source_id
                self.topk_miss_counts_cpu[start:stop] = 1
                self.miss_src_cpu[request, 0] = source_id
                self.miss_dst_cpu[request, 0] = 0
                self.miss_counts_cpu[request] = 1
                source_slot = torch.tensor(source_id, dtype=torch.int64)
                source_block = int(
                    self.dram_block_table_cpu[request, source_slot // BLOCK_SIZE].item()
                )
                source_offset = int((source_slot % BLOCK_SIZE).item())
                destination_block = int(self.hbm_block_table_cpu[request, 0].item())
                self.expected_hbm_cpu[destination_block, 0] = dram_cpu[
                    source_block, source_offset
                ]

        scale = 1.0 / math.sqrt(CKV_DIM + KPE_DIM)
        expected_rows: list[torch.Tensor] = []
        query_row = 0
        for request, query_count in enumerate(query_counts):
            cache_tokens = cache_budgets[request]
            for route in range(query_count):
                visible_kv = int(self.actual_kv_cpu[request]) - (
                    query_count - 1 - route
                )
                logical_slots = torch.cat(
                    (
                        torch.arange(TOPK, dtype=torch.int64),
                        torch.arange(cache_tokens, visible_kv, dtype=torch.int64),
                    )
                )
                selected = _logical_rows(
                    self.expected_hbm_cpu,
                    self.hbm_block_table_cpu,
                    request,
                    logical_slots,
                ).float()
                selected_ckv = selected[:, :CKV_DIM]
                selected_kpe = selected[:, CKV_DIM:]
                scores = (
                    query_cpu[query_row].float() @ selected_ckv.T
                    + query_rope_cpu[query_row].float() @ selected_kpe.T
                ) * scale
                expected_row = torch.softmax(scores, dim=-1) @ selected_ckv
                expected_rows.append(expected_row)
                query_row += 1
        self.expected = torch.stack(expected_rows)

        def to_npu(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.to(self.device)

        self.query = to_npu(query_cpu)
        self.query_rope = to_npu(query_rope_cpu)
        self.initial_hbm = to_npu(initial_hbm_cpu).view(
            self.hbm_blocks, BLOCK_SIZE, 1, TOKEN_STRIDE
        )
        self.hbm = to_npu(initial_hbm_cpu).view(
            self.hbm_blocks, BLOCK_SIZE, 1, TOKEN_STRIDE
        )
        self.actual_q = to_npu(self.actual_q_cpu)
        self.actual_kv = to_npu(self.actual_kv_cpu)
        self.cache_tokens = to_npu(self.cache_tokens_cpu)
        self.sparse_slots = to_npu(self.sparse_slots_cpu)
        self.topk_src = to_npu(self.topk_src_cpu)
        self.topk_miss_counts = to_npu(self.topk_miss_counts_cpu)
        self.miss_src = to_npu(self.miss_src_cpu)
        self.miss_dst = to_npu(self.miss_dst_cpu)
        self.miss_counts = to_npu(self.miss_counts_cpu)
        self.hbm_block_table = to_npu(self.hbm_block_table_cpu)
        self.dram_block_table = to_npu(self.dram_block_table_cpu)
        self.output = torch.empty_like(self.query)
        self.scale = scale
        self._closed = False

    def call(
        self,
        hbm=None,
        topk_src=None,
        topk_miss_counts=None,
        miss_counts=None,
        output=None,
    ):
        torch.ops.npu.fused_scatter_copy_sparse_flash_attention(
            self.query,
            self.query_rope,
            self.hbm if hbm is None else hbm,
            self.actual_q,
            self.actual_kv,
            self.cache_tokens,
            self.topk_src if topk_src is None else topk_src,
            self.sparse_slots,
            self.topk_miss_counts if topk_miss_counts is None else topk_miss_counts,
            self.miss_src,
            self.miss_dst,
            self.miss_counts if miss_counts is None else miss_counts,
            self.hbm_block_table,
            self.dram_block_table,
            self.dram_dev_ptr,
            self.output if output is None else output,
            self.scale,
        )

    def assert_result(self, actual: torch.Tensor, label: str) -> None:
        actual_cpu = actual.cpu().float()
        max_abs = float((actual_cpu - self.expected).abs().max())
        assert max_abs < 0.08, f"{label} mismatch: max_abs={max_abs}"

    def assert_cache_payload(self, cache: torch.Tensor, label: str) -> None:
        torch.testing.assert_close(
            cache.cpu().view(self.hbm_blocks, BLOCK_SIZE, TOKEN_STRIDE),
            self.expected_hbm_cpu,
            rtol=0,
            atol=0,
            msg=label,
        )

    def close(self) -> None:
        if self._closed:
            return
        try:
            torch.npu.synchronize(self.device_id)
        finally:
            # The sgl-kernel-npu allocator currently exposes free-all only.
            # These cases run sequentially and this class owns its sole
            # registered allocation until teardown.
            self.free_shm(self.device_id)
            self._closed = True


def _run_and_check(**kwargs) -> None:
    case = _DynamicTndCase(**kwargs)
    try:
        case.call()
        torch.npu.synchronize(case.device_id)
        case.assert_result(case.output, "dynamic TND eager")
        case.assert_cache_payload(case.hbm, "dynamic TND eager cache")

        graph_hbm = case.initial_hbm.clone()
        graph_topk_src = case.topk_src.clone()
        graph_topk_misses = case.topk_miss_counts.clone()
        graph_miss_counts = case.miss_counts.clone()
        if case.first_fill_all_queries:
            graph_topk_src.copy_(case.sparse_slots)
            graph_topk_misses.zero_()
            graph_miss_counts.zero_()
        elif case.first_fill_requests:
            # Capture the steady branch, then switch stable device metadata to
            # first-fill before replay so the graph reads runtime metadata.
            query_starts = _prefix_sum([0] + case.query_counts[:-1])
            for request in case.first_fill_requests:
                start = query_starts[request]
                stop = start + case.query_counts[request]
                graph_topk_src[start:stop].copy_(case.sparse_slots[start:stop])
                graph_topk_misses[start:stop].zero_()
                graph_miss_counts[request] = 0
        graph_output = torch.empty_like(case.query)
        graph = torch.npu.NPUGraph()
        pool = torch.npu.graph_pool_handle()
        with torch.npu.graph(graph, pool=pool):
            case.call(
                hbm=graph_hbm,
                topk_src=graph_topk_src,
                topk_miss_counts=graph_topk_misses,
                miss_counts=graph_miss_counts,
                output=graph_output,
            )
        torch.npu.synchronize(case.device_id)
        if case.first_fill_requests:
            graph_hbm.copy_(case.initial_hbm)
            graph_topk_src.copy_(case.topk_src)
            graph_topk_misses.copy_(case.topk_miss_counts)
            graph_miss_counts.copy_(case.miss_counts)
            torch.npu.synchronize(case.device_id)
        graph.replay()
        torch.npu.synchronize(case.device_id)
        case.assert_result(graph_output, "dynamic TND graph")
        torch.testing.assert_close(graph_output, case.output, rtol=0, atol=0)
        case.assert_cache_payload(graph_hbm, "dynamic TND graph cache")
    finally:
        case.close()


class _C0DenseMixedGraphCase:
    """Bounded C==0 dense case with a sparse neighbor and a padded request."""

    def __init__(self) -> None:
        ops = _require_npu()
        self.free_shm = ops.free_shm
        self.device_id = torch.npu.current_device()
        self.device = torch.device("npu", self.device_id)
        self.heads = 8
        self.query = torch.randn(3, self.heads, CKV_DIM, dtype=torch.bfloat16)
        self.query_rope = torch.randn(3, self.heads, KPE_DIM, dtype=torch.bfloat16)
        self.actual_q = torch.tensor([1, 2, 3], dtype=torch.int32)
        # Request 0 is a short dense row, request 1 is steady sparse+tail,
        # and request 2 is padding. The graph replay extends request 0 to 2177.
        self.actual_kv_cpu = torch.tensor([129, 2177, 0], dtype=torch.int32)
        self.cache_tokens_cpu = torch.tensor([0, TOPK, 0], dtype=torch.int32)
        self.hbm_blocks_per_request = math.ceil(2177 / BLOCK_SIZE)
        self.hbm_block_table_cpu = torch.arange(
            3 * self.hbm_blocks_per_request, dtype=torch.int32
        ).view(3, self.hbm_blocks_per_request)
        self.hbm_cpu = torch.randn(
            3 * self.hbm_blocks_per_request,
            BLOCK_SIZE,
            1,
            TOKEN_STRIDE,
            dtype=torch.bfloat16,
        )
        self.topk_src_cpu = torch.arange(TOPK, dtype=torch.int32).view(1, 1, TOPK)
        self.topk_src_cpu = self.topk_src_cpu.expand(3, 1, TOPK).clone()
        self.topk_src_cpu[[0, 2]].fill_(-777)
        self.topk_dst_cpu = self.topk_src_cpu.clone()
        self.topk_miss_counts_cpu = torch.tensor(
            [MISS_CAPACITY + 1, 0, MISS_CAPACITY + 1], dtype=torch.int32
        )
        self.miss_src_cpu = torch.full((3, MISS_CAPACITY), -1, dtype=torch.int32)
        self.miss_dst_cpu = torch.full_like(self.miss_src_cpu, -1)
        self.miss_counts_cpu = torch.zeros(3, dtype=torch.int32)

        self.dram, _, self.dram_dev_ptr = ops.create_shm_tensor(
            (1, BLOCK_SIZE, TOKEN_STRIDE),
            torch.bfloat16,
            device_id=self.device_id,
            name="copy_sfa_c0_dense_graph_dram",
        )
        self.dram.zero_()
        self.hbm = self.hbm_cpu.to(self.device)
        self.query = self.query.to(self.device)
        self.query_rope = self.query_rope.to(self.device)
        self.actual_q = self.actual_q.to(self.device)
        self.actual_kv = self.actual_kv_cpu.to(self.device)
        self.cache_tokens = self.cache_tokens_cpu.to(self.device)
        self.topk_src = self.topk_src_cpu.to(self.device)
        self.topk_dst = self.topk_dst_cpu.to(self.device)
        self.topk_miss_counts = self.topk_miss_counts_cpu.to(self.device)
        self.miss_src = self.miss_src_cpu.to(self.device)
        self.miss_dst = self.miss_dst_cpu.to(self.device)
        self.miss_counts = self.miss_counts_cpu.to(self.device)
        self.hbm_block_table = self.hbm_block_table_cpu.to(self.device)
        self.dram_block_table = torch.zeros(
            (3, 1), dtype=torch.int32, device=self.device
        )
        self.sparse_slots = self.topk_dst
        self.output = torch.empty_like(self.query)
        self.scale = 1.0 / math.sqrt(TOKEN_STRIDE)
        self._closed = False

    def call(
        self,
        *,
        hbm,
        actual_kv,
        cache_tokens,
        topk_src,
        topk_miss_counts,
        output,
        topk_dst=None,
    ):
        torch.ops.npu.fused_scatter_copy_sparse_flash_attention(
            self.query,
            self.query_rope,
            hbm,
            self.actual_q,
            actual_kv,
            cache_tokens,
            topk_src,
            self.sparse_slots if topk_dst is None else topk_dst,
            topk_miss_counts,
            self.miss_src,
            self.miss_dst,
            self.miss_counts,
            self.hbm_block_table,
            self.dram_block_table,
            self.dram_dev_ptr,
            output,
            self.scale,
        )

    def expected_output(self, actual_kv_cpu, cache_tokens_cpu, topk_src_cpu):
        rows = []
        for request, query_row in enumerate(range(3)):
            visible = int(actual_kv_cpu[request])
            if visible == 0:
                rows.append(torch.zeros(self.heads, CKV_DIM))
                continue
            cache_tokens = int(cache_tokens_cpu[request])
            if cache_tokens == 0:
                slots = torch.arange(visible, dtype=torch.int64)
            else:
                slots = torch.cat(
                    (
                        topk_src_cpu[query_row, 0].to(torch.int64),
                        torch.arange(cache_tokens, visible, dtype=torch.int64),
                    )
                )
            selected = _logical_rows(
                self.hbm_cpu.view(-1, BLOCK_SIZE, TOKEN_STRIDE),
                self.hbm_block_table_cpu,
                request,
                slots,
            ).float()
            keys = selected[:, :CKV_DIM]
            rope = selected[:, CKV_DIM:]
            scores = (
                self.query[query_row].cpu().float() @ keys.T
                + self.query_rope[query_row].cpu().float() @ rope.T
            ) * self.scale
            rows.append(torch.softmax(scores, dim=-1) @ keys)
        return torch.stack(rows)

    def assert_result(
        self, output, actual_kv_cpu, cache_tokens_cpu, topk_src_cpu, label
    ):
        expected = self.expected_output(actual_kv_cpu, cache_tokens_cpu, topk_src_cpu)
        actual = output.cpu().float()
        max_abs = float((actual - expected).abs().max())
        assert max_abs < 0.08, f"{label} mismatch: max_abs={max_abs}"
        torch.testing.assert_close(
            actual[2], torch.zeros_like(actual[2]), rtol=0, atol=0
        )

    def close(self):
        if self._closed:
            return
        try:
            torch.npu.synchronize(self.device_id)
        finally:
            self.free_shm(self.device_id)
            self._closed = True


def test_copy_sfa_c0_dense_mixed_graph_and_first_long_transition():
    case = _C0DenseMixedGraphCase()
    try:
        case.call(
            hbm=case.hbm,
            actual_kv=case.actual_kv,
            cache_tokens=case.cache_tokens,
            topk_src=case.topk_src,
            topk_miss_counts=case.topk_miss_counts,
            output=case.output,
        )
        torch.npu.synchronize(case.device_id)
        case.assert_result(
            case.output,
            case.actual_kv_cpu,
            case.cache_tokens_cpu,
            case.topk_src_cpu,
            "C0 dense mixed eager",
        )
        torch.testing.assert_close(case.hbm.cpu(), case.hbm_cpu, rtol=0, atol=0)

        graph_hbm = case.hbm.clone()
        graph_actual_kv = case.actual_kv.clone()
        graph_cache_tokens = case.cache_tokens.clone()
        graph_topk_src = case.topk_src.clone()
        graph_topk_dst = case.topk_dst.clone()
        graph_topk_miss_counts = case.topk_miss_counts.clone()
        graph_output = torch.empty_like(case.query)
        graph = torch.npu.NPUGraph()
        pool = torch.npu.graph_pool_handle()
        with torch.npu.graph(graph, pool=pool):
            case.call(
                hbm=graph_hbm,
                actual_kv=graph_actual_kv,
                cache_tokens=graph_cache_tokens,
                topk_src=graph_topk_src,
                topk_dst=graph_topk_dst,
                topk_miss_counts=graph_topk_miss_counts,
                output=graph_output,
            )
        torch.npu.synchronize(case.device_id)
        graph.replay()
        torch.npu.synchronize(case.device_id)
        case.assert_result(
            graph_output,
            case.actual_kv_cpu,
            case.cache_tokens_cpu,
            case.topk_src_cpu,
            "C0 dense mixed graph",
        )
        torch.testing.assert_close(graph_hbm.cpu(), case.hbm_cpu, rtol=0, atol=0)

        # Cross from dense C==0 into the first sparse prefix on the same graph.
        # The initial C0 route has poisoned TopK IDs and an impossible miss count;
        # only the long replay makes those per-query values valid.
        graph_cache_tokens[0] = TOPK
        graph_actual_kv[0] = 2177
        graph_topk_src[0].copy_(
            torch.arange(TOPK, dtype=torch.int32, device=case.device)
        )
        graph_topk_dst[0].copy_(
            torch.arange(TOPK, dtype=torch.int32, device=case.device)
        )
        graph_topk_miss_counts[0] = 0
        graph.replay()
        torch.npu.synchronize(case.device_id)
        transitioned_cache_tokens = case.cache_tokens_cpu.clone()
        transitioned_cache_tokens[0] = TOPK
        transitioned_actual_kv = case.actual_kv_cpu.clone()
        transitioned_actual_kv[0] = 2177
        transitioned_topk_src = case.topk_src_cpu.clone()
        transitioned_topk_src[0].copy_(torch.arange(TOPK, dtype=torch.int32))
        case.assert_result(
            graph_output,
            transitioned_actual_kv,
            transitioned_cache_tokens,
            transitioned_topk_src,
            "C0 to first-long graph transition",
        )
        torch.testing.assert_close(graph_hbm.cpu(), case.hbm_cpu, rtol=0, atol=0)
    finally:
        case.close()


def test_copy_sfa_dynamic_tnd_heterogeneous_mtp_and_cache_budgets():
    _run_and_check(
        query_counts=[3, 4, 9, 16],
        heads=8,
        cache_budgets=[2048, 2176, 2304, 2432],
    )


def test_copy_sfa_dynamic_tnd_head_count_128():
    _run_and_check(query_counts=[1], heads=128)


def test_copy_sfa_dynamic_tnd_fp16():
    _run_and_check(query_counts=[1], heads=8, dtype=torch.float16)


@pytest.mark.parametrize("heads", [16, 32, 64])
def test_copy_sfa_dynamic_tnd_intermediate_head_counts(heads):
    _run_and_check(query_counts=[1], heads=heads)


def test_copy_sfa_dynamic_tnd_heterogeneous_first_fill_graph():
    _run_and_check(
        query_counts=[3, 4, 9],
        heads=8,
        first_fill=True,
        first_fill_requests={0},
        cache_budgets=[2048, 2176, 2304],
    )


def test_copy_sfa_dynamic_tnd_all_request_first_fill_graph():
    _run_and_check(
        query_counts=[3, 4, 9],
        heads=8,
        first_fill=True,
        first_fill_requests={0, 1, 2},
        cache_budgets=[2048, 2176, 2304],
    )


def test_copy_sfa_dynamic_tnd_mixed_first_fill_and_shared_steady_miss():
    _run_and_check(
        query_counts=[3, 4],
        heads=8,
        first_fill=True,
        first_fill_requests={0},
        steady_miss_requests={1},
        cache_budgets=[2048, 2176],
    )


def test_copy_sfa_dynamic_tnd_steady_nonzero_miss():
    _run_and_check(
        query_counts=[3],
        heads=8,
        steady_miss_requests={0},
        cache_budgets=[2176],
    )


def test_copy_sfa_dynamic_tnd_query_causal_poisoned_tail():
    _run_and_check(
        query_counts=[3],
        heads=8,
        cache_budgets=[2048],
        poisoned_future_tail=True,
    )


def test_copy_sfa_dynamic_tnd_tail_crosses_page_boundary():
    # The earliest query sees C+128 tokens, while later MTP routes extend into
    # the next tail page. This checks the causal tail walk across page 128.
    _run_and_check(
        query_counts=[4],
        heads=8,
        cache_budgets=[2048],
        tail_tokens=127,
    )


def test_copy_sfa_dynamic_tnd_lim_sized_first_fill_graph():
    _run_and_check(
        query_counts=[4],
        heads=8,
        first_fill=True,
        cache_budgets=[12288],
    )
