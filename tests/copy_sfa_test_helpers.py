# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared test fixtures for split-cache Copy-SFA.

The upstream matrix is adapted from vllm-ascend PR #16640 at
3326f350c0de30a5170b86ecfce851322e3216e6. These tests require a built native
extension and an available NPU. Each case owns one registered host allocation
and releases it after synchronizing the device.
"""

from __future__ import annotations

import ctypes
import importlib
import math
import mmap
import sys

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
    return sys.modules[__name__]


_registered_host = []


def create_shm_tensor(shape, dtype, device_id=0, name=""):
    """Create test-owned mapped host memory using the public CANN runtime."""
    del name
    torch.npu.set_device(device_id)
    size = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
    memory = mmap.mmap(-1, size)
    address = ctypes.addressof(ctypes.c_char.from_buffer(memory))
    acl = ctypes.CDLL("libascendcl.so")
    acl.aclrtHostRegister.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    acl.aclrtHostRegister.restype = ctypes.c_int
    device_pointer = ctypes.c_void_p()
    status = acl.aclrtHostRegister(address, size, 0, ctypes.byref(device_pointer))
    if status:
        memory.close()
        raise RuntimeError(f"aclrtHostRegister failed: {status}")
    _registered_host.append((acl, address, memory))
    return (
        torch.frombuffer(memory, dtype=dtype).view(shape),
        address,
        device_pointer.value,
    )


def free_shm(device_id=0):
    torch.npu.set_device(device_id)
    torch.npu.synchronize(device_id)
    while _registered_host:
        acl, address, memory = _registered_host.pop()
        acl.aclrtHostUnregister.argtypes = [ctypes.c_void_p]
        acl.aclrtHostUnregister.restype = ctypes.c_int
        status = acl.aclrtHostUnregister(address)
        if status:
            raise RuntimeError(f"aclrtHostUnregister failed: {status}")
        # Tensor views retain the mmap until the test releases them.


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
            .view(1, TOPK)
            .expand(total_query_tokens, TOPK)
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

        # Keep the CKV and KPE reference rows together for the CPU oracle.
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
        self.dram = dram_cpu

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
                self.topk_src_cpu[start:stop, 0] = source_id
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
