"""A5-only Q4, graph, padding and metadata tests for C8 CopySFA.

The numerical tests require a wheel built for Ascend950 and registered host
memory. They intentionally construct CopySFA metadata directly and have no LIM
or vLLM runtime dependency. They were added without local A5 execution.
"""

from __future__ import annotations

import math

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

import sgl_kernel_npu  # noqa: E402,F401
from fused_quant_scatter_copy_sparse_flash_attention_golden import (  # noqa: E402
    BLOCK_SIZE,
    COPY_CAPACITY,
    NOPE_DIM,
    PACKED_DIM,
    POISON,
    TOPK,
    make_packed_bytes,
    unpack_rows,
)


@pytest.fixture()
def require_a5_device():
    import acl

    if acl.rt.get_device_info(0, 601) != (3510, 0):
        pytest.skip("C8 CopySFA numerical tests require Ascend950 (A5)")
    assert hasattr(torch.ops.npu, "fused_quant_scatter_copy_sparse_flash_attention")


def _permutation(size: int, multiplier: int) -> torch.Tensor:
    return torch.tensor(
        [(index * multiplier) % size for index in range(size)], dtype=torch.int32
    )


def _physical_rows(
    table: torch.Tensor, request: int, logical_tokens: torch.Tensor
) -> torch.Tensor:
    logical_tokens = logical_tokens.to(torch.int64)
    return table[request, logical_tokens // BLOCK_SIZE].to(
        torch.int64
    ) * BLOCK_SIZE + logical_tokens.remainder(BLOCK_SIZE)


class _Q4CopySfaCase:
    batch = 3
    total_queries = 9
    live_requests = (0, 2)
    padding_request = 1
    padding_query = 4
    blocks_per_request = TOPK // BLOCK_SIZE

    def __init__(self, heads: int = 8):
        from sgl_kernel_npu.sparsity_driven_kv_offload import create_shm_tensor

        self.device = torch.device("npu:0")
        torch.npu.set_device(self.device)
        generator = torch.Generator().manual_seed(20261031)

        logical_dram = make_packed_bytes(self.blocks_per_request, generator)
        dram_permutation = _permutation(self.blocks_per_request, 5)
        physical_dram = torch.empty_like(logical_dram)
        for logical_page, physical_page in enumerate(dram_permutation.tolist()):
            physical_dram[physical_page].copy_(logical_dram[logical_page])
        self.logical_rows = logical_dram.view(-1, PACKED_DIM)
        self.dram_owner, _, self.dram_dev_ptr = create_shm_tensor(
            physical_dram.shape, torch.int8, device_id=0
        )
        self.dram_owner.copy_(physical_dram)
        self.dram_before = self.dram_owner.clone()

        # Both live requests deliberately share the same physical DRAM pages.
        # HBM has a separately permuted bank per live request and a reserved
        # bank for the padding request, so stray padding writes are observable.
        self.dram_table_cpu = dram_permutation.repeat(self.batch, 1)
        hbm_permutation = _permutation(self.blocks_per_request, 7)
        self.hbm_table_cpu = torch.stack(
            (
                hbm_permutation,
                hbm_permutation + 2 * self.blocks_per_request,
                hbm_permutation + self.blocks_per_request,
            )
        )
        self.dram_table = self.dram_table_cpu.to(self.device)
        self.hbm_table = self.hbm_table_cpu.to(self.device)

        self.sources = torch.randperm(TOPK, generator=generator).to(torch.int32)
        self.destinations = torch.randperm(TOPK, generator=generator).to(torch.int32)
        topk_sources = torch.full((self.total_queries, 1, TOPK), -1, dtype=torch.int32)
        topk_destinations = torch.full_like(topk_sources, -1)
        for query in (*range(4), *range(5, 9)):
            topk_sources[query, 0].copy_(self.sources)
            topk_destinations[query, 0].copy_(self.destinations)
        self.topk_sources = topk_sources.to(self.device)
        self.topk_destinations = topk_destinations.to(self.device)
        self.topk_miss_counts = torch.zeros(
            self.total_queries, dtype=torch.int32, device=self.device
        )

        self.miss_sources = torch.full(
            (self.batch, 1, COPY_CAPACITY),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        self.miss_destinations = torch.full_like(self.miss_sources, -1)
        self.miss_counts = torch.zeros(
            self.batch, dtype=torch.int32, device=self.device
        )
        self.query_ends = torch.tensor([4, 5, 9], dtype=torch.int32, device=self.device)
        self.kv_lengths = torch.tensor(
            [TOPK, 0, TOPK], dtype=torch.int32, device=self.device
        )
        self.cache_tokens = self.kv_lengths.clone()

        query_cpu = torch.randn(
            self.total_queries, heads, NOPE_DIM + 64, generator=generator
        ).to(torch.bfloat16)
        query_cpu[self.padding_query].zero_()
        self.query = query_cpu.to(self.device)
        self.output = torch.empty(
            self.total_queries,
            heads,
            NOPE_DIM,
            dtype=torch.bfloat16,
            device=self.device,
        )
        self.padding_mask = torch.zeros(
            self.total_queries, 1, 1, dtype=torch.bool, device=self.device
        )
        self.padding_mask[self.padding_query] = True
        self.scale = 1.0 / math.sqrt(NOPE_DIM + 64)

        key, value = unpack_rows(self.logical_rows[self.sources.long()])
        self.expected_attention = torch.zeros(
            self.total_queries, heads, NOPE_DIM, dtype=torch.float32
        )
        for query in (*range(4), *range(5, 9)):
            scores = query_cpu[query].float() @ key.T * self.scale
            self.expected_attention[query] = torch.softmax(scores, dim=-1) @ value

        hbm_blocks = self.batch * self.blocks_per_request
        self.hbm = torch.empty(
            hbm_blocks,
            BLOCK_SIZE,
            1,
            PACKED_DIM,
            dtype=torch.int8,
            device=self.device,
        )
        self.expected_hbm = None

    def prepare(self, miss_count: int) -> None:
        if miss_count not in (512, 513):
            raise ValueError("this case targets the 512/513 device-side gate")
        initial = torch.full(self.hbm.shape, POISON, dtype=torch.int8, device="cpu")
        expected = initial.clone()
        for request in self.live_requests:
            physical = _physical_rows(self.hbm_table_cpu, request, self.destinations)
            initial.view(-1, PACKED_DIM)[physical[miss_count:]] = self.logical_rows[
                self.sources[miss_count:].long()
            ]
            expected.view(-1, PACKED_DIM)[physical] = self.logical_rows[
                self.sources.long()
            ]
        self.hbm.copy_(initial.to(self.device))
        self.expected_hbm = expected

        self.topk_miss_counts.zero_()
        for query in (*range(4), *range(5, 9)):
            self.topk_miss_counts[query] = miss_count
        self.miss_sources.fill_(-1)
        self.miss_destinations.fill_(-1)
        self.miss_counts.zero_()
        for request in self.live_requests:
            self.miss_sources[request, 0, :miss_count].copy_(
                self.sources[:miss_count].to(self.device)
            )
            self.miss_destinations[request, 0, :miss_count].copy_(
                self.destinations[:miss_count].to(self.device)
            )
            self.miss_counts[request] = miss_count
        self.output.fill_(float("nan"))

    def enqueue(self) -> None:
        torch.ops.npu.fused_quant_scatter_copy_sparse_flash_attention(
            self.query,
            self.hbm,
            self.dram_owner,
            self.dram_dev_ptr,
            self.hbm_table,
            self.dram_table,
            self.topk_destinations,
            self.topk_sources,
            self.topk_miss_counts,
            self.query_ends,
            self.kv_lengths,
            self.cache_tokens,
            self.miss_sources,
            self.miss_destinations,
            self.miss_counts,
            self.scale,
            self.output,
        )
        # Empty/padding output is caller-managed, just like tail slot mapping.
        self.output.masked_fill_(self.padding_mask, 0)

    def check(self) -> None:
        torch.testing.assert_close(self.hbm.cpu(), self.expected_hbm, rtol=0, atol=0)
        torch.testing.assert_close(
            self.output.cpu().float(),
            self.expected_attention,
            rtol=0.05,
            atol=0.008,
        )
        torch.testing.assert_close(self.dram_owner, self.dram_before, rtol=0, atol=0)
        padding_pages = self.hbm_table_cpu[self.padding_request].long()
        padding_rows = (
            padding_pages[:, None] * BLOCK_SIZE
            + torch.arange(BLOCK_SIZE, dtype=torch.int64)[None, :]
        ).reshape(-1)
        assert torch.all(self.hbm.cpu().view(-1, PACKED_DIM)[padding_rows] == POISON)

    def close(self) -> None:
        from sgl_kernel_npu.sparsity_driven_kv_offload import free_shm

        try:
            torch.npu.synchronize()
        finally:
            free_shm(0)


def test_q4_eager_switches_between_fused_and_union_copy(require_a5_device) -> None:
    case = _Q4CopySfaCase()
    try:
        for miss_count in (512, 513, 512):
            case.prepare(miss_count)
            case.enqueue()
            torch.npu.synchronize()
            case.check()
    finally:
        case.close()


def test_q4_graph_replay_uses_device_side_512_gate(require_a5_device) -> None:
    if not hasattr(torch.npu, "NPUGraph") or not hasattr(torch.npu, "graph"):
        pytest.skip("torch_npu does not expose NPUGraph capture")
    case = _Q4CopySfaCase()
    try:
        case.prepare(512)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            case.enqueue()
        for miss_count in (512, 513, 512):
            case.prepare(miss_count)
            graph.replay()
            torch.npu.synchronize()
            case.check()
    finally:
        case.close()


def _meta_args(heads: int = 8, batch: int = 2, total_queries: int = 5):
    options = {"device": "meta"}
    int_options = {"device": "meta", "dtype": torch.int32}
    return [
        torch.empty(total_queries, heads, 576, dtype=torch.bfloat16, **options),
        torch.empty(32, 128, 1, 656, dtype=torch.int8, **options),
        torch.empty(32, 128, 1, 656, dtype=torch.int8, **options),
        0,
        torch.empty(batch, 16, **int_options),
        torch.empty(batch, 16, **int_options),
        torch.empty(total_queries, 1, TOPK, **int_options),
        torch.empty(total_queries, 1, TOPK, **int_options),
        torch.empty(total_queries, **int_options),
        torch.empty(batch, **int_options),
        torch.empty(batch, **int_options),
        torch.empty(batch, **int_options),
        torch.empty(batch, 1, COPY_CAPACITY, **int_options),
        torch.empty(batch, 1, COPY_CAPACITY, **int_options),
        torch.empty(batch, **int_options),
        1.0 / math.sqrt(576),
        torch.empty(total_queries, heads, 512, dtype=torch.bfloat16, **options),
    ]


@pytest.mark.parametrize("heads", [8, 16, 32, 64, 128])
def test_meta_accepts_all_supported_head_counts(heads: int) -> None:
    torch.ops.npu.fused_quant_scatter_copy_sparse_flash_attention(
        *_meta_args(heads=heads)
    )


@pytest.mark.parametrize("heads", [4, 256])
def test_meta_rejects_unsupported_head_counts(heads: int) -> None:
    with pytest.raises(RuntimeError, match="H in"):
        torch.ops.npu.fused_quant_scatter_copy_sparse_flash_attention(
            *_meta_args(heads=heads)
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda args: args.__setitem__(0, torch.empty(5, 8, 575, device="meta")),
            "query",
        ),
        (
            lambda args: args.__setitem__(
                6, torch.empty(5, TOPK, dtype=torch.int32, device="meta")
            ),
            "topk indices",
        ),
        (
            lambda args: args.__setitem__(
                8, torch.empty(5, dtype=torch.int64, device="meta")
            ),
            "int32",
        ),
    ],
)
def test_meta_rejects_invalid_metadata(mutate, message: str) -> None:
    args = _meta_args()
    mutate(args)
    with pytest.raises(RuntimeError, match=message):
        torch.ops.npu.fused_quant_scatter_copy_sparse_flash_attention(*args)


def test_meta_rejects_more_than_16_queries_per_request() -> None:
    with pytest.raises(RuntimeError, match="B <= T <= 16B"):
        torch.ops.npu.fused_quant_scatter_copy_sparse_flash_attention(
            *_meta_args(batch=2, total_queries=33)
        )
