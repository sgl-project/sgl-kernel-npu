"""Native planner + combined-layout Copy-SFA integration coverage.

These tests need a built A3 sgl-kernel-npu extension and a visible NPU. They
deliberately import the extension only after the NPU availability check so CPU
collection and the pure-Python planner tests do not initialize torch_npu.
"""

from __future__ import annotations

import importlib
import math

import pytest
import torch


def _require_npu():
    try:
        importlib.import_module("torch_npu")
    except ImportError:
        pytest.skip("torch_npu is not installed")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("NPU is required for native Copy-SFA integration tests")
    package = importlib.import_module("sgl_kernel_npu")  # loads native operators
    offload_ops = importlib.import_module("sgl_kernel_npu.sparsity_driven_kv_offload")
    return package, offload_ops


def _page_permutation(page_count: int, multiplier: int) -> list[int]:
    return [(page * multiplier) % page_count for page in range(page_count)]


class _CopySfaCase:
    def __init__(
        self,
        real_request_count: int,
        batch_req_ids: list[int],
        dtype,
        query_heads: int = 8,
    ):
        _, offload_ops = _require_npu()
        self.create_shm_tensor = offload_ops.create_shm_tensor
        self.free_shm = offload_ops.free_shm
        self.device_id = torch.npu.current_device()
        self.device = torch.device("npu", self.device_id)
        self.real_request_count = real_request_count
        self.batch_req_ids = batch_req_ids
        self.row_for_request = {req_id: row for row, req_id in enumerate(batch_req_ids)}
        self.batch = len(batch_req_ids)
        self.max_context_len = 8192
        self.capacity = 2048
        self.page_size = 128
        self.dram_pages = self.max_context_len // self.page_size
        self.hbm_pages_per_request = self.capacity // self.page_size
        self.dtype = dtype
        self.query_heads = query_heads
        self.scale = 1.0 / math.sqrt(512 + 64)

        self.logical_kv = torch.randn(
            (real_request_count, self.max_context_len, 576), dtype=torch.float32
        ).to(dtype)
        self.dram, self.dram_host_ptr, self.dram_dev_ptr = self.create_shm_tensor(
            (real_request_count * self.max_context_len, 576),
            dtype,
            device_id=self.device_id,
            name="copy_sfa_combined_dram",
        )
        self._populate_dram_with_page_permutation()

        # Include one reserved group for req_id == real_request_count so the
        # invalid interior batch row has a canary page-table target.
        self.hbm = torch.full(
            (
                (real_request_count + 1) * self.hbm_pages_per_request,
                self.page_size,
                1,
                576,
            ),
            19.0,
            dtype=dtype,
            device=self.device,
        )
        self.state_source_to_slot = torch.full(
            (real_request_count + 1, self.max_context_len),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        self.state_slot_to_source = torch.full(
            (real_request_count + 1, self.capacity),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        self.req_ids = torch.tensor(
            batch_req_ids, dtype=torch.int32, device=self.device
        )
        self.topk_input = torch.zeros(
            (self.batch, self.capacity), dtype=torch.int32, device=self.device
        )
        self.topk_src = torch.empty(
            (self.batch, 1, self.capacity), dtype=torch.int32, device=self.device
        )
        self.topk_dst = torch.empty_like(self.topk_src)
        self.topk_miss_counts = torch.empty(
            (self.batch,), dtype=torch.int32, device=self.device
        )
        self.miss_src = torch.empty(
            (self.batch, self.capacity), dtype=torch.int32, device=self.device
        )
        self.miss_dst = torch.empty_like(self.miss_src)
        self.miss_counts = torch.empty(
            (self.batch,), dtype=torch.int32, device=self.device
        )
        self.hbm_block_table = torch.empty(
            (self.batch, self.hbm_pages_per_request),
            dtype=torch.int32,
            device=self.device,
        )
        self.dram_block_table = torch.empty(
            (self.batch, self.dram_pages), dtype=torch.int32, device=self.device
        )
        self._fill_block_tables()

        self.query = (
            torch.randn((self.batch, self.query_heads, 512), dtype=torch.float32)
            .to(dtype)
            .to(self.device)
        )
        self.query_rope = (
            torch.randn((self.batch, self.query_heads, 64), dtype=torch.float32)
            .to(dtype)
            .to(self.device)
        )
        self.actual_q = torch.arange(
            1, self.batch + 1, dtype=torch.int32, device=self.device
        )
        self.actual_kv = torch.tensor(
            [
                0 if req == real_request_count else self.capacity
                for req in batch_req_ids
            ],
            dtype=torch.int32,
            device=self.device,
        )
        self.cache_tokens = self.actual_kv.clone()
        self.attention_out = torch.empty_like(self.query)

    def _populate_dram_with_page_permutation(self):
        # Physical host pages are intentionally permuted. The operator receives
        # only the device-mapped base pointer and resolves logical page IDs via
        # the per-request DRAM block table.
        physical = self.dram.view(
            self.real_request_count, self.dram_pages, self.page_size, 576
        )
        permutation = _page_permutation(self.dram_pages, 17)
        for req_id in range(self.real_request_count):
            for logical_page, physical_page in enumerate(permutation):
                start = logical_page * self.page_size
                physical[req_id, physical_page].copy_(
                    self.logical_kv[req_id, start : start + self.page_size]
                )

    def _fill_block_tables(self):
        hbm_permutation = _page_permutation(self.hbm_pages_per_request, 7)
        dram_permutation = _page_permutation(self.dram_pages, 17)
        for row, req_id in enumerate(self.batch_req_ids):
            self.hbm_block_table[row].copy_(
                torch.tensor(
                    [req_id * self.hbm_pages_per_request + p for p in hbm_permutation],
                    dtype=torch.int32,
                    device=self.device,
                )
            )
            if req_id < self.real_request_count:
                dram_pages = [req_id * self.dram_pages + p for p in dram_permutation]
            else:
                # This row must remain unread: it has zero KV/query work and
                # zero misses. Point it at an in-range real host request anyway.
                dram_pages = [p for p in dram_permutation]
            self.dram_block_table[row].copy_(
                torch.tensor(dram_pages, dtype=torch.int32, device=self.device)
            )

    def set_sources(self, row: int, sources: list[int]):
        assert len(sources) == self.capacity
        self.topk_input[row].copy_(
            torch.tensor(sources, dtype=torch.int32, device=self.device)
        )

    def set_request_sources(self, req_id: int, sources: list[int]):
        self.set_sources(self.row_for_request[req_id], sources)

    def _plan(self):
        torch.ops.npu.plan_sparse_kv_cache(
            self.state_source_to_slot,
            self.state_slot_to_source,
            self.req_ids,
            self.topk_input,
            self.topk_src,
            self.topk_dst,
            self.topk_miss_counts,
            self.miss_src,
            self.miss_dst,
            self.miss_counts,
            self.max_context_len,
            self.capacity,
        )

    def _copy_sfa(self):
        torch.ops.npu.fused_scatter_copy_sparse_flash_attention(
            self.query,
            self.query_rope,
            self.hbm,
            self.actual_q,
            self.actual_kv,
            self.cache_tokens,
            self.topk_src,
            self.topk_dst,
            self.topk_miss_counts,
            self.miss_src,
            self.miss_dst,
            self.miss_counts,
            self.hbm_block_table,
            self.dram_block_table,
            self.dram_dev_ptr,
            self.attention_out,
            self.scale,
        )

    def poison_outputs(self):
        self.attention_out.fill_(float("nan"))
        self.topk_src.fill_(-99)
        self.topk_dst.fill_(-99)
        self.topk_miss_counts.fill_(-99)
        self.miss_src.fill_(-99)
        self.miss_dst.fill_(-99)
        self.miss_counts.fill_(-99)
        torch.npu.synchronize(self.device_id)

    def enqueue(self):
        self._plan()
        self._copy_sfa()

    def run(self):
        self.poison_outputs()
        self.enqueue()
        torch.npu.synchronize(self.device_id)

    def assert_counts(self, expected_by_request: dict[int, int]):
        counts = self.miss_counts.cpu().tolist()
        for row, req_id in enumerate(self.batch_req_ids):
            assert counts[row] == expected_by_request.get(req_id, 0)
            assert self.topk_miss_counts[row].item() == counts[row]

    def assert_planned_request(
        self, req_id: int, selected: list[int], misses: set[int]
    ):
        row = self.row_for_request[req_id]
        active_sources = self.topk_src[row, 0].cpu().tolist()
        active_slots = self.topk_dst[row, 0].cpu().tolist()
        miss_count = self.miss_counts[row].item()
        miss_sources = self.miss_src[row, :miss_count].cpu().tolist()
        miss_slots = self.miss_dst[row, :miss_count].cpu().tolist()
        assert set(active_sources) == set(selected)
        assert all(0 <= slot < self.capacity for slot in active_slots)
        assert set(miss_sources) == misses
        assert len(miss_slots) == len(set(miss_slots))
        assert all(0 <= slot < self.capacity for slot in miss_slots)
        source_map = self.state_source_to_slot[req_id]
        for source_id, slot in zip(active_sources, active_slots):
            assert source_map[source_id].item() == slot

    def assert_attention(self, sources_by_request: dict[int, list[int]]):
        output = self.attention_out.cpu().float()
        for row, req_id in enumerate(self.batch_req_ids):
            if req_id == self.real_request_count:
                assert torch.equal(output[row], torch.zeros_like(output[row]))
                continue
            selected = self.logical_kv[req_id, sources_by_request[req_id]].float()
            key = selected[:, :512]
            key_rope = selected[:, 512:]
            scores = (
                self.query[row].cpu().float() @ key.T
                + self.query_rope[row].cpu().float() @ key_rope.T
            ) * self.scale
            expected = torch.softmax(scores, dim=-1) @ key
            torch.testing.assert_close(output[row], expected, atol=0.002, rtol=0.02)

    def assert_state_bijection_and_payload(self):
        source_map = self.state_source_to_slot.cpu()
        inverse = self.state_slot_to_source.cpu()
        hbm = self.hbm.cpu().view(-1, 128, 576)
        hbm_permutation = _page_permutation(self.hbm_pages_per_request, 7)
        for req_id in range(self.real_request_count):
            seen_slots = set()
            for source_id, slot in enumerate(source_map[req_id].tolist()):
                if slot < 0:
                    continue
                assert slot < self.capacity
                assert slot not in seen_slots
                seen_slots.add(slot)
                assert inverse[req_id, slot].item() == source_id
                logical_page, page_offset = divmod(slot, self.page_size)
                physical_page = hbm_permutation[logical_page]
                physical_row = req_id * self.hbm_pages_per_request + physical_page
                actual = hbm[physical_row, page_offset]
                expected = self.logical_kv[req_id, source_id]
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            for slot, source_id in enumerate(inverse[req_id].tolist()):
                if source_id >= 0:
                    assert source_map[req_id, source_id].item() == slot

    def assert_invalid_row_untouched(self, source_before, inverse_before):
        sentinel = self.real_request_count
        assert torch.equal(self.state_source_to_slot[sentinel], source_before)
        assert torch.equal(self.state_slot_to_source[sentinel], inverse_before)
        reserved = self.hbm[
            sentinel
            * self.hbm_pages_per_request : (sentinel + 1)
            * self.hbm_pages_per_request
        ]
        assert torch.all(reserved == 19.0)

    def assert_invalid_outputs(self):
        row = self.row_for_request[self.real_request_count]
        assert torch.all(self.topk_src[row] == -1)
        assert torch.all(self.topk_dst[row] == -1)
        assert torch.all(self.miss_src[row] == -1)
        assert torch.all(self.miss_dst[row] == -1)
        assert self.topk_miss_counts[row].item() == 0
        assert self.miss_counts[row].item() == 0

    def close(self):
        try:
            torch.npu.synchronize(self.device_id)
        finally:
            self.free_shm(self.device_id)


def _choose_next_sources(current: list[int], miss_count: int, next_source: int):
    if miss_count == 2048:
        return list(range(next_source, next_source + 2048)), next_source + 2048
    retained = current[: 2048 - miss_count]
    fresh = list(range(next_source, next_source + miss_count))
    return retained + fresh, next_source + miss_count


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_native_combined_copy_sfa_planner_eager_replacements(dtype):
    # Reverse the two live request rows around the invalid interior row to
    # exercise request ID/table-row permutations as well as page permutations.
    case = _CopySfaCase(real_request_count=2, batch_req_ids=[1, 2, 0], dtype=dtype)
    try:
        sentinel = case.real_request_count
        sentinel_source_before = case.state_source_to_slot[sentinel].clone()
        sentinel_inverse_before = case.state_slot_to_source[sentinel].clone()
        current = list(range(case.capacity))
        fixed_second_request = list(range(case.capacity))

        # Fill both live rows on the first call; the invalid middle row has a
        # reserved page-table row but no state/cache accesses are permitted.
        case.set_request_sources(0, current)
        case.set_request_sources(1, fixed_second_request)
        case.run()
        case.assert_counts({0: 2048, 1: 2048})
        case.assert_planned_request(0, current, set(current))
        case.assert_planned_request(1, fixed_second_request, set(fixed_second_request))
        case.assert_attention({0: current, 1: fixed_second_request})
        case.assert_state_bijection_and_payload()
        case.assert_invalid_row_untouched(
            sentinel_source_before, sentinel_inverse_before
        )
        case.assert_invalid_outputs()

        # Exercise a zero-miss hit-only replay, then true miss counts in the
        # requested range and full replacement. TopK is permuted each time;
        # assertions deliberately check set semantics rather than a victim
        # policy's incidental ordering.
        import random

        randomizer = random.Random(93)
        sequence = [0, 128, 512, 1536, 2048]
        next_source = 2048
        for miss_count in sequence:
            new_sources, next_source = _choose_next_sources(
                current, miss_count, next_source
            )
            randomizer.shuffle(new_sources)
            case.set_request_sources(0, new_sources)
            case.set_request_sources(1, fixed_second_request)
            previous_slots = case.state_source_to_slot[0].cpu().clone()
            case.run()
            case.assert_counts({0: miss_count, 1: 0})
            case.assert_planned_request(
                0, new_sources, set(new_sources).difference(current)
            )
            case.assert_planned_request(1, fixed_second_request, set())
            case.assert_attention({0: new_sources, 1: fixed_second_request})
            case.assert_state_bijection_and_payload()
            current_slots = case.state_source_to_slot[0].cpu()
            for source_id in set(current).intersection(new_sources):
                assert (
                    current_slots[source_id].item() == previous_slots[source_id].item()
                )
            case.assert_invalid_row_untouched(
                sentinel_source_before, sentinel_inverse_before
            )
            case.assert_invalid_outputs()
            current = new_sources

        # Simulate reusing request 0 after manager reset invalidates both maps.
        case.state_source_to_slot[0].fill_(-1)
        case.state_slot_to_source[0].fill_(-1)
        reused = list(range(4096, 6144))
        randomizer.shuffle(reused)
        case.set_request_sources(0, reused)
        case.set_request_sources(1, fixed_second_request)
        case.run()
        case.assert_counts({0: 2048, 1: 0})
        case.assert_planned_request(0, reused, set(reused))
        case.assert_planned_request(1, fixed_second_request, set())
        case.assert_attention({0: reused, 1: fixed_second_request})
        case.assert_state_bijection_and_payload()
    finally:
        case.close()


def test_native_combined_copy_sfa_single_request_first_fill_and_all_hits():
    case = _CopySfaCase(real_request_count=1, batch_req_ids=[0], dtype=torch.bfloat16)
    try:
        sources = list(range(case.capacity))
        case.set_sources(0, sources)
        case.run()
        case.assert_counts({0: case.capacity})
        case.assert_planned_request(0, sources, set(sources))
        case.assert_attention({0: sources})
        case.assert_state_bijection_and_payload()

        # The cache is already populated: this second invocation must leave it
        # intact while returning the same golden attention result.
        before = case.hbm.clone()
        case.run()
        case.assert_counts({0: 0})
        case.assert_planned_request(0, sources, set())
        case.assert_attention({0: sources})
        assert torch.equal(case.hbm, before)
    finally:
        case.close()


def test_native_combined_copy_sfa_npu_graph_replays_changed_topk():
    _require_npu()
    if not hasattr(torch.npu, "NPUGraph"):
        pytest.skip("torch_npu does not expose NPUGraph")
    if not hasattr(torch.npu, "graph"):
        pytest.skip("torch_npu does not expose the NPUGraph capture context")
    case = _CopySfaCase(
        real_request_count=2, batch_req_ids=[1, 2, 0], dtype=torch.bfloat16
    )
    try:
        sentinel = case.real_request_count
        sentinel_source_before = case.state_source_to_slot[sentinel].clone()
        sentinel_inverse_before = case.state_slot_to_source[sentinel].clone()
        initial = list(range(case.capacity))
        case.set_request_sources(0, initial)
        case.set_request_sources(1, initial)

        # Eagerly establish cache and planner state so graph capture behavior
        # cannot determine whether the first replay performs a cache fill.
        case.run()
        case.assert_counts({0: 2048, 1: 2048})
        case.assert_planned_request(0, initial, set(initial))
        case.assert_planned_request(1, initial, set(initial))
        case.assert_attention({0: initial, 1: initial})
        case.assert_state_bijection_and_payload()

        case.poison_outputs()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            case.enqueue()
        case.poison_outputs()
        graph.replay()
        torch.npu.synchronize(case.device_id)
        case.assert_counts({0: 0, 1: 0})
        case.assert_planned_request(0, initial, set())
        case.assert_planned_request(1, initial, set())
        case.assert_attention({0: initial, 1: initial})
        case.assert_state_bijection_and_payload()
        case.assert_invalid_row_untouched(
            sentinel_source_before, sentinel_inverse_before
        )
        case.assert_invalid_outputs()

        # Change A by 512 true misses while B remains all-hit.
        previous_slots_a = case.state_source_to_slot[0].cpu().clone()
        updated_a, next_source = _choose_next_sources(initial, 512, 2048)
        case.set_request_sources(0, updated_a)
        case.set_request_sources(1, initial)
        case.poison_outputs()
        graph.replay()
        torch.npu.synchronize(case.device_id)
        case.assert_counts({0: 512, 1: 0})
        case.assert_planned_request(0, updated_a, set(updated_a).difference(initial))
        case.assert_planned_request(1, initial, set())
        case.assert_attention({0: updated_a, 1: initial})
        case.assert_state_bijection_and_payload()
        current_slots_a = case.state_source_to_slot[0].cpu()
        for source_id in set(initial).intersection(updated_a):
            assert (
                current_slots_a[source_id].item() == previous_slots_a[source_id].item()
            )

        # Change B by 512 misses on the next replay. Validate both outputs and
        # A's unchanged slot ownership after B's request row is updated.
        previous_slots_a = case.state_source_to_slot[0].cpu().clone()
        previous_slots_b = case.state_source_to_slot[1].cpu().clone()
        updated_b, _ = _choose_next_sources(initial, 512, next_source)
        case.set_request_sources(0, updated_a)
        case.set_request_sources(1, updated_b)
        case.poison_outputs()
        graph.replay()
        torch.npu.synchronize(case.device_id)
        case.assert_counts({0: 0, 1: 512})
        case.assert_planned_request(0, updated_a, set())
        case.assert_planned_request(1, updated_b, set(updated_b).difference(initial))
        case.assert_attention({0: updated_a, 1: updated_b})
        case.assert_state_bijection_and_payload()
        case.assert_invalid_row_untouched(
            sentinel_source_before, sentinel_inverse_before
        )
        case.assert_invalid_outputs()
        current_slots_a = case.state_source_to_slot[0].cpu()
        for source_id in updated_a:
            assert (
                current_slots_a[source_id].item() == previous_slots_a[source_id].item()
            )
        current_slots_b = case.state_source_to_slot[1].cpu()
        for source_id in set(initial).intersection(updated_b):
            assert (
                current_slots_b[source_id].item() == previous_slots_b[source_id].item()
            )
    finally:
        case.close()


@pytest.mark.parametrize("query_heads", [2, 4])
def test_native_combined_copy_sfa_small_head_boundaries(query_heads):
    case = _CopySfaCase(
        real_request_count=1,
        batch_req_ids=[0],
        dtype=torch.bfloat16,
        query_heads=query_heads,
    )
    try:
        sources = list(range(case.capacity))
        case.set_sources(0, sources)
        case.run()
        case.assert_counts({0: case.capacity})
        case.assert_planned_request(0, sources, set(sources))
        case.assert_attention({0: sources})
        case.assert_state_bijection_and_payload()

        # Exercise the steady-state hit path before changing TopK. This makes
        # the H2/H4 boundary cases cover more than their first-fill behavior.
        cache_after_fill = case.hbm.clone()
        case.run()
        case.assert_counts({0: 0})
        case.assert_planned_request(0, sources, set())
        case.assert_attention({0: sources})
        case.assert_state_bijection_and_payload()
        assert torch.equal(case.hbm, cache_after_fill)

        # Replace 512 entries and verify only the retained source set's slot
        # ownership. The planner may choose any legal victim/free-slot order.
        previous_slots = case.state_source_to_slot[0].cpu().clone()
        updated_sources, _ = _choose_next_sources(sources, 512, case.capacity)
        case.set_sources(0, updated_sources)
        case.run()
        case.assert_counts({0: 512})
        case.assert_planned_request(
            0, updated_sources, set(updated_sources).difference(sources)
        )
        case.assert_attention({0: updated_sources})
        case.assert_state_bijection_and_payload()
        current_slots = case.state_source_to_slot[0].cpu()
        for source_id in set(sources).intersection(updated_sources):
            assert current_slots[source_id].item() == previous_slots[source_id].item()
    finally:
        case.close()
