import unittest

import sgl_kernel_npu  # noqa: F401
import torch
import torch_npu  # noqa: F401
from sgl_kernel_npu.sparsity_driven_kv_offload import (
    fused_timestamp_lru_metadata_update_with_probation,
    parallel_lru_metadata_write,
    slot_map_lookup,
)


def reference_fused_timestamp_lru_metadata_update_with_probation(
    slot_map,
    req_indices,
    topk_indices,
    device_token_pos,
    device_lru_slots,
    device_lru_slot_stamps,
    device_slot_tokens,
    max_context_len,
    stamp_max=(1 << 24) - 1,
    probation_age=0,
):
    """CPU reference for the fused timestamp-LRU metadata update."""
    slot_map = slot_map.cpu().clone()
    req_indices = req_indices.cpu()
    topk_indices = topk_indices.cpu()
    device_token_pos = device_token_pos.cpu()
    device_lru_slots = device_lru_slots.cpu().clone()
    device_lru_slot_stamps = device_lru_slot_stamps.cpu().clone()
    device_slot_tokens = device_slot_tokens.cpu().clone()
    victim_slots = torch.full_like(topk_indices, -1)

    request_rows, capacity = device_lru_slots.shape
    slot_map_rows = slot_map.size(0)

    for batch_idx, req_id in enumerate(req_indices.tolist()):
        if req_id < 0 or req_id >= request_rows or req_id >= slot_map_rows:
            continue

        tokens = topk_indices[batch_idx]
        positions = device_token_pos[batch_idx]
        valid_miss = (
            (positions == -1)
            & (tokens >= 0)
            & (tokens < max_context_len)
        )
        hit_slots = set(
            positions[(positions >= 0) & (positions < capacity)].tolist()
        )

        # The kernel preserves the existing LRU order within equal-age groups.
        pairs = []
        for slot, stamp in zip(
            device_lru_slots[req_id].tolist(),
            device_lru_slot_stamps[req_id].tolist(),
        ):
            updated_stamp = min(stamp, stamp_max - 1) + 1
            if slot in hit_slots:
                updated_stamp = 0
            pairs.append((slot, updated_stamp))
        pairs.sort(key=lambda pair: pair[1], reverse=True)

        miss_positions = torch.nonzero(valid_miss, as_tuple=False).flatten()
        miss_count = miss_positions.numel()
        for miss_rank, topk_pos in enumerate(miss_positions.tolist()):
            victim = pairs[miss_rank][0]
            new_token = int(tokens[topk_pos].item())
            victim_slots[batch_idx, topk_pos] = victim

            old_token = int(device_slot_tokens[req_id, victim].item())
            if 0 <= old_token < max_context_len:
                slot_map[req_id, old_token] = -1
            device_slot_tokens[req_id, victim] = new_token
            slot_map[req_id, new_token] = victim

        # Victims are the oldest prefix. New fills start at probation_age and
        # are stably reinserted into the descending-age order. probation_age=0
        # reproduces the original MRU insertion behavior.
        victim_pairs = [
            (slot, probation_age) for slot, _ in pairs[:miss_count]
        ]
        rotated_pairs = pairs[miss_count:] + victim_pairs
        rotated_pairs.sort(key=lambda pair: pair[1], reverse=True)
        device_lru_slots[req_id] = torch.tensor(
            [slot for slot, _ in rotated_pairs], dtype=torch.int32
        )
        device_lru_slot_stamps[req_id] = torch.tensor(
            [stamp for _, stamp in rotated_pairs], dtype=torch.int32
        )

    return (
        victim_slots,
        slot_map,
        device_lru_slots,
        device_lru_slot_stamps,
        device_slot_tokens,
    )


class TestFusedTimestampLruMetadataUpdateWithProbation(unittest.TestCase):
    TOPK = 2048
    CAPACITY = 4096
    MAX_CONTEXT_LEN = 8192

    def assert_tensor_equal(self, actual, expected, name):
        actual = actual.cpu()
        expected = expected.cpu()
        mismatch = actual != expected
        if mismatch.any().item():
            indices = torch.nonzero(mismatch, as_tuple=False)[:8]
            details = [
                (
                    tuple(index.tolist()),
                    actual[tuple(index.tolist())].item(),
                    expected[tuple(index.tolist())].item(),
                )
                for index in indices
            ]
            self.fail(
                f"{name} differs from CPU reference; "
                f"first mismatches (index, actual, expected): {details}"
            )

    def setUp(self):
        self.rows = 3
        self.slot_map = torch.full(
            (self.rows + 1, self.MAX_CONTEXT_LEN),
            -1,
            dtype=torch.int32,
            device="npu",
        )
        self.slot_tokens = torch.full(
            (self.rows, self.CAPACITY),
            -1,
            dtype=torch.int32,
            device="npu",
        )
        self.lru_slots = torch.arange(
            self.CAPACITY, dtype=torch.int32, device="npu"
        ).expand(self.rows, self.CAPACITY).clone()
        # Unique non-hit ages make victim order deterministic. Row zero is a
        # valid request row and is left unused by the cases below.
        base_stamps = torch.arange(
            self.CAPACITY - 1, -1, -1, dtype=torch.int32, device="npu"
        )
        self.lru_stamps = base_stamps.expand(self.rows, self.CAPACITY).clone()

        self.slot_tokens[1, 0:3] = torch.tensor(
            [10, 20, 30], dtype=torch.int32, device="npu"
        )
        self.slot_map[1, 10] = 0
        self.slot_map[1, 20] = 1
        self.slot_map[1, 30] = 2

    def _run(self, row1_tokens, row2_tokens=(), stamp_max=(1 << 24) - 1):
        req_indices = torch.tensor(
            [1, 2, self.rows], dtype=torch.int32, device="npu"
        )
        topk = torch.full(
            (3, self.TOPK), -1, dtype=torch.int32, device="npu"
        )
        topk[0, : len(row1_tokens)] = torch.tensor(
            row1_tokens, dtype=torch.int32, device="npu"
        )
        if row2_tokens:
            topk[1, : len(row2_tokens)] = torch.tensor(
                row2_tokens, dtype=torch.int32, device="npu"
            )
        _, device_pos, hit_position_mask = slot_map_lookup(
            self.slot_map,
            req_indices,
            topk,
            pos_mask_size=self.CAPACITY,
        )
        unused_row_slots = self.lru_slots[0].clone()
        unused_row_stamps = self.lru_stamps[0].clone()
        unused_row_tokens = self.slot_tokens[0].clone()

        victims, miss_counts = fused_timestamp_lru_metadata_update_with_probation(
            req_indices,
            topk,
            device_pos,
            hit_position_mask,
            self.lru_slots,
            self.lru_stamps,
            max_context_len=self.MAX_CONTEXT_LEN,
            probation_age=0,
            stamp_max=stamp_max,
        )
        parallel_lru_metadata_write(
            self.slot_map,
            req_indices,
            topk,
            victims,
            miss_counts,
            self.slot_tokens,
            max_context_len=self.MAX_CONTEXT_LEN,
        )
        torch.npu.synchronize()
        return (
            victims.cpu(),
            unused_row_slots.cpu(),
            unused_row_stamps.cpu(),
            unused_row_tokens.cpu(),
        )

    def test_hit_reset_miss_evict_and_invalid_request_skip(self):
        (
            victims,
            unused_row_slots,
            unused_row_stamps,
            unused_row_tokens,
        ) = self._run([10, 40, 41], [50])

        self.assertEqual(victims[0, :4].tolist(), [-1, 1, 2, -1])
        self.assertEqual(victims[1, :2].tolist(), [0, -1])
        # The invalid request row is intentionally undefined. Refill ignores it
        # through its valid mask, so the fused kernel performs no output write.

        self.assertEqual(self.slot_map[1, 10].item(), 0)
        self.assertEqual(self.slot_map[1, 20].item(), -1)
        self.assertEqual(self.slot_map[1, 30].item(), -1)
        self.assertEqual(self.slot_map[1, 40].item(), 1)
        self.assertEqual(self.slot_map[1, 41].item(), 2)
        self.assertEqual(self.slot_tokens[1, :3].cpu().tolist(), [10, 40, 41])

        slots = self.lru_slots[1].cpu()
        stamps = self.lru_stamps[1].cpu()
        self.assertTrue(torch.all(stamps[:-1] >= stamps[1:]).item())
        stamp_by_slot = dict(zip(slots.tolist(), stamps.tolist()))
        self.assertEqual(stamp_by_slot[0], 0)  # hit
        self.assertEqual(stamp_by_slot[1], 0)  # new fill
        self.assertEqual(stamp_by_slot[2], 0)  # new fill
        self.assertEqual(stamp_by_slot[3], self.CAPACITY - 3)

        self.assertTrue(torch.equal(self.lru_slots[0].cpu(), unused_row_slots))
        self.assertTrue(torch.equal(self.lru_stamps[0].cpu(), unused_row_stamps))
        self.assertTrue(torch.equal(self.slot_tokens[0].cpu(), unused_row_tokens))

    def test_duplicate_hits_reset_once_and_are_not_evicted(self):
        victims, *_ = self._run([10, 10, 40])
        self.assertEqual(victims[0, :4].tolist(), [-1, -1, 1, -1])
        slots = self.lru_slots[1].cpu()
        stamps = self.lru_stamps[1].cpu()
        stamp_by_slot = dict(zip(slots.tolist(), stamps.tolist()))
        self.assertEqual(stamp_by_slot[0], 0)
        self.assertEqual(stamp_by_slot[1], 0)

    def test_all_hits_skip_parallel_metadata_writes(self):
        slot_map_before = self.slot_map.clone()
        slot_tokens_before = self.slot_tokens.clone()

        victims, *_ = self._run([10, 20, 30])

        self.assertEqual(victims[0, :4].tolist(), [-1, -1, -1, -1])
        self.assert_tensor_equal(self.slot_map, slot_map_before, "slot_map")
        self.assert_tensor_equal(
            self.slot_tokens,
            slot_tokens_before,
            "device_slot_tokens",
        )

    def test_request_zero_is_valid(self):
        req_indices = torch.tensor([0], dtype=torch.int32, device="npu")
        topk = torch.full(
            (1, self.TOPK), -1, dtype=torch.int32, device="npu"
        )
        topk[0, 0] = 70
        _, device_pos, hit_position_mask = slot_map_lookup(
            self.slot_map,
            req_indices,
            topk,
            pos_mask_size=self.CAPACITY,
        )

        victims, miss_counts = fused_timestamp_lru_metadata_update_with_probation(
            req_indices,
            topk,
            device_pos,
            hit_position_mask,
            self.lru_slots,
            self.lru_stamps,
            max_context_len=self.MAX_CONTEXT_LEN,
            probation_age=0,
        )
        parallel_lru_metadata_write(
            self.slot_map,
            req_indices,
            topk,
            victims,
            miss_counts,
            self.slot_tokens,
            max_context_len=self.MAX_CONTEXT_LEN,
        )
        torch.npu.synchronize()

        self.assertEqual(victims[0, 0].item(), 0)
        self.assertEqual(self.slot_map[0, 70].item(), 0)
        self.assertEqual(self.slot_tokens[0, 0].item(), 70)

    def test_variable_topk_matches_cpu_reference(self):
        topk_width = 1024
        req_indices = torch.tensor([1], dtype=torch.int32, device="npu")
        topk = torch.full(
            (1, topk_width), -1, dtype=torch.int32, device="npu"
        )
        topk[0, :3] = torch.tensor(
            [10, 40, 41], dtype=torch.int32, device="npu"
        )
        device_pos = torch.full_like(topk, -1)
        device_pos[0, 0] = 0
        hit_position_mask = torch.zeros(
            (1, self.CAPACITY), dtype=torch.int32, device="npu"
        )
        hit_position_mask[0, 0] = 1

        slot_map_before = self.slot_map.clone()
        lru_slots_before = self.lru_slots.clone()
        lru_stamps_before = self.lru_stamps.clone()
        slot_tokens_before = self.slot_tokens.clone()
        (
            expected_victims,
            expected_slot_map,
            expected_lru_slots,
            expected_lru_stamps,
            expected_slot_tokens,
        ) = reference_fused_timestamp_lru_metadata_update_with_probation(
            slot_map_before,
            req_indices,
            topk,
            device_pos,
            lru_slots_before,
            lru_stamps_before,
            slot_tokens_before,
            max_context_len=self.MAX_CONTEXT_LEN,
        )

        victims, miss_counts = fused_timestamp_lru_metadata_update_with_probation(
            req_indices,
            topk,
            device_pos,
            hit_position_mask,
            self.lru_slots,
            self.lru_stamps,
            max_context_len=self.MAX_CONTEXT_LEN,
            probation_age=0,
        )
        parallel_lru_metadata_write(
            self.slot_map,
            req_indices,
            topk,
            victims,
            miss_counts,
            self.slot_tokens,
            max_context_len=self.MAX_CONTEXT_LEN,
        )
        torch.npu.synchronize()

        self.assert_tensor_equal(victims, expected_victims, "victim_slots")
        self.assert_tensor_equal(self.slot_map, expected_slot_map, "slot_map")
        self.assert_tensor_equal(
            self.lru_slots, expected_lru_slots, "device_lru_slots"
        )
        self.assert_tensor_equal(
            self.lru_stamps,
            expected_lru_stamps,
            "device_lru_slot_stamps",
        )
        self.assert_tensor_equal(
            self.slot_tokens, expected_slot_tokens, "device_slot_tokens"
        )

    def test_exactly_fifty_percent_hit_rate(self):
        hit_count = self.TOPK // 2
        miss_count = self.TOPK - hit_count

        # Preload 1024 resident tokens. Interleave them with 1024 unique,
        # in-range misses so the effective hit rate is exactly 50%.
        hit_tokens = torch.arange(
            hit_count, dtype=torch.int32, device="npu"
        )
        miss_tokens = torch.arange(
            self.CAPACITY,
            self.CAPACITY + miss_count,
            dtype=torch.int32,
            device="npu",
        )
        self.slot_map[1].fill_(-1)
        self.slot_tokens[1].fill_(-1)
        self.slot_map[1, :hit_count] = hit_tokens
        self.slot_tokens[1, :hit_count] = hit_tokens

        topk = torch.empty(
            (1, self.TOPK), dtype=torch.int32, device="npu"
        )
        topk[0, 0::2] = hit_tokens
        topk[0, 1::2] = miss_tokens
        req_indices = torch.tensor([1], dtype=torch.int32, device="npu")
        _, device_pos, hit_position_mask = slot_map_lookup(
            self.slot_map,
            req_indices,
            topk,
            pos_mask_size=self.CAPACITY,
        )

        self.assertEqual(
            int((device_pos >= 0).sum().item()),
            hit_count,
            "the constructed workload must have exactly 50% cache hits",
        )

        slot_map_before = self.slot_map.clone()
        lru_slots_before = self.lru_slots.clone()
        lru_stamps_before = self.lru_stamps.clone()
        slot_tokens_before = self.slot_tokens.clone()

        (
            expected_victims,
            expected_slot_map,
            expected_lru_slots,
            expected_lru_stamps,
            expected_slot_tokens,
        ) = reference_fused_timestamp_lru_metadata_update_with_probation(
            slot_map_before,
            req_indices,
            topk,
            device_pos,
            lru_slots_before,
            lru_stamps_before,
            slot_tokens_before,
            max_context_len=self.MAX_CONTEXT_LEN,
        )

        victims, miss_counts = fused_timestamp_lru_metadata_update_with_probation(
            req_indices,
            topk,
            device_pos,
            hit_position_mask,
            self.lru_slots,
            self.lru_stamps,
            max_context_len=self.MAX_CONTEXT_LEN,
            probation_age=0,
        )
        parallel_lru_metadata_write(
            self.slot_map,
            req_indices,
            topk,
            victims,
            miss_counts,
            self.slot_tokens,
            max_context_len=self.MAX_CONTEXT_LEN,
        )
        torch.npu.synchronize()

        self.assert_tensor_equal(victims, expected_victims, "victim_slots")
        self.assert_tensor_equal(self.slot_map, expected_slot_map, "slot_map")
        self.assert_tensor_equal(
            self.lru_slots, expected_lru_slots, "device_lru_slots"
        )
        self.assert_tensor_equal(
            self.lru_stamps,
            expected_lru_stamps,
            "device_lru_slot_stamps",
        )
        self.assert_tensor_equal(
            self.slot_tokens, expected_slot_tokens, "device_slot_tokens"
        )

    def test_grid_stride_writeback_matches_cpu_reference(self):
        # Force one AIV to process two requests so the second request reuses
        # the same UB only after the first request's MTE3 writeback completes.
        self.slot_tokens[2, 0] = 50
        self.slot_map[2, 50] = 0
        req_indices = torch.tensor([1, 2], dtype=torch.int32, device="npu")
        topk = torch.full(
            (2, self.TOPK), -1, dtype=torch.int32, device="npu"
        )
        topk[0, :3] = torch.tensor(
            [10, 40, 41], dtype=torch.int32, device="npu"
        )
        topk[1, :2] = torch.tensor(
            [50, 60], dtype=torch.int32, device="npu"
        )
        _, device_pos, hit_position_mask = slot_map_lookup(
            self.slot_map,
            req_indices,
            topk,
            pos_mask_size=self.CAPACITY,
        )

        slot_map_before = self.slot_map.clone()
        lru_slots_before = self.lru_slots.clone()
        lru_stamps_before = self.lru_stamps.clone()
        slot_tokens_before = self.slot_tokens.clone()
        (
            expected_victims,
            expected_slot_map,
            expected_lru_slots,
            expected_lru_stamps,
            expected_slot_tokens,
        ) = reference_fused_timestamp_lru_metadata_update_with_probation(
            slot_map_before,
            req_indices,
            topk,
            device_pos,
            lru_slots_before,
            lru_stamps_before,
            slot_tokens_before,
            max_context_len=self.MAX_CONTEXT_LEN,
        )

        victims, miss_counts = fused_timestamp_lru_metadata_update_with_probation(
            req_indices,
            topk,
            device_pos,
            hit_position_mask,
            self.lru_slots,
            self.lru_stamps,
            max_context_len=self.MAX_CONTEXT_LEN,
            probation_age=0,
            block_dim=1,
        )
        parallel_lru_metadata_write(
            self.slot_map,
            req_indices,
            topk,
            victims,
            miss_counts,
            self.slot_tokens,
            max_context_len=self.MAX_CONTEXT_LEN,
        )
        torch.npu.synchronize()

        self.assert_tensor_equal(victims, expected_victims, "victim_slots")
        self.assert_tensor_equal(self.slot_map, expected_slot_map, "slot_map")
        self.assert_tensor_equal(
            self.lru_slots, expected_lru_slots, "device_lru_slots"
        )
        self.assert_tensor_equal(
            self.lru_stamps,
            expected_lru_stamps,
            "device_lru_slot_stamps",
        )
        self.assert_tensor_equal(
            self.slot_tokens, expected_slot_tokens, "device_slot_tokens"
        )

    def test_capacity_8192_matches_cpu_reference(self):
        capacity = 8192
        req_indices = torch.tensor([0], dtype=torch.int32, device="npu")
        slot_map = torch.full(
            (2, self.MAX_CONTEXT_LEN), -1, dtype=torch.int32, device="npu"
        )
        slot_tokens = torch.full(
            (1, capacity), -1, dtype=torch.int32, device="npu"
        )
        lru_slots = torch.cat(
            (
                torch.arange(7001, capacity, dtype=torch.int32, device="npu"),
                torch.arange(0, 7001, dtype=torch.int32, device="npu"),
            )
        ).unsqueeze(0)
        lru_stamps = torch.arange(
            capacity - 1, -1, -1, dtype=torch.int32, device="npu"
        ).unsqueeze(0)

        slot_tokens[0, 7000] = 10
        slot_tokens[0, 7001] = 20
        slot_map[0, 10] = 7000
        slot_map[0, 20] = 7001
        topk = torch.full(
            (1, self.TOPK), -1, dtype=torch.int32, device="npu"
        )
        topk[0, :3] = torch.tensor(
            [10, 6000, 6001], dtype=torch.int32, device="npu"
        )
        _, device_pos, hit_position_mask = slot_map_lookup(
            slot_map, req_indices, topk, pos_mask_size=capacity
        )

        (
            expected_victims,
            expected_slot_map,
            expected_lru_slots,
            expected_lru_stamps,
            expected_slot_tokens,
        ) = reference_fused_timestamp_lru_metadata_update_with_probation(
            slot_map,
            req_indices,
            topk,
            device_pos,
            lru_slots,
            lru_stamps,
            slot_tokens,
            max_context_len=self.MAX_CONTEXT_LEN,
            probation_age=1024,
        )

        victims, miss_counts = fused_timestamp_lru_metadata_update_with_probation(
            req_indices,
            topk,
            device_pos,
            hit_position_mask,
            lru_slots,
            lru_stamps,
            max_context_len=self.MAX_CONTEXT_LEN,
            probation_age=1024,
        )
        parallel_lru_metadata_write(
            slot_map,
            req_indices,
            topk,
            victims,
            miss_counts,
            slot_tokens,
            max_context_len=self.MAX_CONTEXT_LEN,
        )
        torch.npu.synchronize()

        self.assert_tensor_equal(victims, expected_victims, "victim_slots")
        self.assert_tensor_equal(slot_map, expected_slot_map, "slot_map")
        self.assert_tensor_equal(lru_slots, expected_lru_slots, "device_lru_slots")
        self.assert_tensor_equal(
            lru_stamps, expected_lru_stamps, "device_lru_slot_stamps"
        )
        self.assert_tensor_equal(
            slot_tokens, expected_slot_tokens, "device_slot_tokens"
        )

    def test_stamp_saturates(self):
        self.lru_stamps.fill_(7)
        victims, *_ = self._run([10, 40], stamp_max=7)
        self.assertEqual(victims[0, 0].item(), -1)
        self.assertEqual(victims[0, 1].item(), 1)
        slots = self.lru_slots[1].cpu()
        stamps = self.lru_stamps[1].cpu()
        stamp_by_slot = dict(zip(slots.tolist(), stamps.tolist()))
        self.assertEqual(stamp_by_slot[0], 0)
        self.assertEqual(stamp_by_slot[1], 0)
        self.assertTrue(all(stamp_by_slot[i] == 7 for i in range(2, 32)))

    def test_probation_age_keeps_new_miss_older_than_hit(self):
        probation_age = 1024
        req_indices = torch.tensor([1], dtype=torch.int32, device="npu")
        topk = torch.full(
            (1, self.TOPK), -1, dtype=torch.int32, device="npu"
        )
        topk[0, :2] = torch.tensor([10, 40], dtype=torch.int32, device="npu")
        _, device_pos, hit_position_mask = slot_map_lookup(
            self.slot_map,
            req_indices,
            topk,
            pos_mask_size=self.CAPACITY,
        )

        slot_map_before = self.slot_map.clone()
        lru_slots_before = self.lru_slots.clone()
        lru_stamps_before = self.lru_stamps.clone()
        slot_tokens_before = self.slot_tokens.clone()
        (
            expected_victims,
            expected_slot_map,
            expected_lru_slots,
            expected_lru_stamps,
            expected_slot_tokens,
        ) = reference_fused_timestamp_lru_metadata_update_with_probation(
            slot_map_before,
            req_indices,
            topk,
            device_pos,
            lru_slots_before,
            lru_stamps_before,
            slot_tokens_before,
            max_context_len=self.MAX_CONTEXT_LEN,
            probation_age=probation_age,
        )

        victims, miss_counts = fused_timestamp_lru_metadata_update_with_probation(
            req_indices,
            topk,
            device_pos,
            hit_position_mask,
            self.lru_slots,
            self.lru_stamps,
            max_context_len=self.MAX_CONTEXT_LEN,
            probation_age=probation_age,
        )
        parallel_lru_metadata_write(
            self.slot_map,
            req_indices,
            topk,
            victims,
            miss_counts,
            self.slot_tokens,
            max_context_len=self.MAX_CONTEXT_LEN,
        )
        torch.npu.synchronize()

        self.assert_tensor_equal(victims, expected_victims, "victim_slots")
        self.assert_tensor_equal(self.slot_map, expected_slot_map, "slot_map")
        self.assert_tensor_equal(
            self.lru_slots, expected_lru_slots, "device_lru_slots"
        )
        self.assert_tensor_equal(
            self.lru_stamps,
            expected_lru_stamps,
            "device_lru_slot_stamps",
        )
        self.assert_tensor_equal(
            self.slot_tokens, expected_slot_tokens, "device_slot_tokens"
        )

        slots = self.lru_slots[1].cpu().tolist()
        stamps = self.lru_stamps[1].cpu().tolist()
        stamp_by_slot = dict(zip(slots, stamps))
        self.assertEqual(stamp_by_slot[0], 0)  # hit remains MRU
        self.assertEqual(stamp_by_slot[1], probation_age)  # miss is probationary
        self.assertLess(slots.index(1), slots.index(0))


if __name__ == "__main__":
    unittest.main()
