import unittest

import sgl_kernel_npu  # noqa: F401
import torch
import torch_npu  # noqa: F401
from sgl_kernel_npu.sparsity_driven_kv_offload import slot_map_lookup


def reference_lookup(slot_map, req_indices, topk_indices):
    slot_map_cpu = slot_map.cpu()
    req_indices_cpu = req_indices.cpu()
    topk_indices_cpu = topk_indices.cpu()
    token_on_device = torch.zeros_like(topk_indices_cpu, dtype=torch.bool)
    device_token_pos = torch.full_like(topk_indices_cpu, -1, dtype=torch.int32)

    for batch_idx, req_id in enumerate(req_indices_cpu.tolist()):
        if req_id < 0 or req_id >= slot_map_cpu.size(0):
            continue
        for topk_idx, token_idx in enumerate(topk_indices_cpu[batch_idx].tolist()):
            if token_idx < 0 or token_idx >= slot_map_cpu.size(1):
                continue
            slot = slot_map_cpu[req_id, token_idx]
            device_token_pos[batch_idx, topk_idx] = slot
            token_on_device[batch_idx, topk_idx] = slot >= 0

    return token_on_device, device_token_pos


class TestSlotMapLookup(unittest.TestCase):
    BLOCK_DIMS = (0, 1, 2, 8, 16, 24, 32, 48)

    def setUp(self):
        self.slot_map = torch.full((3, 64), -1, dtype=torch.int32, device="npu")
        self.slot_map[0, 0] = 10
        self.slot_map[0, 63] = 11
        self.slot_map[1, 7] = 20

    def _assert_lookup(self, topk_indices):
        req_indices = torch.tensor([0, 1, -1], dtype=torch.int32, device="npu")
        expected_token, expected_pos = reference_lookup(
            self.slot_map, req_indices, topk_indices
        )

        for block_dim in self.BLOCK_DIMS:
            with self.subTest(block_dim=block_dim):
                actual_token, actual_pos = slot_map_lookup(
                    self.slot_map,
                    req_indices,
                    topk_indices,
                    block_dim=block_dim,
                )
                torch.npu.synchronize()
                self.assertTrue(torch.equal(actual_token.cpu().bool(), expected_token))
                self.assertTrue(torch.equal(actual_pos.cpu(), expected_pos))

    def test_lookup_with_invalid_indices(self):
        topk_indices = torch.zeros((3, 2048), dtype=torch.int32, device="npu")
        topk_indices[0, 0] = 63
        topk_indices[0, 1] = -1
        topk_indices[0, 2] = 64
        topk_indices[1].fill_(7)
        self._assert_lookup(topk_indices)

    def test_rejects_non_fixed_topk(self):
        req_indices = torch.tensor([0], dtype=torch.int32, device="npu")
        topk_indices = torch.zeros((1, 64), dtype=torch.int32, device="npu")
        with self.assertRaisesRegex(RuntimeError, "requires topk=2048"):
            slot_map_lookup(self.slot_map, req_indices, topk_indices)

    def test_rejects_wrong_slot_map_dtype(self):
        req_indices = torch.tensor([0], dtype=torch.int32, device="npu")
        topk_indices = torch.tensor([[0]], dtype=torch.int32, device="npu")
        with self.assertRaisesRegex(RuntimeError, "slot_map must be int32"):
            slot_map_lookup(self.slot_map.to(torch.int64), req_indices, topk_indices)

    def test_position_mask_supports_variable_size_and_duplicate_hits(self):
        slot_map = torch.full((3, 64), -1, dtype=torch.int32, device="npu")
        slot_map[0, 0] = 4094
        slot_map[0, 63] = 4095
        slot_map[1, 7] = 20

        req_indices = torch.tensor([0, 1, -1], dtype=torch.int32, device="npu")
        topk_indices = torch.zeros((3, 2048), dtype=torch.int32, device="npu")
        topk_indices[0, 1] = 63
        # Repeating token 7 verifies that concurrent writes of one to the same
        # position still produce a binary mask.
        topk_indices[1].fill_(7)

        token_on_device, device_token_pos, position_mask = slot_map_lookup(
            slot_map,
            req_indices,
            topk_indices,
            pos_mask_size=4096,
        )
        torch.npu.synchronize()

        self.assertEqual(tuple(position_mask.shape), (3, 4096))
        self.assertEqual(position_mask.dtype, torch.int32)
        self.assertTrue(torch.all(token_on_device[0] == 1).item())
        self.assertEqual(device_token_pos[0, 0].item(), 4094)
        self.assertEqual(device_token_pos[0, 1].item(), 4095)
        # 4094 and 4095 share one 32-byte mask line and may be updated by
        # different workers; both values must survive the atomic merge.
        self.assertEqual(position_mask[0, 4094].item(), 1)
        self.assertEqual(position_mask[0, 4095].item(), 1)
        self.assertEqual(position_mask[0].sum().item(), 2)
        self.assertEqual(position_mask[1, 20].item(), 1)
        self.assertEqual(position_mask[1].sum().item(), 1)
        self.assertEqual(position_mask[2].sum().item(), 0)

    def test_position_mask_ignores_hit_positions_outside_mask(self):
        slot_map = torch.full((1, 64), -1, dtype=torch.int32, device="npu")
        slot_map[0, 0] = 2047
        slot_map[0, 1] = 4095
        req_indices = torch.tensor([0], dtype=torch.int32, device="npu")
        topk_indices = torch.zeros((1, 2048), dtype=torch.int32, device="npu")
        topk_indices[0, 1] = 1

        token_on_device, device_token_pos, position_mask = slot_map_lookup(
            slot_map,
            req_indices,
            topk_indices,
            pos_mask_size=2048,
        )
        torch.npu.synchronize()

        self.assertEqual(token_on_device[0, 1].item(), 1)
        self.assertEqual(device_token_pos[0, 1].item(), 4095)
        self.assertEqual(position_mask[0, 2047].item(), 1)
        self.assertEqual(position_mask.sum().item(), 1)

    def test_rejects_non_positive_position_mask_size(self):
        req_indices = torch.tensor([0], dtype=torch.int32, device="npu")
        topk_indices = torch.zeros((1, 2048), dtype=torch.int32, device="npu")
        with self.assertRaisesRegex(ValueError, "pos_mask_size must be positive"):
            slot_map_lookup(
                self.slot_map,
                req_indices,
                topk_indices,
                pos_mask_size=0,
            )

    def test_rejects_unaligned_position_mask_size(self):
        req_indices = torch.tensor([0], dtype=torch.int32, device="npu")
        topk_indices = torch.zeros((1, 2048), dtype=torch.int32, device="npu")
        with self.assertRaisesRegex(ValueError, "multiple of 8"):
            slot_map_lookup(
                self.slot_map,
                req_indices,
                topk_indices,
                pos_mask_size=2049,
            )


if __name__ == "__main__":
    unittest.main()
