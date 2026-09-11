"""Functional and validation tests for the A5 metadata operator."""

import unittest

import pytest
import sgl_kernel_npu  # noqa: F401
import torch
import torch_npu  # noqa: F401

from utils import require_npu_op

pytestmark = require_npu_op("kv_quant_sparse_attn_sharedkv_metadata")

AIC_SLOTS = 36
FA_WIDTH = 9
FD_WIDTH = 8


def _fa_records(metadata):
    return metadata.cpu().view(-1)[: AIC_SLOTS * FA_WIDTH].view(AIC_SLOTS, FA_WIDTH)


class TestKvQuantSparseAttnSharedkvMetadata(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch_npu.npu.is_available():
            raise unittest.SkipTest("an Ascend NPU is required")
        torch_npu.npu.set_device(0)
        props = torch.npu.get_device_properties(0)
        cls.aic = int(props.cube_core_num)

    def _call(self, *, num_heads_q=64, q_lengths=(1,), kv_lengths=(128,), **kwargs):
        batch_size = len(q_lengths)
        max_seqlen_q = max(q_lengths)
        max_seqlen_kv = max(kv_lengths)
        q_lengths = torch.tensor(q_lengths, dtype=torch.int32, device="npu")
        kv_lengths = torch.tensor(kv_lengths, dtype=torch.int32, device="npu")
        cu_q = torch.cat(
            (torch.zeros(1, dtype=torch.int32, device="npu"), q_lengths.cumsum(0))
        )
        call_args = dict(
            num_heads_q=num_heads_q,
            num_heads_kv=1,
            head_dim=512,
            kv_quant_mode=1,
            cu_seqlens_q=cu_q,
            seqused_kv=kv_lengths,
            batch_size=batch_size,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            ori_topk=0,
            cmp_topk=0,
            tile_size=64,
            rope_head_dim=64,
            cmp_ratio=1,
            ori_mask_mode=4,
            cmp_mask_mode=3,
            ori_win_left=127,
            ori_win_right=0,
            layout_q="TND",
            layout_kv="PA_ND",
            has_ori_kv=True,
            has_cmp_kv=False,
            device="npu",
        )
        call_args.update(kwargs)
        return torch.ops.npu.kv_quant_sparse_attn_sharedkv_metadata(**call_args)

    def test_output_contract_and_fa_layout(self):
        metadata = self._call(q_lengths=(1, 2, 1), kv_lengths=(128, 256, 512))
        self.assertEqual(tuple(metadata.shape), (1024,))
        self.assertEqual(metadata.dtype, torch.int32)
        self.assertEqual(metadata.device.type, "npu")

        fa = _fa_records(metadata)
        enabled = int((fa[:, 0] == 1).sum())
        self.assertGreaterEqual(enabled, 1)
        self.assertLessEqual(enabled, self.aic)
        self.assertTrue(torch.all(fa[:enabled, 0] == 1))
        self.assertTrue(torch.all(fa[enabled:, 0] == 0))
        self.assertEqual(int(fa[enabled - 1, 4]), 3)
        self.assertGreaterEqual(int(fa[:enabled, 8].max()), 1)

        fd_start = AIC_SLOTS * FA_WIDTH
        fd = metadata.cpu().view(-1)[fd_start : fd_start + 72 * FD_WIDTH]
        self.assertTrue(torch.all(fd.view(72, FD_WIDTH)[:, 0] == 0))

    def test_n128_halves_fa_cores_and_duplicates_records(self):
        metadata = _fa_records(
            self._call(num_heads_q=128, q_lengths=(1,), kv_lengths=(128,))
        )
        for i in range(AIC_SLOTS // 2):
            self.assertTrue(torch.equal(metadata[2 * i, :8], metadata[2 * i + 1, :8]))
            self.assertEqual(int(metadata[2 * i, 8]), int(metadata[2 * i + 1, 8]))

    def test_deterministic_for_identical_inputs(self):
        first = self._call(q_lengths=(1, 1), kv_lengths=(4096, 4096)).cpu()
        second = self._call(q_lengths=(1, 1), kv_lengths=(4096, 4096)).cpu()
        self.assertTrue(torch.equal(first, second))

    def test_requires_kv_sequence_lengths(self):
        with self.assertRaisesRegex(RuntimeError, "sequence lengths"):
            torch.ops.npu.kv_quant_sparse_attn_sharedkv_metadata(
                64,
                1,
                512,
                1,
                None,
                None,
                None,
                None,
                None,
                1,
                1,
                128,
                0,
                0,
                64,
                64,
                1,
                4,
                3,
                127,
                0,
                "TND",
                "PA_ND",
                True,
                False,
                "npu",
            )

    def test_rejects_invalid_cmp_parameters(self):
        with self.assertRaises(Exception):
            self._call(has_cmp_kv=True, cmp_topk=64)
        with self.assertRaises(Exception):
            self._call(has_cmp_kv=True, cmp_ratio=7)


if __name__ == "__main__":
    unittest.main()
