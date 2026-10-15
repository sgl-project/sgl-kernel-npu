"""A5 metadata layout tests for the quantized shared-KV attention kernel."""

import unittest

import sgl_kernel_npu  # noqa: F401 - registers the custom operator
import torch
import torch_npu


class TestKvQuantSparseAttnSharedkvMetadata(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch_npu.npu.is_available():
            raise unittest.SkipTest("an Ascend NPU is required")
        if "950" not in torch_npu.npu.get_device_name(0).lower():
            raise unittest.SkipTest("the quantized kernel requires Ascend 950")
        torch_npu.npu.set_device(0)

    @staticmethod
    def _metadata(num_heads_q, cu_seqlens_q=None, seqused_kv=None, batch_size=1):
        return torch.ops.npu.kv_quant_sparse_attn_sharedkv_metadata_host(
            num_heads_q=num_heads_q,
            num_heads_kv=1,
            head_dim=512,
            layout_q="TND",
            layout_kv="PA_ND",
            cu_seqlens_q=cu_seqlens_q,
            seqused_kv=seqused_kv,
            batch_size=batch_size,
            max_seqlen_q=1,
            max_seqlen_kv=128,
            cmp_topk=0,
            cmp_ratio=1,
            ori_mask_mode=4,
            cmp_mask_mode=3,
            ori_win_left=127,
            ori_win_right=0,
            has_ori_kv=True,
            has_cmp_kv=False,
        )

    def test_64_head_metadata_has_nine_int_fa_stride(self):
        cu_q = torch.tensor([0, 1], dtype=torch.int32)
        seq_kv = torch.tensor([128], dtype=torch.int32)
        metadata = self._metadata(64, cu_q, seq_kv)
        self.assertEqual(metadata.device.type, "npu")
        self.assertEqual(metadata.dtype, torch.int32)
        self.assertEqual(metadata.numel(), 1024)
        host = metadata.cpu()
        self.assertEqual(host[0].item(), 1)
        self.assertEqual(host[4].item(), 1)  # BN2 end
        self.assertEqual(host[9].item(), 0)  # next cube core disabled
        self.assertEqual(host[36 * 9].item(), 0)  # first vector core disabled

    def test_128_head_metadata_duplicates_cube_core_pair(self):
        cu_q = torch.tensor([0, 1], dtype=torch.int32)
        seq_kv = torch.tensor([128], dtype=torch.int32)
        host = self._metadata(128, cu_q, seq_kv).cpu().view(-1)
        self.assertEqual(host[:9].tolist(), host[9:18].tolist())
        self.assertEqual(host[0].item(), 1)
        self.assertEqual(host[18].item(), 0)
        self.assertGreater(host[8].item(), 0)  # max S2 blocks for inactive pairs
        self.assertEqual(host[18 + 8].item(), host[8].item())

    def test_rejects_non_int32_sequence_lengths(self):
        with self.assertRaisesRegex(RuntimeError, "cu_seqlens_q"):
            self._metadata(
                64,
                torch.tensor([0, 1], dtype=torch.int64),
                torch.tensor([128], dtype=torch.int32),
            )

    def test_infers_batch_size_from_cu_seqlens(self):
        cu_q = torch.tensor([0, 1], dtype=torch.int32)
        seq_kv = torch.tensor([128], dtype=torch.int32)
        explicit = self._metadata(64, cu_q, seq_kv, batch_size=1)
        inferred = self._metadata(64, cu_q, seq_kv, batch_size=0)
        torch.testing.assert_close(inferred.cpu(), explicit.cpu())


if __name__ == "__main__":
    unittest.main()
