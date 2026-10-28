"""A5 metadata layout, scheduling, and input validation.

Select the NPU with NPU_DEVICE (default: 0), for example:
    NPU_DEVICE=2 python tests/python/sgl_kernel_npu/test_kv_quant_sparse_attn_sharedkv_metadata.py -v
"""

import os
import unittest

import sgl_kernel_npu  # noqa: F401 - registers the custom operator
import torch
import torch_npu


class TestKvQuantSparseAttnSharedkvMetadata(unittest.TestCase):
    CUBE_SLOTS = 36
    VECTOR_SLOTS = 72

    @classmethod
    def setUpClass(cls):
        if not torch_npu.npu.is_available():
            raise unittest.SkipTest("an Ascend NPU is required")
        device_index = int(os.environ.get("NPU_DEVICE", "0"))
        torch_npu.npu.set_device(device_index)
        cls.device = torch.device(f"npu:{device_index}")
        if "950" not in torch_npu.npu.get_device_name(device_index).lower():
            raise unittest.SkipTest("the quantized kernel requires Ascend 950")

    @staticmethod
    def _metadata(
        num_heads_q, cu_seqlens_q=None, seqused_kv=None, batch_size=1, **overrides
    ):
        arguments = dict(
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
            kv_quant_mode=1,
            tile_size=64,
            rope_head_dim=64,
        )
        arguments.update(overrides)
        return torch.ops.npu.kv_quant_sparse_attn_sharedkv_metadata_host(**arguments)

    def test_host_schema_accepts_sglang_quant_kwargs(self):
        schema = (
            torch.ops.npu.kv_quant_sparse_attn_sharedkv_metadata_host.default._schema
        )
        arguments = {argument.name: argument for argument in schema.arguments}
        for name, value in (
            ("kv_quant_mode", 1),
            ("tile_size", 64),
            ("rope_head_dim", 64),
        ):
            with self.subTest(argument=name):
                self.assertEqual(arguments[name].default_value, value)

        cu_q = torch.tensor([0, 1], dtype=torch.int32)
        seq_kv = torch.tensor([128], dtype=torch.int32)
        self._metadata(64, cu_q, seq_kv)
        for name, value in (
            ("kv_quant_mode", 0),
            ("tile_size", 128),
            ("rope_head_dim", 128),
        ):
            with self.subTest(argument=name):
                with self.assertRaisesRegex(RuntimeError, "expected kv_quant_mode"):
                    self._metadata(64, cu_q, seq_kv, **{name: value})

    def test_still_requires_cpu_lengths(self):
        cu_q = torch.tensor([0, 1], dtype=torch.int32, device=self.device)
        seq_kv = torch.tensor([128], dtype=torch.int32)
        with self.assertRaisesRegex(RuntimeError, "contiguous CPU int32"):
            self._metadata(64, cu_q, seq_kv)

    def _assert_valid_schedule(self, metadata, batch_size):
        self.assertEqual(metadata.device, self.device)
        self.assertEqual(metadata.dtype, torch.int32)
        self.assertEqual(tuple(metadata.shape), (1024,))

        host = metadata.cpu().view(-1)
        cube_size = self.CUBE_SLOTS * 9
        cube = host[:cube_size].view(self.CUBE_SLOTS, 9)
        vector = host[cube_size : cube_size + self.VECTOR_SLOTS * 8].view(
            self.VECTOR_SLOTS, 8
        )
        enabled = int((cube[:, 0] == 1).sum())
        self.assertGreaterEqual(enabled, 1)
        self.assertTrue(torch.all(cube[:enabled, 0] == 1).item())
        self.assertTrue(torch.all(cube[enabled:, 0] == 0).item())
        self.assertEqual(cube[enabled - 1, 4].item(), batch_size)
        # Flash Decode splitting is disabled in this metadata scheduler.
        self.assertTrue(torch.all(vector[:, 0] == 0).item())

    def test_64_head_metadata_has_nine_int_fa_stride(self):
        cu_q = torch.tensor([0, 1], dtype=torch.int32)
        seq_kv = torch.tensor([128], dtype=torch.int32)
        metadata = self._metadata(64, cu_q, seq_kv)
        self.assertEqual(metadata.device, self.device)
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

    def test_decode_mixed_sequence_lengths(self):
        batch_size = 8
        cu_q = torch.arange(batch_size + 1, dtype=torch.int32)
        seq_kv = torch.tensor(
            [128, 256, 512, 1024, 2048, 4096, 8192, 12345], dtype=torch.int32
        )
        metadata = self._metadata(64, cu_q, seq_kv, batch_size=batch_size)
        self._assert_valid_schedule(metadata, batch_size)

    def test_all_attention_modes_have_valid_schedules(self):
        batch_size = 8
        cu_q = torch.arange(batch_size + 1, dtype=torch.int32)
        seq_kv = torch.full((batch_size,), 4096, dtype=torch.int32)
        for cmp_ratio, has_cmp_kv, cmp_topk in (
            (1, False, 0),
            (4, True, 512),
            (128, True, 0),
        ):
            with self.subTest(cmp_ratio=cmp_ratio, cmp_topk=cmp_topk):
                metadata = self._metadata(
                    64,
                    cu_q,
                    seq_kv,
                    batch_size=batch_size,
                    max_seqlen_kv=4096,
                    cmp_ratio=cmp_ratio,
                    has_cmp_kv=has_cmp_kv,
                    cmp_topk=cmp_topk,
                )
                self._assert_valid_schedule(metadata, batch_size)

    def test_prefill_uses_multiple_cube_cores(self):
        query_length = 32768
        metadata = self._metadata(
            64,
            torch.tensor([0, query_length], dtype=torch.int32),
            torch.tensor([query_length], dtype=torch.int32),
            max_seqlen_q=query_length,
            max_seqlen_kv=query_length,
        )
        self._assert_valid_schedule(metadata, batch_size=1)
        cube = metadata.cpu()[: self.CUBE_SLOTS * 9].view(self.CUBE_SLOTS, 9)
        self.assertGreater(int((cube[:, 0] == 1).sum()), 1)

    def test_single_request_long_context(self):
        kv_length = 65536
        metadata = self._metadata(
            64,
            torch.tensor([0, 1], dtype=torch.int32),
            torch.tensor([kv_length], dtype=torch.int32),
            max_seqlen_kv=kv_length,
        )
        self._assert_valid_schedule(metadata, batch_size=1)

    def test_deterministic_compressed_schedule(self):
        batch_size = 8
        cu_q = torch.arange(batch_size + 1, dtype=torch.int32)
        seq_kv = torch.full((batch_size,), 4096, dtype=torch.int32)
        arguments = dict(
            batch_size=batch_size,
            max_seqlen_kv=4096,
            cmp_ratio=4,
            has_cmp_kv=True,
            cmp_topk=512,
        )
        first = self._metadata(64, cu_q, seq_kv, **arguments)
        second = self._metadata(64, cu_q, seq_kv, **arguments)
        self.assertTrue(torch.equal(first.cpu(), second.cpu()))

    def test_rejects_invalid_cmp_topk(self):
        cu_q = torch.arange(3, dtype=torch.int32)
        seq_kv = torch.tensor([100, 100], dtype=torch.int32)
        with self.assertRaisesRegex(RuntimeError, "A5 scheduling failed"):
            self._metadata(
                64,
                cu_q,
                seq_kv,
                batch_size=2,
                cmp_ratio=4,
                has_cmp_kv=True,
                cmp_topk=64,
            )

    def test_rejects_invalid_cmp_ratio(self):
        cu_q = torch.arange(3, dtype=torch.int32)
        seq_kv = torch.tensor([100, 100], dtype=torch.int32)
        with self.assertRaisesRegex(RuntimeError, "A5 scheduling failed"):
            self._metadata(
                64,
                cu_q,
                seq_kv,
                batch_size=2,
                cmp_ratio=7,
                has_cmp_kv=True,
                cmp_topk=512,
            )


if __name__ == "__main__":
    unittest.main()
