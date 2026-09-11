"""Correctness and input-contract tests for the A5 attention operator."""

import unittest

import pytest
import sgl_kernel_npu  # noqa: F401
import torch
import torch_npu  # noqa: F401

from utils import require_npu_op

pytestmark = require_npu_op("kv_quant_sparse_attn_sharedkv")


def _single_core_metadata(device, max_s2=1):
    metadata = torch.zeros(1024, dtype=torch.int32, device=device)
    metadata[0] = 1
    metadata[4] = 1
    metadata[8] = max_s2
    return metadata


def _packed_constant_kv(device, value=0.5):
    """Build one PA_ND [rope BF16 | nope FP8 | scale FP8] KV block."""
    raw = torch.zeros((1, 128, 1, 640), dtype=torch.uint8)
    rope = torch.tensor(value, dtype=torch.bfloat16).view(torch.uint8)
    raw[..., :128] = rope.repeat(64)
    nope = torch.tensor(value, dtype=torch.float8_e4m3fn).view(torch.uint8)
    raw[..., 128:576] = nope
    raw[..., 576:] = 127
    return raw.view(torch.float8_e4m3fn).to(device)


class TestKvQuantSparseAttnSharedkv(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch_npu.npu.is_available():
            raise unittest.SkipTest("an Ascend NPU is required")
        torch_npu.npu.set_device(0)

    def _inputs(self, q_len=2):
        device = torch.device("npu:0")
        q = torch.randn((1, q_len, 64, 512), dtype=torch.bfloat16, device=device)
        return {
            "q": q,
            "kv_quant_mode": 1,
            "ori_kv": _packed_constant_kv(device),
            "ori_block_table": torch.zeros((1, 1), dtype=torch.int32, device=device),
            "seqused_kv": torch.tensor([128], dtype=torch.int32, device=device),
            "sinks": torch.zeros(64, dtype=torch.float32, device=device),
            "metadata": _single_core_metadata(device),
            "tile_size": 64,
            "rope_head_dim": 64,
            "softmax_scale": 1.0 / (512**0.5),
            "cmp_ratio": 1,
            "ori_mask_mode": 4,
            "cmp_mask_mode": 3,
            "ori_win_left": 127,
            "ori_win_right": 0,
            "layout_q": "BSND",
            "layout_kv": "PA_ND",
        }

    def test_schema_and_output_contract(self):
        schema = str(torch.ops.npu.kv_quant_sparse_attn_sharedkv.default._schema)
        self.assertIn("Tensor q", schema)
        self.assertIn("Tensor? ori_kv=None", schema)
        self.assertIn("Tensor, Tensor", schema)

        output, lse = torch.ops.npu.kv_quant_sparse_attn_sharedkv(**self._inputs())
        torch_npu.npu.synchronize()
        self.assertEqual(tuple(output.shape), (1, 2, 64, 512))
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertEqual(tuple(lse.shape), (0,))
        self.assertTrue(torch.isfinite(output.float()).all().item())

    def test_constant_kv_is_query_independent(self):
        output, _ = torch.ops.npu.kv_quant_sparse_attn_sharedkv(**self._inputs(q_len=2))
        torch_npu.npu.synchronize()
        torch.testing.assert_close(output[:, 0], output[:, 1], rtol=0, atol=0)

    def test_return_softmax_lse_has_expected_shape_and_dtype(self):
        output, lse = torch.ops.npu.kv_quant_sparse_attn_sharedkv(
            **self._inputs(), return_softmax_lse=True
        )
        torch_npu.npu.synchronize()
        self.assertEqual(tuple(output.shape), (1, 2, 64, 512))
        self.assertEqual(tuple(lse.shape), (1, 2, 64, 1))
        self.assertEqual(lse.dtype, torch.float32)
        self.assertTrue(torch.isfinite(lse).all().item())

    def test_rejects_missing_required_inputs(self):
        inputs = self._inputs()
        inputs.pop("metadata")
        with self.assertRaisesRegex(RuntimeError, "metadata"):
            torch.ops.npu.kv_quant_sparse_attn_sharedkv(**inputs)

    def test_rejects_invalid_q_and_kv_dtypes(self):
        inputs = self._inputs()
        inputs["q"] = inputs["q"].float()
        with self.assertRaisesRegex(RuntimeError, "BF16"):
            torch.ops.npu.kv_quant_sparse_attn_sharedkv(**inputs)

        inputs = self._inputs()
        inputs["ori_kv"] = inputs["ori_kv"].view(torch.uint8)
        with self.assertRaisesRegex(RuntimeError, "FP8_E4M3FN"):
            torch.ops.npu.kv_quant_sparse_attn_sharedkv(**inputs)

    def test_rejects_invalid_metadata_shape(self):
        inputs = self._inputs()
        inputs["metadata"] = inputs["metadata"][:8]
        with self.assertRaisesRegex(RuntimeError, "metadata"):
            torch.ops.npu.kv_quant_sparse_attn_sharedkv(**inputs)


if __name__ == "__main__":
    unittest.main()
