import sys
import unittest
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

attention = import_module(
    "sgl_kernel_npu.operators.attention.mixed_quant_sparse_flash_mla"
)
metadata = import_module(
    "sgl_kernel_npu.operators.attention.mixed_quant_sparse_flash_mla_metadata"
)


class TestA3Int8MixedQuantAttention(unittest.TestCase):
    def test_a3_int8_attn_forwards_signed_pages_and_normalizes_sinks(self):
        q = torch.zeros((1, 2, 512), dtype=torch.bfloat16)
        win = torch.zeros((2, 16, 1, 608), dtype=torch.int8)
        cmp = torch.zeros_like(win)
        out = (q, None)
        op = Mock(return_value=out)
        upstream = SimpleNamespace(mixed_quant_sparse_flash_mla=op)

        with (
            patch.dict(sys.modules, {"cann_ops_transformer": upstream}),
            patch.object(attention, "_cann_a3_int8_fn", None),
        ):
            result = attention._cann_a3_int8_mixed_quant_sparse_flash_mla(
                q, quant_mode=1, ori_kv=win, cmp_kv=cmp
            )

        self.assertIs(result, out)
        self.assertIs(op.call_args.args[0], q)
        args = op.call_args.kwargs
        self.assertIs(args["ori_kv"], win)
        self.assertIs(args["cmp_kv"], cmp)
        self.assertEqual(args["quant_mode"], 1)
        self.assertEqual(args["sinks"].dtype, torch.float32)
        self.assertEqual(args["sinks"].shape, (2,))
        self.assertTrue(torch.isneginf(args["sinks"]).all())

    def test_a3_int8_metadata_preserves_c2_residual(self):
        residual = torch.ones(1, dtype=torch.int32)
        native = Mock(return_value=torch.zeros(1024, dtype=torch.int32))
        upstream = SimpleNamespace(mixed_quant_sparse_flash_mla_metadata=native)
        args = {"quant_mode": 1, "cmp_ratio": 2, "cmp_residual_kv": residual}

        with (
            patch.dict(sys.modules, {"cann_ops_transformer": upstream}),
            patch.object(metadata, "_cann_a3_int8_fn", None),
        ):
            result = metadata._cann_a3_int8_mixed_quant_sparse_flash_mla_metadata(
                **args
            )

        native.assert_called_once_with(**args)
        self.assertIs(result, native.return_value)


if __name__ == "__main__":
    unittest.main()
