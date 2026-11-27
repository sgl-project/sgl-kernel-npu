import unittest
from unittest.mock import Mock, patch

import torch
from sgl_kernel_npu.operators.attention import (
    int8_csa,
    int8_csa_attention,
    int8_csa_metadata,
    int8_csa_write,
)


class TestInt8Csa(unittest.TestCase):
    def test_reserved_page_write_preserves_the_paged_storage(self):
        backing = torch.zeros((3, 4 * 608 + 32), dtype=torch.int8)
        cache = backing[:, :-32].view(3, 4, 1, 608)
        slots = torch.tensor([-1, 0, 3, 4, 11, 12], dtype=torch.int32)
        latent = torch.zeros((6, 1, 512), dtype=torch.bfloat16)
        writer = Mock()
        with patch.object(int8_csa, "_native_ops", return_value=(writer, None, None)):
            int8_csa_write(cache, latent, slots, reserve_page_zero=True)

        dest, rows, valid_slots = writer.call_args.args
        self.assertEqual(dest.dtype, torch.uint8)
        self.assertEqual(dest.data_ptr(), cache.data_ptr())
        self.assertEqual(dest.stride(), cache.stride())
        self.assertEqual(rows.shape, (6, 512))
        self.assertEqual(valid_slots.tolist(), [-1, -1, -1, 4, 11, -1])

    def test_empty_reserved_page_write_skips_dispatch(self):
        with patch.object(int8_csa, "_native_ops") as ops:
            int8_csa_write(
                None,
                torch.empty((0, 512), dtype=torch.bfloat16),
                torch.empty(0, dtype=torch.int32),
                reserve_page_zero=True,
            )
        ops.assert_not_called()

    def test_attention_and_metadata_use_the_same_int8_format(self):
        metadata, attention = Mock(), Mock()
        q = torch.empty((1, 2, 512), dtype=torch.bfloat16)
        with patch.object(
            int8_csa, "_native_ops", return_value=(None, metadata, attention)
        ):
            descriptor = int8_csa_metadata(2, 1, 512, cmp_ratio=2)
            int8_csa_attention(q, metadata=descriptor, cmp_ratio=2)

        metadata.assert_called_once_with(2, 1, 512, 1, rope_head_dim=64, cmp_ratio=2)
        self.assertIs(attention.call_args.args[0], q)
        self.assertEqual(attention.call_args.args[1], 1)
        kwargs = attention.call_args.kwargs
        self.assertIs(kwargs["metadata"], descriptor)
        self.assertEqual(kwargs["rope_head_dim"], 64)
        self.assertEqual(kwargs["topk_value_mode"], 1)
        self.assertEqual(kwargs["sinks"].shape, (2,))
        self.assertTrue(torch.isneginf(kwargs["sinks"]).all())


if __name__ == "__main__":
    unittest.main()
