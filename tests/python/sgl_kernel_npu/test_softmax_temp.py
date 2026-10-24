"""Tests for softmax temp."""

import unittest

import torch
from _sampling_test_utils import NPU_AVAILABLE, SamplingTestCase

if NPU_AVAILABLE:
    from sgl_kernel_npu.sample import softmax_temp_triton


@unittest.skipUnless(NPU_AVAILABLE, "requires Ascend NPU, torch_npu and Triton-Ascend")
class TestSoftmaxTemperature(SamplingTestCase):
    def test_softmax_dtypes_temperature_mapping_and_strides(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for rows_per_request, vocab in ((1, 1), (3, 4097), (17, 8193), (1, 129280)):
                with self.subTest(dtype=dtype, rows=rows_per_request, vocab=vocab):
                    rows = 3 * rows_per_request
                    logits = (torch.randn(rows, vocab + 5, dtype=dtype) * 3).npu()[
                        :, :vocab
                    ]
                    temps = torch.tensor([[0.1, 9.0], [1.0, 9.0], [2.5, 9.0]]).npu()[
                        :, :1
                    ]
                    expected = torch.softmax(
                        logits.cpu().float()
                        / temps.cpu().repeat_interleave(rows_per_request, dim=0),
                        dim=-1,
                    )
                    result = softmax_temp_triton(
                        logits=logits,
                        temperatures=temps,
                        rows_per_request=rows_per_request,
                    )
                    self.assertEqual(result.dtype, torch.float32)
                    self.assertEqual(result.shape, logits.shape)
                    self.assertEqual(result.device, logits.device)
                    torch.testing.assert_close(
                        result.cpu(), expected, rtol=3e-5, atol=1e-7
                    )
                    torch.testing.assert_close(
                        result.cpu().sum(-1), torch.ones(rows), rtol=1e-5, atol=1e-6
                    )

    def test_softmax_large_logits_and_masked_tokens(self):
        logits = torch.tensor(
            [[10000.0, 9999.0, -torch.inf], [-10000.0, -9999.0, -torch.inf]]
        )
        result = softmax_temp_triton(
            logits=logits.npu(), temperatures=torch.ones(2).npu(), rows_per_request=1
        )
        torch.testing.assert_close(
            result.cpu(), logits.softmax(-1), rtol=3e-5, atol=1e-7
        )


if __name__ == "__main__":
    unittest.main()
