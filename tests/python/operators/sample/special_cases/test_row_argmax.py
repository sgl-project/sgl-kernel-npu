"""Tests for row argmax."""

import unittest

import torch

from tests.python.operators.sample.special_cases._sampling_test_utils import (
    NPU_AVAILABLE,
    SamplingTestCase,
)

if NPU_AVAILABLE:
    from sgl_kernel_npu.operators.sample import row_argmax


@unittest.skipUnless(NPU_AVAILABLE, "requires Ascend NPU, torch_npu and Triton-Ascend")
class TestRowArgmax(SamplingTestCase):
    def test_row_argmax_shapes_and_strides(self):
        for rows, vocab in ((1, 1), (3, 2049), (65, 4097), (385, 8193), (2, 129280)):
            with self.subTest(shape=(rows, vocab)):
                x = torch.randn(rows, vocab + 11).npu()[:, :vocab]
                self.assert_output(row_argmax(x), x.cpu().argmax(dim=-1))

    def test_row_argmax_special_values(self):
        # NaN must win over +inf, and a tie must resolve to the first index,
        # for every row count.
        for rows in (65, 9):
            with self.subTest(rows=rows):
                x = torch.full((rows, 4099), -torch.inf)
                x[0, 2047] = x[0, 2048] = 7
                x[1, 0] = x[1, -1] = torch.inf
                x[2, 2048] = x[2, -1] = torch.nan
                x[3, -1] = 1
                x[4].zero_()
                x[5].fill_(torch.nan)
                # The final row is entirely -inf, including a partially padded tile.
                self.assert_output(row_argmax(x.npu()), x.argmax(dim=-1))

    def test_row_argmax_cross_tile_special_values(self):
        for vocab in (8193, 16385, 129280):
            with self.subTest(vocab=vocab):
                x = torch.full((9, vocab), -torch.inf)
                x[0, 8191] = x[0, 8192] = x[0, -1] = 7
                x[1, 0] = torch.inf
                x[1, -1] = torch.nan
                x[2, 8192] = x[2, -1] = torch.nan
                x[3].fill_(torch.nan)
                x[4].zero_()
                self.assert_output(row_argmax(x.npu()), x.argmax(dim=-1))

    def test_row_argmax_cached_launches_and_graph(self):
        vocab = 16385
        for offset in (0, 1):
            with self.subTest(offset=offset):
                x = torch.randn(9, vocab + 13).npu()[:, offset : offset + vocab]
                self.assert_output(row_argmax(x), x.cpu().argmax(dim=-1))
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    output = row_argmax(x)
                for index in (0, 8192, vocab - 1):
                    x.zero_()
                    x[:, index] = 1
                    graph.replay()
                    self.assert_output(
                        output, torch.full((9,), index, dtype=torch.int64)
                    )
                stream = torch.npu.Stream()
                stream.wait_stream(torch.npu.current_stream())
                with torch.npu.stream(stream):
                    output = row_argmax(x)
                stream.synchronize()
                self.assert_output(output, x.cpu().argmax(dim=-1))


if __name__ == "__main__":
    unittest.main()
