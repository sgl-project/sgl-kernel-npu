"""Tests for markov greedy."""

import unittest

import torch

from tests.python.operators.sample.special_cases._sampling_test_utils import (
    NPU_AVAILABLE,
    SamplingTestCase,
)

if NPU_AVAILABLE:
    from sgl_kernel_npu.operators.sample import (
        markov_greedy_step_triton,
    )


def _markov_reference(base, embeds, weight):
    # Use the same NPU projection and intermediate rounding as Torch native.
    return (base + torch.nn.functional.linear(embeds, weight)).argmax(dim=-1).cpu()


@unittest.skipUnless(NPU_AVAILABLE, "requires Ascend NPU, torch_npu and Triton-Ascend")
class TestMarkovGreedy(SamplingTestCase):
    def test_markov_shapes_dtypes_and_strides(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for rows, vocab, rank in (
                (1, 1, 1),
                (3, 257, 33),
                (65, 4097, 64),
                (1, 129280, 32),
            ):
                with self.subTest(dtype=dtype, shape=(rows, vocab, rank)):
                    base = torch.randn(rows, vocab + 7, dtype=dtype).npu()[:, :vocab]
                    weight = (torch.randn(vocab, rank + 3, dtype=dtype) * 0.05).npu()[
                        :, :rank
                    ]
                    embeds = torch.randn(rows, rank * 2, dtype=dtype).npu()[:, ::2]
                    expected = _markov_reference(base, embeds, weight)
                    result = markov_greedy_step_triton(
                        base_logits=base, prev_embeds=embeds, w2_weight=weight
                    )
                    self.assert_output(result, expected)

    def test_markov_addition_rounding(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            base = torch.ones(2, 513, dtype=dtype).npu()
            embeds = torch.ones(2, 33, dtype=dtype).npu()
            weight = torch.zeros(513, 33, dtype=dtype).npu()
            # Torch's FP16/BF16 addition rounds the increment away.
            weight[1, 32] = 2**-13
            base[1].zero_()
            embeds[1].zero_()
            result = markov_greedy_step_triton(
                base_logits=base, prev_embeds=embeds, w2_weight=weight
            )
            expected = _markov_reference(base, embeds, weight)
            self.assert_output(result, expected)

    def test_markov_rows_and_rank_boundaries(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for rows, vocab, rank in (
                (0, 257, 64),
                (1, 513, 64),
                (5, 257, 31),
                (7, 513, 64),
                (9, 259, 127),
                (5, 257, 128),
                (5, 257, 129),
            ):
                for padding in (0, 3):
                    with self.subTest(
                        dtype=dtype, shape=(rows, vocab, rank), padding=padding
                    ):
                        base = torch.randn(rows, vocab, dtype=dtype).npu()
                        embeds = torch.randn(rows, rank, dtype=dtype).npu()
                        weight = (
                            torch.randn(vocab, rank + padding, dtype=dtype) * 0.05
                        ).npu()[:, :rank]
                        expected = _markov_reference(base, embeds, weight)
                        result = markov_greedy_step_triton(
                            base_logits=base, prev_embeds=embeds, w2_weight=weight
                        )
                        self.assert_output(result, expected)

    def test_markov_projection_rounds_to_input_dtype(self):
        for dtype in (torch.float16, torch.bfloat16):
            for rows in (1, 4, 16, 32):
                with self.subTest(dtype=dtype, rows=rows):
                    base = torch.zeros(rows, 257, dtype=dtype).npu()
                    embeds = torch.ones(rows, 64, dtype=dtype).npu()
                    weight = torch.zeros(257, 64, dtype=dtype).npu()
                    weight[0, 0] = weight[256, 0] = 1
                    # Torch's low-precision projection rounds both candidates to 1.
                    weight[256, 32] = 2**-13
                    result = markov_greedy_step_triton(
                        base_logits=base, prev_embeds=embeds, w2_weight=weight
                    )
                    self.assert_output(result, torch.zeros((rows,), dtype=torch.int64))

    def test_markov_cached_launches_and_graph(self):
        for rows, offset in ((5, 0), (5, 1), (33, 0), (33, 1)):
            with self.subTest(rows=rows, offset=offset):
                base = torch.zeros(rows, 272).npu()[:, offset : offset + 257]
                embeds = torch.zeros(rows, 33, dtype=torch.bfloat16).npu()
                weight = torch.zeros(257, 33, dtype=torch.bfloat16).npu()

                def run(logits):
                    return markov_greedy_step_triton(
                        base_logits=logits, prev_embeds=embeds, w2_weight=weight
                    )

                run(base)
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    output = run(base)
                for token in (17, 256):
                    base.zero_()
                    base[:, token] = 1
                    graph.replay()
                    expected = torch.full((rows,), token, dtype=torch.int64)
                    self.assert_output(output, expected)
                    # Same specialization, fresh pointers and a different stream.
                    fresh = torch.zeros_like(base)
                    fresh[:, token] = 1
                    stream = torch.npu.Stream()
                    stream.wait_stream(torch.npu.current_stream())
                    with torch.npu.stream(stream):
                        result = run(fresh)
                    stream.synchronize()
                    self.assert_output(result, expected)

    def test_markov_fp16_projection_matches_native_overflow(self):
        base = torch.zeros(4, 257, dtype=torch.float16).npu()
        embeds = torch.full((4, 64), 4096, dtype=torch.float16).npu()
        weight = torch.ones(257, 64, dtype=torch.float16).npu()
        weight[256, 0] = 2
        result = markov_greedy_step_triton(
            base_logits=base, prev_embeds=embeds, w2_weight=weight
        )
        self.assert_output(result, _markov_reference(base, embeds, weight))

    def test_markov_full_vocab_matches_native_reference(self):
        for dtype in (torch.float16, torch.bfloat16):
            for rows in (1, 4, 16, 32):
                with self.subTest(dtype=dtype, rows=rows):
                    base = torch.randn(rows, 129280).to(dtype).npu()
                    embeds = torch.randn(rows, 64).to(dtype).npu()
                    weight = (torch.randn(129280, 64) * 0.05).to(dtype).npu()
                    expected = _markov_reference(base, embeds, weight)
                    result = markov_greedy_step_triton(
                        base_logits=base,
                        prev_embeds=embeds,
                        w2_weight=weight,
                    )
                    self.assert_output(result, expected)

    def test_markov_mixed_dtype_promotion(self):
        for base_dtype in (torch.float16, torch.bfloat16, torch.float32):
            for dtype in (torch.float16, torch.bfloat16, torch.float32):
                with self.subTest(base_dtype=base_dtype, projection_dtype=dtype):
                    base = torch.ones(3, 8193, dtype=base_dtype).npu()
                    embeds = torch.full((3, 1), 2**-13, dtype=dtype).npu()
                    weight = torch.zeros(8193, 1, dtype=dtype).npu()
                    weight[-1] = 1
                    result = markov_greedy_step_triton(
                        base_logits=base, prev_embeds=embeds, w2_weight=weight
                    )
                    self.assert_output(result, _markov_reference(base, embeds, weight))

    def test_markov_cross_tile_special_values(self):
        for rows in (5, 33):
            with self.subTest(rows=rows):
                base = torch.full((rows, 8193), -torch.inf).npu()
                base[0, 4095] = base[0, 4096] = 7
                base[1, 0] = torch.inf
                base[1, -1] = torch.nan
                base[2, 4095] = base[2, -1] = torch.nan
                base[3].fill_(torch.nan)
                embeds = torch.zeros(rows, 33, dtype=torch.bfloat16).npu()
                weight = torch.zeros(8193, 33, dtype=torch.bfloat16).npu()
                result = markov_greedy_step_triton(
                    base_logits=base, prev_embeds=embeds, w2_weight=weight
                )
                self.assert_output(result, _markov_reference(base, embeds, weight))

    def test_markov_full_vocab_first_index_ties(self):
        for rows in (1, 4, 16, 32):
            with self.subTest(rows=rows):
                base = torch.zeros(rows, 129280, dtype=torch.bfloat16).npu()
                embeds = torch.ones(rows, 64, dtype=torch.bfloat16).npu()
                weight = torch.zeros(129280, 64, dtype=torch.bfloat16).npu()
                weight[8191, 0] = weight[8192, 0] = weight[-1, 0] = 1
                result = markov_greedy_step_triton(
                    base_logits=base, prev_embeds=embeds, w2_weight=weight
                )
                self.assert_output(result, torch.full((rows,), 8191, dtype=torch.int64))


if __name__ == "__main__":
    unittest.main()
