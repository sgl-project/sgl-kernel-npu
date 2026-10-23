"""DSpark greedy draft, target verify and temperature probability kernels."""

import importlib.util
import unittest

import torch

if importlib.util.find_spec("torch_npu") is not None:
    import torch_npu  # noqa: F401

    NPU_AVAILABLE = torch.npu.is_available()
else:
    NPU_AVAILABLE = False

if NPU_AVAILABLE:
    from sgl_kernel_npu.sample import (
        compute_dflash_correct_drafts_and_bonus,
        markov_greedy_step_triton,
        row_argmax,
        softmax_temp_triton,
    )


def _markov_reference(base, embeds, weight):
    # Use the same NPU projection and intermediate rounding as Torch native.
    return (base + torch.nn.functional.linear(embeds, weight)).argmax(dim=-1).cpu()


@unittest.skipUnless(NPU_AVAILABLE, "requires Ascend NPU, torch_npu and Triton-Ascend")
class TestDSparkSampling(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def assert_output(self, actual, expected):
        self.assertEqual(actual.device.type, "npu")
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.shape, expected.shape)
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)

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
                self.assert_output(
                    result, torch.full((rows,), 8191, dtype=torch.int64)
                )

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

    def test_accept_every_possible_length(self):
        for block in (1, 2, 3, 8, 9, 17):
            for dtype in (torch.int32, torch.int64):
                with self.subTest(block=block, dtype=dtype):
                    target = torch.arange(block * block, dtype=dtype).reshape(
                        block, block
                    )
                    candidates = torch.zeros_like(target)
                    candidates[:, 1:] = target[:, :-1]
                    for row in range(block - 1):
                        candidates[row, row + 1] = -100
                    # Non-contiguous inputs must preserve the GPU API behavior.
                    c = torch.empty(block, block * 2, dtype=dtype).npu()[:, ::2]
                    t = torch.empty_like(c, memory_format=torch.contiguous_format)
                    c.copy_(candidates)
                    t.copy_(target)
                    lengths, bonus = compute_dflash_correct_drafts_and_bonus(
                        candidates=c, target_predict=t
                    )
                    expected_len = torch.arange(block, dtype=torch.int32)
                    expected_bonus = target[
                        torch.arange(block), expected_len.long()
                    ].long()
                    self.assert_output(lengths, expected_len)
                    self.assert_output(bonus, expected_bonus)

    def test_accept_multiple_rows_per_core(self):
        target = torch.arange(1025 * 5, dtype=torch.int64).reshape(1025, 5)
        candidates = torch.cat(
            (torch.zeros(1025, 1, dtype=torch.int64), target[:, :-1]), dim=1
        )
        lengths, bonus = compute_dflash_correct_drafts_and_bonus(
            candidates=candidates.npu(), target_predict=target.npu()
        )
        self.assert_output(lengths, torch.full((1025,), 4, dtype=torch.int32))
        self.assert_output(bonus, target[:, -1])

    def test_greedy_target_pipeline(self):
        bs, block, vocab = 5, 9, 4097
        logits = torch.randn(bs * block, vocab)
        target = logits.argmax(-1).view(bs, block)
        candidates = torch.zeros_like(target)
        candidates[:, 1:] = target[:, :-1]
        for row in range(bs):
            candidates[row, row + 1] = -1
        expected_len = (
            (candidates[:, 1:] == target[:, :-1]).int().cumprod(1).sum(1).int()
        )
        expected_bonus = target[torch.arange(bs), expected_len.long()]
        predicted = row_argmax(logits.npu()).view(bs, block)
        lengths, bonus = compute_dflash_correct_drafts_and_bonus(
            candidates=candidates.npu(), target_predict=predicted
        )
        self.assert_output(lengths, expected_len)
        self.assert_output(bonus, expected_bonus)

    def test_multicore_store_stress(self):
        rows, vocab, block = 1025, 4097, 9
        winners = torch.arange(rows).mul(123).remainder(vocab)
        logits = torch.zeros(rows, vocab)
        logits[torch.arange(rows), winners] = 1
        target = torch.arange(rows * block).reshape(rows, block)
        candidates = torch.cat((target[:, :1], target[:, :-1]), dim=1)
        lengths = torch.arange(rows).remainder(block)
        rejected = lengths < block - 1
        candidates[torch.arange(rows)[rejected], lengths[rejected] + 1] = -1
        logits, candidates, npu_target = logits.npu(), candidates.npu(), target.npu()
        for _ in range(10):
            self.assert_output(row_argmax(logits), winners)
            actual_len, bonus = compute_dflash_correct_drafts_and_bonus(
                candidates=candidates, target_predict=npu_target
            )
            self.assert_output(actual_len, lengths.int())
            self.assert_output(bonus, target[torch.arange(rows), lengths])

    def test_accept_invalid_shapes(self):
        for shape, target_shape in (
            ((0, 4), (0, 4)),
            ((2, 0), (2, 0)),
            ((4,), (4,)),
            ((2, 4), (2, 3)),
        ):
            with self.subTest(shape=shape, target_shape=target_shape):
                with self.assertRaises(ValueError):
                    compute_dflash_correct_drafts_and_bonus(
                        candidates=torch.empty(shape, dtype=torch.int64).npu(),
                        target_predict=torch.empty(
                            target_shape, dtype=torch.int64
                        ).npu(),
                    )

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

    def test_empty_rows(self):
        self.assert_output(
            row_argmax(torch.empty(0, 4).npu()), torch.empty(0, dtype=torch.int64)
        )
        result = softmax_temp_triton(
            logits=torch.empty(0, 4).npu(),
            temperatures=torch.empty(0).npu(),
            rows_per_request=1,
        )
        self.assert_output(result, torch.empty(0, 4))
        result = markov_greedy_step_triton(
            base_logits=torch.empty(0, 4).npu(),
            prev_embeds=torch.empty(0, 3).npu(),
            w2_weight=torch.empty(4, 3).npu(),
        )
        self.assert_output(result, torch.empty(0, dtype=torch.int64))

    def test_graph_replay_uses_updated_inputs(self):
        base = torch.zeros(2, 513).npu()
        embeds = torch.zeros(2, 33).npu()
        weight = torch.zeros(513, 33).npu()
        candidates = torch.zeros(2, 3, dtype=torch.int64).npu()
        target = torch.zeros_like(candidates)
        logits = torch.zeros(6, 4097).npu()
        temps = torch.ones(2).npu()

        def run():
            return (
                markov_greedy_step_triton(
                    base_logits=base, prev_embeds=embeds, w2_weight=weight
                ),
                row_argmax(logits),
                *compute_dflash_correct_drafts_and_bonus(
                    candidates=candidates, target_predict=target
                ),
                softmax_temp_triton(
                    logits=logits, temperatures=temps, rows_per_request=3
                ),
            )

        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            for _ in range(3):
                run()
        torch.npu.current_stream().wait_stream(stream)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            outputs = run()
        for token in (17, 512):
            base.zero_()
            base[:, token] = 10
            logits.zero_()
            logits[:, token] = 10
            target.fill_(token)
            candidates[:, 1:].fill_(token)
            temps.fill_(0.7)
            graph.replay()
            expected = run()
            for actual, ref in zip(outputs, expected):
                torch.testing.assert_close(
                    actual.cpu(), ref.cpu(), rtol=3e-5, atol=1e-7
                )


if __name__ == "__main__":
    unittest.main()
