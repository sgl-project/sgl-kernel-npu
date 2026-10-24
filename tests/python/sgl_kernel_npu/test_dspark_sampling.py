"""Tests for dspark sampling."""

import unittest

import torch
from _sampling_test_utils import NPU_AVAILABLE, SamplingTestCase

if NPU_AVAILABLE:
    from sgl_kernel_npu.sample import (
        compute_dflash_correct_drafts_and_bonus,
        markov_greedy_step_triton,
        row_argmax,
        softmax_temp_triton,
    )


@unittest.skipUnless(NPU_AVAILABLE, "requires Ascend NPU, torch_npu and Triton-Ascend")
class TestDSparkSamplingPipeline(SamplingTestCase):
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
