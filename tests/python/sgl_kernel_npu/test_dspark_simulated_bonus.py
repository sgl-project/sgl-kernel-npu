"""Migrated all three GPU simulated-bonus tests from SGLang 7036f4b0ad."""

import types
import unittest

import torch
from test_dspark_kernel_parity import HAS_NPU

if HAS_NPU:
    from sgl_kernel_npu.sample.dspark_simulated_bonus import simulated_bonus_sample


def sample_simulated_bonus(
    *,
    target_logits,
    correct_len,
    greedy_bonus,
    sampling_info,
    bs,
    verify_num_draft_tokens,
):
    return (
        simulated_bonus_sample(
            target_logits=target_logits,
            correct_len=correct_len,
            temperatures=sampling_info.temperatures,
            bs=bs,
            rows_per_request=verify_num_draft_tokens,
        )
        .to(greedy_bonus.dtype)
        .view_as(greedy_bonus)
    )


def _info(temps):
    return types.SimpleNamespace(temperatures=torch.tensor(temps, device="npu"))


@unittest.skipUnless(HAS_NPU, "requires Ascend NPU and torch_npu")
class TestSimulatedBonus(unittest.TestCase):
    def test_picks_row_at_correct_len(self):
        torch.manual_seed(0)
        bs, rows, vocab = 3, 6, 1000
        logits = torch.randn(bs * rows, vocab, device="npu")
        correct_len = torch.tensor([0, 5, 2], device="npu", dtype=torch.int32)
        greedy_bonus = torch.zeros(bs, dtype=torch.int64, device="npu")
        out = sample_simulated_bonus(
            target_logits=logits,
            correct_len=correct_len,
            greedy_bonus=greedy_bonus,
            sampling_info=_info([1e-6] * bs),
            bs=bs,
            verify_num_draft_tokens=rows,
        )
        ref = logits.view(bs, rows, vocab).argmax(-1)[
            torch.arange(bs, device="npu"), correct_len.long()
        ]
        self.assertEqual(out.dtype, torch.int64)
        self.assertTrue(torch.equal(out, ref))

    def test_bf16_full_vocab_low_temperature(self):
        torch.manual_seed(2)
        bs, rows, vocab = 8, 6, 129280
        logits = torch.randn(bs * rows, vocab, device="npu", dtype=torch.bfloat16)
        correct_len = torch.randint(0, rows, (bs,), device="npu", dtype=torch.int64)
        out = sample_simulated_bonus(
            target_logits=logits,
            correct_len=correct_len,
            greedy_bonus=torch.zeros(bs, dtype=torch.int64, device="npu"),
            sampling_info=_info([1e-6] * bs),
            bs=bs,
            verify_num_draft_tokens=rows,
        )
        ref = (
            logits.float()
            .view(bs, rows, vocab)
            .argmax(-1)[torch.arange(bs, device="npu"), correct_len]
        )
        self.assertTrue(torch.equal(out, ref))

    def test_matches_softmax_distribution(self):
        torch.manual_seed(1)
        vocab, n = 8, 200_000
        base = torch.randn(vocab, device="npu")
        logits = base.repeat(n, 1)
        correct_len = torch.zeros(n, dtype=torch.int32, device="npu")
        out = sample_simulated_bonus(
            target_logits=logits,
            correct_len=correct_len,
            greedy_bonus=torch.zeros(n, dtype=torch.int64, device="npu"),
            sampling_info=_info([0.7] * n),
            bs=n,
            verify_num_draft_tokens=1,
        )
        freq = torch.bincount(out, minlength=vocab).float() / n
        want = torch.softmax(base / 0.7, dim=-1)
        self.assertLess((freq - want).abs().max().item(), 5e-3)


if __name__ == "__main__":
    unittest.main()
