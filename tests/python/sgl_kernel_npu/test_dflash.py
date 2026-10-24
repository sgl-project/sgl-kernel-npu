"""Tests for dflash."""

import unittest

import torch
from _sampling_test_utils import NPU_AVAILABLE, SamplingTestCase

if NPU_AVAILABLE:
    from sgl_kernel_npu.sample import compute_dflash_correct_drafts_and_bonus


@unittest.skipUnless(NPU_AVAILABLE, "requires Ascend NPU, torch_npu and Triton-Ascend")
class TestDFlashAcceptance(SamplingTestCase):
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


if __name__ == "__main__":
    unittest.main()
