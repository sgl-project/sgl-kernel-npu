"""DSpark output assembly, including the uncommitted suffix."""

import unittest

import torch
from _sampling_test_utils import NPU_AVAILABLE, SamplingTestCase

if NPU_AVAILABLE:
    from sgl_kernel_npu.sample import build_out_tokens_triton


def reference_build_out_tokens(draft, lengths, bonus):
    draft, lengths, bonus = draft.cpu(), lengths.cpu(), bonus.cpu()
    out = torch.cat(
        (draft.long(), torch.zeros(draft.shape[0], 1, dtype=torch.int64)), 1
    )
    out[torch.arange(draft.shape[0]), lengths.long()] = bonus.long()
    return out


@unittest.skipUnless(NPU_AVAILABLE, "requires Ascend NPU, torch_npu and Triton-Ascend")
class TestBuildOutTokens(SamplingTestCase):
    def test_all_accept_lengths_dtypes_and_strides(self):
        for gamma in (0, 1, 3, 8, 17):
            for dtype in (torch.int32, torch.int64):
                for strided in (False, True):
                    with self.subTest(gamma=gamma, dtype=dtype, strided=strided):
                        bs = gamma + 1
                        draft = torch.arange(bs * gamma, dtype=dtype).reshape(bs, gamma)
                        lengths = torch.arange(bs, dtype=dtype)
                        bonus = -torch.arange(1, bs + 1, dtype=dtype)
                        expected = reference_build_out_tokens(draft, lengths, bonus)
                        if strided:
                            npu_draft = torch.empty(
                                bs, 2 * gamma + 3, dtype=dtype
                            ).npu()[:, 1 : 1 + 2 * gamma : 2]
                            npu_lengths = torch.empty(2 * bs + 1, dtype=dtype).npu()[
                                1::2
                            ]
                            npu_bonus = torch.empty(2 * bs + 1, dtype=dtype).npu()[1::2]
                            npu_draft.copy_(draft)
                            npu_lengths.copy_(lengths)
                            npu_bonus.copy_(bonus)
                        else:
                            npu_draft, npu_lengths, npu_bonus = (
                                draft.npu(),
                                lengths.npu(),
                                bonus.npu(),
                            )
                        actual = build_out_tokens_triton(
                            draft_tokens=npu_draft,
                            correct_len=npu_lengths,
                            bonus=npu_bonus,
                            verify_num_draft_tokens=gamma + 1,
                            gamma=gamma,
                        )
                        self.assert_output(actual, expected)

    def test_int64_bits_and_mixed_integer_dtypes(self):
        draft = torch.tensor(
            [
                [1 << 40, -(1 << 40), (1 << 63) - 1],
                [-(1 << 63), (1 << 32) + 1, -1],
            ]
        )
        lengths = torch.tensor([0, 3], dtype=torch.int32)
        for bonus in (
            torch.tensor([-(1 << 63), (1 << 63) - 1]),
            torch.tensor([-7, 11], dtype=torch.int32),
        ):
            with self.subTest(bonus_dtype=bonus.dtype):
                actual = build_out_tokens_triton(
                    draft_tokens=draft.npu(),
                    correct_len=lengths.npu(),
                    bonus=bonus.npu(),
                    verify_num_draft_tokens=4,
                    gamma=3,
                )
                self.assert_output(
                    actual, reference_build_out_tokens(draft, lengths, bonus)
                )

    def test_empty_batch(self):
        for gamma in (0, 3):
            actual = build_out_tokens_triton(
                draft_tokens=torch.empty(0, gamma, dtype=torch.int64).npu(),
                correct_len=torch.empty(0, dtype=torch.int32).npu(),
                bonus=torch.empty(0, dtype=torch.int64).npu(),
                verify_num_draft_tokens=gamma + 1,
                gamma=gamma,
            )
            self.assert_output(actual, torch.empty(0, gamma + 1, dtype=torch.int64))

    def test_multicore_and_graph_replay_updated_inputs(self):
        bs, gamma = 4097, 17
        draft = torch.arange(bs * gamma, dtype=torch.int64).reshape(bs, gamma).npu()
        lengths = torch.arange(bs, dtype=torch.int32).remainder(gamma + 1).npu()
        bonus = torch.arange(bs, dtype=torch.int64).add(1 << 40).npu()
        kwargs = dict(
            draft_tokens=draft,
            correct_len=lengths,
            bonus=bonus,
            verify_num_draft_tokens=gamma + 1,
            gamma=gamma,
        )
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            for _ in range(3):
                output = build_out_tokens_triton(**kwargs)
        torch.npu.current_stream().wait_stream(stream)
        self.assert_output(output, reference_build_out_tokens(draft, lengths, bonus))
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            output = build_out_tokens_triton(**kwargs)
        for length in (0, gamma // 2, gamma):
            draft.add_(1)
            lengths.fill_(length)
            bonus.add_(7)
            graph.replay()
            self.assert_output(
                output, reference_build_out_tokens(draft, lengths, bonus)
            )

    def test_invalid_metadata(self):
        kwargs = dict(
            draft_tokens=torch.zeros(2, 3, dtype=torch.int64).npu(),
            correct_len=torch.zeros(2, dtype=torch.int32).npu(),
            bonus=torch.zeros(2, dtype=torch.int64).npu(),
            verify_num_draft_tokens=4,
            gamma=3,
        )
        cases = [
            dict(gamma=-1),
            dict(verify_num_draft_tokens=3),
            dict(draft_tokens=torch.zeros(2, dtype=torch.int64).npu()),
            dict(draft_tokens=torch.zeros(2, 4, dtype=torch.int64).npu()),
            dict(correct_len=torch.zeros(2, 1, dtype=torch.int32).npu()),
            dict(bonus=torch.zeros(1, dtype=torch.int64).npu()),
        ]
        for name in ("draft_tokens", "correct_len", "bonus"):
            cases.append({name: kwargs[name].float()})
            cases.append({name: kwargs[name].cpu()})
        for index, change in enumerate(cases):
            with self.subTest(case=index):
                with self.assertRaises(ValueError):
                    build_out_tokens_triton(**dict(kwargs, **change))


if __name__ == "__main__":
    unittest.main()
