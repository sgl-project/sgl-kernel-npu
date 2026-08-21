import torch
import torch_npu  # noqa: F401
from sgl_kernel_npu.sample import chain_speculative_sampling_triton


def chain_rejection_reference(
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    retrive_index,
    uniform_samples,
    uniform_samples_for_final_sampling,
    target_probs,
    draft_probs,
):
    batch_size, num_draft_tokens = candidates.shape
    for req_idx in range(batch_size):
        cur_prob_row = 0
        last_accepted_idx = int(retrive_index[req_idx, 0])
        accept_index[req_idx, 0] = last_accepted_idx
        num_accepted = 0
        all_accepted = True

        for step in range(1, num_draft_tokens):
            draft_token = int(candidates[req_idx, step])
            p = float(target_probs[req_idx, cur_prob_row, draft_token])
            q = float(draft_probs[req_idx, cur_prob_row, draft_token])
            coin = float(uniform_samples[req_idx, step - 1])
            if coin * q < p:
                predicts[last_accepted_idx] = draft_token
                num_accepted += 1
                last_accepted_idx = int(retrive_index[req_idx, step])
                accept_index[req_idx, num_accepted] = last_accepted_idx
                cur_prob_row = step
            else:
                all_accepted = False
                break

        accept_token_num[req_idx] = num_accepted
        residual = target_probs[req_idx, cur_prob_row].clone()
        if not all_accepted:
            residual.sub_(draft_probs[req_idx, cur_prob_row]).clamp_min_(0.0)
        target = float(uniform_samples_for_final_sampling[req_idx]) * float(
            residual.sum()
        )
        sampled_token = int((residual.cumsum(0) <= target).sum())
        if sampled_token == residual.numel():
            positive = torch.nonzero(residual > 0.0).flatten()
            sampled_token = (
                int(positive[-1]) if positive.numel() else residual.numel() - 1
            )
        predicts[last_accepted_idx] = sampled_token


def test_chain_rejection_matches_gpu_algorithm():
    batch_size, num_draft_tokens, vocab_size = 2, 4, 11
    candidates = torch.tensor([[0, 2, 3, 4], [0, 5, 6, 7]])
    retrive_index = torch.arange(batch_size * num_draft_tokens).view(
        batch_size, num_draft_tokens
    )
    target_probs = torch.softmax(
        torch.tensor(
            [
                [
                    [0.1, 0.2, 2.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                    [0.1, 0.2, 0.1, 2.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                    [0.1, 0.2, 0.1, 0.1, 2.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                    [0.1, 0.2, 0.1, 0.1, 2.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                ],
                [
                    [0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 2.0, 0.1, 0.1, 0.1],
                    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 2.0, 0.2, 0.1, 0.1, 0.1],
                    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 2.0, 0.1, 0.1, 0.1],
                    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 2.0, 0.1, 0.1, 0.1],
                ],
            ],
            dtype=torch.float32,
        ),
        dim=-1,
    )
    draft_probs = torch.softmax(
        torch.tensor(
            [
                [[0.1] * vocab_size, [0.1] * vocab_size, [0.1] * vocab_size],
                [[0.1] * vocab_size, [0.1] * vocab_size, [0.1] * vocab_size],
            ],
            dtype=torch.float32,
        ),
        dim=-1,
    )
    draft_probs[1, 0, 5] = 0.9
    draft_probs[1, 0] /= draft_probs[1, 0].sum()
    uniforms = torch.tensor([[0.1, 0.1, 0.1, 0.0], [0.99, 0.0, 0.0, 0.0]])
    final_uniforms = torch.tensor([0.37, 0.61])

    expected_predicts = torch.full(
        (batch_size * num_draft_tokens,), -1, dtype=torch.int32
    )
    expected_accept_index = torch.full(
        (batch_size, num_draft_tokens), -1, dtype=torch.int32
    )
    expected_accept_num = torch.zeros(batch_size, dtype=torch.int32)
    chain_rejection_reference(
        expected_predicts,
        expected_accept_index,
        expected_accept_num,
        candidates,
        retrive_index,
        uniforms,
        final_uniforms,
        target_probs,
        draft_probs,
    )

    predicts = torch.full_like(expected_predicts, -1, device="npu")
    accept_index = torch.full_like(expected_accept_index, -1, device="npu")
    accept_num = torch.zeros_like(expected_accept_num, device="npu")
    next_token = torch.full_like(candidates, -1, device="npu")
    next_sibling = torch.full_like(candidates, -1, device="npu")
    chain_speculative_sampling_triton(
        predicts,
        accept_index,
        accept_num,
        candidates.npu(),
        retrive_index.npu(),
        next_token,
        next_sibling,
        uniforms.npu(),
        final_uniforms.npu(),
        target_probs.npu(),
        draft_probs.npu(),
    )

    torch.testing.assert_close(predicts.cpu(), expected_predicts)
    torch.testing.assert_close(accept_index.cpu(), expected_accept_index)
    torch.testing.assert_close(accept_num.cpu(), expected_accept_num)


def test_chain_rejection_large_vocab_block_loop():
    batch_size, num_draft_tokens, vocab_size = 2, 4, 151552
    candidates = torch.tensor([[0, 11, 22, 33], [0, 44, 55, 66]])
    retrive_index = torch.arange(batch_size * num_draft_tokens).view(
        batch_size, num_draft_tokens
    )
    target_probs = torch.zeros(batch_size, num_draft_tokens, vocab_size)
    draft_probs = torch.zeros(batch_size, num_draft_tokens - 1, vocab_size)

    for row, token in enumerate(candidates[0, 1:]):
        target_probs[0, row, token] = 0.8
        target_probs[0, row, 0] = 0.2
        draft_probs[0, row, token] = 0.5
        draft_probs[0, row, 0] = 0.5
    target_probs[0, -1, 0] = 0.25
    target_probs[0, -1, -1] = 0.75

    target_probs[1, 0, candidates[1, 1]] = 0.1
    target_probs[1, 0, -1] = 0.9
    draft_probs[1, 0, candidates[1, 1]] = 0.9
    draft_probs[1, 0, -1] = 0.1
    for row, token in enumerate(candidates[1, 2:], start=1):
        target_probs[1, row, token] = 1.0
        draft_probs[1, row, token] = 1.0
    target_probs[1, -1, -1] = 1.0

    uniforms = torch.tensor([[0.1, 0.1, 0.1, 0.0], [0.9, 0.0, 0.0, 0.0]])
    final_uniforms = torch.tensor([0.5, 0.5])
    expected_predicts = torch.full(
        (batch_size * num_draft_tokens,), -1, dtype=torch.int32
    )
    expected_accept_index = torch.full(
        (batch_size, num_draft_tokens), -1, dtype=torch.int32
    )
    expected_accept_num = torch.zeros(batch_size, dtype=torch.int32)
    chain_rejection_reference(
        expected_predicts,
        expected_accept_index,
        expected_accept_num,
        candidates,
        retrive_index,
        uniforms,
        final_uniforms,
        target_probs,
        draft_probs,
    )

    predicts = torch.full_like(expected_predicts, -1, device="npu")
    accept_index = torch.full_like(expected_accept_index, -1, device="npu")
    accept_num = torch.zeros_like(expected_accept_num, device="npu")
    chain_speculative_sampling_triton(
        predicts,
        accept_index,
        accept_num,
        candidates.npu(),
        retrive_index.npu(),
        torch.full_like(candidates, -1, device="npu"),
        torch.full_like(candidates, -1, device="npu"),
        uniforms.npu(),
        final_uniforms.npu(),
        target_probs.npu(),
        draft_probs.npu(),
    )

    torch.testing.assert_close(predicts.cpu(), expected_predicts)
    torch.testing.assert_close(accept_index.cpu(), expected_accept_index)
    torch.testing.assert_close(accept_num.cpu(), expected_accept_num)
