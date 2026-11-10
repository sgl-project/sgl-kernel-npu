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


def chain_block_verification_reference(
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
    """
    Python oracle for block verification.
    Mirrors the BLOCK_VERIFICATION branch of the GPU reference kernel.
    """
    batch_size, num_draft_tokens = candidates.shape
    for req_idx in range(batch_size):
        # Pass 1: prefix acceptance probability for every step.
        prefix_prob = 1.0
        prefix_probs = []
        for step in range(1, num_draft_tokens):
            draft_token = int(candidates[req_idx, step])
            p = float(target_probs[req_idx, step - 1, draft_token])
            q = float(draft_probs[req_idx, step - 1, draft_token])
            q_is_prob = 0.0 < q <= 1.0
            prefix_prob = min(prefix_prob * p / q, 1.0) if q_is_prob else 0.0
            prefix_probs.append(prefix_prob)

        # Pass 2: backward scan for the longest acceptable prefix.
        num_accepted = 0
        residual_scale = 1.0
        for step in range(num_draft_tokens - 1, 0, -1):
            if num_accepted != 0:
                break
            prefix_prob = prefix_probs[step - 1]
            coin = float(uniform_samples[req_idx, step - 1])
            if coin < prefix_prob:
                accept_prob = prefix_prob
                if step < num_draft_tokens - 1 and 0.0 < prefix_prob < 1.0:
                    residual_mass = float(
                        (
                            prefix_prob * target_probs[req_idx, step]
                            - draft_probs[req_idx, step]
                        )
                        .clamp_min(0.0)
                        .sum()
                    )
                    accept_prob = residual_mass / (
                        residual_mass + 1.0 - prefix_prob
                    )
                if coin < accept_prob:
                    num_accepted = step
                    residual_scale = prefix_prob

        last_accepted_idx = int(retrive_index[req_idx, 0])
        accept_index[req_idx, 0] = last_accepted_idx
        for step in range(1, num_accepted + 1):
            predicts[last_accepted_idx] = int(candidates[req_idx, step])
            last_accepted_idx = int(retrive_index[req_idx, step])
            accept_index[req_idx, step] = last_accepted_idx
        accept_token_num[req_idx] = num_accepted

        # Final sampling from the scaled residual at row num_accepted.
        residual = target_probs[req_idx, num_accepted].clone()
        if num_accepted != num_draft_tokens - 1:
            q = draft_probs[req_idx, num_accepted]
            q = torch.where((q >= 0.0) & (q <= 1.0), q, torch.zeros_like(q))
            residual = (residual_scale * residual - q).clamp_min(0.0)
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


def _assert_block_verification_matches_oracle(
    candidates,
    retrive_index,
    uniforms,
    final_uniforms,
    target_probs,
    draft_probs,
):
    batch_size, num_draft_tokens = candidates.shape
    expected_predicts = torch.full(
        (batch_size * num_draft_tokens,), -1, dtype=torch.int32
    )
    expected_accept_index = torch.full(
        (batch_size, num_draft_tokens), -1, dtype=torch.int32
    )
    expected_accept_num = torch.zeros(batch_size, dtype=torch.int32)
    chain_block_verification_reference(
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
        block_verification=True,
    )

    torch.testing.assert_close(predicts.cpu(), expected_predicts)
    torch.testing.assert_close(accept_index.cpu(), expected_accept_index)
    torch.testing.assert_close(accept_num.cpu(), expected_accept_num)


def test_block_verification_matches_gpu_algorithm():
    """Small-vocab case covering one distinct branch per request:
    req 0: mid-chain accept through the residual-mass path;
    req 1: invalid draft probability collapses the prefix, nothing accepted;
    req 2: prefix probabilities are exactly 1.0, all drafts accepted;
    req 3: every step rejected, residual sampled from row 0 with scale 1.0.
    """
    batch_size, num_draft_tokens, vocab_size = 4, 4, 11
    candidates = torch.tensor(
        [[0, 2, 3, 4], [0, 5, 6, 7], [0, 8, 9, 10], [0, 1, 2, 3]]
    )
    retrive_index = torch.arange(batch_size * num_draft_tokens).view(
        batch_size, num_draft_tokens
    )
    target_probs = torch.zeros(batch_size, num_draft_tokens, vocab_size)
    draft_probs = torch.zeros(batch_size, num_draft_tokens - 1, vocab_size)

    # req 0: prefix = [0.5, 0.3, 0.15]; coins accept step 1 via residual mass.
    target_probs[0, 0, 2] = 0.2
    target_probs[0, 0, 0] = 0.8
    draft_probs[0, 0, 2] = 0.4
    draft_probs[0, 0, 0] = 0.6
    target_probs[0, 1, 3] = 0.3
    target_probs[0, 1, 0] = 0.7
    draft_probs[0, 1, 3] = 0.1
    draft_probs[0, 1, 0] = 0.9
    target_probs[0, 2, 4] = 0.05
    target_probs[0, 2, 0] = 0.95
    draft_probs[0, 2, 4] = 0.1
    draft_probs[0, 2, 0] = 0.9

    # req 1: draft row 1 has q == 0 (invalid), collapsing prefix from step 2 on;
    # step 1 survives the residual-mass coin flip only to be rejected, so the
    # final sample falls back to the scale-1.0 residual at row 0.
    target_probs[1, 0, 5] = 0.3
    target_probs[1, 0, 0] = 0.7
    draft_probs[1, 0, 5] = 0.6
    draft_probs[1, 0, 0] = 0.4
    target_probs[1, 1, 6] = 0.5
    target_probs[1, 1, 0] = 0.5
    draft_probs[1, 1, 6] = 0.0
    draft_probs[1, 1, 0] = 1.0

    # req 2: p == q on every proposed token, so all prefixes are exactly 1.0
    # and the last step accepts unconditionally (residual = pure target p).
    for row, token in enumerate([8, 9, 10]):
        target_probs[2, row, token] = 0.4
        target_probs[2, row, 0] = 0.6
        draft_probs[2, row, token] = 0.4
        draft_probs[2, row, 0] = 0.6
    target_probs[2, 3, 0] = 0.6
    target_probs[2, 3, 10] = 0.4

    # req 3: every coin exceeds its prefix probability; nothing is accepted.
    for row, token in enumerate([1, 2, 3]):
        target_probs[3, row, token] = 0.1
        target_probs[3, row, 0] = 0.9
        draft_probs[3, row, token] = 0.2
        draft_probs[3, row, 0] = 0.8

    uniforms = torch.tensor(
        [
            [0.02, 0.9, 0.9, 0.0],
            [0.3, 0.1, 0.05, 0.0],
            [0.2, 0.2, 0.2, 0.0],
            [0.99, 0.99, 0.99, 0.0],
        ]
    )
    final_uniforms = torch.tensor([0.5, 0.5, 0.4, 0.5])

    _assert_block_verification_matches_oracle(
        candidates,
        retrive_index,
        uniforms,
        final_uniforms,
        target_probs,
        draft_probs,
    )


def test_block_verification_large_vocab_block_loop():
    """Vocab-block loop coverage: req 0 accepts mid-chain so the residual-mass
    reduction and the final CDF sampling both stride across vocab blocks and
    sample a token near the end of the vocabulary; req 1 accepts every draft.
    """
    batch_size, num_draft_tokens, vocab_size = 2, 4, 151552
    candidates = torch.tensor([[0, 11, 22, 33], [0, 44, 55, 66]])
    retrive_index = torch.arange(batch_size * num_draft_tokens).view(
        batch_size, num_draft_tokens
    )
    target_probs = torch.zeros(batch_size, num_draft_tokens, vocab_size)
    draft_probs = torch.zeros(batch_size, num_draft_tokens - 1, vocab_size)

    # req 0: prefix = [0.5, 0.25, 0.125]; step 1 accepted through the
    # residual-mass path; the residual mass sits at the last vocab entry.
    target_probs[0, 0, 11] = 0.25
    target_probs[0, 0, 0] = 0.75
    draft_probs[0, 0, 11] = 0.5
    draft_probs[0, 0, 0] = 0.5
    target_probs[0, 1, 22] = 0.05
    target_probs[0, 1, 0] = 0.05
    target_probs[0, 1, -1] = 0.9
    draft_probs[0, 1, 22] = 0.1
    draft_probs[0, 1, 0] = 0.4
    draft_probs[0, 1, -1] = 0.4
    target_probs[0, 2, 33] = 0.25
    target_probs[0, 2, 0] = 0.75
    draft_probs[0, 2, 33] = 0.5
    draft_probs[0, 2, 0] = 0.5

    # req 1: p == q on every proposed token; all drafts accepted.
    for row, token in enumerate([44, 55, 66]):
        target_probs[1, row, token] = 0.5
        target_probs[1, row, 0] = 0.5
        draft_probs[1, row, token] = 0.5
        draft_probs[1, row, 0] = 0.5
    target_probs[1, 3, 66] = 0.7
    target_probs[1, 3, 0] = 0.3

    uniforms = torch.tensor([[0.04, 0.9, 0.9, 0.0], [0.3, 0.3, 0.3, 0.0]])
    final_uniforms = torch.tensor([0.5, 0.5])

    _assert_block_verification_matches_oracle(
        candidates,
        retrive_index,
        uniforms,
        final_uniforms,
        target_probs,
        draft_probs,
    )


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
