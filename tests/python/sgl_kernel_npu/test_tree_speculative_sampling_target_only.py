import argparse
import time

import pytest
import torch
import torch_npu  # noqa: F401
from sgl_kernel_npu.sample import tree_speculative_sampling_target_only


def target_only_tree_reference(
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    retrive_index,
    retrive_next_token,
    retrive_next_sibling,
    uniform_samples,
    uniform_samples_for_final_sampling,
    target_probs,
    rejected_probs,
    threshold_single,
    threshold_acc,
):
    """CPU reference translated directly from the GPU CUDA kernel."""
    batch_size, num_draft_tokens = candidates.shape
    num_speculative_tokens = accept_index.shape[1]
    threshold_acc = max(float(threshold_acc), 1e-9)
    rejected_probs.zero_()

    for req_idx in range(batch_size):
        cur_prob_row = 0
        cur_node = 0
        coin = float(uniform_samples[req_idx, 0])
        last_accepted_idx = int(retrive_index[req_idx, 0])
        accept_index[req_idx, 0] = last_accepted_idx
        num_accepted = 0

        for _ in range(1, num_speculative_tokens):
            cur_node = int(retrive_next_token[req_idx, cur_node])
            prob_acc = 0.0
            while cur_node != -1:
                draft_idx = int(retrive_index[req_idx, cur_node])
                draft_token = int(candidates[req_idx, cur_node])
                target_prob = float(target_probs[req_idx, cur_prob_row, draft_token])
                prob_acc += target_prob
                if coin <= prob_acc / threshold_acc or target_prob >= threshold_single:
                    predicts[last_accepted_idx] = draft_token
                    num_accepted += 1
                    accept_index[req_idx, num_accepted] = draft_idx
                    last_accepted_idx = draft_idx
                    cur_prob_row = cur_node
                    coin = float(uniform_samples[req_idx, cur_node])
                    break

                rejected_probs[req_idx, cur_prob_row, draft_token] = target_prob
                cur_node = int(retrive_next_sibling[req_idx, cur_node])

            if cur_node == -1:
                break

        accept_token_num[req_idx] = num_accepted
        residual = (
            target_probs[req_idx, cur_prob_row] - rejected_probs[req_idx, cur_prob_row]
        ).clamp_min(0.0)
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

    return predicts, accept_index, accept_token_num, rejected_probs


def target_only_chain_reference(
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    retrive_index,
    uniform_samples,
    uniform_samples_for_final_sampling,
    target_probs,
    threshold_single,
    threshold_acc,
):
    batch_size, num_draft_tokens = candidates.shape
    threshold_acc = max(float(threshold_acc), 1e-9)

    for req_idx in range(batch_size):
        last_accepted_idx = int(retrive_index[req_idx, 0])
        accept_index[req_idx, 0] = last_accepted_idx
        num_accepted = 0
        rejected_token = -1

        for step in range(1, num_draft_tokens):
            draft_token = int(candidates[req_idx, step])
            target_prob = float(target_probs[req_idx, step - 1, draft_token])
            coin = float(uniform_samples[req_idx, step - 1])
            if coin <= target_prob / threshold_acc or target_prob >= threshold_single:
                predicts[last_accepted_idx] = draft_token
                num_accepted += 1
                last_accepted_idx = int(retrive_index[req_idx, step])
                accept_index[req_idx, num_accepted] = last_accepted_idx
            else:
                rejected_token = draft_token
                break

        accept_token_num[req_idx] = num_accepted
        final_probs = target_probs[req_idx, num_accepted].clone().float()
        if rejected_token >= 0:
            final_probs[rejected_token] = 0.0
        final_probs.clamp_min_(0.0)
        target = float(uniform_samples_for_final_sampling[req_idx]) * float(
            final_probs.sum()
        )
        sampled_token = int((final_probs.cumsum(0) <= target).sum())
        sampled_token = min(sampled_token, final_probs.numel() - 1)
        predicts[last_accepted_idx] = sampled_token

    return predicts, accept_index, accept_token_num


@pytest.mark.parametrize(
    "threshold_single,threshold_acc", [(1.0, 1.0), (0.0, 0.0), (0.5, 0.8)]
)
def test_general_tree_matches_gpu_algorithm_reference(threshold_single, threshold_acc):
    candidates = torch.tensor(
        [[0, 1, 2, 3, 4, 5], [7, 8, 9, 10, 11, 12]], dtype=torch.int64
    )
    retrive_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]], dtype=torch.int64
    )
    retrive_next_token = torch.tensor(
        [[1, 2, -1, 4, 5, -1], [4, 2, 3, -1, 5, -1]],
        dtype=torch.int64,
    )
    retrive_next_sibling = torch.tensor(
        [[-1, 3, -1, -1, -1, -1], [-1, -1, -1, -1, 1, -1]],
        dtype=torch.int64,
    )
    batch_size, num_draft_tokens = candidates.shape
    vocab_size = 20
    target_probs = torch.full(
        (batch_size, num_draft_tokens, vocab_size), 0.01, dtype=torch.float32
    )
    target_probs[0, 0, 1] = 0.12
    target_probs[0, 0, 3] = 0.72
    target_probs[0, 3, 4] = 0.82
    target_probs[0, 4, 5] = 0.75
    target_probs[1, 0, 11] = 0.68
    target_probs[1, 0, 8] = 0.14
    target_probs[1, 4, 12] = 0.77
    target_probs /= target_probs.sum(dim=-1, keepdim=True)
    uniforms = torch.tensor(
        [[0.55, 0.2, 0.8, 0.4, 0.3, 0.9], [0.6, 0.2, 0.8, 0.7, 0.3, 0.4]],
        dtype=torch.float32,
    )
    final_uniforms = torch.tensor([0.25, 0.75], dtype=torch.float32)

    ref_predicts = torch.full((12,), -1, dtype=torch.int32)
    ref_accept_index = torch.full((2, 4), -1, dtype=torch.int32)
    ref_accept_num = torch.zeros(2, dtype=torch.int32)
    ref_rejected = torch.empty_like(target_probs)
    target_only_tree_reference(
        ref_predicts,
        ref_accept_index,
        ref_accept_num,
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        uniforms,
        final_uniforms,
        target_probs,
        ref_rejected,
        threshold_single,
        threshold_acc,
    )

    npu_predicts = torch.full_like(ref_predicts, -1, device="npu")
    npu_accept_index = torch.full_like(ref_accept_index, -1, device="npu")
    npu_accept_num = torch.zeros_like(ref_accept_num, device="npu")
    npu_rejected = torch.empty_like(target_probs, device="npu")
    tree_speculative_sampling_target_only(
        npu_predicts,
        npu_accept_index,
        npu_accept_num,
        candidates.npu(),
        retrive_index.npu(),
        retrive_next_token.npu(),
        retrive_next_sibling.npu(),
        uniforms.npu(),
        final_uniforms.npu(),
        target_probs.npu(),
        npu_rejected,
        threshold_single,
        threshold_acc,
        True,
    )

    torch.testing.assert_close(npu_predicts.cpu(), ref_predicts, rtol=0, atol=0)
    torch.testing.assert_close(npu_accept_index.cpu(), ref_accept_index, rtol=0, atol=0)
    torch.testing.assert_close(npu_accept_num.cpu(), ref_accept_num, rtol=0, atol=0)
    torch.testing.assert_close(npu_rejected.cpu(), ref_rejected, rtol=0, atol=0)


def target_only_chain_torch(
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    retrive_index,
    uniform_samples,
    uniform_samples_for_final_sampling,
    target_probs,
):
    batch_size, num_draft_tokens = candidates.shape
    device = candidates.device
    draft_tokens = candidates[:, 1:].long()
    step_probs = torch.gather(
        target_probs[:, :-1, :], 2, draft_tokens.unsqueeze(-1)
    ).squeeze(-1)
    accept_steps = uniform_samples[:, : num_draft_tokens - 1] <= step_probs
    reject_count = (~accept_steps).to(torch.int32).cumsum(dim=1)
    num_correct = (reject_count == 0).to(torch.int32).sum(dim=1)

    accept_token_num.copy_(num_correct)
    accept_index.fill_(-1)
    positions = torch.arange(num_draft_tokens, device=device).view(1, -1)
    valid_accept = positions <= num_correct.view(-1, 1)
    accept_index.copy_(
        torch.where(
            valid_accept,
            retrive_index.to(torch.int32),
            torch.full_like(accept_index, -1),
        )
    )

    predicts.zero_()
    parent_positions = torch.arange(num_draft_tokens - 1, device=device).view(1, -1)
    valid_parent = parent_positions < num_correct.view(-1, 1)
    parent_indices = retrive_index[:, :-1].reshape(-1).long()
    parent_values = candidates[:, 1:].to(torch.int32).reshape(-1)
    predicts[parent_indices] = torch.where(
        valid_parent.reshape(-1), parent_values, predicts[parent_indices]
    )

    rows = torch.arange(batch_size, device=device)
    final_rows = num_correct.long()
    final_probs = target_probs[rows, final_rows].clone()
    rejected = num_correct < num_draft_tokens - 1
    rejected_positions = (num_correct.long() + 1).clamp_max(num_draft_tokens - 1)
    rejected_tokens = candidates[rows, rejected_positions].long()
    final_probs[rejected, rejected_tokens[rejected]] = 0.0

    probability_sums = final_probs.sum(dim=-1, keepdim=True)
    targets = uniform_samples_for_final_sampling.view(-1, 1) * probability_sums
    final_tokens = (
        (final_probs.cumsum(dim=-1) <= targets)
        .to(torch.int32)
        .sum(dim=-1)
        .clamp_max(target_probs.shape[-1] - 1)
    )
    final_indices = retrive_index[rows, final_rows].long()
    predicts[final_indices] = final_tokens.to(torch.int32)


def make_chain_indices(batch_size, num_draft_tokens, device):
    retrive_index = torch.arange(
        batch_size * num_draft_tokens, dtype=torch.int64, device=device
    ).view(batch_size, num_draft_tokens)
    retrive_next_token = torch.arange(
        1, num_draft_tokens + 1, dtype=torch.int64, device=device
    ).repeat(batch_size, 1)
    retrive_next_token[:, -1] = -1
    retrive_next_sibling = torch.full_like(retrive_next_token, -1)
    return retrive_index, retrive_next_token, retrive_next_sibling


def make_stable_chain_final_uniforms(
    candidates,
    uniform_samples,
    target_probs,
):
    """Choose final-sampling coins away from inverse-CDF boundaries."""
    batch_size, num_draft_tokens = candidates.shape
    final_uniforms = torch.empty(batch_size, dtype=torch.float32)

    for req_idx in range(batch_size):
        num_accepted = 0
        rejected_token = -1
        for step in range(1, num_draft_tokens):
            draft_token = int(candidates[req_idx, step])
            target_prob = float(target_probs[req_idx, step - 1, draft_token])
            if float(uniform_samples[req_idx, step - 1]) <= target_prob:
                num_accepted += 1
            else:
                rejected_token = draft_token
                break

        final_probs = target_probs[req_idx, num_accepted].double().clone()
        if rejected_token >= 0:
            final_probs[rejected_token] = 0.0

        sampled_token = int(final_probs.argmax())
        probability_sum = final_probs.sum()
        cdf_before = final_probs[:sampled_token].sum()
        cdf_midpoint = cdf_before + final_probs[sampled_token] * 0.5
        final_uniforms[req_idx] = (cdf_midpoint / probability_sum).float()

    return final_uniforms


@pytest.mark.parametrize("batch_size", [1, 4, 17])
@pytest.mark.parametrize("num_draft_tokens", [2, 5])
@pytest.mark.parametrize("vocab_size", [20, 32000, 151552])
def test_target_only_chain_matches_reference(batch_size, num_draft_tokens, vocab_size):
    torch.manual_seed(20260717 + batch_size + num_draft_tokens + vocab_size)
    candidates = torch.randint(
        0, vocab_size, (batch_size, num_draft_tokens), dtype=torch.int64
    )
    logits = torch.randn(batch_size, num_draft_tokens, vocab_size)
    target_probs = torch.softmax(logits, dim=-1).float()

    # Give some draft tokens meaningful acceptance probability.
    for req_idx in range(batch_size):
        for step in range(1, num_draft_tokens):
            token = int(candidates[req_idx, step])
            target_probs[req_idx, step - 1] *= 0.35
            target_probs[req_idx, step - 1, token] += 0.65
            target_probs[req_idx, step - 1] /= target_probs[req_idx, step - 1].sum()

    uniform_samples = torch.rand(batch_size, num_draft_tokens)
    # A random coin can land within FP32 reduction error of a CDF boundary for
    # large vocabularies. Use the midpoint of a high-mass token's interval so
    # exact token equality tests the algorithm instead of reduction order.
    final_uniform_samples = make_stable_chain_final_uniforms(
        candidates,
        uniform_samples,
        target_probs,
    )
    retrive_index, retrive_next_token, retrive_next_sibling = make_chain_indices(
        batch_size, num_draft_tokens, "cpu"
    )

    ref_predicts = torch.full((batch_size * num_draft_tokens,), -1, dtype=torch.int32)
    ref_accept_index = torch.full((batch_size, num_draft_tokens), -1, dtype=torch.int32)
    ref_accept_num = torch.zeros(batch_size, dtype=torch.int32)
    target_only_chain_reference(
        ref_predicts,
        ref_accept_index,
        ref_accept_num,
        candidates,
        retrive_index,
        uniform_samples,
        final_uniform_samples,
        target_probs,
        1.0,
        1.0,
    )

    npu_predicts = torch.full_like(ref_predicts, -1, device="npu")
    npu_accept_index = torch.full_like(ref_accept_index, -1, device="npu")
    npu_accept_num = torch.zeros_like(ref_accept_num, device="npu")
    candidates_npu = candidates.npu()
    retrive_index_npu = retrive_index.npu()
    next_token_npu = retrive_next_token.npu()
    next_sibling_npu = retrive_next_sibling.npu()
    target_probs_npu = target_probs.npu()

    tree_speculative_sampling_target_only(
        predicts=npu_predicts,
        accept_index=npu_accept_index,
        accept_token_num=npu_accept_num,
        candidates=candidates_npu,
        retrive_index=retrive_index_npu,
        retrive_next_token=next_token_npu,
        retrive_next_sibling=next_sibling_npu,
        uniform_samples=uniform_samples.npu(),
        uniform_samples_for_final_sampling=final_uniform_samples.npu(),
        target_probs=target_probs_npu,
        draft_probs=torch.empty_like(target_probs_npu),
        threshold_single=1.0,
        threshold_acc=1.0,
        deterministic=True,
    )

    torch.testing.assert_close(npu_predicts.cpu(), ref_predicts, rtol=0, atol=0)
    torch.testing.assert_close(npu_accept_index.cpu(), ref_accept_index, rtol=0, atol=0)
    torch.testing.assert_close(npu_accept_num.cpu(), ref_accept_num, rtol=0, atol=0)


@pytest.mark.parametrize(
    "threshold_single,threshold_acc",
    [(1.0, 1.0), (0.0, 0.0), (0.5, 0.8)],
)
def test_target_only_thresholds(threshold_single, threshold_acc):
    batch_size, num_draft_tokens, vocab_size = 2, 4, 32
    candidates = torch.tensor([[0, 3, 4, 5], [0, 7, 8, 9]], dtype=torch.int64)
    target_probs = torch.full(
        (batch_size, num_draft_tokens, vocab_size), 1.0 / vocab_size
    )
    for req_idx in range(batch_size):
        for step in range(1, num_draft_tokens):
            token = int(candidates[req_idx, step])
            target_probs[req_idx, step - 1] *= 0.2
            target_probs[req_idx, step - 1, token] += 0.8
            target_probs[req_idx, step - 1] /= target_probs[req_idx, step - 1].sum()

    uniforms = torch.tensor([[0.1, 0.9, 0.2, 0.0], [0.7, 0.2, 0.95, 0.0]])
    final_uniforms = torch.tensor([0.25, 0.75])
    retrive_index, next_token, next_sibling = make_chain_indices(
        batch_size, num_draft_tokens, "cpu"
    )

    ref_predicts = torch.full((batch_size * num_draft_tokens,), -1, dtype=torch.int32)
    ref_accept_index = torch.full((batch_size, num_draft_tokens), -1, dtype=torch.int32)
    ref_accept_num = torch.zeros(batch_size, dtype=torch.int32)
    target_only_chain_reference(
        ref_predicts,
        ref_accept_index,
        ref_accept_num,
        candidates,
        retrive_index,
        uniforms,
        final_uniforms,
        target_probs,
        threshold_single,
        threshold_acc,
    )

    npu_predicts = torch.full_like(ref_predicts, -1, device="npu")
    npu_accept_index = torch.full_like(ref_accept_index, -1, device="npu")
    npu_accept_num = torch.zeros_like(ref_accept_num, device="npu")
    target_probs_npu = target_probs.npu()
    tree_speculative_sampling_target_only(
        npu_predicts,
        npu_accept_index,
        npu_accept_num,
        candidates.npu(),
        retrive_index.npu(),
        next_token.npu(),
        next_sibling.npu(),
        uniforms.npu(),
        final_uniforms.npu(),
        target_probs_npu,
        torch.empty_like(target_probs_npu),
        threshold_single,
        threshold_acc,
        True,
    )

    torch.testing.assert_close(npu_predicts.cpu(), ref_predicts, rtol=0, atol=0)
    torch.testing.assert_close(npu_accept_index.cpu(), ref_accept_index, rtol=0, atol=0)
    torch.testing.assert_close(npu_accept_num.cpu(), ref_accept_num, rtol=0, atol=0)


def run_benchmark(batch_size, num_draft_tokens, vocab_size, warmup, iterations):
    candidates = torch.randint(
        0,
        vocab_size,
        (batch_size, num_draft_tokens),
        dtype=torch.int64,
        device="npu",
    )
    target_probs = torch.softmax(
        torch.randn(
            batch_size,
            num_draft_tokens,
            vocab_size,
            dtype=torch.float32,
            device="npu",
        ),
        dim=-1,
    )
    retrive_index, next_token, next_sibling = make_chain_indices(
        batch_size, num_draft_tokens, "npu"
    )
    uniforms = torch.rand(
        batch_size, num_draft_tokens, dtype=torch.float32, device="npu"
    )
    final_uniforms = torch.rand(batch_size, dtype=torch.float32, device="npu")
    draft_probs = torch.empty_like(target_probs)
    predicts = torch.zeros(
        batch_size * num_draft_tokens, dtype=torch.int32, device="npu"
    )
    accept_index = torch.full(
        (batch_size, num_draft_tokens), -1, dtype=torch.int32, device="npu"
    )
    accept_num = torch.zeros(batch_size, dtype=torch.int32, device="npu")

    def run_kernel():
        tree_speculative_sampling_target_only(
            predicts,
            accept_index,
            accept_num,
            candidates,
            retrive_index,
            next_token,
            next_sibling,
            uniforms,
            final_uniforms,
            target_probs,
            draft_probs,
            1.0,
            1.0,
            True,
        )

    def run_torch():
        target_only_chain_torch(
            predicts,
            accept_index,
            accept_num,
            candidates,
            retrive_index,
            uniforms,
            final_uniforms,
            target_probs,
        )

    def benchmark(fn):
        for _ in range(warmup):
            fn()
        torch.npu.synchronize()
        started = time.perf_counter()
        for _ in range(iterations):
            fn()
        torch.npu.synchronize()
        return (time.perf_counter() - started) * 1000 / iterations

    kernel_latency_ms = benchmark(run_kernel)
    torch_latency_ms = benchmark(run_torch)
    print(
        f"batch={batch_size} drafts={num_draft_tokens} vocab={vocab_size} "
        f"kernel_ms={kernel_latency_ms:.4f} torch_ms={torch_latency_ms:.4f} "
        f"speedup={torch_latency_ms / kernel_latency_ms:.2f}x"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--perf", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-draft-tokens", type=int, default=5)
    parser.add_argument("--vocab-size", type=int, default=151552)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    if args.perf:
        run_benchmark(
            args.batch_size,
            args.num_draft_tokens,
            args.vocab_size,
            args.warmup,
            args.iterations,
        )
    else:
        raise SystemExit(pytest.main([__file__]))
