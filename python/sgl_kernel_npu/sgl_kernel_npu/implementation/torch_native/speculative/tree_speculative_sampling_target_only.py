import torch


def tree_speculative_sampling_target_only_torch(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    uniform_samples: torch.Tensor,
    uniform_samples_for_final_sampling: torch.Tensor,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    threshold_single: float = 1.0,
    threshold_acc: float = 1.0,
    deterministic: bool = True,
) -> None:
    """Independent PyTorch reference for target-only tree sampling."""
    del deterministic
    rejected_probs = draft_probs

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
        # Roundoff near a CDF boundary can change the sampled token. Use the
        # same FP64 cumulative sum for the total and the token comparison.
        cdf = residual.cumsum(0, dtype=torch.float64)
        target = float(uniform_samples_for_final_sampling[req_idx]) * float(cdf[-1])
        sampled_token = int((cdf <= target).sum())
        if sampled_token == residual.numel():
            positive = torch.nonzero(residual > 0.0).flatten()
            sampled_token = (
                int(positive[-1]) if positive.numel() else residual.numel() - 1
            )
        predicts[last_accepted_idx] = sampled_token
