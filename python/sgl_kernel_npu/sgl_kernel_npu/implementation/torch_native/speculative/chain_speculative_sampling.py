import torch


def chain_speculative_sampling_torch(
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
    """Independent PyTorch reference for classic chain rejection sampling."""
    del retrive_next_token, retrive_next_sibling
    del threshold_single, threshold_acc, deterministic

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
