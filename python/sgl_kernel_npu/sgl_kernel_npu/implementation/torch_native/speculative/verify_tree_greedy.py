import torch


def _verify_tree_greedy_native(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    target_predict: torch.Tensor,
    topk: int = -1,
):
    batch_size, num_draft_tokens = candidates.shape

    # Optimized common case for performance.
    if num_draft_tokens == 2 and accept_index.shape[1] == 2 and topk == 1:
        comparison_result = candidates[:, 1] == target_predict[:, 0]

        predicts = target_predict.flatten()

        accept_index = torch.arange(
            0,
            num_draft_tokens * batch_size,
            device=candidates.device,
            dtype=torch.int32,
        ).reshape(batch_size, num_draft_tokens)
        comparison_result = comparison_result.to(torch.int64)
        accept_index_mask = accept_index[:, 1] * comparison_result
        accept_index[:, 1] = accept_index_mask - (1 - comparison_result)

        accept_token_num = comparison_result.int()
        return predicts, accept_index, accept_token_num

    # BFS
    for bx in range(batch_size):
        cur_candidates = candidates[bx]
        cur_retrive_index = retrive_index[bx]
        cur_next_token = retrive_next_token[bx]
        cur_next_sibling = retrive_next_sibling[bx]
        cur_target = target_predict[bx]

        last_accepted_idx = cur_retrive_index[0]
        accept_index[bx, 0] = last_accepted_idx
        num_accepted = 0
        cur_node = 0

        for _ in range(1, num_draft_tokens):
            cur_node = cur_next_token[cur_node]
            found = False
            while cur_node != -1:
                draft_idx = cur_retrive_index[cur_node]
                draft_token = cur_candidates[cur_node]
                target_token = cur_target[last_accepted_idx - num_draft_tokens * bx]

                # Compare the draft_token with the target_token
                # If they match, update the prediction result, accept the index, and the number of accepted markers
                # If no match is found, terminate the loop.
                if draft_token == target_token:
                    predicts[last_accepted_idx] = target_token
                    num_accepted += 1
                    accept_index[bx, num_accepted] = draft_idx
                    last_accepted_idx = draft_idx
                    found = True
                    break
                else:
                    cur_node = cur_next_sibling[cur_node]
            if not found:
                break

        # Update accept_token_num and predicts
        accept_token_num[bx] = num_accepted
        predicts[last_accepted_idx] = cur_target[
            last_accepted_idx - num_draft_tokens * bx
        ]
    return predicts, accept_index, accept_token_num


def verify_tree_greedy_torch(
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    retrive_index,
    retrive_next_token,
    retrive_next_sibling,
    target_predict,
):
    """Run the existing native default path, adapting its tuple return to None."""
    # The original default topk=-1 uses in-place traversal. Its topk=1 fast path
    # allocates replacement tensors, which is a different output contract.
    _verify_tree_greedy_native(
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        target_predict,
    )
