import triton
import triton.language as tl


@triton.jit
def verify_tree_greedy_kernel(
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    retrive_index,
    retrive_next_token,
    retrive_next_sibling,
    target_predict,
    num_draft_tokens: tl.constexpr,
    pad_num_draft_tokens: tl.constexpr,
):
    """
    predicts: [tot_num_draft_tokens] -> ((bs * (self.spec_steps + 1))
    accept_index: [bs, num_spec_step + 1]
    accept_token_num: [bs]
    candidates: [bs, num_draft_tokens]
    retrive_index: [bs, num_draft_tokens]
    retrive_next_token: [bs, num_draft_tokens]
    retrive_next_sibling: [bs, num_draft_tokens]
    target_predict: [bs, num_draft_tokens]
    """
    req_idx = tl.program_id(0)
    tokens_offset = tl.arange(0, pad_num_draft_tokens)
    cur_tokens_offset = req_idx * num_draft_tokens + tokens_offset
    mask_draft_mask = tokens_offset < num_draft_tokens
    cur_candidates = tl.load(
        candidates + cur_tokens_offset, mask=mask_draft_mask, other=0
    )
    cur_retrive_index = tl.load(
        retrive_index + cur_tokens_offset, mask=mask_draft_mask, other=0
    )
    cur_target = tl.load(
        target_predict + cur_tokens_offset, mask=mask_draft_mask, other=0
    )

    last_accepted_idx = tl.get_element(cur_retrive_index, (0,))
    tl.store(accept_index + req_idx * num_draft_tokens, last_accepted_idx)
    num_accepted = 0

    rejected = False
    for i in range(1, num_draft_tokens):
        if not rejected:
            draft_token = tl.get_element(cur_candidates, (i,))
            target_token = tl.get_element(cur_target, (i - 1,))

            if draft_token == target_token:
                draft_idx = tl.get_element(cur_retrive_index, (i,))
                tl.store(predicts + last_accepted_idx, target_token)
                num_accepted += 1
                tl.store(
                    accept_index + req_idx * num_draft_tokens + num_accepted, draft_idx
                )
                last_accepted_idx = draft_idx
            else:
                rejected = True
    target_token = tl.get_element(
        cur_target, (last_accepted_idx - num_draft_tokens * req_idx,)
    )
    tl.store(accept_token_num + req_idx, num_accepted)
    tl.store(predicts + last_accepted_idx, target_token)


def verify_tree_greedy(
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    retrive_index,
    retrive_next_token,
    retrive_next_sibling,
    target_predict,
):
    bs = candidates.shape[0]
    num_draft_tokens = candidates.shape[1]
    verify_tree_greedy_kernel[(bs,)](
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        target_predict=target_predict,
        num_draft_tokens=num_draft_tokens,
        pad_num_draft_tokens=triton.next_power_of_2(num_draft_tokens),
    )
