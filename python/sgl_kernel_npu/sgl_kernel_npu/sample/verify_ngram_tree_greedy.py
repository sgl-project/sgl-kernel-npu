"""Greedy verification for NGRAM irregular trees."""

import triton
import triton.language as tl


@triton.jit
def verify_ngram_tree_greedy_kernel(
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    retrive_index,
    retrive_next_token,
    retrive_next_sibling,
    target_predict,
    num_draft_tokens: tl.constexpr,
    accept_index_stride: tl.constexpr,
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
    base = req_idx * num_draft_tokens
    last_accepted_idx = tl.load(retrive_index + base)
    accept_base = req_idx * accept_index_stride
    tl.store(accept_index + accept_base, last_accepted_idx)
    num_accepted = 0

    current = 0
    child = tl.load(retrive_next_token + base).to(tl.int32)
    while child != -1 and num_accepted < accept_index_stride - 1:
        target_token = tl.load(target_predict + last_accepted_idx)
        selected = -1
        # Children need not be adjacent in the flattened candidate array.
        # In particular, NGRAM merges several independently matched subtrees.
        while child != -1 and selected == -1:
            draft_token = tl.load(candidates + base + child)
            if draft_token == target_token:
                selected = child
            else:
                child = tl.load(retrive_next_sibling + base + child).to(tl.int32)
        if selected != -1:
            tl.store(predicts + last_accepted_idx, target_token)
            num_accepted += 1
            last_accepted_idx = tl.load(retrive_index + base + selected)
            tl.store(accept_index + accept_base + num_accepted, last_accepted_idx)
            current = selected
            child = tl.load(retrive_next_token + base + current).to(tl.int32)
    target_token = tl.load(target_predict + last_accepted_idx)
    tl.store(accept_token_num + req_idx, num_accepted)
    tl.store(predicts + last_accepted_idx, target_token)


def verify_ngram_tree_greedy(
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
    if not bs:
        return
    if accept_index.shape[1] == 1:
        # Avoid a zero-trip nested loop in Ascend's Triton lowering. This also
        # covers root-only verification (draft_token_num=1).
        indices = retrive_index[:, 0].long()
        accept_index[:, 0].copy_(indices)
        accept_token_num.zero_()
        predicts[indices] = target_predict.reshape(-1)[indices].to(predicts.dtype)
        return
    verify_ngram_tree_greedy_kernel[(bs,)](
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        target_predict=target_predict,
        num_draft_tokens=num_draft_tokens,
        accept_index_stride=accept_index.shape[1],
    )
