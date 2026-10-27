import triton
import triton.language as tl
import triton.language.extra.cann.extension as al


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

    Tree walk (matches ``verify_tree_greedy_native``):
      start at the root (node 0, the already-verified bonus token); at each
      depth descend to the first child and scan siblings until one matches the
      *parent's* target argmax; accept it and descend. The first depth with no
      match stops the walk; the leftover slot stores the parent's argmax as the
      bonus token.

    A chain (topk=1) is just the degenerate tree, so this covers both.
    """
    req_idx = tl.program_id(0)
    row_offset = req_idx * num_draft_tokens
    # accept_index is [bs, max_tree_depth], not [bs, num_draft_tokens]: with a
    # tree the two differ, and using num_draft_tokens here wrote past the row.
    ai_row = req_idx * accept_index_stride
    tokens_offset = tl.arange(0, pad_num_draft_tokens)
    mask_draft_mask = tokens_offset < num_draft_tokens

    cur_retrive_index = tl.load(
        retrive_index + row_offset + tokens_offset, mask=mask_draft_mask, other=0
    )

    # Node 0 is the root: already accepted from the previous iteration.
    last_accepted_idx = al.get_element(cur_retrive_index, (0,))
    tl.store(accept_index + ai_row, last_accepted_idx)

    num_accepted = tl.full((), 0, tl.int32)
    path_active = tl.full((), 1, tl.int32)
    cur_node = tl.full((), 0, tl.int64)

    for _depth in range(1, num_draft_tokens):
        if path_active == 1:
            cur_node = tl.load(retrive_next_token + row_offset + cur_node).to(tl.int64)
            if cur_node == -1:
                path_active = 0

        # Walk the sibling list of this depth until one child is accepted.
        found = tl.full((), 0, tl.int32)
        for _sibling in range(0, num_draft_tokens):
            if (path_active == 1) & (found == 0) & (cur_node != -1):
                # The token to match is always the *parent's* prediction.
                last_local = last_accepted_idx - num_draft_tokens * req_idx
                target_token = tl.load(target_predict + row_offset + last_local).to(
                    tl.int64
                )
                draft_token = tl.load(candidates + row_offset + cur_node).to(tl.int64)

                if draft_token == target_token:
                    draft_idx = tl.load(retrive_index + row_offset + cur_node).to(
                        tl.int64
                    )
                    tl.store(predicts + last_accepted_idx, target_token.to(tl.int32))
                    num_accepted += 1
                    tl.store(
                        accept_index + ai_row + num_accepted,
                        draft_idx.to(tl.int32),
                    )
                    last_accepted_idx = draft_idx
                    found = 1
                else:
                    cur_node = tl.load(
                        retrive_next_sibling + row_offset + cur_node
                    ).to(tl.int64)

        if found == 0:
            path_active = 0

    # Bonus token: the target argmax at the last accepted node.
    last_local = last_accepted_idx - num_draft_tokens * req_idx
    target_token = tl.load(target_predict + row_offset + last_local).to(tl.int64)
    tl.store(predicts + last_accepted_idx, target_token.to(tl.int32))
    tl.store(accept_token_num + req_idx, num_accepted)


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
        accept_index_stride=accept_index.shape[1],
    )
