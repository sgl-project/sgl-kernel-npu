def reconstruct_indices_from_tree_mask(
    tree_mask,
    verified_seq_len,
    positions,
    retrive_index,
    retrive_next_token,
    retrive_next_sibling,
    batch_size,
    draft_token_num,
):
    if not batch_size:
        return
    if draft_token_num <= 0:
        raise ValueError("draft_token_num must be positive")
    d = draft_token_num
    mask = tree_mask.reshape(batch_size, d, d).cpu().tolist()
    lengths = verified_seq_len.cpu().tolist()
    outputs = [[], [], [], []]
    for b in range(batch_size):
        for i in range(d):
            ancestors = [j for j in range(i) if mask[b][i][j]]
            parent = ancestors[-1] if ancestors else -1
            child = next((j for j in range(i + 1, d) if mask[b][j][i]), -1)
            sibling = next(
                (
                    j
                    for j in range(i + 1, d)
                    if parent >= 0
                    and mask[b][j][parent]
                    and not any(mask[b][j][parent + 1 : j])
                ),
                -1,
            )
            for out, val in zip(
                outputs, (lengths[b] + len(ancestors), b * d + i, child, sibling)
            ):
                out.append(val)
    for dst, vals in zip(
        (positions, retrive_index, retrive_next_token, retrive_next_sibling), outputs
    ):
        dst.copy_(dst.new_tensor(vals).reshape(dst.shape))
