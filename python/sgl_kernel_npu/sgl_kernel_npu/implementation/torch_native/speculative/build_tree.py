"""Independent verification-tree reference, with the low-level in-place API.

This implementation derives parent edges, depths and ancestor masks directly;
it does not call a custom kernel or the old speculative.py reference.
QLEN_ONLY follows the mathematical layout, exposing the retained device bug.
"""

import torch


def _raw_span(tensor):
    # Preserve the old pointer-based storage interpretation, including offsets.
    # No contiguous temporary or copyback is introduced for mutable buffers.
    return tensor.as_strided((tensor.numel(),), (1,))


def build_tree(
    parent_list,
    selected_index,
    verified_seq_len,
    tree_mask,
    positions,
    retrive_index,
    retrive_next_token,
    retrive_next_sibling,
    topk,
    depth,
    draft_token_num,
    tree_mask_mode,
):
    """Write the ancestor-closed tree into the five supplied buffers.

    For node i>0, q=selected_index[i-1]//topk identifies its parent.
    q=0 denotes the root; otherwise parent_list[q] is searched in selected_index.
    Valid inputs require unique selected IDs and all ancestors to be present.
    Malformed trees have no defined legacy result and are not a reference case.
    """
    if tree_mask_mode == 2:
        raise NotImplementedError("QLEN_ONLY_BITPACKING is not implemented")
    tensors = (
        parent_list,
        selected_index,
        verified_seq_len,
        tree_mask,
        positions,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
    )
    for index, tensor in enumerate(tensors):
        if tensor.dtype != (torch.bool if index == 3 else torch.int64):
            raise ValueError("build_tree requires int64 tensors and a bool tree_mask")

    parents, selected, lengths, mask, pos, indices, children, siblings = map(
        _raw_span, tensors
    )
    batch = parent_list.shape[0]
    width = draft_token_num
    parent_stride = topk * (depth - 1) + 1
    full_mask_base = 0
    for b in range(batch):
        ids = selected[b * (width - 1) : (b + 1) * (width - 1)].tolist()
        selected_to_node = {value: index + 1 for index, value in enumerate(ids)}
        parent_nodes = [-1]
        for value in ids:
            parent_group = value // topk
            parent_nodes.append(
                0
                if parent_group == 0
                else selected_to_node[int(parents[b * parent_stride + parent_group])]
            )
        base = b * width
        children[base : base + width].fill_(-1)
        siblings[base : base + width].fill_(-1)
        # Reverse head insertion yields increasing child/sibling node indices.
        for node in range(width - 1, 0, -1):
            parent = parent_nodes[node]
            siblings[base + node] = children[base + parent]
            children[base + parent] = node

        prefix_length = int(lengths[b])
        for node in range(width):
            ancestors = []
            current = node
            while current != -1:
                if current in ancestors:
                    raise ValueError("selected nodes must form an acyclic tree")
                ancestors.append(current)
                current = parent_nodes[current]
            indices[base + node] = base + node
            pos[base + node] = prefix_length + len(ancestors) - 1
            if tree_mask_mode == 0:
                start = full_mask_base + node * (prefix_length + width) + prefix_length
            else:
                # Correct QLEN_ONLY rows intentionally differ from the legacy and current native backends.
                start = b * width * width + node * width
            row = mask[start : start + width]
            row.fill_(False)
            row[ancestors] = True
        full_mask_base += width * (prefix_length + width)
