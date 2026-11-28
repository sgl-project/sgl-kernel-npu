"""Reconstruct NGRAM tree links and positions from its ancestor mask."""

import torch
import triton
import triton.language as tl
from sgl_kernel_npu.utils.triton_utils import get_device_properties


@triton.jit
def _reconstruct_indices_from_tree_mask_kernel(
    tree_mask_ptr,
    verified_seq_len_ptr,
    positions_ptr,
    retrive_index_ptr,
    retrive_next_token_ptr,
    retrive_next_sibling_ptr,
    batch_size,
    D: tl.constexpr,
    batch_per_core: tl.constexpr,
):
    core_id = tl.program_id(0)
    tid = tl.program_id(1)

    start_b = core_id * batch_per_core
    end_b = start_b + batch_per_core
    end_b = min(end_b, batch_size)

    for bid in tl.range(start_b, end_b):
        base_offset = D * D
        token_idx = bid * D
        tree_mask_offset = bid * base_offset

        depth = 0
        parent_idx = -1
        next_token_idx = -1
        next_sibling_idx = -1

        # 1. depth & parent_idx
        start_idx = tree_mask_offset + tid * D
        for i in range(tid - 1, -1, -1):
            tm_ptr = tree_mask_ptr + start_idx + i
            tm_val = tl.load(tm_ptr)
            if tm_val:
                depth += 1
                if parent_idx == -1:
                    parent_idx = i

        # 2. retrive_index & positions
        ri_val = token_idx + tid
        pos_val = depth + tl.load(verified_seq_len_ptr + bid)

        # 3. next_token_idx
        for i in range(tid + 1, D):
            if next_token_idx == -1:
                tm_ptr = tree_mask_ptr + tree_mask_offset + i * D + tid
                tm_val = tl.load(tm_ptr)
                if tm_val:
                    next_token_idx = i

        # 4. next_sibling_idx
        if parent_idx != -1:
            for i in range(tid + 1, D):
                if next_sibling_idx == -1:
                    tm_parent_ptr = (
                        tree_mask_ptr + tree_mask_offset + i * D + parent_idx
                    )
                    tm_parent_val = tl.load(tm_parent_ptr)
                    if tm_parent_val:
                        is_sibling = True
                        for j in range(parent_idx + 1, i):
                            if is_sibling:
                                tm_j_ptr = tree_mask_ptr + tree_mask_offset + i * D + j
                                tm_j_val = tl.load(tm_j_ptr)
                                if tm_j_val:
                                    is_sibling = False
                        if is_sibling:
                            next_sibling_idx = i

        # save
        tl.store(retrive_index_ptr + token_idx + tid, ri_val)
        tl.store(positions_ptr + token_idx + tid, pos_val)
        tl.store(retrive_next_token_ptr + token_idx + tid, next_token_idx)
        tl.store(retrive_next_sibling_ptr + token_idx + tid, next_sibling_idx)


def reconstruct_indices_from_tree_mask(
    tree_mask: torch.Tensor,
    verified_seq_len: torch.Tensor,
    positions: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    batch_size: int,
    draft_token_num: int,
) -> None:
    D = draft_token_num

    if batch_size == 0:
        return
    if D <= 0:
        raise ValueError("draft_token_num must be positive.")

    num_vectorcore = get_device_properties()[1]
    grid_0 = min(batch_size, num_vectorcore)
    grid_1 = D

    batch_per_core = triton.cdiv(batch_size, grid_0)

    grid = (grid_0, grid_1)

    _reconstruct_indices_from_tree_mask_kernel[grid](
        tree_mask,
        verified_seq_len,
        positions,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        batch_size,
        D,
        batch_per_core=batch_per_core,
    )
