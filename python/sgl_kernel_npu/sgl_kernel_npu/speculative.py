import logging
from enum import IntEnum
from typing import Tuple

import torch
import triton
import triton.language as tl
from sgl_kernel_npu.utils.triton_utils import get_device_properties

logger = logging.getLogger(__name__)

tensor_one = None


class TreeMaskMode(IntEnum):
    FULL_MASK = 0
    QLEN_ONLY = 1
    QLEN_ONLY_BITPACKING = 2


def build_tree_efficient_native(
    parent_list: torch.Tensor,
    selected_index: torch.Tensor,
    verified_seq_len: torch.Tensor,
    tree_mask: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    topk: int,
    draft_token_num: int,
    tree_mask_mode: int,
    bs: int,
):
    # Generate batch and token index ranges
    bs_range = torch.arange(bs, device=tree_mask.device)
    draft_token_num_range = torch.arange(draft_token_num, device=tree_mask.device)

    # Optimized common case for performance.
    if draft_token_num == 2 and topk == 1 and tree_mask_mode == TreeMaskMode.FULL_MASK:
        positions = verified_seq_len.repeat_interleave(draft_token_num)
        positions = (positions.view(bs, -1) + draft_token_num_range).view(-1)

        retrive_index[:] = (
            bs_range.view(-1, 1) * draft_token_num + draft_token_num_range
        )
        retrive_next_token[:, 0] = 1
        retrive_next_token[:, 1] = -1
        return (
            positions,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            tree_mask,
        )

    # Precompute sequence tree indices
    draft_token_num_range1 = torch.arange(draft_token_num - 1, device=tree_mask.device)
    cum_seq_len = torch.cumsum(verified_seq_len * draft_token_num, dim=0)
    cum_seq_len = torch.cat((torch.tensor([0], device=tree_mask.device), cum_seq_len))
    cum_seq_len = cum_seq_len[:-1]
    seq_tree_idx = draft_token_num * draft_token_num * bs_range + cum_seq_len

    # Batch processing for tree mask
    if tree_mask_mode == TreeMaskMode.FULL_MASK:
        token_tree_base = (
            seq_tree_idx.view(-1, 1)
            + (verified_seq_len.view(-1, 1) + draft_token_num) * draft_token_num_range
        )
        token_tree_indices = token_tree_base + verified_seq_len.view(-1, 1) + 1
    else:
        token_tree_indices = (
            bs_range.view(-1, 1) * draft_token_num**2
            + draft_token_num_range * draft_token_num
            + 1
        )

    tree_mask[token_tree_indices.flatten() - 1] = True
    indices = token_tree_indices.unsqueeze(-1) + draft_token_num_range1.view(1, 1, -1)
    tree_mask[indices.view(-1)] = False

    positions = verified_seq_len.repeat_interleave(draft_token_num)
    parent_tb_indices = selected_index // topk
    retrive_index[:] = bs_range.view(-1, 1) * draft_token_num + draft_token_num_range
    tree_mask[token_tree_indices.view(-1, 1) + draft_token_num_range1] = True

    # Process root and non-root nodes, update retrieve_next_token and retrive_next_sibling
    # Calculate the positional information of each token.
    for bid in range(bs):
        for tid in range(draft_token_num):
            position = 0
            if tid == 0:
                # Process root node
                for i in range(draft_token_num - 1, 0, -1):
                    parent_position = 0
                    parent_tb_idx = parent_tb_indices[bid][i - 1]
                    if parent_tb_idx > 0:
                        parent_token_idx = parent_list[bid][parent_tb_idx]
                        loop_num = draft_token_num - parent_position
                        for _ in range(loop_num):
                            if selected_index[bid][parent_position] == parent_token_idx:
                                parent_position += 1
                                break
                            parent_position += 1
                    if parent_position == draft_token_num:
                        logger.warning(
                            "WARNING: invalid eagle tree!! Detected a token with no parent token selected."
                            "Please check if the logprob has nan. The token will be ignored to keep "
                            "proceeding."
                        )
                        continue

                    if retrive_next_token[bid][parent_position] != -1:
                        retrive_next_sibling[bid][i] = retrive_next_token[bid][
                            parent_position
                        ]
                    retrive_next_token[bid][parent_position] = i
            else:
                # Process no-root nodes
                cur_position = tid - 1
                while True:
                    position += 1
                    if cur_position >= draft_token_num:
                        tree_mask[token_tree_indices + cur_position] = True
                        parent_tb_idx = selected_index[bid][cur_position] // topk
                    else:
                        parent_tb_idx = parent_tb_indices[bid][cur_position]
                    if parent_tb_idx == 0:
                        break
                    token_idx = parent_list[bid][parent_tb_idx]
                    cur_position = 0
                    for _ in range(draft_token_num):
                        if selected_index[bid][cur_position] == token_idx:
                            break
                        cur_position += 1
                positions[bid * draft_token_num + tid] += position
    return positions, retrive_index, retrive_next_token, retrive_next_sibling, tree_mask


def verify_tree_greedy_native(
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


@triton.jit
def reconstruct_indices_from_tree_mask_triton_kernel(
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


def reconstruct_indices_from_tree_mask_triton(
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

    num_vectorcore = get_device_properties()[1]
    grid_0 = min(batch_size, num_vectorcore)
    grid_1 = D

    batch_per_core = triton.cdiv(batch_size, grid_0)

    grid = (grid_0, grid_1)

    reconstruct_indices_from_tree_mask_triton_kernel[grid](
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


def reconstruct_indices_from_tree_mask_torch(
    tree_mask: torch.Tensor,
    verified_seq_len: torch.Tensor,
    batch_size: int,
    draft_token_num: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    D = draft_token_num
    bs = batch_size

    tree_mask = tree_mask.view(bs, D, D)
    device = tree_mask.device
    dtype = torch.int64

    retrive_index = torch.arange(bs * D, device=device, dtype=dtype).view(bs, D)

    t_indices = torch.arange(D, device=device).unsqueeze(0).expand(D, D)
    i_lt_t_mask = t_indices < torch.arange(D, device=device).unsqueeze(1)
    i_lt_t_mask = i_lt_t_mask.unsqueeze(0).expand(bs, D, D)

    depth = (tree_mask & i_lt_t_mask).sum(dim=-1)

    reversed_tree_mask = tree_mask.flip(dims=[-1]) & i_lt_t_mask.flip(dims=[-1])
    parent_idx = torch.argmax(reversed_tree_mask.int(), dim=-1)
    has_parent = reversed_tree_mask.sum(dim=-1) > 0
    parent_idx = (
        torch.arange(D, device=device).unsqueeze(0).expand(bs, D) - 1
    ) - parent_idx
    parent_idx = torch.where(has_parent, parent_idx, -torch.ones_like(parent_idx))

    positions = depth + verified_seq_len.unsqueeze(1)

    i_indices = torch.arange(D, device=device).reshape(1, D, 1).expand(bs, D, D)
    t_indices = torch.arange(D, device=device).reshape(1, 1, D).expand(bs, D, D)
    i_gt_t_mask = i_indices > t_indices

    next_token_candidates = tree_mask & i_gt_t_mask
    next_token_idx = torch.argmax(next_token_candidates.int(), dim=1)
    has_next_token = next_token_candidates.sum(dim=1) > 0
    retrive_next_token = torch.where(
        has_next_token, next_token_idx, -torch.ones_like(next_token_idx)
    )

    parent_expanded = parent_idx.unsqueeze(1).expand(bs, D, D)
    cond_a = torch.gather(tree_mask, dim=2, index=parent_expanded) & i_gt_t_mask
    j_indices = torch.arange(D, device=device).reshape(1, 1, D).expand(bs, D, D)
    cond_b_mask = (j_indices > parent_expanded) & (j_indices < i_indices)
    cond_b = (tree_mask & cond_b_mask).sum(dim=2) == 0
    cond_b = cond_b.unsqueeze(2).expand(bs, D, D)
    sibling_candidates = cond_a & cond_b
    next_sibling_idx = torch.argmax(sibling_candidates.int(), dim=1)
    has_next_sibling = (sibling_candidates.sum(dim=1) > 0) & (parent_idx != -1)
    retrive_next_sibling = torch.where(
        has_next_sibling, next_sibling_idx, -torch.ones_like(next_sibling_idx)
    )

    retrive_index = retrive_index.reshape(-1)
    positions = positions.reshape(-1)
    retrive_next_token = retrive_next_token.reshape(-1)
    retrive_next_sibling = retrive_next_sibling.reshape(-1)

    return retrive_index, positions, retrive_next_token, retrive_next_sibling


def reconstruct_indices_from_tree_mask_torch_wrapper(
    tree_mask: torch.Tensor,
    verified_seq_len: torch.Tensor,
    positions: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    batch_size: int,
    draft_token_num: int,
) -> None:
    origin_retrive_index_shape = retrive_index.shape
    origin_positions_shape = positions.shape
    origin_retrive_next_token_shape = retrive_next_token.shape
    origin_retrive_next_sibling_shape = retrive_next_sibling.shape

    ri, pos, ntk, nsb = reconstruct_indices_from_tree_mask_torch(
        tree_mask, verified_seq_len, batch_size, draft_token_num
    )
    ri = ri.contiguous().view(*origin_retrive_index_shape)
    pos = pos.contiguous().view(*origin_positions_shape)
    ntk = ntk.contiguous().view(*origin_retrive_next_token_shape)
    nsb = nsb.contiguous().view(*origin_retrive_next_sibling_shape)

    retrive_index.copy_(ri)
    positions.copy_(pos)
    retrive_next_token.copy_(ntk)
    retrive_next_sibling.copy_(nsb)
