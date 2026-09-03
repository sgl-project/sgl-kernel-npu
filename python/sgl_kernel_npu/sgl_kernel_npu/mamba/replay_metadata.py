"""Graph-recordable Kimi-K3 hybrid-attention replay metadata update."""

import torch
import triton
import triton.language as tl


@triton.jit
def _replay_metadata_kernel(
    req_pool_indices_ptr,
    seq_lens_ptr,
    mamba_index_mapping_ptr,
    state_indices_ptr,
    query_start_loc_ptr,
    BS: tl.constexpr,
    QUERY_STEP: tl.constexpr,
    META_BLOCK: tl.constexpr,
):
    offs = tl.arange(0, META_BLOCK)
    row_mask = offs < BS
    seq_len = tl.load(seq_lens_ptr + offs, mask=row_mask, other=0)
    valid = row_mask & (seq_len != 0)

    # Graph padding is suffix-only. Count the live prefix once and use it for
    # both state-slot masking and the static query indptr.
    valid_bs = tl.sum(valid.to(tl.int32), axis=0)
    req_idx = tl.load(req_pool_indices_ptr + offs, mask=valid, other=0)
    state_idx = tl.load(mamba_index_mapping_ptr + req_idx, mask=valid, other=0)
    tl.store(state_indices_ptr + offs, state_idx.to(tl.int32), mask=valid)
    tl.store(state_indices_ptr + offs, 0, mask=row_mask & ~valid)
    tl.store(req_pool_indices_ptr + offs, 0, mask=row_mask & ~valid)

    query_mask = offs <= BS
    query_row = tl.where(offs < valid_bs, offs, valid_bs)
    tl.store(
        query_start_loc_ptr + offs,
        query_row.to(tl.int32) * QUERY_STEP,
        mask=query_mask,
    )


def update_replay_metadata(
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    mamba_index_mapping: torch.Tensor,
    state_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    *,
    query_step: int,
) -> None:
    """Refresh graph-stable Mamba slots and query indptr in one launch.

    This function is intended for ``init_forward_metadata_in_graph``: the
    launch is captured once and reads the replay input buffers thereafter.
    Padded rows are identified from the graph-stable ``seq_lens`` buffer.
    """
    bs = state_indices.numel()
    if req_pool_indices.numel() < bs or seq_lens.numel() < bs:
        raise ValueError("replay input buffers are shorter than state_indices")
    if query_start_loc.numel() != bs + 1:
        raise ValueError("query_start_loc must have exactly bs + 1 entries")
    _replay_metadata_kernel[(1,)](
        req_pool_indices,
        seq_lens,
        mamba_index_mapping,
        state_indices,
        query_start_loc,
        BS=bs,
        QUERY_STEP=query_step,
        META_BLOCK=triton.next_power_of_2(bs + 1),
    )
