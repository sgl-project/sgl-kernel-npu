"""Build the per-expert group_list from raw dispatch metadata, fully on device.

The DeepEP ``intranode_dispatch`` returns the raw per-round received-token counts
as a device tensor shaped ``[round, num_local_experts]`` (round-major, int32).
This module reduces it into the ``group_list`` consumed by the grouped MoE
kernels (int64, one entry per local expert) without any host round-trip, so the
runner no longer stalls on a D2H/H2D copy to build the group_list.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _sum_rounds_kernel(
    recv_ptr,  # int32 [ROUND, NEL], round-major
    out_ptr,   # int64 [NEL]
    ROUND: tl.constexpr,
    NEL: tl.constexpr,
    ROUND_ALIGN: tl.constexpr,
):
    e = tl.program_id(0)
    if e < NEL:
        r_offsets = tl.arange(0, ROUND_ALIGN)
        r_mask = r_offsets < ROUND
        vals = tl.load(
            recv_ptr + r_offsets * NEL + e, mask=r_mask, other=0
        ).to(tl.int32)
        total = tl.sum(vals, axis=0)
        tl.store(out_ptr + e, total.to(out_ptr.dtype.element_ty))


@triton.jit
def _prefix_sum_kernel(
    in_ptr,    # int64 [NEL]
    out_ptr,   # int64 [NEL]
    NEL: tl.constexpr,
    NEL_ALIGN: tl.constexpr,
):
    offsets = tl.arange(0, NEL_ALIGN)
    mask = offsets < NEL
    vals = tl.load(in_ptr + offsets, mask=mask, other=0).to(tl.int64)
    acc = tl.cumsum(vals, axis=0)
    tl.store(out_ptr + offsets, acc.to(out_ptr.dtype.element_ty), mask=mask)


def recv_tokens_to_group_list(
    recv_tokens_per_expert: torch.Tensor,
    expert_token_nums_type: int = 1,
) -> torch.Tensor:
    """Parse raw dispatch metadata into the expert ``group_list``.

    Args:
        recv_tokens_per_expert: int32 device tensor shaped
            ``[round, num_local_experts]``, round-major, as returned by
            ``intranode_dispatch``.
        expert_token_nums_type: ``1`` = per-expert received token count (default),
            ``0`` = prefix sum of the received token counts.

    Returns:
        int64 device tensor shaped ``[num_local_experts]``.
    """
    assert recv_tokens_per_expert.dim() == 2
    assert recv_tokens_per_expert.dtype == torch.int32
    assert expert_token_nums_type in (0, 1)

    round_, num_local_experts = recv_tokens_per_expert.shape
    device = recv_tokens_per_expert.device
    counts = torch.empty((num_local_experts,), dtype=torch.int64, device=device)

    _sum_rounds_kernel[(num_local_experts,)](
        recv_tokens_per_expert,
        counts,
        ROUND=round_,
        NEL=num_local_experts,
        ROUND_ALIGN=triton.next_power_of_2(round_),
    )

    if expert_token_nums_type == 0:
        group_list = torch.empty_like(counts)
        _prefix_sum_kernel[(1,)](
            counts,
            group_list,
            NEL=num_local_experts,
            NEL_ALIGN=triton.next_power_of_2(num_local_experts),
        )
        return group_list

    return counts
