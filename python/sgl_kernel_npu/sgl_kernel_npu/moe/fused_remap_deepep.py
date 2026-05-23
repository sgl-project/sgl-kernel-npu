import torch
import triton
import triton.language as tl


@triton.jit
def _fused_remap_deepep_kernel(
    topk_ids_ptr,
    topk_weights_ptr,
    out_ids_ptr,
    out_weights_ptr,
    N,
    K,
    K_PLUS_N,
    num_local_routed,
    num_local_experts,
    ep_rank,
    shared_weight,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= N:
        return

    cols = tl.arange(0, BLOCK)
    mask = cols < K_PLUS_N
    is_routed = cols < K

    safe_cols = tl.where(is_routed, cols, tl.zeros_like(cols))

    routed_ids = tl.load(topk_ids_ptr + row * K + safe_cols, mask=is_routed, other=0)
    remapped_ids = routed_ids + routed_ids // num_local_routed

    shared_idx = cols - K
    shared_ids = ep_rank * num_local_experts + num_local_routed + shared_idx

    final_ids = tl.where(is_routed, remapped_ids, shared_ids)
    tl.store(out_ids_ptr + row * K_PLUS_N + cols, final_ids, mask=mask)

    routed_weights = tl.load(
        topk_weights_ptr + row * K + safe_cols, mask=is_routed, other=0.0
    )
    final_weights = tl.where(is_routed, routed_weights, shared_weight)
    tl.store(out_weights_ptr + row * K_PLUS_N + cols, final_weights, mask=mask)


def fused_remap_deepep(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_fused_shared_experts: int,
    num_physical_routed_experts: int,
    ep_rank: int,
    ep_size: int,
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused Triton kernel: expand routed topk with shared experts and remap
    to DeepEP interleaved layout.

    Args:
        topk_ids: [N, K] routed expert IDs (int).
        topk_weights: [N, K] routed expert weights (float).
        num_fused_shared_experts: number of shared experts to fuse.
        num_physical_routed_experts: total physical routed experts across EP.
        ep_rank: expert parallel rank of this worker.
        ep_size: expert parallel world size.
        routed_scaling_factor: scaling factor for routed experts (0 or 1 means no scaling).
    """
    if topk_ids.shape[0] == 0:
        return topk_ids, topk_weights

    num_local_routed = num_physical_routed_experts // ep_size
    num_local_experts = num_local_routed + num_fused_shared_experts

    N = topk_ids.shape[0]
    K = topk_ids.shape[1]
    K_PLUS_N = K + num_fused_shared_experts

    shared_weight = 1.0 if not routed_scaling_factor else 1.0 / routed_scaling_factor

    out_ids = topk_ids.new_empty((N, K_PLUS_N), dtype=topk_ids.dtype)
    out_weights = topk_weights.new_empty((N, K_PLUS_N), dtype=topk_weights.dtype)

    BLOCK = max(32, K_PLUS_N)
    _fused_remap_deepep_kernel[(N,)](
        topk_ids,
        topk_weights,
        out_ids,
        out_weights,
        N=N,
        K=K,
        K_PLUS_N=K_PLUS_N,
        num_local_routed=num_local_routed,
        num_local_experts=num_local_experts,
        ep_rank=ep_rank,
        shared_weight=shared_weight,
        BLOCK=BLOCK,
    )

    return out_ids, out_weights
