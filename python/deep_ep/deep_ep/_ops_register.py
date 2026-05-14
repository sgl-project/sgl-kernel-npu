import os

import torch
from deep_ep_cpp import fused_deep_moe_no_buffer

m = torch.library.Library("npu", "FRAGMENT")

m.define(
    "fused_deep_moe_op("
    "Tensor x, "
    "Tensor topk_ids, "
    "Tensor gmm1_permuted_weight, "
    "Tensor gmm1_permuted_weight_scale, "
    "Tensor gmm2_weight, "
    "Tensor gmm2_weight_scale, "
    "Tensor topk_weights, "
    "int num_max_dispatch_tokens_per_rank, "
    "int num_experts, "
    "int quant_mode, "
    "int rank, "
    "int num_ranks, "
    "str moe_all_to_all_group_name"
    ") -> (Tensor, Tensor)"
)


@torch.library.impl(m, "fused_deep_moe_op", "Meta")
def fused_deep_moe_meta(
    x,
    topk_ids,
    gmm1_permuted_weight,
    gmm1_permuted_weight_scale,
    gmm2_weight,
    gmm2_weight_scale,
    topk_weights,
    num_max_dispatch_tokens_per_rank,
    num_experts,
    quant_mode,
    rank,
    num_ranks,
    moe_all_to_all_group_name,
):
    output = torch.empty_like(x)
    shared_expert_rank_num = int(os.environ.get("MOE_SHARED_EXPERT_RANK_NUM", "0"))
    is_share_expert = rank < shared_expert_rank_num
    if is_share_expert:
        recv_count_out_shape = num_ranks
    else:
        recv_count_out_shape = num_ranks * (
            num_experts // (num_ranks - shared_expert_rank_num)
        )
    ep_recv_count = torch.empty(
        (recv_count_out_shape,), dtype=topk_ids.dtype, device="meta"
    )
    return output, ep_recv_count


@torch.library.impl(m, "fused_deep_moe_op", "PrivateUse1")
def fused_deep_moe_meta(
    x,
    topk_ids,
    gmm1_permuted_weight,
    gmm1_permuted_weight_scale,
    gmm2_weight,
    gmm2_weight_scale,
    topk_weights,
    num_max_dispatch_tokens_per_rank,
    num_experts,
    quant_mode,
    rank,
    num_ranks,
    moe_all_to_all_group_name,
):
    return fused_deep_moe_no_buffer(
        x,
        topk_ids,
        gmm1_permuted_weight,
        gmm1_permuted_weight_scale,
        gmm2_weight,
        gmm2_weight_scale,
        topk_weights,
        num_max_dispatch_tokens_per_rank,
        num_experts,
        quant_mode,
        rank,
        num_ranks,
        moe_all_to_all_group_name,
    )
