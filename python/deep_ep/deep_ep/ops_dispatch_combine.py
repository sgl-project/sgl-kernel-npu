import os
import torch
import torch_npu
import torch.distributed as dist


class MoeDistributeBuffer:
    def __init__(self, group_name, group_size, rank, ccl_buffer_size: int = 0, comm_alg: str = ""):
        self.rank = rank
        self.group_size = group_size
        self.group_name = group_name
        comm_alg_support_list = ["hierarchy", "fullmesh_v1", "fullmesh_v2", ""]
        torch._check(comm_alg in comm_alg_support_list,
                     lambda: (f"comm_alg only support {comm_alg_support_list=}, but got {comm_alg=}."))

    def npu_low_latency_dispatch(self, x, topk_idx, num_experts: int, *,
                             quant_mode=0, comm_alg="", x_smooth_scale=None,
                             x_active_mask=None, topk_weights=None, expert_shard_type=0, shared_expert_num=0,
                             shared_expert_rank_num=0, num_max_dispatch_tokens_per_rank=0):

        shared_expert_rank_num = os.getenv("MOE_SHARED_EXPERT_RANK_NUM", 0)
        expert_token_nums_type = os.getenv("MOE_EXPERT_TOKEN_NUMS_TYPE", 1)
        global_bs = num_max_dispatch_tokens_per_rank * self.group_size
        if comm_alg == 'hierarchy':
            assert shared_expert_num == 0, "When comm_alg='hierarchy', shared_expert_num must be 0."

        (expand_x, dynamic_scales, expand_idx, expert_token_nums, ep_recv_counts, tp_recv_counts, expand_scales) \
            = torch_npu.npu_moe_distribute_dispatch_v2(
                x=x,
                expert_ids=topk_idx,
                group_ep=self.group_name,
                ep_world_size=self.group_size,
                ep_rank_id=self.rank,
                moe_expert_num=num_experts,
                scales=x_smooth_scale,
                x_active_mask=x_active_mask,
                expert_scales=topk_weights,
                performance_info=None,
                expert_shard_type=expert_shard_type,
                shared_expert_num=shared_expert_num,
                shared_expert_rank_num=shared_expert_rank_num,
                quant_mode=quant_mode,
                expert_token_nums_type=expert_token_nums_type,
                global_bs=global_bs,
                comm_alg=comm_alg,  # A3: 支持""，"fullmesh_v1"，"fullmesh_v2", "hierarchy"
            )

        return expand_x, dynamic_scales, expert_token_nums, expand_idx, ep_recv_counts, expand_scales

    def npu_low_latency_combine(self, x, topk_idx, topk_weights, assist_info_for_combine, ep_send_counts, *,
                            num_experts=0, comm_alg="", comm_quant_mode=0, x_active_mask=None, expand_scales=None,
                            shared_expert_x=None, expert_shared_type=0, shared_expert_num=0,
                            shared_expert_rank_num=0, num_max_dispatch_tokens_per_rank=0):

        shared_expert_rank_num = os.getenv("MOE_SHARED_EXPERT_RANK_NUM", 0)
        global_bs = num_max_dispatch_tokens_per_rank * self.group_size
        if comm_alg == 'hierarchy':
            assert shared_expert_num == 0, "When comm_alg='hierarchy', shared_expert_num must be 0."

        combine_x = torch_npu.npu_moe_distribute_combine_v2(
                    expand_x=x,
                    expert_ids=topk_idx,
                    assist_info_for_combine=assist_info_for_combine,
                    ep_send_counts=ep_send_counts,
                    expert_scales=topk_weights,
                    group_ep=self.group_name,
                    ep_world_size=self.group_size,
                    ep_rank_id=self.rank,
                    moe_expert_num=num_experts,
                    tp_send_counts=None,
                    x_active_mask=x_active_mask,
                    expand_scales=expand_scales,
                    shared_expert_x=shared_expert_x,
                    performance_info=None,
                    expert_shard_type=expert_shared_type,
                    shared_expert_num=shared_expert_num,
                    shared_expert_rank_num=shared_expert_rank_num,
                    global_bs=global_bs,
                    comm_quant_mode=comm_quant_mode,
                    comm_alg=comm_alg,  # A3: 支持""，"hierarchy"两种
                )

        return combine_x

