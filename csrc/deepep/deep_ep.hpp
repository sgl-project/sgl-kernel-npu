#pragma once

#include <torch/types.h>
#include <torch/python.h>
#include <tuple>
#include <vector>
#include <optional>
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"
#include "aclnn/opdev/platform.h"

#include "config.hpp"
#include "event.hpp"

namespace deep_ep {

struct Buffer {
    int64_t rank, rdma_rank, nvl_rank;
    int64_t num_ranks, num_rdma_ranks, num_nvl_ranks;
    op::SocVersion soc_version;

    int64_t num_nvl_bytes;
    int64_t num_rdma_bytes;

    bool low_latency_mode = false;
    bool is_padding = false;
    int padding_cnt = 0;
    at::Tensor ori_x;
    at::Tensor new_topk_idx;
    at::Tensor new_scales;
    at::Tensor notify_send_data;  // only for internode notify
    at::Tensor send_token_idx_small;
    int notify_send_data_size;  // only for internode notify

    int64_t shared_expert_rank_num;
    int64_t shared_expert_num = 1;

private:
    std::string moe_all_to_all_group_name;

    int device_id;

    HcclComm ep_comm;

    bool available = false;

public:
    Buffer(int64_t rank, int64_t num_ranks, int64_t num_nvl_bytes, int64_t num_rdma_bytes, bool low_latency_mode,
           std::string moe_all_to_all_group_name);

    ~Buffer() noexcept(false);

    bool is_available() const;

    int get_num_rdma_ranks() const;

    int get_rdma_rank() const;

    std::tuple<torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, torch::Tensor, std::optional<EventHandle>>
    get_dispatch_layout(const torch::Tensor &topk_idx, int num_experts, std::optional<EventHandle> &previous_event,
                        bool async, bool allocate_on_comm_stream);

    torch::Tensor get_notify_send_data();

    std::tuple<at::Tensor, std::optional<at::Tensor>, std::optional<at::Tensor>, std::optional<at::Tensor>,
               std::vector<int>, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, std::optional<EventHandle>>
    intranode_dispatch(const at::Tensor &x, const std::optional<at::Tensor> &x_scales,
                       const std::optional<at::Tensor> &topk_idx, const std::optional<at::Tensor> &topk_weights,
                       const std::optional<at::Tensor> &num_tokens_per_rank, const at::Tensor &is_token_in_rank,
                       const std::optional<at::Tensor> &num_tokens_per_expert, int cached_num_recv_tokens,
                       const std::optional<at::Tensor> &cached_rank_prefix_matrix,
                       const std::optional<at::Tensor> &cached_channel_prefix_matrix,
                       const std::optional<at::Tensor> &dispatch_wait_recv_cost_stats, int expert_alignment,
                       int num_worst_tokens, const Config &config, std::optional<EventHandle> &previous_event,
                       bool async, bool allocate_on_comm_stream, bool use_quant);

    void clean_low_latency_buffer(int num_max_dispatch_tokens_per_rank, int hidden, int num_experts);

    std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<EventHandle>>
    intranode_combine(const torch::Tensor &x, const torch::Tensor &topk_idx,
                      const std::optional<torch::Tensor> &topk_weights, const torch::Tensor &src_idx,
                      const torch::Tensor &send_head, const std::optional<at::Tensor> &combine_send_cost_stats);

    std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<torch::Tensor>, std::optional<torch::Tensor>,
               std::vector<int>, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
               torch::Tensor, std::optional<EventHandle>>
    internode_dispatch(const torch::Tensor &x, const std::optional<torch::Tensor> &x_scales,
                       const std::optional<torch::Tensor> &topk_idx, const std::optional<torch::Tensor> &topk_weights,
                       const std::optional<torch::Tensor> &num_tokens_per_rank,
                       const std::optional<torch::Tensor> &num_tokens_per_rdma_rank,
                       const torch::Tensor &is_token_in_rank, const std::optional<torch::Tensor> &num_tokens_per_expert,
                       const Config &config, std::optional<EventHandle> &previous_event, bool async,
                       bool allocate_on_comm_stream, bool use_quant);

    std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<EventHandle>> internode_combine(
        const torch::Tensor &x, const torch::Tensor &topk_idx, const std::optional<torch::Tensor> &topk_weights,
        const torch::Tensor &src_idx, const torch::Tensor &send_head, const torch::Tensor &offsetInner,
        const torch::Tensor &offsetOuter, const torch::Tensor &countOuter, const torch::Tensor &expand_scales);

    std::tuple<at::Tensor, std::optional<at::Tensor>, at::Tensor, at::Tensor, at::Tensor, std::optional<EventHandle>,
               std::optional<std::function<void()>>>
    low_latency_dispatch(const at::Tensor &x, const at::Tensor &topk_idx,
                         const std::optional<at::Tensor> &cumulative_local_expert_recv_stats,
                         int64_t num_max_dispatch_tokens_per_rank, int64_t num_experts, bool use_fp8, bool round_scale,
                         bool use_ue8m0, bool async, bool return_recv_hook);

    std::tuple<at::Tensor, std::optional<EventHandle>, std::optional<std::function<void()>>> low_latency_combine(
        const at::Tensor &x, const at::Tensor &topk_idx, const at::Tensor &topk_weights, const at::Tensor &src_info,
        const at::Tensor &layout_range, int64_t num_max_dispatch_tokens_per_rank, int64_t num_experts,
        const at::Tensor &packed_recv_count, bool zero_copy, bool async, bool return_recv_hook,
        const std::optional<at::Tensor> &out);

    std::vector<at::Tensor> fused_deep_moe(const at::Tensor &x, const at::Tensor &expertIds,
                                           const at::Tensor &gmm1PermutedWeight,
                                           const at::Tensor &gmm1PermutedWeightScale, const at::Tensor &gmm2Weight,
                                           const at::Tensor &gmm2WeightScale, const at::Tensor &expertScalesOptional,
                                           int64_t num_max_dispatch_tokens_per_rank, int64_t num_experts,
                                           int quant_mode);
};
}  // namespace deep_ep
