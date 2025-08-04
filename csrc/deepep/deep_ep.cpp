#include <memory>
#include <pybind11/functional.h>

#include "hccl/hccl.h"
#include "exception.hpp"
#include "deep_ep.hpp"
#include "pytorch_npu_helper.hpp"


namespace deep_ep {

Buffer::Buffer(int64_t rank, int64_t num_ranks, int64_t num_nvl_bytes, int64_t num_rdma_bytes, bool low_latency_mode,
               std::string moe_all_to_all_group_name)
    : rank(rank),
      num_ranks(num_ranks),
      num_nvl_bytes(num_nvl_bytes),
      num_rdma_bytes(num_rdma_bytes),
      low_latency_mode(low_latency_mode),
      moe_all_to_all_group_name(moe_all_to_all_group_name)
{
    rdma_rank = rank;
    EP_HOST_ASSERT(0 <= rank and rank < num_ranks);

    if (moe_all_to_all_group_name.empty()) {
        char *ranktable_file = std::getenv("RANK_TABLE_FILE");
        EP_HOST_ASSERT(ranktable_file != nullptr)
        ACL_CHECK(aclrtGetDevice(&device_id));

        // ep domain
        HCCL_CHECK(HcclCommInitClusterInfo(ranktable_file, device_id, &ep_comm));
    } else {
        EP_HOST_ASSERT(moe_all_to_all_group_name.size() < 128);
    }
}

Buffer::~Buffer() noexcept(false) {
}

bool Buffer::is_available() const {
    return available;
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, torch::Tensor, std::optional<EventHandle>>
Buffer::get_dispatch_layout(const torch::Tensor& topk_idx, int num_experts,
                            std::optional<EventHandle>& previous_event, bool async, bool allocate_on_comm_stream) {
    EP_HOST_ASSERT(topk_idx.dim() == 2);
    EP_HOST_ASSERT(topk_idx.is_contiguous());
    EP_HOST_ASSERT(num_experts > 0);

    const int num_tokens = topk_idx.size(0);
    const int num_topk = topk_idx.size(1);

    auto options_cpu = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    auto num_tokens_per_expert_cpu = torch::zeros({num_experts}, options_cpu);
    auto num_tokens_per_rank_cpu = torch::zeros({num_ranks}, options_cpu);
    auto is_token_in_rank_cpu = torch::zeros({num_tokens, num_ranks}, torch::kBool);

    auto topk_cpu = topk_idx.to(torch::kCPU);
    const int64_t* topk_data = topk_cpu.data_ptr<int64_t>();
    int32_t* global_expert_acc = num_tokens_per_expert_cpu.data_ptr<int32_t>();
    int32_t* global_rank_acc = num_tokens_per_rank_cpu.data_ptr<int32_t>();
    bool* global_in_rank = is_token_in_rank_cpu.data_ptr<bool>();

    std::optional<torch::Tensor> num_tokens_per_rdma_rank = std::nullopt;
    std::optional<EventHandle> output_event = std::nullopt;

    const int experts_per_rank = num_experts / num_ranks;

    #pragma omp parallel
    {
        // Private buffer for each thread
        std::vector<int32_t> local_expert_acc(num_experts, 0);
        std::vector<int32_t> local_rank_acc(num_ranks, 0);
        std::vector<uint8_t> local_in_rank(num_tokens * num_ranks, 0);

        #pragma omp for nowait
        for (int i = 0; i < num_tokens; ++i) {
            std::vector<uint8_t> seen_rank(num_ranks, 0);
            for (int j = 0; j < num_topk; ++j) {
                int64_t expert_idx = topk_data[i * num_topk + j];
                if (expert_idx >= 0) {
                    local_expert_acc[expert_idx]++;
                    int rank_id = expert_idx / experts_per_rank;
                    if (!seen_rank[rank_id]) {
                        local_rank_acc[rank_id]++;
                        local_in_rank[i * num_ranks + rank_id] = 1;
                        seen_rank[rank_id] = 1;
                    }
                }
            }
        }

        #pragma omp critical
        {
            for (int i = 0; i < num_experts; ++i)
                global_expert_acc[i] += local_expert_acc[i];
            for (int i = 0; i < num_ranks; ++i)
                global_rank_acc[i] += local_rank_acc[i];
            for (int i = 0; i < num_tokens * num_ranks; ++i)
                if (local_in_rank[i])
                    global_in_rank[i] = true;
        }
    }

    auto num_tokens_per_expert = num_tokens_per_expert_cpu.to(topk_idx.device());
    auto num_tokens_per_rank = num_tokens_per_rank_cpu.to(topk_idx.device());
    auto is_token_in_rank = is_token_in_rank_cpu.to(topk_idx.device());

    return std::make_tuple(
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
        is_token_in_rank,
        output_event
    );
}

std::tuple<at::Tensor, std::optional<at::Tensor>, at::Tensor, at::Tensor, at::Tensor, std::optional<EventHandle>,
    std::optional<std::function<void()>>>
    Buffer::low_latency_dispatch(const at::Tensor &x, const at::Tensor &topk_idx,
        const std::optional<at::Tensor> &cumulative_local_expert_recv_stats, int64_t num_max_dispatch_tokens_per_rank,
        int64_t num_experts, bool use_fp8, bool round_scale, bool use_ue8m0, bool async, bool return_recv_hook)
{
    this->is_padding = false;
    EP_HOST_ASSERT(low_latency_mode);
    at::Tensor new_x = x;
    this->new_topk_idx = topk_idx;
    if (topk_idx.size(0) <= 2) {
        this->is_padding = true;
        this->padding_cnt = 3 - topk_idx.size(0);
        std::vector<at::Tensor> x_blocks;
        std::vector<at::Tensor> topk_blocks;
        if (topk_idx.size(0) != 0) {
            x_blocks.emplace_back(x);
            topk_blocks.emplace_back(topk_idx);
        } else {
            this->ori_x = x.clone();
        }
        for (int i = 0; i < this->padding_cnt; i++) {
            at::Tensor tmp_x = torch::ones({1, 7168}, x.options());
            at::Tensor tmp_topk = torch::arange(0, 8, topk_idx.options()).reshape({1, 8});
            x_blocks.emplace_back(tmp_x);
            topk_blocks.emplace_back(tmp_topk);
        }
        new_x = torch::cat(x_blocks, 0);
        this->new_topk_idx = torch::cat(topk_blocks, 0);
    }

    auto num_tokens = static_cast<int>(new_x.size(0)), hidden = static_cast<int>(new_x.size(1));
    auto num_scales = hidden / 128, num_topk = static_cast<int>(new_topk_idx.size(1));
    auto num_local_experts = num_experts / num_ranks;
    auto max_size = 
        num_tokens * num_topk > num_local_experts * num_ranks * num_max_dispatch_tokens_per_rank * 128 ?
        num_tokens * num_topk : num_local_experts * num_ranks * num_max_dispatch_tokens_per_rank * 128;

    // Allocate packed tensors
    auto device = new_x.device();
    auto packed_recv_x = at::empty({num_local_experts * num_ranks * num_max_dispatch_tokens_per_rank, hidden},
        new_x.options().dtype(use_fp8 ? at::kChar : at::kBFloat16));
    auto packed_recv_x_scales = at::empty(
        {num_local_experts * num_ranks * num_max_dispatch_tokens_per_rank}, at::dtype(at::kFloat).device(device));
    auto assist_info_for_combine = at::empty({max_size}, at::dtype(at::kInt).device(device));
    auto packed_recv_count = at::empty({num_local_experts * num_ranks}, at::dtype(at::kInt).device(device));
    auto tp_recv_count = at::empty({1}, at::dtype(at::kInt).device(device));
    auto expertTokenNumsOut = at::empty({num_local_experts}, at::dtype(at::kLong).device(device));
    auto expandScales = at::empty({1}, at::dtype(at::kFloat).device(device));
    at::Tensor scales;
    at::Tensor activateMask;
    auto expert_scales = at::empty({1}, at::dtype(at::kFloat).device(device));
    int64_t quant_mode = use_fp8 ? 2 : 0;
    int64_t tp_size = 1;
    int64_t tp_rank = 0;
    int64_t expert_shard_type = 0;
    int64_t shared_expert_num = 1;
    int64_t expert_token_nums_type = 1;
    int64_t global_bs = num_max_dispatch_tokens_per_rank * num_ranks;
    int64_t shared_expert_rank_num = 0;
    std::string comm_log = "0";
    char *comm_log_ptr = const_cast<char *>(comm_log.c_str());

    // get ep & tp name
    char hcom_ep_name[128];
    if (!moe_all_to_all_group_name.empty()) {
        std:memcpy(hcom_ep_name, moe_all_to_all_group_name.data(), moe_all_to_all_group_name.size() + 1);
    } else {
        HCCL_CHECK(HcclGetCommName(ep_comm, hcom_ep_name));
    }

    EXEC_NPU_CMD(aclnnMoeDistributeDispatchV2,
        new_x,
        new_topk_idx,
        scales,         // smooth scales,
        activateMask,   // activateMask
        expert_scales,  // expert_scales
        hcom_ep_name,     // ep
        num_ranks,      // rankSize
        rank,           // rankId
        num_experts,
        hcom_ep_name,           // tp
        tp_size,               // tp_size
        tp_rank,               // tp_rank
        expert_shard_type,            // expert_shard_type
        shared_expert_num,      // shared_expert_num
        shared_expert_rank_num,  // shared_expert_rank_num
        quant_mode,
        global_bs,             // global_bs
        expert_token_nums_type,  // expert_token_nums_type
        comm_log_ptr,
        packed_recv_x,
        packed_recv_x_scales,  // dynamicScalesOut
        assist_info_for_combine,
        expertTokenNumsOut,
        packed_recv_count,
        tp_recv_count,
        expandScales);

    // Wait streams
    std::optional<EventHandle> event;

    // Return values
    return {packed_recv_x, packed_recv_x_scales, packed_recv_count, assist_info_for_combine, expertTokenNumsOut, event, std::function<void()>([]{})};
}

int Buffer::get_rdma_rank() const {
    return rdma_rank;
}

std::tuple<at::Tensor, std::optional<EventHandle>, std::optional<std::function<void()>>> Buffer::low_latency_combine(
    const at::Tensor &x, const at::Tensor &topk_idx, const at::Tensor &topk_weights, const at::Tensor &src_info,
    const at::Tensor &layout_range, int64_t num_max_dispatch_tokens_per_rank, int64_t num_experts,
    const at::Tensor &ep_send_count, bool zero_copy, bool async, bool return_recv_hook,
    const std::optional<at::Tensor> &out)
{
    at::Tensor new_idx = topk_idx;
    at::Tensor new_scales = topk_weights;
    if (this->is_padding) {
        std::vector<at::Tensor> scales_blocks;
        if (this->padding_cnt != 3) {
            scales_blocks.emplace_back(topk_weights);
        }
        for (int i = 0; i < this->padding_cnt; i++) {
            at::Tensor tmp_scales = torch::zeros({1, 8}, topk_weights.options());
            scales_blocks.emplace_back(tmp_scales);
        }
        new_idx = this->new_topk_idx;
        this->new_scales = torch::cat(scales_blocks, 0);
        new_scales = this->new_scales;
    }
    // Tensor checks
    EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous() and x.scalar_type() == at::kBFloat16);
    // EP_HOST_ASSERT(x.size(0) == num_experts / num_ranks);

    // get ep & tp name
    char hcom_ep_name[128];
    if (!moe_all_to_all_group_name.empty()) {
        std:memcpy(hcom_ep_name, moe_all_to_all_group_name.data(), moe_all_to_all_group_name.size() + 1);
    } else {
        HCCL_CHECK(HcclGetCommName(ep_comm, hcom_ep_name));
    }

    auto device = x.device();
    at::Tensor expand_x = x;
    at::Tensor expert_ids = new_idx;
    at::Tensor assist_info_for_combine = src_info; // handle[0] = src_info
    at::Tensor ep_send_counts = ep_send_count;
    at::Tensor expert_scales = new_scales;
    at::Tensor tp_send_counts = at::empty({1}, at::dtype(at::kInt).device(device));
    at::Tensor x_active_mask, activation_scale, weight_scale, group_list, expand_scales;

    int64_t tp_world_size = 1;
    int64_t tp_rankId = 0;
    int64_t expert_shared_type = 0;
    int64_t shared_expert_num = 1;
    int64_t global_bs = num_max_dispatch_tokens_per_rank * num_ranks;
    int64_t shared_expert_rank_num = 0;
    int64_t out_dtype = 0;
    int64_t comm_quant_mode = 0;
    int64_t group_list_type = 0;

    auto num_combined_tokens = static_cast<int>(new_scales.size(0));
    auto hidden = static_cast<int>(x.size(1));
    at::Tensor shared_expert_x{nullptr};
    at::Tensor combined_x = at::empty({num_combined_tokens, hidden}, x.options());
    std::optional<EventHandle> event;
    std::string comm_log = "0";
    char *comm_log_ptr = const_cast<char *>(comm_log.c_str());

    EXEC_NPU_CMD(aclnnMoeDistributeCombineV2,
        expand_x,
        expert_ids,
        assist_info_for_combine,
        ep_send_counts,
        expert_scales,
        tp_send_counts,
        x_active_mask,
        activation_scale,
        weight_scale,
        group_list,
        expand_scales,
        shared_expert_x,
        hcom_ep_name,
        num_ranks,
        rank,
        num_experts,
        hcom_ep_name,
        tp_world_size,
        tp_rankId,
        expert_shared_type,
        shared_expert_num,
        shared_expert_rank_num,
        global_bs,
        out_dtype,
        comm_quant_mode,
        group_list_type,
        comm_log_ptr,
        combined_x);
    if (this->is_padding) {
        if (this->padding_cnt == 3) {
            combined_x = this->ori_x;
        } else {
            combined_x = combined_x.slice(0, 0, 3 - this->padding_cnt);
        }
    }
    return {combined_x, event, std::function<void()>([]{})};
}

} // namespace deep_ep
