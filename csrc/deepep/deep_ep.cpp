#include <memory>
#include <cmath>
#include <pybind11/functional.h>

#include "hccl/hccl.h"
#include "exception.hpp"
#include "deep_ep.hpp"
#include "pytorch_npu_helper.hpp"

namespace deep_ep {
constexpr int PADDING_SIZE = 3;
constexpr size_t HCOMM_NAME_LEN = 128;
constexpr uint32_t NO_SCALES = 0;
constexpr uint32_t DYNAMIC_SCALES = 2;

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
        EP_HOST_ASSERT(moe_all_to_all_group_name.size() < HCOMM_NAME_LEN);
    }

    this->shared_expert_rank_num = get_value_from_env("MOE_SHARED_EXPERT_RANK_NUM", 0);
}

Buffer::~Buffer() noexcept(false) {}

bool Buffer::is_available() const
{
    return available;
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, torch::Tensor, std::optional<EventHandle>>
Buffer::get_dispatch_layout(const torch::Tensor &topk_idx, int num_experts, std::optional<EventHandle> &previous_event,
                            bool async, bool allocate_on_comm_stream)
{
    EP_HOST_ASSERT(topk_idx.dim() == 2);
    EP_HOST_ASSERT(topk_idx.is_contiguous());
    EP_HOST_ASSERT(num_experts > 0);

    this->new_topk_idx = topk_idx;
    // for padding
    if (topk_idx.size(0) < PADDING_SIZE) {
        this->is_padding = true;
        this->padding_cnt = PADDING_SIZE - topk_idx.size(0);
        std::vector<at::Tensor> topk_blocks;
        if (topk_idx.size(0) != 0) {
            topk_blocks.emplace_back(topk_idx);
        }
        int topk = static_cast<int>(topk_idx.size(1));
        for (int i = 0; i < this->padding_cnt; i++) {
            at::Tensor tmp_topk = torch::arange(0, topk, topk_idx.options()).reshape({1, topk});
            topk_blocks.emplace_back(tmp_topk);
        }
        this->new_topk_idx = torch::cat(topk_blocks, 0);
    }

    const int num_tokens = new_topk_idx.size(0);
    const int num_topk = new_topk_idx.size(1);

    auto device = new_topk_idx.device();
    auto num_tokens_per_expert = at::zeros({num_experts}, at::dtype(at::kInt).device(device));
    auto num_tokens_per_rank = at::zeros({num_ranks}, at::dtype(at::kInt).device(device));
    auto is_token_in_rank = at::zeros({num_tokens, num_ranks}, at::dtype(at::kInt).device(device));

    EXEC_NPU_CMD(aclnnDispatchLayout, new_topk_idx, num_tokens, num_ranks, num_experts, num_topk, num_tokens_per_rank,
                 num_tokens_per_expert, is_token_in_rank);

    std::optional<torch::Tensor> num_tokens_per_rdma_rank = std::nullopt;
    std::optional<EventHandle> output_event = std::nullopt;
    auto is_token_in_rank_bool = is_token_in_rank.to(at::kBool);

    return std::make_tuple(num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank_bool,
                           output_event);
}

std::tuple<at::Tensor, std::optional<at::Tensor>, std::optional<at::Tensor>, std::optional<at::Tensor>,
           std::vector<int>, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, std::optional<EventHandle>>
Buffer::intranode_dispatch(const at::Tensor &x, const std::optional<at::Tensor> &x_scales,
                           const std::optional<at::Tensor> &topk_idx, const std::optional<at::Tensor> &topk_weights,
                           const std::optional<at::Tensor> &num_tokens_per_rank, const at::Tensor &is_token_in_rank,
                           const std::optional<at::Tensor> &num_tokens_per_expert, int cached_num_recv_tokens,
                           const std::optional<at::Tensor> &cached_rank_prefix_matrix,
                           const std::optional<at::Tensor> &cached_channel_prefix_matrix,
                           const std::optional<at::Tensor> &dispatch_wait_recv_cost_stats, int expert_alignment,
                           int num_worst_tokens, const Config &config, std::optional<EventHandle> &previous_event,
                           bool async, bool allocate_on_comm_stream, bool use_quant)
{
    // One channel use two blocks, even-numbered blocks for sending, odd-numbered blocks for receiving.
    EP_HOST_ASSERT(config.num_sms % 2 == 0);
    int num_channels = config.num_sms / 2;

    at::Tensor new_x = x;
    // for padding
    if (topk_idx->size(0) < PADDING_SIZE) {
        this->is_padding = true;
        this->padding_cnt = PADDING_SIZE - topk_idx->size(0);
        std::vector<at::Tensor> x_blocks;
        if (topk_idx->size(0) != 0) {
            x_blocks.emplace_back(x);
        } else {
            this->ori_x = x.clone();
        }
        for (int i = 0; i < this->padding_cnt; i++) {
            at::Tensor tmp_x = torch::ones({1, x.size(1)}, x.options()) * (i + 1) * 2;
            x_blocks.emplace_back(tmp_x);
        }
        new_x = torch::cat(x_blocks, 0);
    }

    EP_HOST_ASSERT(num_tokens_per_rank.has_value());
    EP_HOST_ASSERT(num_tokens_per_expert.has_value());

    // Type checks
    EP_HOST_ASSERT(is_token_in_rank.scalar_type() == at::kBool);
    EP_HOST_ASSERT(num_tokens_per_expert->scalar_type() == at::kInt);
    EP_HOST_ASSERT(num_tokens_per_rank->scalar_type() == at::kInt);

    // Shape and contiguous checks
    EP_HOST_ASSERT(new_x.dim() == 2 and new_x.is_contiguous());
    // EP_HOST_ASSERT((x.size(1) * x.element_size()) % sizeof(int4) == 0);
    EP_HOST_ASSERT(is_token_in_rank.dim() == 2 and is_token_in_rank.is_contiguous());
    EP_HOST_ASSERT(is_token_in_rank.size(0) == new_x.size(0) and is_token_in_rank.size(1) == num_ranks);
    EP_HOST_ASSERT(num_tokens_per_expert->dim() == 1 and num_tokens_per_expert->is_contiguous());
    EP_HOST_ASSERT(num_tokens_per_expert->size(0) % num_ranks == 0);
    EP_HOST_ASSERT(num_tokens_per_rank->dim() == 1 and num_tokens_per_rank->is_contiguous());
    EP_HOST_ASSERT(num_tokens_per_rank->size(0) == num_ranks);

    auto num_tokens = static_cast<int>(new_x.size(0)), hidden = static_cast<int>(new_x.size(1));
    auto num_experts = static_cast<int64_t>(num_tokens_per_expert->size(0));
    auto num_local_experts = static_cast<int>(num_experts / num_ranks);

    // Top-k checks
    int num_topk = 0;
    EP_HOST_ASSERT(topk_idx.has_value());
    if (topk_idx.has_value()) {
        num_topk = static_cast<int>(topk_idx->size(1));
        EP_HOST_ASSERT(num_experts > 0);
        EP_HOST_ASSERT(topk_idx->dim() == 2 and topk_idx->is_contiguous());
        EP_HOST_ASSERT(topk_weights->dim() == 2 and topk_weights->is_contiguous());
        EP_HOST_ASSERT(num_tokens == new_topk_idx.size(0));
        EP_HOST_ASSERT(num_topk == topk_weights->size(1));
        EP_HOST_ASSERT(topk_weights->scalar_type() == at::kFloat);
    }

    // FP8 scales checks
    float *x_scales_ptr = nullptr;
    int num_scales = 0, scale_token_stride = 0, scale_hidden_stride = 0;
    if (x_scales.has_value()) {
        EP_HOST_ASSERT(new_x.element_size() == 1);
        EP_HOST_ASSERT(x_scales->scalar_type() == at::kFloat or x_scales->scalar_type() == at::kInt);
        EP_HOST_ASSERT(x_scales->dim() == 2);
        EP_HOST_ASSERT(x_scales->size(0) == num_tokens);
        num_scales = x_scales->dim() == 1 ? 1 : static_cast<int>(x_scales->size(1));
        x_scales_ptr = static_cast<float *>(x_scales->data_ptr());
        scale_token_stride = static_cast<int>(x_scales->stride(0));
        scale_hidden_stride = static_cast<int>(x_scales->stride(1));
    }

    at::Tensor dispatch_wait_recv_cost_stats_out;
    if (dispatch_wait_recv_cost_stats.has_value()) {
        EP_HOST_ASSERT(dispatch_wait_recv_cost_stats->scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(dispatch_wait_recv_cost_stats->dim() == 1 and dispatch_wait_recv_cost_stats->is_contiguous());
        EP_HOST_ASSERT(dispatch_wait_recv_cost_stats->size(0) == num_ranks);
        dispatch_wait_recv_cost_stats_out = dispatch_wait_recv_cost_stats.value();
    }

    int send_per_group = 3;  // (send_to_expert_num, send_to_expert_offset, send_rank_tokens)

    auto send_data = at::zeros({num_experts * send_per_group}, at::dtype(at::kInt).device(x.device()));
    int64_t send_count = send_per_group * num_local_experts * num_ranks;

    auto send_data_offset = at::zeros({num_experts}, at::dtype(at::kInt).device(x.device()));
    at::Tensor recv_data = at::zeros({num_experts * send_per_group}, at::dtype(at::kInt).device(x.device()));

    // get ep name
    char hcom_ep_name[HCOMM_NAME_LEN];
    if (!moe_all_to_all_group_name.empty()) {
        std::memcpy(hcom_ep_name, moe_all_to_all_group_name.data(), moe_all_to_all_group_name.size() + 1);
    } else {
        HCCL_CHECK(HcclGetCommName(ep_comm, hcom_ep_name));
    }

    int64_t local_rank_size = num_ranks;
    int64_t local_rank_id = rank % local_rank_size;
    auto new_num_tokens_per_expert = num_tokens_per_expert.value();

    EXEC_NPU_CMD(aclnnNotifyDispatch, send_data, new_num_tokens_per_expert, send_count, num_tokens,
                 hcom_ep_name,  // commGroup
                 num_ranks,     // rankSize
                 rank,          // rankId
                 local_rank_size, local_rank_id, send_data_offset, recv_data);

    auto options_cpu = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    std::vector<int32_t> local_expert_acc(num_experts, 0);
    auto send_token_idx_cpu = at::zeros({num_tokens, num_topk}, options_cpu);
    auto send_token_idx_ptr = send_token_idx_cpu.data_ptr<int>();

    auto topk_idx_cpu = new_topk_idx.to(at::kCPU);
    auto topk_idx_ptr = topk_idx_cpu.data_ptr<int64_t>();
    for (int i = 0; i < num_tokens; ++i) {
        for (int j = 0; j < num_topk; ++j) {
            int64_t expert_idx = topk_idx_ptr[i * num_topk + j];
            if (expert_idx >= 0) {
                int32_t cnt = local_expert_acc[expert_idx];
                send_token_idx_ptr[i * num_topk + j] = cnt;
                local_expert_acc[expert_idx]++;
            }
        }
    }

    EP_HOST_ASSERT(recv_data.dim() == 1 and recv_data.is_contiguous());
    EP_HOST_ASSERT(recv_data.size(0) % num_experts == 0);
    at::Tensor recv_offset_cpu = at::zeros({num_experts}, options_cpu);
    at::Tensor recv_count_cpu = at::zeros({num_experts}, options_cpu);
    auto recv_data_cpu = recv_data.to(at::kCPU);
    auto recv_data_ptr = recv_data_cpu.data_ptr<int>();
    auto recv_count_ptr = recv_count_cpu.data_ptr<int>();
    auto recv_offset_ptr = recv_offset_cpu.data_ptr<int>();
    int total_recv_tokens = 0;
    int num_max_dispatch_tokens_per_rank = 0;
    std::vector<int> num_recv_tokens_per_expert_list;

    for (int64_t local_e = 0; local_e < num_local_experts; ++local_e) {
        int64_t local_expert_recv_tokens = 0;
        for (int64_t src_rank = 0; src_rank < num_ranks; ++src_rank) {
            int64_t index = local_e * num_ranks + src_rank;
            int64_t pair_idx = send_per_group * (src_rank * num_local_experts + local_e);

            int recv_cnt = recv_data_ptr[pair_idx];             // count from this src_rank for this global_expert
            int recv_off = recv_data_ptr[pair_idx + 1];         // offset in that src_rank's window
            int send_num_tokens = recv_data_ptr[pair_idx + 2];  // all bs from rank

            total_recv_tokens += recv_cnt;
            recv_count_ptr[index] = total_recv_tokens;
            recv_offset_ptr[index] = recv_off;
            num_max_dispatch_tokens_per_rank = std::max(num_max_dispatch_tokens_per_rank, send_num_tokens);

            local_expert_recv_tokens += recv_cnt;
        }
        num_recv_tokens_per_expert_list.push_back(local_expert_recv_tokens);
    }

    at::Tensor expert_ids = new_topk_idx.to(at::kInt);
    int64_t tp_size = 1;
    int64_t tp_rank = 0;
    int64_t quant_mode = use_quant ? DYNAMIC_SCALES : NO_SCALES;
    int64_t global_bs = static_cast<int64_t>(
        std::max(num_max_dispatch_tokens_per_rank * num_ranks, static_cast<int64_t>(num_worst_tokens)));

    auto send_token_idx = send_token_idx_cpu.to(x.device());
    auto recv_offset = recv_offset_cpu.to(x.device());
    auto recv_count = recv_count_cpu.to(x.device());

    int num_recv_tokens = (total_recv_tokens == 0) ? 1 : total_recv_tokens;
    auto expandx_out = use_quant ? at::zeros({num_recv_tokens, hidden}, at::dtype(at::kChar).device(x.device()))
                                 : at::zeros({num_recv_tokens, hidden}, x.options());
    auto dynamic_scales_out = at::zeros({num_recv_tokens}, at::dtype(at::kFloat).device(x.device()));
    auto expand_idx_out = at::zeros({num_recv_tokens * 3}, at::dtype(at::kInt).device(x.device()));

    EXEC_NPU_CMD(aclnnCamMoeDispatchNormal, new_x, expert_ids, send_data_offset, send_token_idx, recv_offset,
                 recv_count, hcom_ep_name,
                 num_ranks,  // rankSize
                 rank,       // rankId
                 hcom_ep_name, tp_size, tp_rank, num_experts, quant_mode, global_bs, expandx_out, dynamic_scales_out,
                 expand_idx_out, dispatch_wait_recv_cost_stats_out);

    auto recv_topk_idx = std::optional<at::Tensor>();
    auto recv_topk_weights = std::optional<at::Tensor>();
    auto expand_idx_out_cpu = expand_idx_out.to(torch::kCPU);
    if (topk_idx.has_value()) {
        recv_topk_idx = at::empty({total_recv_tokens, num_topk}, topk_idx->options());
        recv_topk_weights = at::empty({total_recv_tokens, num_topk}, topk_weights->options());
    }
    // Wait streams
    std::optional<EventHandle> event;

    auto rank_prefix_matrix = at::empty({num_ranks, num_ranks}, at::dtype(at::kInt).device(x.device()));
    auto channel_prefix_matrix = at::empty({num_ranks, num_channels}, at::dtype(at::kInt).device(x.device()));
    auto recv_channel_prefix_matrix = at::empty({num_ranks, num_channels}, at::dtype(at::kInt).device(x.device()));

    // Return values
    return {expandx_out,
            dynamic_scales_out,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
            rank_prefix_matrix,
            channel_prefix_matrix,
            recv_channel_prefix_matrix,
            expand_idx_out,
            recv_count,
            event};
}

void Buffer::clean_low_latency_buffer(int num_max_dispatch_tokens_per_rank, int hidden, int num_experts)
{
    return;
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<EventHandle>>
Buffer::intranode_combine(const torch::Tensor &x, const torch::Tensor &topk_idx,
                          const std::optional<torch::Tensor> &topk_weights, const torch::Tensor &src_idx,
                          const torch::Tensor &send_head, const std::optional<at::Tensor> &combine_send_cost_stats)
{
    EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous());
    at::Tensor recv_x = x;

    at::Tensor topk_idx_p = topk_idx;
    if (this->is_padding) {
        topk_idx_p = this->new_topk_idx;
    }

    auto topk_idx_int32 = topk_idx_p.to(at::kInt);
    at::Tensor expand_ids = topk_idx_int32;
    at::Tensor token_src_info = src_idx;
    at::Tensor ep_send_counts = send_head;
    auto device = x.device();

    const int num_tokens = topk_idx_p.size(0);
    const int num_topk = topk_idx_p.size(1);
    at::Tensor expert_scales;
    // for padding
    if (topk_weights.has_value()) {
        if (!this->is_padding) {
            expert_scales = topk_weights.value();
        } else {
            std::vector<at::Tensor> weight_blocks;
            if (topk_weights->size(0) != 0) {
                weight_blocks.emplace_back(topk_weights.value());
            }
            for (int i = 0; i < this->padding_cnt; i++) {
                at::Tensor tmp_weight = torch::arange(0, num_topk, topk_weights->options()).reshape({1, num_topk});
                weight_blocks.emplace_back(tmp_weight);
            }
            expert_scales = torch::cat(weight_blocks, 0);
        }
    } else {
        expert_scales = at::ones({num_tokens, num_topk}, at::dtype(at::kFloat).device(device));
    }

    at::Tensor combine_send_cost_stats_out;
    if (combine_send_cost_stats.has_value()) {
        EP_HOST_ASSERT(combine_send_cost_stats->scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(combine_send_cost_stats->dim() == 1 and combine_send_cost_stats->is_contiguous());
        EP_HOST_ASSERT(combine_send_cost_stats->size(0) == num_ranks);
        combine_send_cost_stats_out = combine_send_cost_stats.value();
    }

    int64_t hidden = static_cast<int>(recv_x.size(1));
    at::Tensor tp_send_counts = at::empty({1}, at::dtype(at::kInt).device(device));
    int64_t tp_world_size = 1;
    int64_t tp_rankId = 0;
    int64_t moe_expert_number = send_head.size(0);
    int64_t global_bs = topk_idx_p.size(0) * num_ranks;

    // get ep & tp name
    char hcom_ep_name[HCOMM_NAME_LEN];
    if (!moe_all_to_all_group_name.empty()) {
        std::memcpy(hcom_ep_name, moe_all_to_all_group_name.data(), moe_all_to_all_group_name.size() + 1);
    } else {
        HCCL_CHECK(HcclGetCommName(ep_comm, hcom_ep_name));
    }

    // Combine data
    auto combined_x = torch::empty({expert_scales.size(0), hidden}, x.options());
    std::optional<torch::Tensor> recv_topk_weights;
    std::optional<EventHandle> event;

    EXEC_NPU_CMD(aclnnCamMoeCombineNormal, recv_x, token_src_info, ep_send_counts, expert_scales, tp_send_counts,
                 hcom_ep_name, num_ranks, rank, hcom_ep_name, tp_world_size, tp_rankId, moe_expert_number, global_bs,
                 combined_x, combine_send_cost_stats_out);

    if (this->is_padding) {
        if (this->padding_cnt == PADDING_SIZE) {
            combined_x = this->ori_x;
        } else {
            combined_x = combined_x.slice(0, 0, PADDING_SIZE - this->padding_cnt);
        }
        is_padding = false;
    }

    return {combined_x, recv_topk_weights, event};
}

std::tuple<at::Tensor, std::optional<at::Tensor>, at::Tensor, at::Tensor, at::Tensor, std::optional<EventHandle>,
           std::optional<std::function<void()>>>
Buffer::low_latency_dispatch(const at::Tensor &x, const at::Tensor &topk_idx,
                             const std::optional<at::Tensor> &cumulative_local_expert_recv_stats,
                             int64_t num_max_dispatch_tokens_per_rank, int64_t num_experts, bool use_fp8,
                             bool round_scale, bool use_ue8m0, bool async, bool return_recv_hook)
{
    this->is_padding = false;
    EP_HOST_ASSERT(low_latency_mode);
    at::Tensor new_x = x;
    this->new_topk_idx = topk_idx;
    if (topk_idx.size(0) < PADDING_SIZE) {
        this->is_padding = true;
        this->padding_cnt = PADDING_SIZE - topk_idx.size(0);
        std::vector<at::Tensor> x_blocks;
        std::vector<at::Tensor> topk_blocks;
        if (topk_idx.size(0) != 0) {
            x_blocks.emplace_back(x);
            topk_blocks.emplace_back(topk_idx);
        } else {
            this->ori_x = x.clone();
        }
        int topk = static_cast<int>(new_topk_idx.size(1));
        for (int i = 0; i < this->padding_cnt; i++) {
            at::Tensor tmp_x = torch::ones({1, x.size(1)}, x.options());
            at::Tensor tmp_topk = torch::arange(0, topk, topk_idx.options()).reshape({1, topk});
            x_blocks.emplace_back(tmp_x);
            topk_blocks.emplace_back(tmp_topk);
        }
        new_x = torch::cat(x_blocks, 0);
        this->new_topk_idx = torch::cat(topk_blocks, 0);
    }

    auto num_tokens = static_cast<int>(new_x.size(0)), hidden = static_cast<int>(new_x.size(1));
    auto num_scales = hidden / 128, num_topk = static_cast<int>(new_topk_idx.size(1));
    auto num_local_experts = num_experts / (num_ranks - shared_expert_rank_num);

    int64_t global_bs = std::max(new_topk_idx.size(0), num_max_dispatch_tokens_per_rank) * num_ranks;
    auto num_max_tokens = 0;
    if (rank < shared_expert_rank_num) {
        num_max_tokens = global_bs / shared_expert_rank_num;
        num_local_experts = 1;
    } else {  // moe expert
        num_max_tokens = global_bs * num_local_experts;
    }
    auto max_size = std::max(num_tokens * num_topk, num_max_tokens * 128);

    // Allocate packed tensors
    auto device = new_x.device();
    auto packed_recv_x =
        at::empty({num_max_tokens, hidden}, new_x.options().dtype(use_fp8 ? at::kChar : at::kBFloat16));
    auto packed_recv_x_scales = at::empty({num_max_tokens}, at::dtype(at::kFloat).device(device));
    auto expandIdx = at::empty({max_size}, at::dtype(at::kInt).device(device));
    auto ep_recv_count = at::empty({num_local_experts * num_ranks}, at::dtype(at::kInt).device(device));
    auto tp_recv_count = at::empty({1}, at::dtype(at::kInt).device(device));
    auto packed_recv_count = at::empty({num_local_experts}, at::dtype(at::kLong).device(device));
    auto expandScales = at::empty({1}, at::dtype(at::kFloat).device(device));
    at::Tensor scales;
    at::Tensor activateMask;
    auto expert_scales = at::empty({1}, at::dtype(at::kFloat).device(device));
    int64_t quant_mode = use_fp8 ? 2 : 0;
    int64_t tp_size = 1;
    int64_t tp_rank = 0;
    int64_t expert_shard_type = 0;
    int64_t expert_token_nums_type = 1;

    std::string comm_log = "0";
    std::vector<char> comm_log_buf(comm_log.begin(), comm_log.end());
    comm_log_buf.push_back('\0');
    char *comm_log_ptr = comm_log_buf.data();

    // get ep & tp name
    char hcom_ep_name[HCOMM_NAME_LEN];
    if (!moe_all_to_all_group_name.empty()) {
        std::memcpy(hcom_ep_name, moe_all_to_all_group_name.data(), moe_all_to_all_group_name.size() + 1);
    } else {
        HCCL_CHECK(HcclGetCommName(ep_comm, hcom_ep_name));
    }
    char hcom_tp_name[HCOMM_NAME_LEN] = {0};

    EXEC_NPU_CMD(aclnnMoeDistributeDispatchV2, new_x, new_topk_idx,
                 scales,         // smooth scales,
                 activateMask,   // activateMask
                 expert_scales,  // expert_scales
                 hcom_ep_name,   // ep
                 num_ranks,      // rankSize
                 rank,           // rankId
                 num_experts,
                 hcom_tp_name,            // tp
                 tp_size,                 // tp_size
                 tp_rank,                 // tp_rank
                 expert_shard_type,       // expert_shard_type
                 shared_expert_num,       // shared_expert_num
                 shared_expert_rank_num,  // shared_expert_rank_num
                 quant_mode,
                 global_bs,               // global_bs
                 expert_token_nums_type,  // expert_token_nums_type
                 comm_log_ptr, packed_recv_x,
                 packed_recv_x_scales,  // dynamicScalesOut
                 expandIdx,
                 packed_recv_count,  // expertTokenNumsOut
                 ep_recv_count, tp_recv_count, expandScales);

    // Wait streams
    std::optional<EventHandle> event;

    // Return values
    return {packed_recv_x, packed_recv_x_scales,        packed_recv_count, expandIdx, ep_recv_count,
            event,         std::function<void()>([] {})};
}

int Buffer::get_rdma_rank() const
{
    return rdma_rank;
}

std::tuple<at::Tensor, std::optional<EventHandle>, std::optional<std::function<void()>>> Buffer::low_latency_combine(
    const at::Tensor &x, const at::Tensor &topk_idx, const at::Tensor &topk_weights, const at::Tensor &src_info,
    const at::Tensor &layout_range, int64_t num_max_dispatch_tokens_per_rank, int64_t num_experts,
    const at::Tensor &packed_recv_count, bool zero_copy, bool async, bool return_recv_hook,
    const std::optional<at::Tensor> &out)
{
    at::Tensor new_idx = topk_idx;
    at::Tensor new_scales = topk_weights;
    if (this->is_padding) {
        std::vector<at::Tensor> scales_blocks;
        if (this->padding_cnt != PADDING_SIZE) {
            scales_blocks.emplace_back(topk_weights);
        }
        for (int i = 0; i < this->padding_cnt; i++) {
            at::Tensor tmp_scales = torch::zeros({1, topk_weights.size(1)}, topk_weights.options());
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
    char hcom_ep_name[HCOMM_NAME_LEN];
    if (!moe_all_to_all_group_name.empty()) {
        std::memcpy(hcom_ep_name, moe_all_to_all_group_name.data(), moe_all_to_all_group_name.size() + 1);
    } else {
        HCCL_CHECK(HcclGetCommName(ep_comm, hcom_ep_name));
    }
    char hcom_tp_name[HCOMM_NAME_LEN] = {0};

    auto device = x.device();
    at::Tensor expand_x = x;
    at::Tensor expert_ids = new_idx;
    at::Tensor expand_idx = src_info;  // handle[0] = src_info
    at::Tensor ep_send_counts = layout_range;
    at::Tensor expert_scales = new_scales;
    at::Tensor tp_send_counts = at::empty({1}, at::dtype(at::kInt).device(device));
    at::Tensor x_active_mask, activation_scale, weight_scale, group_list, expand_scales;

    int64_t tp_world_size = 1;
    int64_t tp_rankId = 0;
    int64_t expert_shared_type = 0;
    int64_t global_bs = std::max(new_idx.size(0), num_max_dispatch_tokens_per_rank) * num_ranks;
    int64_t out_dtype = 0;
    int64_t comm_quant_mode = 0;
    int64_t group_list_type = 0;

    auto num_combined_tokens = static_cast<int>(new_scales.size(0));
    auto hidden = static_cast<int>(x.size(1));
    at::Tensor shared_expert_x{nullptr};
    at::Tensor combined_x = at::empty({num_combined_tokens, hidden}, x.options());
    std::optional<EventHandle> event;
    std::string comm_log = "0";
    std::vector<char> comm_log_buf(comm_log.begin(), comm_log.end());
    comm_log_buf.push_back('\0');
    char *comm_log_ptr = comm_log_buf.data();

    EXEC_NPU_CMD(aclnnMoeDistributeCombineV2, expand_x, expert_ids, expand_idx, ep_send_counts, expert_scales,
                 tp_send_counts, x_active_mask, activation_scale, weight_scale, group_list, expand_scales,
                 shared_expert_x, hcom_ep_name, num_ranks, rank, num_experts, hcom_tp_name, tp_world_size, tp_rankId,
                 expert_shared_type, shared_expert_num, shared_expert_rank_num, global_bs, out_dtype, comm_quant_mode,
                 group_list_type, comm_log_ptr, combined_x);
    if (this->is_padding) {
        if (this->padding_cnt == PADDING_SIZE) {
            combined_x = this->ori_x;
        } else {
            combined_x = combined_x.slice(0, 0, PADDING_SIZE - this->padding_cnt);
        }
        is_padding = false;
    }
    return {combined_x, event, std::function<void()>([] {})};
}

std::tuple<at::Tensor, std::optional<EventHandle>, std::optional<std::function<void()>>> Buffer::fused_deep_moe(
    const at::Tensor &x, const at::Tensor &expertIds, const at::Tensor &gmm1PermutedWeight,
    const at::Tensor &gmm1PermutedWeightScale, const at::Tensor &gmm2Weight, const at::Tensor &gmm2WeightScale,
    const at::Tensor &expertScalesOptional, int64_t num_max_dispatch_tokens_per_rank,
    int64_t num_experts, bool use_fp8)
{
    EP_HOST_ASSERT(expertIds.dim() == 2); 
    EP_HOST_ASSERT(expertScalesOptional.dim() == 2);

    this->is_padding = false;
    at::Tensor new_x = x;
    this->new_topk_idx = expertIds;
    at::Tensor new_scales = expertScalesOptional;

    if (expertIds.size(0) < PADDING_SIZE) {
        this->is_padding = true;
        this->padding_cnt = PADDING_SIZE - expertIds.size(0);

        std::vector<at::Tensor> x_blocks;
        std::vector<at::Tensor> idx_blocks;

        if (expertIds.size(0) != 0) {
            x_blocks.emplace_back(x);
            idx_blocks.emplace_back(expertIds);
        } else {
            this->ori_x = x.clone();  //  Store the original input when the batch is completely empty.
        }

        int topk = static_cast<int>(expertIds.size(1));
        for (int i = 0; i < this->padding_cnt; i++) {
            at::Tensor tmp_x = torch::ones({1, x.size(1)}, x.options());
            at::Tensor tmp_idx = torch::arange(0, topk, expertIds.options()).reshape({1, topk});
            x_blocks.emplace_back(tmp_x);
            idx_blocks.emplace_back(tmp_idx);
        }

        new_x = torch::cat(x_blocks, 0);
        this->new_topk_idx = torch::cat(idx_blocks, 0);

        // padding expertScalesOptional
        std::vector<at::Tensor> scales_blocks;
        if (this->padding_cnt != PADDING_SIZE) {
            scales_blocks.emplace_back(expertScalesOptional);
        }
        for (int i = 0; i < this->padding_cnt; i++) {
            at::Tensor tmp_scales = torch::zeros({1, expertScalesOptional.size(1)}, expertScalesOptional.options());
            scales_blocks.emplace_back(tmp_scales);
        }

        new_scales = torch::cat(scales_blocks, 0);
    }

    char hcom_ep_name[128];
    if (!moe_all_to_all_group_name.empty()) {
        std::memcpy(hcom_ep_name, moe_all_to_all_group_name.data(), moe_all_to_all_group_name.size() + 1);
    } else {
        HCCL_CHECK(HcclGetCommName(ep_comm, hcom_ep_name));
    }

    int64_t quantMode = use_fp8 ? 2 : 0;
    int64_t globalBs = std::max(new_topk_idx.size(0), num_max_dispatch_tokens_per_rank) * num_ranks;

    auto xShape = x.sizes();
    auto expertIdsShape = expertIds.sizes();
    int h = xShape[1];
        int bs = this->new_topk_idx.size(0);
    at::Tensor output = at::empty({bs, h}, x.options());

    EXEC_NPU_CMD(aclnnFusedDeepMoe,
        // input
        x,
        expertIds,
        gmm1PermutedWeight,
        gmm1PermutedWeightScale,
        gmm2Weight,
        gmm2WeightScale,
        static_cast<const std::nullptr_t&>(nullptr),
        new_scales,
        // attr
        hcom_ep_name,
        num_ranks,
        rank,
        num_experts,
        shared_expert_num,
        shared_expert_rank_num,
        quantMode,
        globalBs,
        // output 
        output);

     // ---------- Unpadding ----------
    if (this->is_padding) {
        if (expertIds.size(0) == 0) {
            output = this->ori_x;
        } else {
            output = output.slice(0, 0, PADDING_SIZE - this->padding_cnt);
        }
        this->is_padding = false;
    }

    std::optional<EventHandle> event;
    return {output, event, std::function<void()>([]{})};
}
}  // namespace deep_ep
