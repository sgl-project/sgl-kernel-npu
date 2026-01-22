#include <memory>
#include <cmath>
#include <pybind11/functional.h>

#include "hccl/hccl.h"
#include "exception.hpp"
#include "deep_ep.hpp"
#include "pytorch_npu_helper.hpp"

namespace deep_ep {
constexpr int PADDING_SIZE = 1;
constexpr size_t HCOMM_NAME_LEN = 128;
constexpr uint32_t NO_SCALES = 0;
constexpr uint32_t DYNAMIC_SCALES = 2;
constexpr int LOCAL_RANK_SIZE = 8;
constexpr int MAX_BATCH_SIZE = 4096;
constexpr int EXPERT_DATA_SIZE = 1 + MAX_BATCH_SIZE;  // 4097
constexpr int A3_MAX_HCCS_PEERS = 384;
constexpr int A2_MAX_HCCS_PEERS = 8;
constexpr uint32_t MAX_ROUNDS = 256;
constexpr uint32_t MIN_TOKENS_PER_ROUND = 32;
constexpr uint32_t MAX_TOKENS_PER_ROUND = 8192;
constexpr uint32_t MAX_TOTAL_TOKENS = 131072;

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
    const char *roundEnv = std::getenv("DEEPEP_NORMAL_LONG_SEQ_ROUND");
    const char *tokensEnv = std::getenv("DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS");
    this->combine_enable_long_seq = get_value_from_env("DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ", 0);
    bool roundSet = (roundEnv != nullptr);
    bool tokensSet = (tokensEnv != nullptr);

    // 检查设置状态
    EP_HOST_ASSERT(roundSet == tokensSet);

    // 未设置时使用默认值
    if (!roundSet && !tokensSet) {
        this->round = 1;
        this->per_round_tokens = 8192;
    } else {
        // 转换并验证数值
        char *end;
        long r = std::strtol(roundEnv, &end, 10);
        EP_HOST_ASSERT(*end == '\0' && r >= 1 && r <= MAX_ROUNDS);
        long t = std::strtol(tokensEnv, &end, 10);
        EP_HOST_ASSERT(*end == '\0' && t >= MIN_TOKENS_PER_ROUND && t <= MAX_TOKENS_PER_ROUND);
        // 验证乘积限制
        EP_HOST_ASSERT(r * t <= 131072);
        round = static_cast<int>(r);
        per_round_tokens = static_cast<int>(t);
    }

    soc_version = op::GetCurrentPlatformInfo().GetSocVersion();
    num_rdma_ranks = 1;
    num_nvl_ranks = num_ranks;
    rdma_rank = rank;
    nvl_rank = rank;
    if (soc_version == op::SocVersion::ASCEND910B) {
        num_rdma_ranks = std::max(static_cast<int64_t>(1), num_ranks / A2_MAX_HCCS_PEERS);
        num_nvl_ranks = std::min(num_ranks, static_cast<int64_t>(A2_MAX_HCCS_PEERS));
        rdma_rank = rank / A2_MAX_HCCS_PEERS;
        nvl_rank = rank % A2_MAX_HCCS_PEERS;
    }
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
    EP_HOST_ASSERT(topk_idx.size(0) <= round * per_round_tokens);

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
    const int local_ranksize = LOCAL_RANK_SIZE;
    auto server_num = num_ranks / local_ranksize;

    auto device = new_topk_idx.device();

    auto num_tokens_per_expert = at::zeros({round, num_experts}, at::dtype(at::kInt).device(device));
    auto num_tokens_per_rank = at::zeros({num_ranks}, at::dtype(at::kInt).device(device));
    auto is_token_in_rank = at::zeros({num_tokens, num_ranks}, at::dtype(at::kInt).device(device));
    const int notify_send_data_size =
        num_experts * EXPERT_DATA_SIZE + server_num + MAX_BATCH_SIZE * (1 + 2 * server_num + num_experts);
    /*
    The notify send data is constructed by 7 parameters and the 7 parameters are ordered as follows:
    1. the number of the tokens that every expert received from this NPU.
       size:[numExpert]
    2. The number of tokens received by each server from this NPU (deduplicated).
       size:[serverNum]
    3. The number of tokens sent from this NPU to each server (without deduplication).
       size:[MAX_BS, serverNum]
    4. The number of servers each token is sent to by this NPU.
       size:[MAX_BS]
    5. The order in which each token of this NPU is sent to various servers.
       size:[MAX_BS, serverNum]
    6. The order in which each token is sent to the expert.
       size:[MAX_BS, numTopk]
    7. The server offset of tokens received by each expert from this NPU.
       size:[numExpert, MAX_BS]
    */
    auto send_token_idx_small = at::zeros({num_tokens, num_topk}, at::dtype(at::kInt).device(device));
    auto notify_send_data = at::zeros({notify_send_data_size}, at::dtype(at::kInt).device(device));
    int32_t rank_id = static_cast<int>(rank);
    EXEC_NPU_CMD(aclnnDispatchLayout, new_topk_idx, num_tokens, num_ranks, num_experts, num_topk, local_ranksize,
                 per_round_tokens, rank_id, num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank,
                 notify_send_data, send_token_idx_small);

    this->notify_send_data = notify_send_data;
    this->send_token_idx_small = send_token_idx_small;
    this->notify_send_data_size = notify_send_data_size;

    std::optional<torch::Tensor> num_tokens_per_rdma_rank = std::nullopt;
    std::optional<EventHandle> output_event = std::nullopt;

    auto num_tokens_per_expert_one_dim = num_tokens_per_expert.flatten();
    return std::make_tuple(num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert_one_dim,
                           is_token_in_rank, output_event);
}

torch::Tensor Buffer::get_notify_send_data()
{
    return this->notify_send_data;
}

int Buffer::get_num_rdma_ranks() const
{
    return num_rdma_ranks;
}

int Buffer::get_rdma_rank() const
{
    return rdma_rank;
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

    at::Tensor expert_ids = new_topk_idx.to(at::kInt);
    int64_t tp_size = 1;
    int64_t tp_rank = 0;
    int64_t quant_mode = use_quant ? DYNAMIC_SCALES : NO_SCALES;
    auto recv_topk_idx = std::optional<at::Tensor>();
    auto recv_topk_weights = std::optional<at::Tensor>();
    // Wait streams
    std::optional<EventHandle> event;
    auto rank_prefix_matrix = at::empty({num_ranks, num_ranks}, at::dtype(at::kInt).device(x.device()));
    auto channel_prefix_matrix = at::empty({num_ranks, num_channels}, at::dtype(at::kInt).device(x.device()));
    auto recv_channel_prefix_matrix = at::empty({num_ranks, num_channels}, at::dtype(at::kInt).device(x.device()));

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
    EP_HOST_ASSERT(num_tokens_per_expert->scalar_type() == at::kInt);
    EP_HOST_ASSERT(num_tokens_per_rank->scalar_type() == at::kInt);

    // Shape and contiguous checks
    EP_HOST_ASSERT(new_x.dim() == 2 and new_x.is_contiguous());
    EP_HOST_ASSERT(num_tokens_per_expert->dim() == 1 and num_tokens_per_expert->is_contiguous());
    EP_HOST_ASSERT(num_tokens_per_expert->size(0) % num_ranks == 0);
    EP_HOST_ASSERT(num_tokens_per_rank->dim() == 1 and num_tokens_per_rank->is_contiguous());
    EP_HOST_ASSERT(num_tokens_per_rank->size(0) == num_ranks);

    auto num_tokens = static_cast<int>(new_x.size(0)), hidden = static_cast<int>(new_x.size(1));
    auto num_experts = static_cast<int64_t>(num_tokens_per_expert->size(0) / round);
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

    auto send_data = torch::empty({round, num_experts * send_per_group}, at::dtype(at::kInt).device(x.device()));
    int64_t send_count = send_per_group * num_local_experts * num_ranks * round;

    auto send_data_offset = torch::empty({round, num_experts}, at::dtype(at::kInt).device(x.device()));
    at::Tensor recv_data = torch::empty({round, num_experts * send_per_group}, at::dtype(at::kInt).device(x.device()));
    // get ep name
    char hcom_ep_name[HCOMM_NAME_LEN];
    if (!moe_all_to_all_group_name.empty()) {
        std::memcpy(hcom_ep_name, moe_all_to_all_group_name.data(), moe_all_to_all_group_name.size() + 1);
    } else {
        HCCL_CHECK(HcclGetCommName(ep_comm, hcom_ep_name));
    }
    at::Tensor total_recv_token = torch::empty({1}, at::dtype(at::kInt).device(x.device()));
    at::Tensor recv_offset = at::empty({round, num_experts}, at::dtype(at::kInt).device(x.device()));
    at::Tensor recv_count = at::empty({round, num_experts}, at::dtype(at::kInt).device(x.device()));
    at::Tensor max_bs = torch::empty({1}, at::dtype(at::kInt).device(x.device()));
    at::Tensor recv_tokens_per_expert =
        torch::empty({round * num_local_experts}, at::dtype(at::kInt).device(x.device()));
    at::Tensor expert_global_offset = at::empty({num_local_experts}, at::dtype(at::kInt).device(x.device()));
    at::Tensor srcrank_in_expert_offset =
        at::empty({num_local_experts * num_ranks}, at::dtype(at::kInt).device(x.device()));
    at::Tensor r_in_srcrank_offset =
        at::empty({num_local_experts * num_ranks * round}, at::dtype(at::kInt).device(x.device()));

    int64_t local_rank_size = num_ranks;
    int64_t local_rank_id = rank % local_rank_size;
    auto new_num_tokens_per_expert = num_tokens_per_expert.value();
    std::vector<int> num_recv_tokens_per_expert_list;
    // indicates the value type of the output num_recv_tokens_per_expert_list, with a range of [0, 1]
    // 0 means the prefix sum of the number of tokens received by each expert;
    // 1 means the number of tokens received by each expert (default)
    int expert_token_nums_type = get_value_from_env("MOE_EXPERT_TOKEN_NUMS_TYPE", 1);
    EP_HOST_ASSERT(expert_token_nums_type == 1 or expert_token_nums_type == 0);

    EXEC_NPU_CMD(aclnnNotifyDispatch, send_data, new_num_tokens_per_expert, send_count, num_tokens,
                 hcom_ep_name,  // commGroup
                 num_ranks,     // rankSize
                 rank,          // rankId
                 local_rank_size, local_rank_id, round, per_round_tokens, send_data_offset, recv_data, recv_count,
                 recv_offset, expert_global_offset, srcrank_in_expert_offset, r_in_srcrank_offset, total_recv_token,
                 max_bs, recv_tokens_per_expert);
    auto send_token_idx_small = this->send_token_idx_small;

    real_max_bs = static_cast<int64_t>(std::max(max_bs.item<int>(), static_cast<int>(num_worst_tokens)));

    // dispatch算子内部按照 min(per_round_tokens, real_max_bs)来预留显存
    int64_t global_bs = static_cast<int64_t>(std::min(static_cast<int64_t>(per_round_tokens), real_max_bs) * num_ranks);

    int64_t trt = total_recv_token.item<int>();
    int num_recv_tokens = (trt == 0) ? 1 : trt;
    auto expandx_out = use_quant ? torch::empty({num_recv_tokens, hidden}, at::dtype(at::kChar).device(x.device()))
                                 : torch::empty({num_recv_tokens, hidden}, x.options());
    auto dynamic_scales_out = torch::empty({num_recv_tokens}, at::dtype(at::kFloat).device(x.device()));
    auto expand_idx_out = torch::empty({num_recv_tokens * 3}, at::dtype(at::kInt).device(x.device()));
    if (topk_idx.has_value()) {
        recv_topk_idx = at::empty({trt, num_topk}, topk_idx->options());
        recv_topk_weights = at::empty({trt, num_topk}, topk_weights->options());
    }
    EXEC_NPU_CMD(aclnnCamMoeDispatchNormal, new_x, expert_ids, send_data_offset, send_token_idx_small, recv_offset,
                 recv_count, expert_global_offset, srcrank_in_expert_offset, r_in_srcrank_offset, hcom_ep_name,
                 num_ranks,  // rankSize
                 rank,       // rankId
                 hcom_ep_name, tp_size, tp_rank, num_experts, quant_mode, real_max_bs, global_bs, round,
                 per_round_tokens, expandx_out, dynamic_scales_out, expand_idx_out, dispatch_wait_recv_cost_stats_out);
    auto recv_token_per_exp_cpu = recv_tokens_per_expert.to(at::kCPU);
    auto recv_token_per_exp_ptr = recv_token_per_exp_cpu.data_ptr<int32_t>();

    int token_cnt = 0;
    // 多轮处理为一维
    std::vector<int> round_recv_tokens_per_expert;
    round_recv_tokens_per_expert.resize(num_local_experts);
    for (int r = 0; r < round; r++) {
        for (int local_e = 0; local_e < num_local_experts; ++local_e) {
            int current_tokens = static_cast<int>(recv_token_per_exp_ptr[r * num_local_experts + local_e]);
            token_cnt = round_recv_tokens_per_expert[local_e] + current_tokens;
            round_recv_tokens_per_expert[local_e] = token_cnt;
        }
    }

    token_cnt = 0;
    for (int local_e = 0; local_e < num_local_experts; ++local_e) {
        int current_tokens = static_cast<int>(round_recv_tokens_per_expert[local_e]);
        token_cnt = (expert_token_nums_type == 0) ? token_cnt + current_tokens : current_tokens;
        num_recv_tokens_per_expert_list.emplace_back(token_cnt);
    }

    auto recv_count_one_dim = recv_count.sum(0, false).to(at::kInt);
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
            recv_count_one_dim,
            event};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
Buffer::notify_verify(const at::Tensor &x, const std::optional<at::Tensor> &x_scales,
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

    at::Tensor expert_ids = new_topk_idx.to(at::kInt);
    int64_t tp_size = 1;
    int64_t tp_rank = 0;
    int64_t quant_mode = use_quant ? DYNAMIC_SCALES : NO_SCALES;
    auto recv_topk_idx = std::optional<at::Tensor>();
    auto recv_topk_weights = std::optional<at::Tensor>();
    // Wait streams
    std::optional<EventHandle> event;
    auto rank_prefix_matrix = at::empty({num_ranks, num_ranks}, at::dtype(at::kInt).device(x.device()));
    auto channel_prefix_matrix = at::empty({num_ranks, num_channels}, at::dtype(at::kInt).device(x.device()));
    auto recv_channel_prefix_matrix = at::empty({num_ranks, num_channels}, at::dtype(at::kInt).device(x.device()));

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
    EP_HOST_ASSERT(num_tokens_per_expert->scalar_type() == at::kInt);
    EP_HOST_ASSERT(num_tokens_per_rank->scalar_type() == at::kInt);

    // Shape and contiguous checks
    EP_HOST_ASSERT(new_x.dim() == 2 and new_x.is_contiguous());
    EP_HOST_ASSERT(num_tokens_per_expert->dim() == 1 and num_tokens_per_expert->is_contiguous());
    EP_HOST_ASSERT(num_tokens_per_expert->size(0) % num_ranks == 0);
    EP_HOST_ASSERT(num_tokens_per_rank->dim() == 1 and num_tokens_per_rank->is_contiguous());
    EP_HOST_ASSERT(num_tokens_per_rank->size(0) == num_ranks);

    auto num_tokens = static_cast<int>(new_x.size(0)), hidden = static_cast<int>(new_x.size(1));
    auto num_experts = static_cast<int64_t>(num_tokens_per_expert->size(0) / round);
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

    auto send_data = torch::empty({round, num_experts * send_per_group}, at::dtype(at::kInt).device(x.device()));
    int64_t send_count = send_per_group * num_local_experts * num_ranks * round;

    auto send_data_offset = torch::empty({round, num_experts}, at::dtype(at::kInt).device(x.device()));
    at::Tensor recv_data = torch::empty({round, num_experts * send_per_group}, at::dtype(at::kInt).device(x.device()));
    // get ep name
    char hcom_ep_name[HCOMM_NAME_LEN];
    if (!moe_all_to_all_group_name.empty()) {
        std::memcpy(hcom_ep_name, moe_all_to_all_group_name.data(), moe_all_to_all_group_name.size() + 1);
    } else {
        HCCL_CHECK(HcclGetCommName(ep_comm, hcom_ep_name));
    }
    at::Tensor total_recv_token = torch::empty({1}, at::dtype(at::kInt).device(x.device()));
    at::Tensor recv_offset = at::empty({round, num_experts}, at::dtype(at::kInt).device(x.device()));
    at::Tensor recv_count = at::empty({round, num_experts}, at::dtype(at::kInt).device(x.device()));
    at::Tensor max_bs = torch::empty({1}, at::dtype(at::kInt).device(x.device()));
    at::Tensor recv_tokens_per_expert =
        torch::empty({round * num_local_experts}, at::dtype(at::kInt).device(x.device()));
    at::Tensor expert_global_offset = at::empty({num_local_experts}, at::dtype(at::kInt).device(x.device()));
    at::Tensor srcrank_in_expert_offset =
        at::empty({num_local_experts * num_ranks}, at::dtype(at::kInt).device(x.device()));
    at::Tensor r_in_srcrank_offset =
        at::empty({num_local_experts * num_ranks * round}, at::dtype(at::kInt).device(x.device()));

    int64_t local_rank_size = num_ranks;
    int64_t local_rank_id = rank % local_rank_size;
    auto new_num_tokens_per_expert = num_tokens_per_expert.value();
    std::vector<int> num_recv_tokens_per_expert_list;
    // indicates the value type of the output num_recv_tokens_per_expert_list, with a range of [0, 1]
    // 0 means the prefix sum of the number of tokens received by each expert;
    // 1 means the number of tokens received by each expert (default)
    int expert_token_nums_type = get_value_from_env("MOE_EXPERT_TOKEN_NUMS_TYPE", 1);
    EP_HOST_ASSERT(expert_token_nums_type == 1 or expert_token_nums_type == 0);

    EXEC_NPU_CMD(aclnnNotifyDispatch, send_data, new_num_tokens_per_expert, send_count, num_tokens,
                 hcom_ep_name,  // commGroup
                 num_ranks,     // rankSize
                 rank,          // rankId
                 local_rank_size, local_rank_id, round, per_round_tokens, send_data_offset, recv_data, recv_count,
                 recv_offset, expert_global_offset, srcrank_in_expert_offset, r_in_srcrank_offset, total_recv_token,
                 max_bs, recv_tokens_per_expert);

    return {recv_data,           recv_count,       recv_offset, expert_global_offset,  srcrank_in_expert_offset,
            r_in_srcrank_offset, total_recv_token, max_bs,      recv_tokens_per_expert};
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
                if (topk_weights.has_value()) {
                    at::Tensor tmp_weight = torch::arange(0, num_topk, topk_weights->options()).reshape({1, num_topk});
                    weight_blocks.emplace_back(tmp_weight);
                }
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

    int32_t round = this->combine_enable_long_seq ? this->round : 1;
    int32_t per_round_tokens = this->combine_enable_long_seq ? this->per_round_tokens : MAX_TOKENS_PER_ROUND;
    EXEC_NPU_CMD(aclnnCamMoeCombineNormal, recv_x, token_src_info, ep_send_counts, expert_scales, topk_idx_int32,
                 tp_send_counts, hcom_ep_name, num_ranks, rank, hcom_ep_name, tp_world_size, tp_rankId,
                 moe_expert_number, real_max_bs, round, per_round_tokens, combined_x, combine_send_cost_stats_out);

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

std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<torch::Tensor>, std::optional<torch::Tensor>,
           std::vector<int>, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           std::optional<EventHandle>>
Buffer::internode_dispatch(
    const torch::Tensor &x, const std::optional<torch::Tensor> &x_scales, const std::optional<torch::Tensor> &topk_idx,
    const std::optional<torch::Tensor> &topk_weights, const std::optional<torch::Tensor> &num_tokens_per_rank,
    const std::optional<torch::Tensor> &num_tokens_per_rdma_rank, const torch::Tensor &is_token_in_rank,
    const std::optional<torch::Tensor> &num_tokens_per_expert, const Config &config,
    std::optional<EventHandle> &previous_event, bool async, bool allocate_on_comm_stream, bool use_quant)
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
            at::Tensor tmp_x = torch::zeros({1, x.size(1)}, x.options());
            x_blocks.emplace_back(tmp_x);
        }
        new_x = torch::cat(x_blocks, 0);
    }
    EP_HOST_ASSERT(num_tokens_per_rank.has_value());
    EP_HOST_ASSERT(num_tokens_per_expert.has_value());

    // Type checks
    EP_HOST_ASSERT(num_tokens_per_expert->scalar_type() == at::kInt);
    EP_HOST_ASSERT(num_tokens_per_rank->scalar_type() == at::kInt);

    // Shape and contiguous checks
    EP_HOST_ASSERT(new_x.dim() == 2 and new_x.is_contiguous());
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

    auto device = x.device();
    at::Tensor new_topk_weights;
    // for padding
    if (topk_weights.has_value()) {
        if (!this->is_padding) {
            new_topk_weights = topk_weights.value();
        } else {
            std::vector<at::Tensor> weight_blocks;
            if (topk_weights->size(0) != 0) {
                weight_blocks.emplace_back(topk_weights.value());
            }
            for (int i = 0; i < this->padding_cnt; i++) {
                at::Tensor tmp_weight = torch::arange(0, num_topk, topk_weights->options()).reshape({1, num_topk});
                weight_blocks.emplace_back(tmp_weight);
            }
            new_topk_weights = torch::cat(weight_blocks, 0);
        }
    } else {
        new_topk_weights = at::ones({num_tokens, num_topk}, at::dtype(at::kFloat).device(device));
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

    // dispatch normal param
    int64_t tp_size = 1;
    int64_t tp_rank = 0;
    int64_t expertShardType = 0;
    int64_t sharedExpertNum = 1;
    int64_t sharedExpertRankNum = 0;
    int64_t expertTokenNumsType = 0;

    int64_t quant_mode = use_quant ? DYNAMIC_SCALES : NO_SCALES;
    int64_t global_bs = static_cast<int64_t>(MAX_BATCH_SIZE * num_ranks);
    at::Tensor expert_ids = new_topk_idx.to(at::kInt);
    at::Tensor xActiveMask = at::empty({1}, at::dtype(at::kInt).device(x.device()));

    auto expertTokenNums = at::zeros({1}, at::dtype(at::kLong).device(x.device()));
    auto epRecvCount = at::zeros({1}, at::dtype(at::kInt).device(x.device()));
    auto tpRecvCount = at::zeros({1}, at::dtype(at::kInt).device(x.device()));
    at::Tensor dispatch_wait_recv_cost_stats_out;
    auto recv_topk_idx = std::optional<at::Tensor>();
    auto recv_topk_weights = std::optional<at::Tensor>();
    // Wait streams
    std::optional<EventHandle> event;

    int64_t local_rank_size = A2_MAX_HCCS_PEERS;
    int32_t server_num = num_ranks / local_rank_size;
    int64_t local_rank_id = rank % local_rank_size;
    auto new_num_tokens_per_expert = num_tokens_per_expert.value();
    std::vector<int> num_recv_tokens_per_expert_list;
    // indicates the value type of the output num_recv_tokens_per_expert_list, with a range of [0, 1]
    // 0 means the prefix sum of the number of tokens received by each expert;
    // 1 means the number of tokens received by each expert (default)
    int expert_token_nums_type = get_value_from_env("MOE_EXPERT_TOKEN_NUMS_TYPE", 1);
    EP_HOST_ASSERT(expert_token_nums_type == 1 or expert_token_nums_type == 0);

    // Corresponding to the output data and length of the layout
    auto new_send_data = this->notify_send_data;
    int send_count = this->notify_send_data_size;

    auto send_data_offset = at::empty({num_experts}, at::dtype(at::kInt).device(x.device()));
    at::Tensor tmp_data =
        at::empty({send_count * num_ranks}, at::dtype(at::kInt).device(x.device()));  // 给notify算子用来临时存数的空间
    at::Tensor recv_data = at::empty({send_count * num_ranks}, at::dtype(at::kInt).device(x.device()));
    at::Tensor token_server_idx =
        at::empty({MAX_BATCH_SIZE, server_num}, at::dtype(at::kInt).device(x.device()));  // offset_outer
    at::Tensor token_unique_per_server = at::empty({server_num}, at::dtype(at::kInt).device(x.device()));
    at::Tensor ep_rank_token_cnt =
        at::empty({num_experts, num_ranks}, at::dtype(at::kInt).device(x.device()));  // 包含全局的
    // The number of tokens received by each expert on this rank, not a prefix sum
    at::Tensor recv_tokens_per_expert = at::empty({num_local_experts}, at::dtype(at::kLong).device(x.device()));
    at::Tensor src_offset_rank_token_idx =
        at::empty({num_experts, num_ranks, MAX_BATCH_SIZE}, at::dtype(at::kInt).device(x.device()));
    at::Tensor dst_offset_rank_token_idx =
        at::empty({num_experts, num_ranks, MAX_BATCH_SIZE}, at::dtype(at::kInt).device(x.device()));
    // The offsetInner for the current rank and the peer rank
    at::Tensor offset_inner = at::empty({2, MAX_BATCH_SIZE, num_experts}, at::dtype(at::kInt).device(x.device()));
    at::Tensor count_outer = at::empty({MAX_BATCH_SIZE}, at::dtype(at::kInt).device(x.device()));
    at::Tensor expand_idx = at::empty({MAX_BATCH_SIZE, num_experts}, at::dtype(at::kInt).device(x.device()));
    at::Tensor total_recv_token = torch::empty({1}, at::dtype(at::kInt).device(x.device()));

    // get ep name
    char hcom_ep_name[HCOMM_NAME_LEN];
    if (!moe_all_to_all_group_name.empty()) {
        std::memcpy(hcom_ep_name, moe_all_to_all_group_name.data(), moe_all_to_all_group_name.size() + 1);
    } else {
        HCCL_CHECK(HcclGetCommName(ep_comm, hcom_ep_name));
    }

    EXEC_NPU_CMD(aclnnNotifyDispatchA2, new_send_data, new_num_tokens_per_expert, tmp_data, send_count, num_tokens,
                 num_topk, num_experts,
                 hcom_ep_name,  // commGroup
                 num_ranks,     // rankSize
                 rank,          // rankId
                 local_rank_size, local_rank_id,
                 send_data_offset,  // A2 not use
                 recv_data, token_server_idx, token_unique_per_server, ep_rank_token_cnt, recv_tokens_per_expert,
                 src_offset_rank_token_idx, dst_offset_rank_token_idx, offset_inner, count_outer, expand_idx,
                 total_recv_token);

    int total_count = total_recv_token.item<int>();
    int num_recv_tokens = (total_count == 0) ? 1 : total_count;

    auto expandx_out = use_quant ? at::empty({num_recv_tokens, hidden}, at::dtype(at::kChar).device(x.device()))
                                 : at::empty({num_recv_tokens, hidden}, x.options());
    auto dynamic_scales_out = at::empty({num_recv_tokens}, at::dtype(at::kFloat).device(x.device()));
    auto expand_scales = at::empty({num_recv_tokens}, at::dtype(at::kFloat).device(x.device()));
    if (topk_idx.has_value()) {
        recv_topk_idx = at::empty({total_count, num_topk}, topk_idx->options());
        recv_topk_weights = at::empty({total_count, num_topk}, topk_weights->options());
    }

    EXEC_NPU_CMD(aclnnDispatchNormalA2, new_x, expert_ids, x_scales, xActiveMask, new_topk_weights, token_server_idx,
                 token_unique_per_server, ep_rank_token_cnt, src_offset_rank_token_idx, dst_offset_rank_token_idx,
                 hcom_ep_name, num_ranks, rank, num_experts, hcom_ep_name, tp_size, tp_rank, expertShardType,
                 sharedExpertNum, sharedExpertRankNum, quant_mode, global_bs, expertTokenNumsType, expandx_out,
                 dynamic_scales_out, expand_idx, expertTokenNums, epRecvCount, expand_scales,
                 dispatch_wait_recv_cost_stats_out);

    auto recv_token_per_exp_cpu = recv_tokens_per_expert.to(at::kCPU);
    auto recv_token_per_exp_ptr = recv_token_per_exp_cpu.data_ptr<int64_t>();

    int token_cnt = 0;
    for (int local_e = 0; local_e < num_local_experts; ++local_e) {
        int current_tokens = static_cast<int>(recv_token_per_exp_ptr[local_e]);
        token_cnt = (expert_token_nums_type == 0) ? token_cnt + current_tokens : current_tokens;
        num_recv_tokens_per_expert_list.emplace_back(token_cnt);
    }

    return {expandx_out,
            dynamic_scales_out,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
            expand_idx,
            ep_rank_token_cnt,
            offset_inner,
            token_server_idx,
            count_outer,
            expand_scales,
            event};
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<EventHandle>> Buffer::internode_combine(
    const torch::Tensor &x, const torch::Tensor &topk_idx, const std::optional<torch::Tensor> &topk_weights,
    const torch::Tensor &src_idx, const torch::Tensor &send_head, const torch::Tensor &offsetInner,
    const torch::Tensor &offsetOuter, const torch::Tensor &countOuter, const torch::Tensor &expand_scales)
{
    EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous());
    at::Tensor recv_x = x;

    at::Tensor topk_idx_p = topk_idx;
    if (this->is_padding) {
        topk_idx_p = this->new_topk_idx;
    }

    auto topk_idx_int32 = topk_idx_p.to(at::kInt);
    at::Tensor expert_ids = topk_idx_int32;
    // In the A2 implementation, the tensor is expanded from [bs, k] to [bs, num_expert].
    at::Tensor expand_idx = src_idx;
    // A2 needs global send counts, [num_expert, num_rank]
    at::Tensor ep_send_counts = send_head;
    auto device = x.device();

    const int num_tokens = topk_idx_p.size(0);
    const int num_topk = topk_idx_p.size(1);
    at::Tensor expert_scales = at::empty({1}, at::dtype(at::kFloat).device(x.device()));

    int64_t hidden = static_cast<int>(recv_x.size(1));
    at::Tensor tp_send_counts = at::empty({1}, at::dtype(at::kInt).device(device));
    int64_t tp_world_size = 1;
    int64_t tp_rankId = 0;
    int64_t moe_expert_number = send_head.size(0);
    int64_t global_bs = static_cast<int64_t>(MAX_BATCH_SIZE * num_ranks);

    // get ep & tp name
    char hcom_ep_name[HCOMM_NAME_LEN];
    if (!moe_all_to_all_group_name.empty()) {
        std::memcpy(hcom_ep_name, moe_all_to_all_group_name.data(), moe_all_to_all_group_name.size() + 1);
    } else {
        HCCL_CHECK(HcclGetCommName(ep_comm, hcom_ep_name));
    }

    // Combine data
    auto combined_x = torch::empty({new_topk_idx.size(0), hidden}, x.options());
    std::optional<torch::Tensor> recv_topk_weights;
    std::optional<EventHandle> event;
    at::Tensor x_active_mask, activation_scale, weight_scale, group_list;
    int64_t expert_shared_type = 0;
    int64_t out_dtype = 0;
    int64_t comm_quant_mode = 0;
    int64_t group_list_type = 0;

    EXEC_NPU_CMD(aclnnMoeDistributeCombineA2, recv_x, expert_ids, expand_idx, ep_send_counts, expert_scales,
                 tp_send_counts, x_active_mask, activation_scale, weight_scale, group_list, expand_scales, offsetInner,
                 offsetOuter, countOuter, hcom_ep_name, num_ranks, rank, moe_expert_number, hcom_ep_name, tp_world_size,
                 tp_rankId, expert_shared_type, shared_expert_num, shared_expert_rank_num, global_bs, out_dtype,
                 comm_quant_mode, group_list_type, combined_x);

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

    EP_HOST_ASSERT(num_max_dispatch_tokens_per_rank >= new_x.size(0));

    auto num_tokens = static_cast<int>(new_x.size(0)), hidden = static_cast<int>(new_x.size(1));
    auto num_scales = hidden / 128, num_topk = static_cast<int>(new_topk_idx.size(1));
    auto num_local_experts = num_experts / (num_ranks - shared_expert_rank_num);
    int64_t global_bs = num_max_dispatch_tokens_per_rank * num_ranks;
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

    int32_t server_num = num_ranks / LOCAL_RANK_SIZE;
    at::Tensor ep_recv_count =
        at::empty({num_local_experts * num_ranks}, at::dtype(at::kInt).device(device));  // A2 non-layered / A3
    auto tp_recv_count = at::empty({1}, at::dtype(at::kInt).device(device));
    auto packed_recv_count = at::empty({num_local_experts}, at::dtype(at::kLong).device(device));
    at::Tensor scales;
    at::Tensor active_mask;
    int enable_neg_one = get_value_from_env("MOE_ENABLE_TOPK_NEG_ONE", 0);
    int64_t quant_mode = use_fp8 ? 2 : 0;
    int64_t tp_size = 1;
    int64_t tp_rank = 0;
    int64_t expert_shard_type = 0;
    int outType = get_value_from_env("MOE_EXPERT_TOKEN_NUMS_TYPE", 1);
    char *comm_alg;
    int64_t expert_token_nums_type = outType;

    // get ep & tp name
    char hcom_ep_name[HCOMM_NAME_LEN];
    if (!moe_all_to_all_group_name.empty()) {
        std::memcpy(hcom_ep_name, moe_all_to_all_group_name.data(), moe_all_to_all_group_name.size() + 1);
    } else {
        HCCL_CHECK(HcclGetCommName(ep_comm, hcom_ep_name));
    }
    char hcom_tp_name[HCOMM_NAME_LEN] = {0};
    // Wait streams
    std::optional<EventHandle> event;
    bool isLayered = false;

    if (soc_version == op::SocVersion::ASCEND910B) {
        const char *hcclIntraPcieEnable = getenv("HCCL_INTRA_PCIE_ENABLE");
        const char *hcclIntraRoceEnable = getenv("HCCL_INTRA_ROCE_ENABLE");
        if (hcclIntraPcieEnable != nullptr && hcclIntraRoceEnable != nullptr && strcmp(hcclIntraPcieEnable, "1") == 0 &&
            strcmp(hcclIntraRoceEnable, "0") == 0) {  // A2 layered
            isLayered = true;
            int64_t recv_count_tensor_size = num_experts + 2 * global_bs * num_topk * server_num;
            ep_recv_count = at::empty({recv_count_tensor_size}, at::dtype(at::kInt).device(device));
        }
    }

    if (soc_version == op::SocVersion::ASCEND910B) {
        comm_alg = "fullmesh";
    } else {
        comm_alg = "fullmesh_v1";
    }

    if (enable_neg_one) {
        EP_HOST_ASSERT(isLayered == false);
        active_mask = (new_topk_idx >= 0).to(torch::kBool);
    }

    EXEC_NPU_CMD(aclnnMoeDistributeDispatchV2, new_x, new_topk_idx,
                 scales,        // smooth scales,
                 active_mask,   // active_mask
                 hcom_ep_name,  // ep
                 num_ranks,     // rankSize
                 rank,          // rankId
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
                 comm_alg, packed_recv_x,
                 packed_recv_x_scales,  // dynamicScalesOut
                 expandIdx,
                 packed_recv_count,  // expertTokenNumsOut
                 ep_recv_count, tp_recv_count);

    // Return values
    return {packed_recv_x, packed_recv_x_scales,        packed_recv_count, expandIdx, ep_recv_count,
            event,         std::function<void()>([] {})};
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
    EP_HOST_ASSERT(num_max_dispatch_tokens_per_rank >= new_idx.size(0));
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
    int enable_neg_one = get_value_from_env("MOE_ENABLE_TOPK_NEG_ONE", 0);
    int64_t tp_world_size = 1;
    int64_t tp_rankId = 0;
    int64_t expert_shared_type = 0;
    int64_t global_bs = num_max_dispatch_tokens_per_rank * num_ranks;
    int64_t out_dtype = 0;
    int64_t comm_quant_mode = 0;
    int64_t group_list_type = 0;
    bool isLayered = false;
    char *comm_alg;

    auto num_combined_tokens = static_cast<int>(new_scales.size(0));
    auto hidden = static_cast<int>(x.size(1));
    at::Tensor shared_expert_x{nullptr};
    at::Tensor combined_x = at::empty({num_combined_tokens, hidden}, x.options());
    std::optional<EventHandle> event;
    if (soc_version == op::SocVersion::ASCEND910B) {
        const char *hcclIntraPcieEnable = getenv("HCCL_INTRA_PCIE_ENABLE");
        const char *hcclIntraRoceEnable = getenv("HCCL_INTRA_ROCE_ENABLE");
        if (hcclIntraPcieEnable != nullptr && hcclIntraRoceEnable != nullptr && strcmp(hcclIntraPcieEnable, "1") == 0 &&
            strcmp(hcclIntraRoceEnable, "0") == 0) {  // A2 layered
            isLayered = true;
        }
    }

    if (soc_version == op::SocVersion::ASCEND910B) {
        comm_alg = "fullmesh";
    } else {
        comm_alg = "fullmesh_v1";
    }

    if (enable_neg_one) {
        EP_HOST_ASSERT(isLayered == false);
        x_active_mask = (new_topk_idx >= 0).to(torch::kBool);
    }

    EXEC_NPU_CMD(aclnnMoeDistributeCombineV2, expand_x, expert_ids, expand_idx, ep_send_counts, expert_scales,
                 tp_send_counts, x_active_mask, activation_scale, weight_scale, group_list, expand_scales,
                 shared_expert_x, hcom_ep_name, num_ranks, rank, num_experts, hcom_tp_name, tp_world_size, tp_rankId,
                 expert_shared_type, shared_expert_num, shared_expert_rank_num, global_bs, out_dtype, comm_quant_mode,
                 group_list_type, comm_alg, combined_x);
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

std::vector<at::Tensor> Buffer::fused_deep_moe(const at::Tensor &x, const at::Tensor &expert_ids,
                                               const at::Tensor &gmm1_permuted_weight,
                                               const at::Tensor &gmm1_permuted_weight_scale,
                                               const at::Tensor &gmm2_weight, const at::Tensor &gmm2_weight_scale,
                                               const at::Tensor &expert_scales_optional,
                                               int64_t num_max_dispatch_tokens_per_rank, int64_t num_experts,
                                               int quant_mode)
{
    EP_HOST_ASSERT(expert_ids.dim() == 2);
    EP_HOST_ASSERT(expert_scales_optional.dim() == 2);

    this->is_padding = false;
    at::Tensor new_x = x;
    this->new_topk_idx = expert_ids;
    at::Tensor new_scales = expert_scales_optional;

    if (expert_ids.size(0) < PADDING_SIZE) {
        this->is_padding = true;
        this->padding_cnt = PADDING_SIZE - expert_ids.size(0);

        std::vector<at::Tensor> x_blocks;
        std::vector<at::Tensor> idx_blocks;

        if (expert_ids.size(0) != 0) {
            x_blocks.emplace_back(x);
            idx_blocks.emplace_back(expert_ids);
        } else {
            this->ori_x = x.clone();  // store the original input when the batch is completely empty
        }

        int topk = static_cast<int>(expert_ids.size(1));
        for (int i = 0; i < this->padding_cnt; i++) {
            at::Tensor tmp_x = torch::ones({1, x.size(1)}, x.options());
            at::Tensor tmp_idx = torch::arange(0, topk, expert_ids.options()).reshape({1, topk});
            x_blocks.emplace_back(tmp_x);
            idx_blocks.emplace_back(tmp_idx);
        }
        new_x = torch::cat(x_blocks, 0);
        this->new_topk_idx = torch::cat(idx_blocks, 0);

        // padding expert_scales_optional
        std::vector<at::Tensor> scales_blocks;
        if (this->padding_cnt != PADDING_SIZE) {
            scales_blocks.emplace_back(expert_scales_optional);
        }
        for (int i = 0; i < this->padding_cnt; i++) {
            at::Tensor tmp_scales = torch::zeros({1, expert_scales_optional.size(1)}, expert_scales_optional.options());
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

    int64_t global_bs = std::max(new_topk_idx.size(0), num_max_dispatch_tokens_per_rank) * num_ranks;

    auto x_shape = x.sizes();
    int h = x_shape[1];
    int bs = this->new_topk_idx.size(0);

    at::Tensor output = at::empty({bs, h}, x.options());

    bool is_shared_expert = (rank < shared_expert_rank_num);
    int64_t num_local_experts = is_shared_expert ? 1 : num_experts / (num_ranks - shared_expert_rank_num);
    at::Tensor ep_recv_count = at::empty({num_local_experts * num_ranks}, expert_ids.options());

    EXEC_NPU_CMD(aclnnFusedDeepMoe,
                 // input
                 new_x, this->new_topk_idx, gmm1_permuted_weight, gmm1_permuted_weight_scale, gmm2_weight,
                 gmm2_weight_scale, static_cast<const std::nullptr_t &>(nullptr), new_scales,
                 // attr
                 hcom_ep_name, num_ranks, rank, num_experts, shared_expert_num, shared_expert_rank_num, quant_mode,
                 global_bs,
                 // output
                 output, ep_recv_count);

    // ---------- unpadding ----------
    if (this->is_padding) {
        if (expert_ids.size(0) == 0) {
            output = this->ori_x;
        } else {
            output = output.slice(0, 0, PADDING_SIZE - this->padding_cnt);
        }
        this->is_padding = false;
    }

    return {output, ep_recv_count};
}
}  // namespace deep_ep
