// Experimental single-node A2, single-round, INT8 dispatch with device metadata.
#include "deep_ep.hpp"
#include "exception.hpp"
#include "pytorch_npu_helper.hpp"

#include <algorithm>
#include <cstring>

namespace deep_ep {

std::vector<at::Tensor> Buffer::intranode_dispatch_fused_decode(
    const at::Tensor &x, const at::Tensor &ids)
{
    // Collective contract: all ranks have the same N in [1,4].
    TORCH_CHECK(num_ranks == 8 && round == 1, "fused decode requires EP8, one round");
    TORCH_CHECK(x.dim() == 2 && x.size(0) >= 1 && x.size(0) <= 4 && x.size(1) == 4096 &&
                x.is_contiguous() && x.scalar_type() == at::kBFloat16, "invalid fused decode input");
    const int64_t n = x.size(0);
    TORCH_CHECK(n <= per_round_tokens && ids.dim() == 2 && ids.size(0) == n && ids.size(1) == 8 &&
                ids.is_contiguous() && ids.scalar_type() == at::kInt && ids.device() == x.device(),
                "invalid fused decode Top8 IDs");
    TORCH_CHECK(!moe_all_to_all_group_name.empty() && moe_all_to_all_group_name.size() < 128,
                "fused decode requires a named HCCL group");
    char group[128];
    std::memcpy(group, moe_all_to_all_group_name.c_str(), moe_all_to_all_group_name.size() + 1);
    auto values = at::empty({n * 64, 4096}, x.options().dtype(at::kChar));
    auto scales = at::empty({n * 64}, x.options().dtype(at::kFloat));
    auto ends = at::empty({36}, x.options().dtype(at::kLong));
    auto source = at::empty({n * 64 * 3}, x.options().dtype(at::kInt));
    auto counts = at::empty({288}, x.options().dtype(at::kInt));
    auto total = at::empty({1}, x.options().dtype(at::kInt));
    real_max_bs = n;
    const int64_t sendCount = 864;
    EXEC_NPU_CMD(aclnnFusedDispatchDecode, x, ids, sendCount, n, group, num_ranks, rank,
                 num_ranks, rank, round, per_round_tokens, values, scales, ends, source, counts, total);
    return {values, scales, ends, source, counts, total};
}

std::vector<at::Tensor> Buffer::intranode_dispatch_decode(
    const at::Tensor &x, const at::Tensor &topk_idx)
{
    // Collective contract: EVERY rank has exactly one token. Uneven senders
    // must use intranode_dispatch_device_counts with a common capacity bound.
    TORCH_CHECK(x.dim() == 2 && x.size(0) == 1 && topk_idx.scalar_type() == at::kInt,
                "fused decode dispatch requires one token and INT32 Top8 IDs");
    return dispatch_device_counts_impl(x, topk_idx, at::Tensor(), 1, std::nullopt, true);
}

std::vector<at::Tensor> Buffer::intranode_dispatch_device_counts(
    const at::Tensor &x, const at::Tensor &topk_idx,
    const at::Tensor &num_tokens_per_expert, int64_t max_tokens_per_rank,
    const std::optional<at::Tensor> &send_token_indices)
{
    return dispatch_device_counts_impl(x, topk_idx, num_tokens_per_expert,
                                       max_tokens_per_rank, send_token_indices, false);
}

std::vector<at::Tensor> Buffer::dispatch_device_counts_impl(
    const at::Tensor &x, const at::Tensor &topk_idx,
    const at::Tensor &num_tokens_per_expert, int64_t max_tokens_per_rank,
    const std::optional<at::Tensor> &send_token_indices, bool fused_decode)
{
    // Caller must provide the SAME bound on every rank, >= each rank's rows.
    // GLM's DeepEP prepare path pads to equal TP8 slices before routing.
    // Restrict this first implementation to one round and the tested A2 shape.
    TORCH_CHECK(num_ranks == 8 && round == 1, "device-count dispatch requires EP8 and one round");
    TORCH_CHECK(x.dim() == 2 && x.is_contiguous() && x.scalar_type() == at::kBFloat16,
                "expected contiguous BF16 input");
    TORCH_CHECK(topk_idx.dim() == 2 && topk_idx.is_contiguous() && topk_idx.size(0) == x.size(0),
                "invalid topk indices");
    TORCH_CHECK((topk_idx.scalar_type() == at::kLong || topk_idx.scalar_type() == at::kInt)
                && topk_idx.device() == x.device(), "expected INT32/INT64 indices on input device");
    if (!fused_decode) TORCH_CHECK(num_tokens_per_expert.scalar_type() == at::kInt && num_tokens_per_expert.is_contiguous()
                && num_tokens_per_expert.device() == x.device(), "invalid layout counts");
    TORCH_CHECK(max_tokens_per_rank >= x.size(0) && max_tokens_per_rank > 0
                && max_tokens_per_rank <= 512 && max_tokens_per_rank <= per_round_tokens,
                "device-count path supports a shared rank bound in [1,512]");
    const int64_t experts = fused_decode ? 288 : num_tokens_per_expert.numel();
    TORCH_CHECK(experts == 288 && topk_idx.size(1) == 8 && x.size(1) == 4096,
                "device-count path currently tested only for GLM 288 experts, top8, hidden4096");
    const at::Tensor send_indices = fused_decode ? at::empty_like(topk_idx)
        : (send_token_indices.has_value() ? send_token_indices.value() : send_token_idx_small);
    TORCH_CHECK(send_indices.defined() && send_indices.sizes() == topk_idx.sizes()
                && send_indices.scalar_type() == at::kInt && send_indices.is_contiguous()
                && send_indices.device() == x.device(),
                "expected INT32 send indices from compact or standard dispatch layout");
    const int64_t local_experts = experts / num_ranks;
    const int64_t capacity = max_tokens_per_rank * num_ranks * std::min(topk_idx.size(1), local_experts);
    auto int_options = x.options().dtype(at::kInt);
    auto send_data = at::empty({1, experts * 3}, int_options);
    auto send_offset = at::empty({1, experts}, int_options);
    auto recv_data = at::empty({1, experts * 3}, int_options);
    auto recv_count = at::empty({1, experts}, int_options);
    auto recv_offset = at::empty({1, experts}, int_options);
    auto expert_offset = at::empty({local_experts}, int_options);
    auto source_offset = at::empty({experts}, int_options);
    auto round_offset = at::empty({experts}, int_options);
    auto total_recv = at::empty({1}, int_options);
    auto max_bs = at::empty({1}, int_options);
    auto expert_counts = at::empty({local_experts}, int_options);
    char group_name[128];
    TORCH_CHECK(!moe_all_to_all_group_name.empty() && moe_all_to_all_group_name.size() < sizeof(group_name),
                "device-count dispatch requires a named HCCL group");
    std::memcpy(group_name, moe_all_to_all_group_name.c_str(), moe_all_to_all_group_name.size() + 1);

    const int64_t send_count = experts * 3, num_tokens = x.size(0);
    at::Tensor end_offsets;
    if (fused_decode) {
        end_offsets = at::empty({local_experts}, x.options().dtype(at::kLong));
        EXEC_NPU_CMD(aclnnNotifyDispatchDecode, send_data, topk_idx, send_count, num_tokens,
                     group_name, num_ranks, rank, num_ranks, rank, round, per_round_tokens,
                     send_offset, recv_data, recv_count, recv_offset, expert_offset, source_offset,
                     round_offset, total_recv, max_bs, expert_counts, send_indices, end_offsets);
    } else {
    EXEC_NPU_CMD(aclnnNotifyDispatch, send_data, num_tokens_per_expert, send_count, num_tokens,
                 group_name, num_ranks, rank, num_ranks, rank, round, per_round_tokens,
                 send_offset, recv_data, recv_count, recv_offset, expert_offset, source_offset,
                 round_offset, total_recv, max_bs, expert_counts);
    }

    // Capacity bounds every possible route, including all top8 landing here.
    // Dispatch writes only valid rows. Both GMMs use device end offsets; Combine
    // uses recv_count to bound its reads and never inspects the unwritten tail.
    auto recv_x = at::empty({capacity, x.size(1)}, x.options().dtype(at::kChar));
    auto scales = at::empty({capacity}, x.options().dtype(at::kFloat));
    auto src_idx = at::empty({capacity * 3}, int_options);
    auto ids = topk_idx.to(at::kInt);
    at::Tensor wait_stats;
    real_max_bs = max_tokens_per_rank;
    const int64_t tp_size = 1, tp_rank = 0, quant_mode = 2;
    const int64_t global_bs = max_tokens_per_rank * num_ranks;
    EXEC_NPU_CMD(aclnnCamMoeDispatchNormal, x, ids, send_offset, send_indices,
                 recv_offset, recv_count, expert_offset, source_offset, round_offset,
                 group_name, num_ranks, rank, group_name, tp_size, tp_rank, experts,
                 quant_mode, real_max_bs, global_bs, round, per_round_tokens,
                 recv_x, scales, src_idx, wait_stats);

    // Notify already computed start offsets. This is vector add + cast, no
    // Cumsum or CPU transfer. INT64 cumulative offsets are required by GMM1.
    if (!fused_decode) end_offsets = (expert_offset + expert_counts).to(at::kLong);
    return {recv_x, scales, end_offsets, src_idx, recv_count.view({experts}), total_recv};
}

}  // namespace deep_ep
