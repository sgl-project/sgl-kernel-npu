// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under the BSD 3-Clause License.

#include "op_host/fused_copy_sfa.h"

#include "aclnn_runtime.h"
#include "aclnn_sgl_fused_copy_sfa.h"
#include <array>
#include <torch/library.h>

namespace sglang::npu_kernel {
void LaunchFusedCopySfa(const at::Tensor &query, const at::Tensor &query_rope, at::Tensor &hbm_ckv,
                        at::Tensor &hbm_rope, const at::Tensor &actual_seq_lengths_query,
                        const at::Tensor &actual_seq_lengths_kv, const at::Tensor &num_cache_tokens,
                        const at::Tensor &topk_src_ids, const at::Tensor &topk_dst_slots,
                        const at::Tensor &topk_miss_counts, const at::Tensor &miss_src_ids,
                        const at::Tensor &miss_dst_slots, const at::Tensor &miss_counts,
                        const at::Tensor &hbm_block_table, const at::Tensor &dram_block_table, int64_t dram_kv_dev_ptr,
                        at::Tensor &attention_out, double scale_value, bool full_visible, bool resident_refresh_tail,
                        const c10::optional<at::Tensor> &current, const c10::optional<at::Tensor> &positions,
                        const c10::optional<at::Tensor> &valid, const c10::optional<at::Tensor> &request_slots,
                        const c10::optional<at::Tensor> &req_to_token, const c10::optional<at::Tensor> &resident_end,
                        int64_t host_token_capacity, int64_t dram_rope_dev_ptr)
{
    // The caller owns the two registered host allocations until stream completion.
    auto stream = c10_npu::getCurrentNPUStream();
    const auto floating = query.scalar_type() == at::kHalf ? ACL_FLOAT16 : ACL_BF16;
    std::array<at::Tensor, 22> tensors = {query,
                                          query_rope,
                                          hbm_ckv,
                                          hbm_rope,
                                          actual_seq_lengths_query,
                                          actual_seq_lengths_kv,
                                          num_cache_tokens,
                                          topk_src_ids,
                                          topk_dst_slots,
                                          topk_miss_counts,
                                          miss_src_ids,
                                          miss_dst_slots,
                                          miss_counts,
                                          hbm_block_table,
                                          dram_block_table,
                                          current.value_or(at::Tensor{}),
                                          positions.value_or(at::Tensor{}),
                                          valid.value_or(at::Tensor{}),
                                          request_slots.value_or(at::Tensor{}),
                                          req_to_token.value_or(at::Tensor{}),
                                          resident_end.value_or(num_cache_tokens),
                                          attention_out};
    const std::array<aclDataType, 22> types = {floating,  floating,  floating,  floating,  ACL_INT32, ACL_INT32,
                                               ACL_INT32, ACL_INT32, ACL_INT32, ACL_INT32, ACL_INT32, ACL_INT32,
                                               ACL_INT32, ACL_INT32, ACL_INT32, floating,  ACL_INT64, ACL_BOOL,
                                               ACL_INT32, ACL_INT32, ACL_INT32, floating};
    std::array<std::shared_ptr<aclTensor>, 22> desc;
    for (size_t i = 0; i < tensors.size(); ++i) {
        if (tensors[i].defined()) {
            tensors[i].record_stream(stream);
            desc[i] = sgl_cann::tensor_descriptor(tensors[i], types[i]);
        }
    }
    static auto prepare = reinterpret_cast<decltype(&aclnnSglFusedCopySfaGetWorkspaceSize)>(
        sgl_cann::op_api_symbol("aclnnSglFusedCopySfaGetWorkspaceSize"));
    static auto execute =
        reinterpret_cast<decltype(&aclnnSglFusedCopySfa)>(sgl_cann::op_api_symbol("aclnnSglFusedCopySfa"));
    uint64_t bytes = 0;
    aclOpExecutor *executor = nullptr;
    auto status =
        prepare(desc[0].get(), desc[1].get(), desc[2].get(), desc[3].get(), desc[4].get(), desc[5].get(), desc[6].get(),
                desc[7].get(), desc[8].get(), desc[9].get(), desc[10].get(), desc[11].get(), desc[12].get(),
                desc[13].get(), desc[14].get(), desc[15].get(), desc[16].get(), desc[17].get(), desc[18].get(),
                desc[19].get(), desc[20].get(), static_cast<double>(scale_value), dram_kv_dev_ptr, dram_rope_dev_ptr,
                full_visible, resident_refresh_tail, host_token_capacity, desc[21].get(), &bytes, &executor);
    TORCH_CHECK(status == ACL_SUCCESS, "Copy-SFA workspace query failed: ", aclGetRecentErrMsg());
    TORCH_CHECK(bytes <= INT64_MAX, "Copy-SFA workspace exceeds the Tensor allocation limit");
    at::Tensor workspace;
    if (bytes) {
        workspace = at::empty({static_cast<int64_t>(bytes)}, query.options().dtype(at::kByte));
    }
    auto aclStream = stream.stream(false);
    auto launch = [tensors, desc, workspace, bytes, executor, aclStream]() -> int {
        auto result = execute(bytes ? workspace.data_ptr() : nullptr, bytes, executor, aclStream);
        TORCH_CHECK(result == ACL_SUCCESS, "Copy-SFA execution failed: ", aclGetRecentErrMsg());
        return result;
    };
    at_npu::native::OpCommand::RunOpApi("aclnnSglFusedCopySfa", launch);
}
}  // namespace sglang::npu_kernel

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def(
        "fused_scatter_copy_sparse_flash_attention_split(Tensor query, Tensor query_rope, "
        "Tensor(a!) hbm_ckv, Tensor(e!) hbm_rope, Tensor actual_seq_lengths_query, Tensor actual_seq_lengths_kv, "
        "Tensor num_cache_tokens, Tensor topk_src_ids, Tensor topk_dst_slots, Tensor topk_miss_counts, "
        "Tensor miss_src_ids, Tensor miss_dst_slots, Tensor miss_counts, Tensor hbm_block_table, "
        "Tensor dram_block_table, int dram_kv_dev_ptr, Tensor(b!) attention_out, float scale_value, bool "
        "full_visible=False, bool resident_refresh_tail=False, Tensor? current=None, Tensor? positions=None, "
        "Tensor? valid=None, Tensor? request_slots=None, Tensor? req_to_token=None, "
        "Tensor(c!)? resident_end=None, int host_token_capacity=0, "
        "int dram_rope_dev_ptr=0) -> ()");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("fused_scatter_copy_sparse_flash_attention_split",
           &sglang::npu_kernel::fused_scatter_copy_sparse_flash_attention_split);
}
