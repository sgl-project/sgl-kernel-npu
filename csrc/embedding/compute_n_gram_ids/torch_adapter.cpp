#include "aclnn_runtime.h"
#include "aclnn_sgl_compute_n_gram_ids.h"

#include <array>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>

namespace {

// Preserve the input contract of the existing sgl-kernel-npu binding.
void CheckComputeNGramIdsInputs(const at::Tensor &oe_weights, const at::Tensor &oe_mods,
                                const at::Tensor &exclusive_oe_embeder_size_sums, const at::Tensor &tokens,
                                const at::Tensor &exclusive_req_len_sums, const at::Tensor &oe_token_table,
                                const at::Tensor &row_indices, const at::Tensor &column_starts)
{
    TORCH_CHECK(oe_weights.scalar_type() == at::kInt, "oe_weights must be int32");
    TORCH_CHECK(oe_mods.scalar_type() == at::kInt, "oe_mods must be int32");
    TORCH_CHECK(exclusive_oe_embeder_size_sums.scalar_type() == at::kInt,
                "exclusive_oe_embeder_size_sums must be int32");
    TORCH_CHECK(tokens.scalar_type() == at::kInt, "tokens must be int32");
    TORCH_CHECK(exclusive_req_len_sums.scalar_type() == at::kInt, "exclusive_req_len_sums must be int32");
    TORCH_CHECK(oe_token_table.scalar_type() == at::kInt, "oe_token_table must be int32");
    TORCH_CHECK(row_indices.scalar_type() == at::kLong, "row_indices must be int64");
    TORCH_CHECK(column_starts.scalar_type() == at::kInt, "column_starts must be int32");

    TORCH_CHECK(oe_weights.is_contiguous(), "oe_weights must be contiguous");
    TORCH_CHECK(oe_mods.is_contiguous(), "oe_mods must be contiguous");
    TORCH_CHECK(exclusive_oe_embeder_size_sums.is_contiguous(), "exclusive_oe_embeder_size_sums must be contiguous");
    TORCH_CHECK(tokens.is_contiguous(), "tokens must be contiguous");
    TORCH_CHECK(exclusive_req_len_sums.is_contiguous(), "exclusive_req_len_sums must be contiguous");
    TORCH_CHECK(oe_token_table.is_contiguous(), "oe_token_table must be contiguous");
    TORCH_CHECK(row_indices.is_contiguous(), "row_indices must be contiguous");
    TORCH_CHECK(column_starts.is_contiguous(), "column_starts must be contiguous");
}

at::Tensor compute_n_gram_ids(const at::Tensor &oe_weights, const at::Tensor &oe_mods,
                              const at::Tensor &exclusive_oe_embeder_size_sums, const at::Tensor &tokens,
                              const at::Tensor &exclusive_req_len_sums, const at::Tensor &oe_token_table,
                              const at::Tensor &row_indices, const at::Tensor &column_starts, int64_t batch_size,
                              int64_t oe_n, int64_t oe_k, int64_t max_context_len)
{
    CheckComputeNGramIdsInputs(oe_weights, oe_mods, exclusive_oe_embeder_size_sums, tokens, exclusive_req_len_sums,
                               oe_token_table, row_indices, column_starts);
    TORCH_CHECK(batch_size > 0, "batch_size must be positive");
    TORCH_CHECK(oe_n > 1, "oe_n must be greater than 1");
    TORCH_CHECK(oe_k > 0, "oe_k must be positive");
    TORCH_CHECK(max_context_len > 0, "max_context_len must be positive");
    const c10::DeviceGuard device_guard(tokens.device());
    auto output = at::empty({tokens.size(0), (oe_n - 1) * oe_k}, tokens.options().dtype(at::kInt));

    static auto query = reinterpret_cast<decltype(&aclnnSglComputeNGramIdsGetWorkspaceSize)>(
        sgl_cann::op_api_symbol("aclnnSglComputeNGramIdsGetWorkspaceSize"));
    static auto execute =
        reinterpret_cast<decltype(&aclnnSglComputeNGramIds)>(sgl_cann::op_api_symbol("aclnnSglComputeNGramIds"));
    std::array<at::Tensor, 9> tensors = {oe_weights,
                                         oe_mods,
                                         exclusive_oe_embeder_size_sums,
                                         tokens,
                                         exclusive_req_len_sums,
                                         oe_token_table,
                                         row_indices,
                                         column_starts,
                                         output};
    std::array<std::shared_ptr<aclTensor>, 9> descriptors;
    for (size_t i = 0; i < tensors.size(); ++i) {
        descriptors[i] = sgl_cann::tensor_descriptor(tensors[i], i == 6 ? ACL_INT64 : ACL_INT32);
    }
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    auto status = query(descriptors[0].get(), descriptors[1].get(), descriptors[2].get(), descriptors[3].get(),
                        descriptors[4].get(), descriptors[5].get(), descriptors[6].get(), descriptors[7].get(),
                        batch_size, oe_n, oe_k, max_context_len, descriptors[8].get(), &workspace_size, &executor);
    TORCH_CHECK(status == ACL_SUCCESS, "ComputeNGramIds workspace query failed: ", aclGetRecentErrMsg());
    auto workspace = at::empty({static_cast<int64_t>(workspace_size)}, output.options().dtype(at::kByte));
    auto stream = c10_npu::getCurrentNPUStream();
    for (auto &tensor : tensors) {
        tensor.record_stream(stream);
    }
    workspace.record_stream(stream);
    auto acl_stream = stream.stream(false);
    // Keep Tensor storage and ACL descriptors alive until the host queue executes the call.
    auto launch = [tensors, descriptors, workspace, workspace_size, executor, acl_stream]() -> int {
        auto result = execute(workspace_size ? workspace.data_ptr() : nullptr, workspace_size, executor, acl_stream);
        TORCH_CHECK(result == ACL_SUCCESS, "ComputeNGramIds ACLNN execution failed: ", aclGetRecentErrMsg());
        return result;
    };
    at_npu::native::OpCommand::RunOpApi("aclnnSglComputeNGramIds", launch);
    return output;
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def(
        "compute_n_gram_ids(Tensor oe_weights, Tensor oe_mods, Tensor exclusive_oe_embeder_size_sums, "
        "Tensor tokens, Tensor exclusive_req_len_sums, Tensor oe_token_table, Tensor row_indices, "
        "Tensor column_starts, int batch_size, int oe_n, int oe_k, int max_context_len) -> Tensor");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("compute_n_gram_ids", &compute_n_gram_ids);
}
