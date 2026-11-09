#include "aclnn_runtime.h"
#include "aclnn_sgl_update_oe_token_table.h"

#include <array>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>

namespace {

// Preserve the input contract of the existing sgl-kernel-npu binding.
void CheckUpdateOeTokenTableInputs(const at::Tensor &tokens, const at::Tensor &req_lens, const at::Tensor &row_indices,
                                   const at::Tensor &column_starts, const at::Tensor &ignore_tokens,
                                   const at::Tensor &oe_token_table, int64_t batch_size, int64_t max_context_len)
{
    TORCH_CHECK(tokens.scalar_type() == at::kInt, "tokens must be int32");
    TORCH_CHECK(req_lens.scalar_type() == at::kInt, "req_lens must be int32");
    TORCH_CHECK(row_indices.scalar_type() == at::kLong, "row_indices must be int64");
    TORCH_CHECK(column_starts.scalar_type() == at::kInt, "column_starts must be int32");
    TORCH_CHECK(ignore_tokens.scalar_type() == at::kInt, "ignore_tokens must be int32");
    TORCH_CHECK(oe_token_table.scalar_type() == at::kInt, "oe_token_table must be int32");

    TORCH_CHECK(tokens.is_contiguous(), "tokens must be contiguous");
    TORCH_CHECK(req_lens.is_contiguous(), "req_lens must be contiguous");
    TORCH_CHECK(row_indices.is_contiguous(), "row_indices must be contiguous");
    TORCH_CHECK(column_starts.is_contiguous(), "column_starts must be contiguous");
    TORCH_CHECK(ignore_tokens.is_contiguous(), "ignore_tokens must be contiguous");
    TORCH_CHECK(oe_token_table.is_contiguous(), "oe_token_table must be contiguous");

    TORCH_CHECK(tokens.dim() == 1, "tokens must be a 1D tensor");
    TORCH_CHECK(req_lens.dim() == 1, "req_lens must be a 1D tensor");
    TORCH_CHECK(row_indices.dim() == 1, "row_indices must be a 1D tensor");
    TORCH_CHECK(column_starts.dim() == 1, "column_starts must be a 1D tensor");
    TORCH_CHECK(ignore_tokens.dim() == 1, "ignore_tokens must be a 1D tensor");
    TORCH_CHECK(oe_token_table.dim() == 2, "oe_token_table must be a 2D tensor");

    TORCH_CHECK(batch_size > 0, "batch_size must be positive");
    TORCH_CHECK(max_context_len > 0, "max_context_len must be positive");
    TORCH_CHECK(req_lens.numel() == batch_size, "req_lens length must match batch_size");
    TORCH_CHECK(row_indices.numel() == batch_size, "row_indices length must match batch_size");
    TORCH_CHECK(column_starts.numel() == batch_size, "column_starts length must match batch_size");
    TORCH_CHECK(oe_token_table.size(1) >= max_context_len,
                "oe_token_table second dimension must be >= max_context_len");
    TORCH_CHECK(tokens.device() == req_lens.device(), "tokens and req_lens must be on the same device");
    TORCH_CHECK(tokens.device() == row_indices.device(), "tokens and row_indices must be on the same device");
    TORCH_CHECK(tokens.device() == column_starts.device(), "tokens and column_starts must be on the same device");
    TORCH_CHECK(tokens.device() == ignore_tokens.device(), "tokens and ignore_tokens must be on the same device");
    TORCH_CHECK(tokens.device() == oe_token_table.device(), "tokens and oe_token_table must be on the same device");
}

at::Tensor update_oe_token_table(const at::Tensor &tokens, const at::Tensor &req_lens, const at::Tensor &row_indices,
                                 const at::Tensor &column_starts, const at::Tensor &ignore_tokens, int64_t batch_size,
                                 int64_t max_context_len, const at::Tensor &oe_token_table)
{
    CheckUpdateOeTokenTableInputs(tokens, req_lens, row_indices, column_starts, ignore_tokens, oe_token_table,
                                  batch_size, max_context_len);
    const c10::DeviceGuard device_guard(oe_token_table.device());
    static auto query = reinterpret_cast<decltype(&aclnnSglUpdateOeTokenTableGetWorkspaceSize)>(
        sgl_cann::op_api_symbol("aclnnSglUpdateOeTokenTableGetWorkspaceSize"));
    static auto execute =
        reinterpret_cast<decltype(&aclnnSglUpdateOeTokenTable)>(sgl_cann::op_api_symbol("aclnnSglUpdateOeTokenTable"));
    std::array<at::Tensor, 6> tensors = {tokens, req_lens, row_indices, column_starts, ignore_tokens, oe_token_table};
    std::array<std::shared_ptr<aclTensor>, 6> descriptors;
    for (size_t i = 0; i < tensors.size(); ++i) {
        descriptors[i] = sgl_cann::tensor_descriptor(tensors[i], i == 2 ? ACL_INT64 : ACL_INT32);
    }
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    // The official CANN operator has an output-only table. Supply the existing table storage.
    auto status =
        query(descriptors[0].get(), descriptors[1].get(), descriptors[2].get(), descriptors[3].get(),
              descriptors[4].get(), batch_size, max_context_len, descriptors[5].get(), &workspace_size, &executor);
    TORCH_CHECK(status == ACL_SUCCESS, "UpdateOeTokenTable workspace query failed: ", aclGetRecentErrMsg());
    auto workspace = at::empty({static_cast<int64_t>(workspace_size)}, oe_token_table.options().dtype(at::kByte));
    auto stream = c10_npu::getCurrentNPUStream();
    for (auto &tensor : tensors) {
        tensor.record_stream(stream);
    }
    workspace.record_stream(stream);
    auto acl_stream = stream.stream(false);
    auto launch = [tensors, descriptors, workspace, workspace_size, executor, acl_stream]() -> int {
        auto result = execute(workspace_size ? workspace.data_ptr() : nullptr, workspace_size, executor, acl_stream);
        TORCH_CHECK(result == ACL_SUCCESS, "UpdateOeTokenTable ACLNN execution failed: ", aclGetRecentErrMsg());
        return result;
    };
    at_npu::native::OpCommand::RunOpApi("aclnnSglUpdateOeTokenTable", launch);
    return oe_token_table;
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def(
        "update_oe_token_table(Tensor tokens, Tensor req_lens, Tensor row_indices, Tensor column_starts, "
        "Tensor ignore_tokens, int batch_size, int max_context_len, Tensor(a!) oe_token_table) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("update_oe_token_table", &update_oe_token_table);
}
