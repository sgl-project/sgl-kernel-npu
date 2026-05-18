#include "compute_n_gram_ids_tiling.h"

#include <algorithm>

#include "defines.h"
#include "torch_helper.h"
#include "acl/acl.h"
#include "aclrtlaunch_compute_n_gram_ids.h"
#include "tiling/platform/platform_ascendc.h"

namespace sglang::npu_kernel {
namespace {

constexpr uint32_t PADDING_BYTE = 32U;

void CheckComputeNGramIdsInputs(const at::Tensor &oe_weights, const at::Tensor &oe_mods,
                                const at::Tensor &exclusive_oe_embeder_size_sums,
                                const at::Tensor &tokens,
                                const at::Tensor &exclusive_req_len_sums,
                                const at::Tensor &oe_token_table,
                                const at::Tensor &row_indices,
                                const at::Tensor &column_starts)
{
    TORCH_CHECK(oe_weights.scalar_type() == at::kInt, "oe_weights must be int32");
    TORCH_CHECK(oe_mods.scalar_type() == at::kInt, "oe_mods must be int32");
    TORCH_CHECK(exclusive_oe_embeder_size_sums.scalar_type() == at::kInt,
                "exclusive_oe_embeder_size_sums must be int32");
    TORCH_CHECK(tokens.scalar_type() == at::kInt, "tokens must be int32");
    TORCH_CHECK(exclusive_req_len_sums.scalar_type() == at::kInt,
                "exclusive_req_len_sums must be int32");
    TORCH_CHECK(oe_token_table.scalar_type() == at::kInt, "oe_token_table must be int32");
    TORCH_CHECK(row_indices.scalar_type() == at::kLong, "row_indices must be int64");
    TORCH_CHECK(column_starts.scalar_type() == at::kInt, "column_starts must be int32");

    TORCH_CHECK(oe_weights.is_contiguous(), "oe_weights must be contiguous");
    TORCH_CHECK(oe_mods.is_contiguous(), "oe_mods must be contiguous");
    TORCH_CHECK(exclusive_oe_embeder_size_sums.is_contiguous(),
                "exclusive_oe_embeder_size_sums must be contiguous");
    TORCH_CHECK(tokens.is_contiguous(), "tokens must be contiguous");
    TORCH_CHECK(exclusive_req_len_sums.is_contiguous(),
                "exclusive_req_len_sums must be contiguous");
    TORCH_CHECK(oe_token_table.is_contiguous(), "oe_token_table must be contiguous");
    TORCH_CHECK(row_indices.is_contiguous(), "row_indices must be contiguous");
    TORCH_CHECK(column_starts.is_contiguous(), "column_starts must be contiguous");
}

at::Tensor BuildTilingTensor(uint32_t &block_dim, int64_t batch_size, int64_t oe_n,
                             int64_t oe_k, int64_t max_context_len)
{
    auto ascendc_platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t aiv_num = static_cast<uint32_t>(ascendc_platform->GetCoreNumAiv());
    uint32_t total_task = static_cast<uint32_t>(batch_size * (oe_n - 1) * oe_k);
    block_dim = std::min(aiv_num, std::max(total_task, 1U));

    int32_t tiling_size =
        static_cast<int32_t>(((sizeof(ComputeNGramIdsTilingData) + PADDING_BYTE - 1) / PADDING_BYTE) * PADDING_BYTE);
    auto tiling_buffer = at::empty({tiling_size}, at::TensorOptions().dtype(at::kByte).device(at::kCPU));

    auto *tiling_data =
        reinterpret_cast<ComputeNGramIdsTilingData *>(tiling_buffer.data_ptr<uint8_t>());
    tiling_data->coreNum = block_dim;
    tiling_data->batchSize = static_cast<uint32_t>(batch_size);
    tiling_data->totalTask = total_task;
    tiling_data->oeN = static_cast<uint32_t>(oe_n);
    tiling_data->oeK = static_cast<uint32_t>(oe_k);
    tiling_data->maxContextLen = static_cast<uint32_t>(max_context_len);

    return TorchNpuHelper::CopyTensorHostToDevice(tiling_buffer);
}

}  // namespace

HOST_API at::Tensor compute_n_gram_ids(
    const at::Tensor &oe_weights, const at::Tensor &oe_mods,
    const at::Tensor &exclusive_oe_embeder_size_sums, const at::Tensor &tokens,
    const at::Tensor &exclusive_req_len_sums, const at::Tensor &oe_token_table,
    const at::Tensor &row_indices, const at::Tensor &column_starts, int64_t batch_size,
    int64_t oe_n, int64_t oe_k, int64_t max_context_len)
{
    CheckComputeNGramIdsInputs(oe_weights, oe_mods, exclusive_oe_embeder_size_sums,
                               tokens, exclusive_req_len_sums, oe_token_table,
                               row_indices, column_starts);
    TORCH_CHECK(batch_size > 0, "batch_size must be positive");
    TORCH_CHECK(oe_n > 1, "oe_n must be greater than 1");
    TORCH_CHECK(oe_k > 0, "oe_k must be positive");
    TORCH_CHECK(max_context_len > 0, "max_context_len must be positive");

    auto output = at::empty({tokens.size(0), (oe_n - 1) * oe_k},
                            tokens.options().dtype(at::kInt));

    uint32_t block_dim = 1;
    auto tiling_tensor = BuildTilingTensor(block_dim, batch_size, oe_n, oe_k, max_context_len);
    auto workspace_tensor = at::empty({1}, at::TensorOptions().dtype(at::kByte).device(tokens.device()));

    EXEC_KERNEL_CMD(compute_n_gram_ids, block_dim, oe_weights, oe_mods,
                    exclusive_oe_embeder_size_sums, tokens, exclusive_req_len_sums,
                    oe_token_table, row_indices, column_starts, output,
                    workspace_tensor, tiling_tensor);
    return output;
}

}  // namespace sglang::npu_kernel
