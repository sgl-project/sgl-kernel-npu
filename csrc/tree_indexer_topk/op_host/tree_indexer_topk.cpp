// SPDX-License-Identifier: BSD-3-Clause
#include <algorithm>
#include <limits>

#include "defines.h"
#include "torch_helper.h"
#include "aclrtlaunch_tree_indexer_topk.h"
#include "tiling/platform/platform_ascendc.h"

namespace sglang::npu_kernel {
HOST_API at::Tensor tree_indexer_topk(const at::Tensor &scores, int64_t k)
{
    TORCH_CHECK(scores.device().type() == c10::DeviceType::PrivateUse1, "scores must be on NPU");
    TORCH_CHECK(scores.scalar_type() == at::kFloat && scores.dim() == 2 && scores.is_contiguous(),
                "scores must be contiguous float32 [rows, slots]");
    TORCH_CHECK(k >= 1 && k <= 1024, "tree_indexer_topk supports 1 <= k <= 1024");
    TORCH_CHECK(scores.size(1) > 0 && scores.size(1) <= std::numeric_limits<int32_t>::max(),
                "slots must be positive and fit int32 indices");
    TORCH_CHECK(scores.size(0) <= std::numeric_limits<int32_t>::max(), "too many rows");
    auto out = at::empty({scores.size(0), 1, k}, scores.options().dtype(at::kInt));
    if (scores.size(0) == 0) return out;
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t cores = platform->GetCoreNumAiv();
    TORCH_CHECK(cores > 0, "no vector cores available");
    uint32_t rows = static_cast<uint32_t>(scores.size(0));
    uint32_t slots = static_cast<uint32_t>(scores.size(1));
    uint32_t count = static_cast<uint32_t>(k);
    uint32_t blockDim = std::min(rows, cores);
    EXEC_KERNEL_CMD(tree_indexer_topk, blockDim, scores, out, rows, slots, count);
    return out;
}
}  // namespace sglang::npu_kernel
