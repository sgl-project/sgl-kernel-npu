#include "defines.h"
#include "torch_helper.h"
#include "tiling/platform/platform_ascendc.h"

#include "aclrtlaunch_plan_sparse_kv_cache.h"

#include <limits>

namespace sglang {
namespace npu_kernel {

HOST_API void plan_sparse_kv_cache(const at::Tensor &source_to_slot, const at::Tensor &slot_to_source,
                                   const at::Tensor &req_ids, const at::Tensor &topk_input, at::Tensor &topk_src_ids,
                                   at::Tensor &topk_dst_slots, at::Tensor &topk_miss_counts, at::Tensor &miss_src_ids,
                                   at::Tensor &miss_dst_slots, at::Tensor &miss_counts, int64_t max_context_len,
                                   int64_t cache_capacity)
{
    const at::Tensor *tensors[] = {&source_to_slot, &slot_to_source,   &req_ids,      &topk_input,     &topk_src_ids,
                                   &topk_dst_slots, &topk_miss_counts, &miss_src_ids, &miss_dst_slots, &miss_counts};
    const char *names[] = {"source_to_slot", "slot_to_source",   "req_ids",      "topk_input",     "topk_src_ids",
                           "topk_dst_slots", "topk_miss_counts", "miss_src_ids", "miss_dst_slots", "miss_counts"};
    for (size_t i = 0; i < 10; ++i) {
        TORCH_CHECK(tensors[i]->device().type() == at::DeviceType::PrivateUse1, names[i], " must be on NPU");
        TORCH_CHECK(tensors[i]->scalar_type() == at::kInt, names[i], " must be int32");
        TORCH_CHECK(tensors[i]->is_contiguous(), names[i], " must be contiguous");
        TORCH_CHECK(tensors[i]->device() == source_to_slot.device(), names[i], " must share source_to_slot device");
    }
    TORCH_CHECK(source_to_slot.dim() == 2 && slot_to_source.dim() == 2, "state maps must be 2-D");
    TORCH_CHECK(req_ids.dim() == 1 && topk_input.dim() == 2, "req_ids/topk_input must be [B] and [B,K]");
    const int64_t batch = req_ids.size(0);
    const int64_t realReqCount = source_to_slot.size(0) - 1;
    const int64_t sourceStride = source_to_slot.size(1);
    TORCH_CHECK(realReqCount > 0 && sourceStride >= max_context_len && max_context_len > 0,
                "invalid source map dimensions");
    TORCH_CHECK(max_context_len <= (1LL << 24), "FP32 planner sort keys require max_context_len <= 2^24");
    TORCH_CHECK(slot_to_source.size(0) == realReqCount + 1 && slot_to_source.size(1) == cache_capacity,
                "slot_to_source shape must be [R+1, cache_capacity]");
    TORCH_CHECK(cache_capacity == 2048, "planner requires cache_capacity=2048");
    TORCH_CHECK(batch > 0 && topk_input.size(0) == batch && topk_input.size(1) == cache_capacity,
                "topk_input must have shape [B, 2048]");
    TORCH_CHECK(topk_src_ids.numel() == batch * cache_capacity && topk_dst_slots.numel() == batch * cache_capacity,
                "top-k output buffers must contain B*2048 elements");
    TORCH_CHECK(miss_src_ids.numel() == batch * cache_capacity && miss_dst_slots.numel() == batch * cache_capacity,
                "miss output buffers must contain B*2048 elements");
    TORCH_CHECK(topk_miss_counts.numel() == batch && miss_counts.numel() == batch,
                "count output buffers must have shape [B]");

    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    const uint32_t coreCount = static_cast<uint32_t>(platform->GetCoreNumAiv());
    TORCH_CHECK(coreCount > 0, "failed to query AIV core count");
    TORCH_CHECK(realReqCount <= std::numeric_limits<uint32_t>::max() &&
                    sourceStride <= std::numeric_limits<uint32_t>::max() &&
                    batch <= std::numeric_limits<uint32_t>::max(),
                "planner dimensions exceed uint32 launch limits");
    const uint32_t reqCountArg = static_cast<uint32_t>(realReqCount);
    const uint32_t contextLenArg = static_cast<uint32_t>(max_context_len);
    const uint32_t sourceStrideArg = static_cast<uint32_t>(sourceStride);
    const uint32_t capacityArg = static_cast<uint32_t>(cache_capacity);
    const uint32_t batchArg = static_cast<uint32_t>(batch);
    EXEC_KERNEL_CMD(plan_sparse_kv_cache, coreCount, source_to_slot, slot_to_source, req_ids, topk_input, topk_src_ids,
                    topk_dst_slots, topk_miss_counts, miss_src_ids, miss_dst_slots, miss_counts, reqCountArg,
                    contextLenArg, sourceStrideArg, capacityArg, batchArg);
}

}  // namespace npu_kernel
}  // namespace sglang
