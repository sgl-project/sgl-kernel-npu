// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License");
// you may not use this file except in compliance with the License.

#include "defines.h"
#include "torch_helper.h"
#include "tiling/platform/platform_ascendc.h"

#include "aclrtlaunch_parallel_lru_metadata_write.h"

#include <algorithm>
#include <limits>

namespace sglang {
namespace npu_kernel {

namespace {

constexpr uint32_t kCacheCapacityUnit = 2048;
constexpr uint32_t kMinCacheCapacity = kCacheCapacityUnit;
constexpr uint32_t kMaxCacheCapacity = 4 * kCacheCapacityUnit;
constexpr uint32_t kMetadataTileElements = 32;
constexpr uint32_t kAlignment = 8;
constexpr uint64_t kUint32Max = std::numeric_limits<uint32_t>::max();

void CheckNpuTensor(const at::Tensor &tensor, const char *name)
{
    TORCH_CHECK(tensor.device().type() == at::DeviceType::PrivateUse1,
                name, " must be on an NPU device");
}

void CheckSameDevice(const at::Tensor &tensor, const at::Tensor &reference,
                     const char *name)
{
    TORCH_CHECK(tensor.device() == reference.device(), name,
                " must be on the same device as the reference tensor");
}

void CheckInt32Contiguous(const at::Tensor &tensor, const char *name)
{
    CheckNpuTensor(tensor, name);
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.scalar_type() == at::kInt, name,
                " must be int32, got ", tensor.scalar_type());
}

void CheckFitsUint32(int64_t value, const char *name)
{
    TORCH_CHECK(value >= 0 && static_cast<uint64_t>(value) <= kUint32Max,
                name, " exceeds uint32 range: ", value);
}

}  // namespace

void parallel_lru_metadata_write(
    at::Tensor &slot_map, const at::Tensor &req_indices,
    const at::Tensor &topk_indices, const at::Tensor &victim_slots,
    const at::Tensor &miss_counts, at::Tensor &device_slot_tokens,
    int64_t max_context_len, int64_t block_dim)
{
    CheckInt32Contiguous(slot_map, "slot_map");
    CheckInt32Contiguous(req_indices, "req_indices");
    CheckInt32Contiguous(topk_indices, "topk_indices");
    CheckInt32Contiguous(victim_slots, "victim_slots");
    CheckInt32Contiguous(miss_counts, "miss_counts");
    CheckInt32Contiguous(device_slot_tokens, "device_slot_tokens");

    CheckSameDevice(req_indices, slot_map, "req_indices");
    CheckSameDevice(topk_indices, slot_map, "topk_indices");
    CheckSameDevice(victim_slots, slot_map, "victim_slots");
    CheckSameDevice(miss_counts, slot_map, "miss_counts");
    CheckSameDevice(device_slot_tokens, slot_map, "device_slot_tokens");

    TORCH_CHECK(slot_map.dim() == 2, "slot_map must be 2-D");
    TORCH_CHECK(req_indices.dim() == 1, "req_indices must be 1-D");
    TORCH_CHECK(topk_indices.dim() == 2, "topk_indices must be 2-D");
    TORCH_CHECK(victim_slots.dim() == 2, "victim_slots must be 2-D");
    TORCH_CHECK(miss_counts.dim() == 1, "miss_counts must be 1-D");
    TORCH_CHECK(device_slot_tokens.dim() == 2,
                "device_slot_tokens must be 2-D");

    const int64_t batchSize64 = req_indices.size(0);
    const int64_t requestRows64 = device_slot_tokens.size(0);
    const int64_t cacheCapacity64 = device_slot_tokens.size(1);
    const int64_t slotMapRows64 = slot_map.size(0);
    const int64_t slotMapWidth64 = slot_map.size(1);
    const int64_t topk64 = topk_indices.size(1);

    TORCH_CHECK(batchSize64 > 0, "batch size must be positive");
    TORCH_CHECK(topk_indices.size(0) == batchSize64,
                "topk_indices dim0 must match req_indices");
    TORCH_CHECK(topk64 > 0 && topk64 <= cacheCapacity64 &&
                    topk64 % kMetadataTileElements == 0,
                "parallel LRU metadata write requires topk in (0, cache_capacity] and a multiple of ",
                kMetadataTileElements, ", got topk=", topk64,
                " with cache_capacity=", cacheCapacity64);
    TORCH_CHECK(victim_slots.sizes() == topk_indices.sizes(),
                "victim_slots shape must match topk_indices");
    TORCH_CHECK(miss_counts.size(0) == batchSize64,
                "miss_counts shape must be [batch]");
    TORCH_CHECK(cacheCapacity64 >= kMinCacheCapacity &&
                    cacheCapacity64 <= kMaxCacheCapacity &&
                    cacheCapacity64 % kCacheCapacityUnit == 0,
                "device_slot_tokens width must be in {2048, 4096, 6144, 8192}, got ",
                cacheCapacity64);
    TORCH_CHECK(slotMapRows64 >= requestRows64,
                "slot_map must have at least as many rows as device_slot_tokens");
    TORCH_CHECK(max_context_len > 0 && max_context_len <= slotMapWidth64,
                "max_context_len must be in (0, slot_map.size(1)], got ",
                max_context_len);
    TORCH_CHECK(slotMapWidth64 % kAlignment == 0,
                "slot_map row width must be a multiple of ", kAlignment,
                ", got ", slotMapWidth64);
    TORCH_CHECK(block_dim >= 0,
                "block_dim must be non-negative, got ", block_dim);

    CheckFitsUint32(batchSize64, "batch size");
    CheckFitsUint32(requestRows64, "request rows");
    CheckFitsUint32(slotMapRows64, "slot_map rows");
    CheckFitsUint32(slotMapWidth64, "slot_map width");
    CheckFitsUint32(max_context_len, "max_context_len");
    TORCH_CHECK(static_cast<uint64_t>(slotMapRows64) *
                        static_cast<uint64_t>(slotMapWidth64) <=
                    kUint32Max,
                "slot_map storage exceeds the kernel uint32 address range");
    TORCH_CHECK(static_cast<uint64_t>(requestRows64) *
                        static_cast<uint64_t>(cacheCapacity64) <=
                    kUint32Max,
                "device_slot_tokens storage exceeds the kernel uint32 address range");
    TORCH_CHECK(static_cast<uint64_t>(batchSize64) * static_cast<uint64_t>(topk64) <= kUint32Max,
                "batch metadata storage exceeds the kernel uint32 address range");

    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    const uint32_t maxAivCoreNum =
        static_cast<uint32_t>(platform->GetCoreNumAiv());
    TORCH_CHECK(maxAivCoreNum > 0,
                "failed to get the available AIV core count");
    TORCH_CHECK(block_dim <= static_cast<int64_t>(maxAivCoreNum),
                "block_dim must not exceed available AIV cores ",
                maxAivCoreNum, ", got ", block_dim);

    const uint32_t batchSize = static_cast<uint32_t>(batchSize64);
    const uint32_t requestRows = static_cast<uint32_t>(requestRows64);
    const uint32_t cacheCapacity = static_cast<uint32_t>(cacheCapacity64);
    const uint32_t topk = static_cast<uint32_t>(topk64);
    const uint32_t slotMapWidth = static_cast<uint32_t>(slotMapWidth64);
    const uint32_t maxContextLen = static_cast<uint32_t>(max_context_len);
    const uint32_t metadataTilesPerBatch = topk / kMetadataTileElements;
    TORCH_CHECK(static_cast<uint64_t>(batchSize) * metadataTilesPerBatch <= kUint32Max,
                "metadata task count exceeds the kernel uint32 range");
    const uint32_t metadataTaskCount = batchSize * metadataTilesPerBatch;
    uint32_t effectiveBlockDim =
        block_dim > 0 ? static_cast<uint32_t>(block_dim)
                      : std::min(metadataTaskCount, maxAivCoreNum);
    effectiveBlockDim = std::max(effectiveBlockDim, 1U);

    auto npuStream = c10_npu::getCurrentNPUStream();
    slot_map.record_stream(npuStream);
    req_indices.record_stream(npuStream);
    topk_indices.record_stream(npuStream);
    victim_slots.record_stream(npuStream);
    miss_counts.record_stream(npuStream);
    device_slot_tokens.record_stream(npuStream);

    EXEC_KERNEL_CMD(parallel_lru_metadata_write, effectiveBlockDim, slot_map,
                    req_indices, topk_indices, victim_slots, miss_counts,
                    device_slot_tokens, batchSize, requestRows, cacheCapacity, topk,
                    slotMapWidth, maxContextLen);
}

}  // namespace npu_kernel
}  // namespace sglang
