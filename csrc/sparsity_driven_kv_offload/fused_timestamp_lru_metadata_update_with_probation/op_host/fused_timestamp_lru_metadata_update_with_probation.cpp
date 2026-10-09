// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License");
// you may not use this file except in compliance with the License.

#include "defines.h"
#include "torch_helper.h"
#include "tiling/platform/platform_ascendc.h"

#include "aclrtlaunch_fused_timestamp_lru_metadata_update_with_probation.h"

#include <algorithm>
#include <limits>
#include <tuple>

namespace sglang {
namespace npu_kernel {

namespace {

constexpr uint32_t kFixedTopk = 2048;
constexpr uint32_t kCacheCapacityUnit = 2048;
constexpr uint32_t kMinCacheCapacity = kCacheCapacityUnit;
constexpr uint32_t kMaxCacheCapacity = 4 * kCacheCapacityUnit;
constexpr uint32_t kPipeReserveBytes = 8 * 1024;
constexpr uint64_t kUint32Max = std::numeric_limits<uint32_t>::max();

uint32_t GetRequiredWorkUbBytes(uint32_t cacheCapacity)
{
    constexpr uint32_t kBytesPerInt = sizeof(int32_t);
    const uint32_t patternBytes = cacheCapacity / 8;
    const uint32_t compactStageABytes =
        5 * cacheCapacity * kBytesPerInt + 2 * patternBytes;
    const uint32_t victimStageBytes =
        2 * cacheCapacity * kBytesPerInt + 7 * kFixedTopk * kBytesPerInt;
    const uint32_t writebackStageBytes =
        5 * cacheCapacity * kBytesPerInt;
    return std::max({compactStageABytes, victimStageBytes, writebackStageBytes});
}

void CheckNpuTensor(const at::Tensor &tensor, const char *name)
{
    TORCH_CHECK(tensor.device().type() == at::DeviceType::PrivateUse1, name, " must be on an NPU device");
}

void CheckSameDevice(const at::Tensor &tensor, const at::Tensor &reference, const char *name)
{
    TORCH_CHECK(tensor.device() == reference.device(), name, " must be on the same device as the reference tensor");
}

void CheckInt32Contiguous(const at::Tensor &tensor, const char *name)
{
    CheckNpuTensor(tensor, name);
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.scalar_type() == at::kInt, name, " must be int32, got ", tensor.scalar_type());
}

void CheckFitsUint32(int64_t value, const char *name)
{
    TORCH_CHECK(value >= 0 && static_cast<uint64_t>(value) <= kUint32Max,
                name, " exceeds uint32 range: ", value);
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> fused_timestamp_lru_metadata_update_with_probation(
    const at::Tensor &req_indices, const at::Tensor &topk_indices,
    const at::Tensor &device_token_pos, const at::Tensor &hit_position_mask,
    at::Tensor &device_lru_slots, at::Tensor &device_lru_slot_stamps,
    int64_t max_context_len, int64_t probation_age, int64_t stamp_max,
    int64_t block_dim)
{
    CheckInt32Contiguous(req_indices, "req_indices");
    CheckInt32Contiguous(topk_indices, "topk_indices");
    CheckInt32Contiguous(device_token_pos, "device_token_pos");
    CheckInt32Contiguous(hit_position_mask, "hit_position_mask");
    CheckInt32Contiguous(device_lru_slots, "device_lru_slots");
    CheckInt32Contiguous(device_lru_slot_stamps, "device_lru_slot_stamps");

    CheckSameDevice(topk_indices, req_indices, "topk_indices");
    CheckSameDevice(device_token_pos, req_indices, "device_token_pos");
    CheckSameDevice(hit_position_mask, req_indices, "hit_position_mask");
    CheckSameDevice(device_lru_slots, req_indices, "device_lru_slots");
    CheckSameDevice(device_lru_slot_stamps, req_indices, "device_lru_slot_stamps");

    TORCH_CHECK(req_indices.dim() == 1, "req_indices must be 1-D");
    TORCH_CHECK(topk_indices.dim() == 2, "topk_indices must be 2-D");
    TORCH_CHECK(device_token_pos.dim() == 2, "device_token_pos must be 2-D");
    TORCH_CHECK(hit_position_mask.dim() == 2, "hit_position_mask must be 2-D");
    TORCH_CHECK(device_lru_slots.dim() == 2, "device_lru_slots must be 2-D");
    TORCH_CHECK(device_lru_slot_stamps.dim() == 2, "device_lru_slot_stamps must be 2-D");

    const int64_t batchSize64 = req_indices.size(0);
    const int64_t requestRows64 = device_lru_slots.size(0);
    const int64_t cacheCapacity64 = device_lru_slots.size(1);
    const int64_t topk64 = topk_indices.size(1);

    TORCH_CHECK(batchSize64 > 0, "batch size must be positive");
    TORCH_CHECK(topk64 == kFixedTopk, "fused timestamp LRU requires topk=", kFixedTopk, ", got ", topk64);
    TORCH_CHECK(cacheCapacity64 >= kMinCacheCapacity && cacheCapacity64 <= kMaxCacheCapacity &&
                    cacheCapacity64 % kCacheCapacityUnit == 0,
                "fused timestamp LRU requires cache capacity in {2048, 4096, 6144, 8192}, got ",
                cacheCapacity64);
    TORCH_CHECK(topk_indices.size(0) == batchSize64, "topk_indices dim0 must match req_indices");
    TORCH_CHECK(device_token_pos.sizes() == topk_indices.sizes(),
                "device_token_pos shape must match topk_indices");
    TORCH_CHECK(hit_position_mask.size(0) == batchSize64 &&
                    hit_position_mask.size(1) == cacheCapacity64,
                "hit_position_mask shape must be [batch, cache_capacity]");
    TORCH_CHECK(device_lru_slot_stamps.sizes() == device_lru_slots.sizes(),
                "device_lru_slot_stamps shape must match device_lru_slots");
    TORCH_CHECK(max_context_len > 0 && max_context_len <= std::numeric_limits<int32_t>::max(),
                "max_context_len must fit positive int32, got ", max_context_len);
    TORCH_CHECK(stamp_max > 0 && stamp_max <= std::numeric_limits<int32_t>::max(),
                "stamp_max must fit positive int32, got ", stamp_max);
    TORCH_CHECK(probation_age >= 0 && probation_age <= stamp_max,
                "probation_age must be in [0, stamp_max], got ", probation_age,
                " with stamp_max=", stamp_max);
    TORCH_CHECK(block_dim >= 0, "block_dim must be non-negative, got ", block_dim);

    CheckFitsUint32(batchSize64, "batch size");
    CheckFitsUint32(requestRows64, "request rows");
    CheckFitsUint32(max_context_len, "max_context_len");
    CheckFitsUint32(stamp_max, "stamp_max");
    TORCH_CHECK(static_cast<uint64_t>(requestRows64) * static_cast<uint64_t>(cacheCapacity64) <=
                    kUint32Max,
                "device LRU storage exceeds the kernel uint32 address range");
    TORCH_CHECK(static_cast<uint64_t>(batchSize64) * kFixedTopk <= kUint32Max,
                "batch output storage exceeds the kernel uint32 address range");
    TORCH_CHECK(static_cast<uint64_t>(batchSize64) * static_cast<uint64_t>(cacheCapacity64) <=
                    kUint32Max,
                "hit_position_mask storage exceeds the kernel uint32 address range");

    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    const uint32_t maxAivCoreNum = static_cast<uint32_t>(platform->GetCoreNumAiv());
    TORCH_CHECK(maxAivCoreNum > 0, "failed to get the available AIV core count");
    TORCH_CHECK(block_dim <= static_cast<int64_t>(maxAivCoreNum),
                "block_dim must not exceed available AIV cores ",
                maxAivCoreNum, ", got ", block_dim);
    uint64_t ubSize = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    const uint32_t cacheCapacity = static_cast<uint32_t>(cacheCapacity64);
    const uint32_t workUbBytes = GetRequiredWorkUbBytes(cacheCapacity);
    TORCH_CHECK(ubSize >= static_cast<uint64_t>(workUbBytes + kPipeReserveBytes),
                "fused timestamp LRU requires at least ", workUbBytes + kPipeReserveBytes,
                " bytes of UB, got ", ubSize);
    // Allocate only the verified arena instead of consuming all UB left after
    // the pipe reserve.
    const uint32_t batchSize = static_cast<uint32_t>(batchSize64);
    const uint32_t requestRows = static_cast<uint32_t>(requestRows64);
    const uint32_t maxContextLen = static_cast<uint32_t>(max_context_len);
    const uint32_t stampMax = static_cast<uint32_t>(stamp_max);
    const uint32_t probationAge = static_cast<uint32_t>(probation_age);
    uint32_t effectiveBlockDim = block_dim > 0 ? static_cast<uint32_t>(block_dim)
                                               : std::min(batchSize, maxAivCoreNum);
    effectiveBlockDim = std::max(effectiveBlockDim, 1U);

    auto victimSlots = at::empty_like(topk_indices);
    auto missCounts = at::empty({batchSize64}, topk_indices.options());
    auto npuStream = c10_npu::getCurrentNPUStream();
    req_indices.record_stream(npuStream);
    topk_indices.record_stream(npuStream);
    device_token_pos.record_stream(npuStream);
    hit_position_mask.record_stream(npuStream);
    device_lru_slots.record_stream(npuStream);
    device_lru_slot_stamps.record_stream(npuStream);
    victimSlots.record_stream(npuStream);
    missCounts.record_stream(npuStream);

    EXEC_KERNEL_CMD(fused_timestamp_lru_metadata_update_with_probation, effectiveBlockDim,
                    req_indices, topk_indices, device_token_pos, hit_position_mask,
                    device_lru_slots, device_lru_slot_stamps, victimSlots, missCounts,
                    batchSize, requestRows, cacheCapacity, maxContextLen, stampMax, probationAge,
                    workUbBytes);
    return std::make_tuple(victimSlots, missCounts);
}

}  // namespace npu_kernel
}  // namespace sglang
