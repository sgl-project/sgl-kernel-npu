// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under the Apache License, Version 2.0.
#ifndef SGL_KERNEL_NPU_LIM_TILING_DATA_H
#define SGL_KERNEL_NPU_LIM_TILING_DATA_H

#include <cstddef>
#include <cstdint>

// The first ten uint32 fields retain the upstream serialized tiling layout.
struct FusedLightningIndexerManageTilingData {
    uint32_t bSize;
    uint32_t tSize;
    uint32_t s2Size;
    uint32_t usedCoreNum;
    uint32_t blockSize;
    uint32_t maxBlockNumPerBatch;
    uint32_t poolSize;
    uint32_t n1Size;
    uint32_t cacheSlotsSize;
    uint32_t scheduleMode;
    uint32_t dtype;
    // Explicit zeroed fields keep every cache slot 32-byte aligned and avoid
    // unspecified padding bytes in the serialized tiling-cache key.
    uint32_t reserved[5];
};
static_assert(offsetof(FusedLightningIndexerManageTilingData, dtype) == 40);
static_assert(sizeof(FusedLightningIndexerManageTilingData) == 64);

// Identical MTP workspace regions to upstream DoTiling/MtpWorkspace. The
// system prefix is consumed by GetUserWorkspace in the device entry.
inline uint64_t LimWorkspaceBytes(const FusedLightningIndexerManageTilingData &t, uint64_t systemBytes)
{
    constexpr uint64_t topk = 2048;
    uint64_t bytes = systemBytes;
    bytes += uint64_t(t.usedCoreNum) * 2 * (8 * t.n1Size) * 512 * sizeof(float);
    bytes += uint64_t(t.usedCoreNum) * 2 * 8 * 2 * topk * sizeof(float);
    bytes += uint64_t(t.usedCoreNum) * 2 * 8 * 16 * sizeof(int64_t);
    bytes += uint64_t(t.bSize) * (4 * topk) * sizeof(float) * 2;
    bytes += uint64_t(t.tSize) * ((uint64_t(t.s2Size) + 511) / 512 * 512) * sizeof(float);
    bytes += uint64_t(t.tSize) * 8 * sizeof(float);
    bytes += uint64_t(t.tSize) * 8 * sizeof(int32_t);
    return bytes;
}
#endif
