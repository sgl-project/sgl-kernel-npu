// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// CANN-2.0. Serialized layout from PR #18490; zeroed padding for tiling cache.
#pragma once
#include <cstdint>
struct FusedQuantScatterCopySparseFlashAttentionBaseParams {
    uint32_t batchSize, seqSize, qSeqSize, nNumOfQInOneGroup;
    uint32_t sparseBlockCount, sparseMode, outputLayout, blockSize;
    uint32_t maxBlockNumPerBatch, isActualLenDimsNull, isActualLenDimsKVNull, dSizeVInput;
    float scaleValue;
    // CANN aligns each nested tiling section to eight bytes.
    uint32_t reserved;
};
struct FusedQuantScatterCopySparseFlashAttentionSingleCoreParams {
    uint32_t usedCoreNum, reserved;
};
struct FusedQuantScatterCopySparseFlashAttentionTilingData {
    FusedQuantScatterCopySparseFlashAttentionBaseParams baseParams;
    FusedQuantScatterCopySparseFlashAttentionSingleCoreParams singleCoreParams;
    uint32_t dramMaxBlocks, hbmPhysicalBlocks, dramPhysicalBlocks;
    uint32_t reserved[1];
};
static_assert(sizeof(FusedQuantScatterCopySparseFlashAttentionBaseParams) == 56);
static_assert(sizeof(FusedQuantScatterCopySparseFlashAttentionSingleCoreParams) == 8);
static_assert(sizeof(FusedQuantScatterCopySparseFlashAttentionTilingData) == 80);
