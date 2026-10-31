// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// CANN-2.0. Serialized layout from PR #18490; zeroed padding for tiling cache.
#pragma once
#include <cstdint>
struct FusedQuantScatterCopySparseFlashAttentionBaseParams {
    uint32_t batchSize, seqSize, qSeqSize, nNumOfQInOneGroup;
    uint32_t sparseBlockCount, sparseMode, outputLayout, blockSize;
    uint32_t maxBlockNumPerBatch, isActualLenDimsNull, isActualLenDimsKVNull, dSizeVInput;
    float scaleValue;
};
struct FusedQuantScatterCopySparseFlashAttentionSingleCoreParams { uint32_t usedCoreNum; };
struct FusedQuantScatterCopySparseFlashAttentionTilingData {
    FusedQuantScatterCopySparseFlashAttentionBaseParams baseParams;
    FusedQuantScatterCopySparseFlashAttentionSingleCoreParams singleCoreParams;
    uint32_t dramMaxBlocks, hbmPhysicalBlocks, dramPhysicalBlocks;
    uint32_t reserved[7];
};
static_assert(sizeof(FusedQuantScatterCopySparseFlashAttentionBaseParams) == 52);
static_assert(sizeof(FusedQuantScatterCopySparseFlashAttentionTilingData) == 96);
