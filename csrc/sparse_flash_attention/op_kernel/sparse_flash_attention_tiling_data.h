/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef SGL_KERNEL_NPU_SPARSE_FLASH_ATTENTION_TILING_DATA_H
#define SGL_KERNEL_NPU_SPARSE_FLASH_ATTENTION_TILING_DATA_H

#include <cstdint>

// These device-side POD types mirror the serialized layout produced by the
// host-side BEGIN_TILING_DATA_DEF declarations. Keep their field order in sync.
struct alignas(8) SparseFlashAttentionBaseParamsMla {
    uint32_t batchSize;
    uint32_t seqSize;
    uint32_t qSeqSize;
    int64_t blockSize;
    uint32_t maxBlockNumPerBatch;
    float scaleValue;
    uint32_t nNumOfQInOneGroup;
    uint32_t actualLenDimsQ;
    uint32_t actualLenDimsKV;
    uint32_t outputLayout;
    uint32_t sparseMode;
    int64_t preTokens;
    int64_t nextTokens;
    uint32_t attentionMode;
    uint32_t returnSoftmaxLse;
    int64_t sparseBlockSize;
    uint32_t sparseBlockCount;
    uint32_t isActualLenDimsNull;
    uint32_t isActualLenDimsKVNull;
    uint32_t dispatchKey;
};

struct alignas(8) SparseFlashAttentionSingleCoreParamsMla {
    uint32_t usedCoreNum;
};

struct alignas(8) SparseFlashAttentionSingleCoreTensorSizeMla {
    uint32_t mmResUbSize;
    uint32_t bmm2ResUbSize;
};

struct alignas(8) SparseFlashAttentionSplitKVParamsMla {
    uint32_t s2;
    uint32_t accumOutSize;
    uint32_t logSumExpSize;
};

struct alignas(8) SparseFlashAttentionInnerSplitParams {
    uint32_t mBaseSize;
    uint32_t s2BaseSize;
};

struct alignas(8) SparseFlashAttentionTilingDataMla {
    SparseFlashAttentionBaseParamsMla baseParams;
    SparseFlashAttentionSplitKVParamsMla splitKVParams;
    SparseFlashAttentionSingleCoreParamsMla singleCoreParams;
    SparseFlashAttentionSingleCoreTensorSizeMla singleCoreTensorSize;
    SparseFlashAttentionInnerSplitParams innerSplitParams;
};

static_assert(sizeof(SparseFlashAttentionBaseParamsMla) == 104);
static_assert(sizeof(SparseFlashAttentionSplitKVParamsMla) == 16);
static_assert(sizeof(SparseFlashAttentionSingleCoreParamsMla) == 8);
static_assert(sizeof(SparseFlashAttentionTilingDataMla) == 144);

#endif  // SGL_KERNEL_NPU_SPARSE_FLASH_ATTENTION_TILING_DATA_H
