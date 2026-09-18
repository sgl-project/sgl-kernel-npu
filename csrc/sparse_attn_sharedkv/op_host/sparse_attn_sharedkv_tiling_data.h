/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#ifndef SGL_KERNEL_NPU_SPARSE_ATTN_SHAREDKV_TILING_DATA_H
#define SGL_KERNEL_NPU_SPARSE_ATTN_SHAREDKV_TILING_DATA_H

#include <cstdint>

namespace optiling {

struct SparseAttnSharedkvSwaParams {
    uint32_t batchSize = 0;
    uint32_t qSeqSize = 0;
    uint32_t kvSeqSize = 0;
    int64_t paBlockSize = 0;
    int64_t oriBlockSize = 0;
    int64_t cmpBlockSize = 0;
    uint32_t oriMaxBlockNumPerBatch = 0;
    uint32_t nNumOfQInOneGroup = 0;
    uint32_t actualLenDimsQ = 0;
    uint32_t actualLenDimsKV = 0;
    float softmaxScale = 1.0F;
    uint32_t outputLayout = 0;
    uint64_t oriMaskMode = 4;
    int64_t oriKvStride = 0;
    int64_t oriWinLeft = 127;
    int64_t oriWinRight = 0;
    int64_t sparseBlockSize = 1;
    bool hasOriSparseIndices = false;
    uint32_t oriSparseIndexWidth = 0;
    uint32_t usedCoreNum = 0;
    uint32_t mmResUbSize = 0;
    uint32_t bmm2ResUbSize = 0;
    uint32_t mBaseSize = 0;
    uint32_t s2BaseSize = 0;
    bool returnSoftmaxLse = false;
};

struct SparseAttnSharedkvCmpParams {
    uint32_t cmpMaxBlockNumPerBatch = 0;
    uint32_t sparseBlockCount = 0;
    int64_t cmpRatio = 1;
    uint64_t cmpMaskMode = 3;
    int64_t cmpKvStride = 0;
};

struct SparseAttnSharedkvTilingData {
    SparseAttnSharedkvSwaParams baseParams{};
    SparseAttnSharedkvCmpParams cmpParams{};
    uint32_t dispatchKey = 0;
};

enum class SASDType : uint32_t { FP16 = 0, BF16 = 1 };

constexpr uint32_t MakeSASDispatchKey(SASDType dtype, uint32_t qLayout, uint32_t kvLayout, uint32_t mode)
{
    return (static_cast<uint32_t>(dtype) << 24U) | (qLayout << 16U) | (kvLayout << 8U) | mode;
}

}  // namespace optiling
#endif
