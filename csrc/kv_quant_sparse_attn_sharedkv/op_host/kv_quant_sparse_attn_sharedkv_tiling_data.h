/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#ifndef SGL_KERNEL_NPU_KV_QUANT_SPARSE_ATTN_SHAREDKV_TILING_DATA_H
#define SGL_KERNEL_NPU_KV_QUANT_SPARSE_ATTN_SHAREDKV_TILING_DATA_H

#include <cstdint>

namespace optiling {

// Keep the source operator's field order and types: the AICore kernel reads
// this object directly from the device tiling buffer.
struct KvQuantSparseAttnSharedkvBaseParams {
    uint32_t batchSize = 0;
    uint32_t qSeqSize = 0;
    uint32_t kvSeqSize = 0;
    uint32_t paOriBlockSize = 0;
    uint32_t paCmpBlockSize = 0;
    uint32_t oriMaxBlockNumPerBatch = 0;
    uint32_t cmpMaxBlockNumPerBatch = 0;
    uint32_t nNumOfQInOneGroup = 0;
    uint32_t sparseBlockCount = 0;
    float softmaxScale = 0.0F;
    int32_t oriKvStride = 0;
    int32_t cmpKvStride = 0;
    uint32_t tileSize = 0;
    uint32_t ropeHeadDim = 0;
    uint32_t cmpRatio = 0;
    uint32_t oriMaskMode = 0;
    uint32_t cmpMaskMode = 0;
    int32_t oriWinLeft = 0;
    int32_t oriWinRight = 0;
    uint32_t sparseBlockSize = 0;
    bool hasOriSparseIndices = false;
    uint32_t oriSparseIndexWidth = 0;
    uint32_t dSize = 0;
    uint32_t dSizeVInput = 0;
};

struct KvQuantSparseAttnSharedkvTilingData {
    uint32_t dispatchKey = 0;
    KvQuantSparseAttnSharedkvBaseParams baseParams{};
};

// FLASH_DECODE is fixed to zero and KV_LAYOUT_T is fixed to PA_ND.
// qLayout: BSND=0, TND=1; mode: SWA=0, CFA=1, SCFA=2; splitG: 0/1.
constexpr uint32_t MakeKvQuantSparseAttnSharedkvDispatchKey(uint32_t qLayout, uint32_t mode, bool splitG)
{
    return (qLayout << 8U) | (mode << 4U) | static_cast<uint32_t>(splitG);
}

constexpr bool IsKvQuantSparseAttnSharedkvDispatchKeySupported(uint32_t key)
{
    const uint32_t qLayout = key >> 8U;
    const uint32_t mode = (key >> 4U) & 0xFU;
    const uint32_t splitG = key & 0xFU;
    return qLayout <= 1U && mode <= 2U && splitG <= 1U &&
           key == MakeKvQuantSparseAttnSharedkvDispatchKey(qLayout, mode, splitG != 0U);
}

}  // namespace optiling
#endif
