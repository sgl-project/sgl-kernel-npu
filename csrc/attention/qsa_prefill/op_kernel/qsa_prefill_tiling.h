// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under the CANN Open Software License Agreement Version 2.0.
#ifndef SGL_QSA_PREFILL_TILING_H
#define SGL_QSA_PREFILL_TILING_H

#include <cstdint>

namespace optiling {

// Shared host/kernel tiling layouts. Field order and types must stay
// byte-identical across Tiling and Kernel translation units.
struct SglQsaPrefillPrepareTilingData {
    uint32_t rows;
    uint32_t blocks;
    uint32_t tableStride;
    uint32_t length;
    int32_t base;
    uint32_t cap;
    uint32_t outWidth;
    uint32_t reserved;
};

static_assert(sizeof(SglQsaPrefillPrepareTilingData) == 32, "SglQsaPrefillPrepareTilingData layout mismatch");

struct SglQsaPrefillRunsTilingData {
    uint64_t mm1OutSize;
    uint64_t smOnlineOutSize;
    uint64_t mm2OutSize;
    uint64_t updateSize;
    uint64_t workSpaceSize;
    uint64_t tilingKey;
    uint32_t batch;
    uint32_t numHeads;
    uint32_t kvHeads;
    uint32_t embeddingSize;
    uint32_t blockSize;
    uint32_t topK;
    uint32_t maxBlocksPerBatch;
    uint32_t totalQTokens;
    uint32_t totalTaskNum;
    uint32_t firstBatchTaskNum;
    float scaleValue;
    uint32_t maxQSeqlen;
    uint32_t groupSize;
    uint32_t qBaseTile;
    uint32_t kvBaseTile;
    uint32_t reserved;
};

static_assert(sizeof(SglQsaPrefillRunsTilingData) == 112, "SglQsaPrefillRunsTilingData layout mismatch");

}  // namespace optiling

#endif  // SGL_QSA_PREFILL_TILING_H
