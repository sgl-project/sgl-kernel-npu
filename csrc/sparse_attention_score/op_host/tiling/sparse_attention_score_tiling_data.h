/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software; you can redistribute it and/or modify it under the terms of
 * CANN Open Software License Agreement Version 2.0 ("the License").
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR
 * FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the
 * full text of the License.
 */

#ifndef SPARSE_ATTENTION_SCORE_TILING_DATA_H
#define SPARSE_ATTENTION_SCORE_TILING_DATA_H

#include <cstdint>

namespace sglang {
namespace SAHost {

// Host-side mirror of SparseAttn::SparseAttentionScoreTilingData (kernel_common.hpp).
// Field order/type MUST stay byte-identical: the kernel reinterprets the GM tiling
// blob as that struct. Natural alignment under aarch64 ABI; no #pragma pack needed.
struct SATilingData {
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
    uint32_t innerPrecise;
    uint32_t maxQSeqlen;
    uint64_t mm1OutSize;
    uint64_t smOnlineOutSize;
    uint64_t mm2OutSize;
    uint64_t updateSize;
    uint64_t workSpaceSize;
    uint64_t tilingKey;
    uint32_t groupSize;
    uint32_t qBaseTile;
    uint32_t kvBaseTile;
    uint32_t mm1L1TileM;
    uint32_t mm1L1TileN;
    uint32_t mm1L1TileKLeft;
    uint32_t mm1L1TileKRight;
    uint32_t mm2L1TileM;
    uint32_t mm2L1TileN;
    uint32_t mm2L1TileKLeft;
    uint32_t mm2L1TileKRight;
    uint32_t qL1BufNum;
    uint32_t kL1BufNum;
    uint32_t vL1BufNum;
    uint32_t pL1BufNum;
};

static_assert(sizeof(SATilingData) == 168,
              "SATilingData layout drifted from kernel_common.hpp SparseAttentionScoreTilingData");

}  // namespace SAHost
}  // namespace sglang

#endif  // SPARSE_ATTENTION_SCORE_TILING_DATA_H
