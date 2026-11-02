/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software; you can redistribute it and/or modify it under the terms of
 * CANN Open Software License Agreement Version 2.0 ("the License").
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR
 * FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the
 * full text of the License.
 */

#include "sparse_attention_score_tiling.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace sglang {
namespace SAHost {

void SATiling::DoTiling(const SAInfo &info, SATilingData &tilingData, size_t &workspaceSize, uint32_t &blockDim)
{
    // CalculateTaskSplit: totalTaskNum = totalQTokens * kvHeads; blockDim = min(totalTaskNum, aicNum).
    // One (Q token, KV head) pair per task; cube-core count drives the 1D split.
    uint32_t totalTaskNum = info.totalQTokens * info.kvHeads;
    blockDim = std::min(totalTaskNum, info.aicNum);
    if (blockDim == 0) {
        blockDim = 1;
    }

    // CalculateWorkSpace: full workspace = libapi + identityIdx + per-phase buffers.
    // Kernel recovers user part via GetUserWorkspace; HOST_API allocates exactly this size.
    uint64_t mm1OutSize = 0;
    uint64_t smOnlineOutSize = 0;
    uint64_t mm2OutSize = 0;
    uint64_t updateSize = 0;
    uint64_t workSpaceSize = 0;
    if (info.socVer != SOC_VER_950_CODE) {
        constexpr uint32_t WORKSPACE_BLOCK_SIZE_DB = 131072;
        constexpr uint32_t NUM3 = 3;
        mm1OutSize = static_cast<uint64_t>(blockDim) * WORKSPACE_BLOCK_SIZE_DB * sizeof(float) * NUM3;
        smOnlineOutSize = static_cast<uint64_t>(blockDim) * WORKSPACE_BLOCK_SIZE_DB * sizeof(uint16_t) * NUM3;
        mm2OutSize = static_cast<uint64_t>(blockDim) * WORKSPACE_BLOCK_SIZE_DB * sizeof(float) * NUM3;
        updateSize = static_cast<uint64_t>(blockDim) * WORKSPACE_BLOCK_SIZE_DB * sizeof(float) * NUM3;
        uint64_t identityIdxSize = static_cast<uint64_t>(info.topK) * sizeof(int32_t);
        workSpaceSize = info.libapiSize + identityIdxSize + mm1OutSize + smOnlineOutSize + mm2OutSize + updateSize;
    } else {
        uint32_t dtypeSize = (info.dataType == ge::DT_FLOAT8_E4M3FN) ? 1 : 2;
        uint64_t perTaskWorkspace =
            static_cast<uint64_t>(info.topK) * info.blockSize * info.embeddingSize * dtypeSize * 2;
        uint64_t identityIdxSize = static_cast<uint64_t>(info.topK) * sizeof(int32_t);
        workSpaceSize = info.libapiSize + identityIdxSize + static_cast<uint64_t>(blockDim) * perTaskWorkspace;
    }
    workspaceSize = static_cast<size_t>(workSpaceSize);

    // ---- FillTilingData ----
    tilingData.batch = info.batch;
    tilingData.numHeads = info.numHeads;
    tilingData.kvHeads = info.kvHeads;
    tilingData.embeddingSize = info.embeddingSize;
    tilingData.blockSize = info.blockSize;
    tilingData.topK = info.topK;
    tilingData.maxBlocksPerBatch = info.maxBlocksPerBatch;
    tilingData.totalQTokens = info.totalQTokens;
    tilingData.totalTaskNum = totalTaskNum;
    tilingData.firstBatchTaskNum = info.kvHeads;
    tilingData.scaleValue = info.scaleValue;
    tilingData.innerPrecise = info.innerPrecise;
    tilingData.maxQSeqlen = info.maxQSeqlen;
    tilingData.mm1OutSize = mm1OutSize;
    tilingData.smOnlineOutSize = smOnlineOutSize;
    tilingData.mm2OutSize = mm2OutSize;
    tilingData.updateSize = updateSize;
    tilingData.workSpaceSize = workSpaceSize;
    uint32_t groupSize = (info.kvHeads > 0) ? (info.numHeads / info.kvHeads) : 1;
    tilingData.groupSize = groupSize;
    uint64_t tilingKey = GenerateTilingKey(info);
    tilingData.tilingKey = tilingKey;

    // BaseTileInfo
    uint32_t qBaseTile = (info.embeddingSize <= 128) ? 128 : 64;
    uint32_t kvBaseTile = info.blockSize;
    tilingData.qBaseTile = qBaseTile;
    tilingData.kvBaseTile = kvBaseTile;
    // MmPhaseL1TileInfo: QK matmul L1 tile = [qBaseTile, kvBaseTile, embed]
    tilingData.mm1L1TileM = qBaseTile;
    tilingData.mm1L1TileN = kvBaseTile;
    tilingData.mm1L1TileKLeft = info.embeddingSize;
    tilingData.mm1L1TileKRight = info.embeddingSize;
    // PV matmul L1 tile = [qBaseTile, embed, kvBaseTile]
    tilingData.mm2L1TileM = qBaseTile;
    tilingData.mm2L1TileN = info.embeddingSize;
    tilingData.mm2L1TileKLeft = kvBaseTile;
    tilingData.mm2L1TileKRight = kvBaseTile;
    // Buffer counts
    tilingData.qL1BufNum = 1;
    tilingData.kL1BufNum = 1;
    tilingData.vL1BufNum = 1;
    tilingData.pL1BufNum = 3;  // PRE_LAUNCH + 1
}

uint64_t SATiling::GenerateTilingKey(const SAInfo &info) const
{
    if (info.socVer != SOC_VER_950_CODE) {
        if (info.dataType == ge::DT_BF16 && info.embeddingSize == 128 && info.blockSize == 128) {
            return SASA_BF16_D128_ARCH22_TILING;
        }
        if (info.dataType == ge::DT_FLOAT16 && info.embeddingSize == 128 && info.blockSize == 128) {
            return SASA_FP16_D128_ARCH22_TILING;
        }
        return SASA_FP16_D128_ARCH22_TILING;
    }
    if (info.dataType == ge::DT_FLOAT8_E4M3FN && info.embeddingSize == 128 && info.blockSize == 128) {
        return SASA_FP8_D128_TILING;
    }
    if (info.dataType == ge::DT_BF16 && info.embeddingSize == 128 && info.blockSize == 128) {
        return SASA_BF16_D128_TILING;
    }
    if (info.dataType == ge::DT_FLOAT16 && info.embeddingSize == 128 && info.blockSize == 128) {
        return SASA_FP16_D128_TILING;
    }
    return SASA_FP16_D128_TILING;
}

}  // namespace SAHost
}  // namespace sglang
