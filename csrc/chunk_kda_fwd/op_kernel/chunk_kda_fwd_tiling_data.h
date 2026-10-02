/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file chunk_kda_fwd_tiling_data.h
 * \brief direct-launch tiling data of chunk_kda_fwd
 *
 * Plain-struct mirror of the aclnn ChunkKdaFwdTilingData (register/tilingdata_base.h
 * TILING_DATA_FIELD_DEF layout). Field order/type must be kept in sync with the host
 * side filling routine so that the packed blob copied GM->stack in the kernel entry
 * matches exactly what the host serializes.
 */

#ifndef CHUNK_KDA_FWD_TILING_DATA_H
#define CHUNK_KDA_FWD_TILING_DATA_H

#include <cstdint>

namespace ChunkKdaFwd {

#pragma pack(push, 8)
struct ChunkKdaFwdTilingData {
    int64_t batch;
    int64_t seqNum;
    int64_t qHeadNum;
    int64_t vHeadNum;
    int64_t seqlen;
    int64_t kHeadDim;
    int64_t vHeadDim;
    int64_t chunkSize;
    int64_t totalChunks;
    int64_t inputRank;

    float scale;
    float lowerBound;

    bool hasInitialState;
    bool isVarLen;
    bool safeGate;
    bool inputSequenceMajor;
    bool useGateInKernel;
    bool hasALog;
    bool hasDtBias;
    bool computeGateInPrepare;
    bool fusePostWu;
    bool fusePostWuIntoFwdH;
    bool useDenseFwdH;
    bool storeFinalState;
    bool storeGk;
    bool storeW;
    bool storeU;
    bool storeQG;
    bool storeKg;
    bool storeVNew;
    bool storeH;

    int64_t gateDataType;
    int64_t gateUsedCoreNum;
    int64_t prepareUsedCoreNum;
    int64_t postWuUsedCoreNum;
    int64_t outputUsedCoreNum;

    int64_t gkStorageOffset;
    int64_t finalStateStorageOffset;
    int64_t wStorageOffset;
    int64_t uStorageOffset;
    int64_t qgStorageOffset;
    int64_t kgStorageOffset;
    int64_t vNewStorageOffset;
    int64_t hStorageOffset;
    int64_t qgScaledOffset;
    int64_t prepareAqkFp32Offset;
    int64_t prepareAkkFp32Offset;
    int64_t prepareScratchOffset;
    int64_t postWuScratchOffset;
    int64_t outputScratchOffset;
    int64_t fwdHWorkspaceBaseOffset;
    int64_t vWorkspaceOffset;
    int64_t vUpdateWorkspaceOffset;
    int64_t kDecayWorkspaceOffset;
    int64_t hWorkspaceOffset;
    int64_t numSeqWorkspaceOffset;
    int64_t numChunksWorkspaceOffset;

    // direct-launch only: beta storage dtype (0 = fp32, 1 = bf16)
    int64_t betaDataType;
};
#pragma pack(pop)

}  // namespace ChunkKdaFwd

#endif  // CHUNK_KDA_FWD_TILING_DATA_H
