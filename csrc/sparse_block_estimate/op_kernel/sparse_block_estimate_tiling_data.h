/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 *
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#pragma once
#include "kernel_tiling/kernel_tiling.h"
// Device layout of the host tiling records. Scalar fields are naturally
// aligned and every record is padded to eight bytes, as in CANN tiling data.
#pragma pack(push, 1)
struct SparseBlockEstimateSeqParams {
    uint32_t coreHeadNumTail[64];
    uint32_t actualS1[64];
    uint32_t actualCoreNums[64];
    uint32_t singleCoreHeadNumSize[64];
    uint32_t coreSeqPosStart[64];
    uint32_t coreSeqPosEnd[64];
};

struct SparseBlockEstimateTilingData {
    uint32_t actualCoreNums;
    uint32_t coreNumAic;
    uint32_t batchSize;
    uint32_t seqLenQ;
    uint32_t seqLenK;
    uint32_t actualSeqLengthsSize;
    uint32_t actualSeqLengthsKVSize;
    uint32_t headNumQ;
    uint32_t headNumKV;
    uint32_t dim;
    uint32_t stride;
    uint32_t sparseSize;
    uint32_t sInnerFactor;
    uint32_t sOuterFactor;
    float scaleFactor;
    float threshold;
    bool causal;
    bool setFirstCol;
    bool setDiag;
    uint8_t rowSparsePadding[1];
    float rowSparse;
    SparseBlockEstimateSeqParams sparseBlockEstimateSeqParams;
    TCubeTiling cubeTilingData;
};

#pragma pack(pop)
