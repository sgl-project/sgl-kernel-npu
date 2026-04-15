/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#ifndef SPARSE_BLOCK_ESTIMATE_H
#define SPARSE_BLOCK_ESTIMATE_H

#include "exe_graph/runtime/tiling_context.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

struct SparseBlockEstimateCompileInfo {
    uint64_t l0ASize;
    uint64_t l0BSize;
    uint64_t l0CSize;
    uint64_t l1Size;
    uint64_t ubSize;
    uint32_t maxAicCoresNum;
    uint32_t maxAivCoresNum;
    size_t defaultSysWorkspaceSize;
};

BEGIN_TILING_DATA_DEF(SparseBlockEstimateSeqParams)                  // 不同的核心的首尾
TILING_DATA_FIELD_DEF_ARR(uint32_t, 64, coreHeadNumTail);        // coreNStart
TILING_DATA_FIELD_DEF_ARR(uint32_t, 64, actualS1);               // coreNEnd
TILING_DATA_FIELD_DEF_ARR(uint32_t, 64, actualCoreNums);         // coreSidStart
TILING_DATA_FIELD_DEF_ARR(uint32_t, 64, singleCoreHeadNumSize);  // coreSidEnd
TILING_DATA_FIELD_DEF_ARR(uint32_t, 64, coreSeqPosStart);
TILING_DATA_FIELD_DEF_ARR(uint32_t, 64, coreSeqPosEnd);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(SparseBlockEstimateSeqParamsOp, SparseBlockEstimateSeqParams)

BEGIN_TILING_DATA_DEF(SparseBlockEstimateTilingData)
TILING_DATA_FIELD_DEF(uint32_t, actualCoreNums);  // 分核后实际使用核心数量
TILING_DATA_FIELD_DEF(uint32_t, coreNumAic);
TILING_DATA_FIELD_DEF(uint32_t, batchSize);
TILING_DATA_FIELD_DEF(uint32_t, seqLenQ);
TILING_DATA_FIELD_DEF(uint32_t, seqLenK);
TILING_DATA_FIELD_DEF(uint32_t, actualSeqLengthsSize);    // 默认值0
TILING_DATA_FIELD_DEF(uint32_t, actualSeqLengthsKVSize);  // 默认值0
TILING_DATA_FIELD_DEF(uint32_t, headNumQ);
TILING_DATA_FIELD_DEF(uint32_t, headNumKV);
TILING_DATA_FIELD_DEF(uint32_t, dim);
TILING_DATA_FIELD_DEF(uint32_t, stride);
TILING_DATA_FIELD_DEF(uint32_t, sparseSize);
TILING_DATA_FIELD_DEF(uint32_t, sInnerFactor);
TILING_DATA_FIELD_DEF(uint32_t, sOuterFactor);
TILING_DATA_FIELD_DEF(float, scaleFactor);
TILING_DATA_FIELD_DEF(float, threshold);
TILING_DATA_FIELD_DEF(bool, causal);
TILING_DATA_FIELD_DEF(bool, setFirstCol);
TILING_DATA_FIELD_DEF(bool, setDiag);
TILING_DATA_FIELD_DEF(float, rowSparse);
TILING_DATA_FIELD_DEF_STRUCT(SparseBlockEstimateSeqParams, sparseBlockEstimateSeqParams);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, cubeTilingData);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SparseBlockEstimate, SparseBlockEstimateTilingData)
}  // namespace optiling
#endif // SPARSE_BLOCK_ESTIMATE_H