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
 * \file moe_init_routing_v3_arch35_tiling_def.h
 * \brief
 */

#ifndef __MOE_INIT_ROUTING_V3_ARCH35_TILING_DEF_H__
#define __MOE_INIT_ROUTING_V3_ARCH35_TILING_DEF_H__

// 单核排序阶段
struct MoeV3Arch35VBSComputeTilingData {
    int64_t needCoreNum{0};
    int64_t perCoreElements{0};
    int64_t perCoreLoops{0};
    int64_t perCorePerLoopElements{0};
    int64_t perCoreLastLoopElements{0};
    int64_t lastCoreElements{0};
    int64_t lastCoreLoops{0};
    int64_t lastCorePerLoopElements{0};
    int64_t lastCoreLastLoopElements{0};
    int64_t oneLoopMaxElements{0};
};

// 多核归并排序阶段
struct MoeV3Arch35VMSMiddleComputeTilingData {
    int64_t needCoreNum{0};
};

// 排序输出阶段
struct MoeV3Arch35SortOutComputeTilingData {
    int64_t oneLoopMaxElements{0};
};

// ExpertTokensCount阶段
struct MoeV3Arch35ExpertTokensCountTilingData {
    int64_t needCoreNum{0};
    int64_t perCoreElements{0};
    int64_t lastCoreElements{0};
    int64_t perCoreLoops{0};
    int64_t perCorePerLoopElements{0};
    int64_t perCoreLastLoopElements{0};
    int64_t lastCoreLoops{0};
    int64_t lastCorePerLoopElements{0};
    int64_t lastCoreLastLoopElements{0};
};

// 拷出阶段
struct MoeV3Arch35GatherOutComputeTilingData {
    int64_t needCoreNum{0};
    int64_t perCoreIndicesElements{0};
    int64_t lastCoreIndicesElements{0};
    int64_t perCoreIndicesLoops{0};
    int64_t perCorePerLoopIndicesElements{0};
    int64_t perCoreLastLoopIndicesElements{0};
    int64_t lastCoreIndicesLoops{0};
    int64_t lastCorePerLoopIndicesElements{0};
    int64_t lastCoreLastLoopIndicesElements{0};
    int64_t colsLoops{0};
    int64_t perLoopCols{0};
    int64_t lastLoopCols{0};
    int64_t activeNum{0};
};

// Arch35用的TilingData
struct MoeInitRoutingV3Arch35TilingData {
    int64_t coreNum{0};
    int64_t n{0};
    int64_t cols{0};
    int64_t k{0};
    int64_t expertStart{0};
    int64_t expertEnd{0};
    int64_t actualExpertNum{0};
    int64_t quantMode{0};
    int64_t rowIdxType{0};
    int64_t isInputScale{0};
    int64_t isInputOffset{0};
    int64_t expertNum{0};
    int64_t expertTokensNumType{0};
    int64_t expertTokensNumFlag{0};
    int64_t gatherFirstFullload{0};
    int64_t epFullload{0};  // 未使用
    int64_t activeNum{0};
    int64_t dropPadMode{0};
    int64_t smoothType{0};  // 未使用
    int64_t worldSize{0};
    uint64_t tilingKey;
    MoeV3Arch35VBSComputeTilingData vbsComputeParamsOp;
    MoeV3Arch35VMSMiddleComputeTilingData vmsMiddleComputeParamsOp;
    MoeV3Arch35SortOutComputeTilingData sortOutComputeParamsOp;
    MoeV3Arch35ExpertTokensCountTilingData expertTokensCountTilingDataOp;
    MoeV3Arch35GatherOutComputeTilingData gatherOutComputeParamsOp;
};

#endif  // __MOE_INIT_ROUTING_V3_ARCH35_TILING_DEF_H__
