/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SGL_HC_PRE_TILING_DATA_H
#define SGL_HC_PRE_TILING_DATA_H
#include <cstdint>

struct HcPreTilingData {
    int64_t bs;
    int64_t hcMix;
    int64_t hcMult;
    int64_t d;
    int64_t hcMultAlign;
    int64_t rowOfFormerBlock;
    int64_t rowOfTailBlock;
    int64_t rowLoopOfFormerBlock;
    int64_t rowLoopOfTailBlock;
    int64_t rowFactor;
    int64_t tailRowFactorOfFormerBlock;
    int64_t tailRowFactorOfTailBlock;
    int64_t dLoop;
    int64_t dFactor;
    int64_t tailDFactor;
    int64_t iterTimes;
    float hcEps;
    float normEps;
    int64_t kBlockFactor;
    int64_t kFactor;
    int64_t tailKFactor;
    int64_t kLoop;
    int64_t stage2RowFactor;
    int64_t k;
    int64_t kLoopOfFormerBlock;
    int64_t kLoopOfTailBlock;
    int64_t stage1KFactor;
    int64_t kFactorOfFormerBlock;
    int64_t kL1Size;
    int64_t mL1Size;
    int64_t cubeBlockDimM;
    int64_t cubeBlockDimK;
    int64_t kUbSize;
    int64_t cvLoopKSize;
    int64_t multCoreSplitKSize;
    int64_t multCoreSplitMSize;
    int64_t tailKSizeOfFormerBlock;
    int64_t tailKSizeOfTailBlock;
    int64_t mLoopOfFormerBlock;
    int64_t mLoopOfTailBlock;
    int64_t formerMSize;
    int64_t tailMSizeOfFormerBlock;
    int64_t tailMSizeOfTailBlock;
    int64_t firstUsedCoreNum;
    int64_t secondUsedCoreNum;
    int64_t rowInnerFactor;
    int64_t cubeCoreNum;
    int64_t stage1MFactor;
    int64_t bufferPool0Size;
    int64_t bufferPool1Size;
    int64_t mUbSize;
    int64_t hasPreMix;
    int64_t hasPreOut;
};
static_assert(sizeof(HcPreTilingData) == 416);
#endif
