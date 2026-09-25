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
struct CopyTransposeTiling {
    uint32_t dstShapeB;
    uint32_t dstShapeN;
    uint32_t dstShapeS;
    uint32_t dstShapeHN;
    uint32_t dstShapeH;
    uint32_t srcShapeB;
    uint32_t srcShapeN;
    uint32_t srcShapeS;
    uint32_t srcShapeHN;
    uint32_t originalShapeNLen;
    uint32_t shapeSHValue;
    uint32_t shapeNsValue;
    uint32_t shapeNsnValue;
    uint32_t invalidParamCopyTransposeTiling;
    uint32_t shapeBHValue;
    uint32_t paramsAlign;
};

struct PromptAttentionBaseParams {
    uint8_t causal;
    uint8_t sparseSizePadding[3];
    uint32_t sparseSize;
    uint32_t sparseMaskS1;
    uint32_t sparseMaskS2;
    uint32_t batchSize;
    uint32_t headNumSize;
    uint32_t seqSize;
    uint32_t headSize;
    float scaleValue;
    int32_t preTokens;
    int32_t nextTokens;
    int32_t blockSize;
    int32_t blockTableDim2;
    int32_t PABlockNumSum;
    uint32_t dimNumOfseq;
    uint32_t typeByteNum;
    uint32_t seqInnerSize;
    uint32_t prefixSeqInnerSize;
    uint32_t usePseShift;
    uint32_t useMask;
    uint32_t headNumRatio;
    uint32_t attenMaskElemType;
    uint32_t pseShiftTypeByteNum;
    uint32_t pseMaskMaxSize;
    uint32_t maskTypeByteNum;
    uint32_t outputTypeByteNum;
    uint32_t softmaxTypeByteNum;
    uint32_t sparseMode;
    uint32_t alignedHeadSize;
    uint32_t splitS2;
    uint32_t splitD;
    uint32_t layoutType;
    uint32_t PAlayoutType;
    uint32_t pseShiftS1Size;
    uint32_t pseShiftS2Size;
    uint32_t maskKVsSize;
    uint32_t maskQsSize;
    uint32_t isLayoutSH;
    uint32_t isActualSeqLengthsNull;
    uint32_t isActualSeqLengthsKVNull;
    uint32_t actualSeqLengthsSize;
    uint32_t actualSeqLengthsKVSize;
    uint32_t deqScaleFlag;
    uint32_t deqScale2Flag;
    uint32_t isAntiPerchannel;
    uint32_t isRowInvalid;
    uint32_t softmaxOuterSize;
    uint32_t isQuant2Perchannel;
    uint32_t isQuant2BF16;
    uint32_t isKvContinuous;
    uint32_t fromFused;
    uint32_t isBSNDOut;
    uint32_t isIFA;
    uint32_t isSoftMaxLseEnable;
    uint32_t isActualSharedPrefixLenNull;
    uint32_t isQHasLeftPadding;
    uint32_t isKVHasLeftPadding;
    uint8_t keyAntiquantModePadding[4];
    int64_t keyAntiquantMode;
    int64_t valueAntiquantMode;
    uint32_t hasKeyAntiquantOffset;
    uint32_t isMsd;
    uint32_t isQuant2FP16;
    uint32_t ropeHeadSize;
    uint32_t qkHeadSize;
    uint32_t vHeadSize;
    uint32_t gOfMla;
    uint8_t padding[4];
};

struct PromptAttentionSeqParams {
    uint32_t CoreHeadNumTail[64];
    uint32_t actualS1[64];
    uint32_t actualCoreNums[64];
    uint32_t singleCoreHeadNumSize[64];
    uint32_t coreSeqPosStart[64];
    uint32_t coreSeqPosEnd[64];
};

struct PromptAttentionSingleCoreParams {
    uint32_t singleProcessSInnerSize;
    uint32_t singleProcessSOuterSize;
    uint32_t multiSmaxsInnerLoopTimes;
    uint32_t actualCoreNums;
    uint32_t pseShiftBatch;
    uint32_t attenMaskBatch;
    uint32_t kvAntiquantSInnerSize;
    uint8_t padding[4];
};

struct PromptAttentionSingleCoreTensorSize {
    uint32_t mmResUbSize;
    uint32_t pseShiftUbSize;
    uint32_t attenMaskUbSize;
    uint32_t maskSize;
    uint32_t softmaxMaxSize;
    uint32_t softmaxSumSize;
    uint32_t softmaxExpSize;
    uint32_t softmaxValueSize;
    uint32_t spmTmpSize;
    uint32_t scmTmpSize;
    uint32_t bmm2ResUbSize;
    uint32_t tmpMMResBmm2PreUbSize;
    uint32_t tmpSoftmaxBmm2UbSize;
    uint32_t selectSpaceUbSize;
    uint32_t tmpSoftMaxV2Size;
    uint32_t mm1TmpUbSize;
    uint32_t mm2TmpUbSize;
    uint32_t kvAntiquantUbSize;
    uint32_t bmm2ResUbMsdSize;
    uint32_t tempBmm2QueueMsdSize;
    uint32_t msdInQueueSize;
    uint32_t msdQRowSumBuffSize;
    uint32_t msdAMaxTmpBuffSize;
    uint32_t msdAMaxResBuffSize;
    uint32_t msdSoftmaxResAmaxBuffSize;
    uint32_t msdSoftmaxRowSumScaleBuffSize;
    uint32_t msdScaleBuffSize;
    uint32_t msdOffsetBuffSize;
    uint32_t msdTmpMm1BuffSize;
    uint32_t msdTmpMm2BuffSize;
    uint32_t msdOutQueueSize;
    uint32_t msdComputeLines;
};

struct PromptAttentionInitOutputParams {
    uint32_t singleCoreSize;
    uint8_t totalOutputSizePadding[4];
    int64_t totalOutputSize;
    int64_t totalSoftMaxLseOutputSize;
    uint32_t needInit;
    uint32_t isOneN;
};

struct AdaBlockSparseAttentionTilingData {
    TCubeTiling bmm1TilingDataRect;
    TCubeTiling bmm2TilingDataRect;
    PromptAttentionBaseParams promptAttentionBaseParams;
    PromptAttentionSeqParams promptAttentionSeqParams;
    PromptAttentionSingleCoreParams promptAttentionSingleCoreParams;
    PromptAttentionSingleCoreTensorSize promptAttentionTensorSizeRect;
    PromptAttentionInitOutputParams promptAttentionInitOutputParams;
    SoftMaxTiling softmaxTilingDataRect;
    SoftMaxTiling softmaxFlashTilingDataRect;
    CopyTransposeTiling transposeTilingDataRect;
};

#pragma pack(pop)
