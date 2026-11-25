/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
// Provenance: csrc/attentions/csrc/ops/ada_block_sparse_attention/op_host/
//   ada_block_sparse_attention_tiling.h and data_copy_transpose_tiling_def.h
// The TilingData macros are replaced by plain structs shared by Host and Kernel.
// Field names, order and types are unchanged, and the legacy get_/set_ accessors
// are kept so the copied Tiling logic needs no rewrite.

#ifndef ADA_BLOCK_SPARSE_ATTENTION_TILING_DATA_H
#define ADA_BLOCK_SPARSE_ATTENTION_TILING_DATA_H

#include <cstdint>
#include <cstring>

#include "kernel_tiling/kernel_tiling.h"

namespace optiling {

// Generated from the legacy TilingData macros; fields, order and types are unchanged.
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
    uint32_t get_dstShapeB() const
    {
        return dstShapeB;
    }
    void set_dstShapeB(uint32_t value)
    {
        dstShapeB = value;
    }
    uint32_t get_dstShapeN() const
    {
        return dstShapeN;
    }
    void set_dstShapeN(uint32_t value)
    {
        dstShapeN = value;
    }
    uint32_t get_dstShapeS() const
    {
        return dstShapeS;
    }
    void set_dstShapeS(uint32_t value)
    {
        dstShapeS = value;
    }
    uint32_t get_dstShapeHN() const
    {
        return dstShapeHN;
    }
    void set_dstShapeHN(uint32_t value)
    {
        dstShapeHN = value;
    }
    uint32_t get_dstShapeH() const
    {
        return dstShapeH;
    }
    void set_dstShapeH(uint32_t value)
    {
        dstShapeH = value;
    }
    uint32_t get_srcShapeB() const
    {
        return srcShapeB;
    }
    void set_srcShapeB(uint32_t value)
    {
        srcShapeB = value;
    }
    uint32_t get_srcShapeN() const
    {
        return srcShapeN;
    }
    void set_srcShapeN(uint32_t value)
    {
        srcShapeN = value;
    }
    uint32_t get_srcShapeS() const
    {
        return srcShapeS;
    }
    void set_srcShapeS(uint32_t value)
    {
        srcShapeS = value;
    }
    uint32_t get_srcShapeHN() const
    {
        return srcShapeHN;
    }
    void set_srcShapeHN(uint32_t value)
    {
        srcShapeHN = value;
    }
    uint32_t get_originalShapeNLen() const
    {
        return originalShapeNLen;
    }
    void set_originalShapeNLen(uint32_t value)
    {
        originalShapeNLen = value;
    }
    uint32_t get_shapeSHValue() const
    {
        return shapeSHValue;
    }
    void set_shapeSHValue(uint32_t value)
    {
        shapeSHValue = value;
    }
    uint32_t get_shapeNsValue() const
    {
        return shapeNsValue;
    }
    void set_shapeNsValue(uint32_t value)
    {
        shapeNsValue = value;
    }
    uint32_t get_shapeNsnValue() const
    {
        return shapeNsnValue;
    }
    void set_shapeNsnValue(uint32_t value)
    {
        shapeNsnValue = value;
    }
    uint32_t get_invalidParamCopyTransposeTiling() const
    {
        return invalidParamCopyTransposeTiling;
    }
    void set_invalidParamCopyTransposeTiling(uint32_t value)
    {
        invalidParamCopyTransposeTiling = value;
    }
    uint32_t get_shapeBHValue() const
    {
        return shapeBHValue;
    }
    void set_shapeBHValue(uint32_t value)
    {
        shapeBHValue = value;
    }
    uint32_t get_paramsAlign() const
    {
        return paramsAlign;
    }
    void set_paramsAlign(uint32_t value)
    {
        paramsAlign = value;
    }
};

struct PromptAttentionBaseParams {
    uint8_t causal;
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
    int64_t keyAntiquantMode;
    int64_t valueAntiquantMode;
    uint32_t hasKeyAntiquantOffset;
    uint32_t isMsd;
    uint32_t isQuant2FP16;
    uint32_t ropeHeadSize;
    uint32_t qkHeadSize;
    uint32_t vHeadSize;
    uint32_t gOfMla;
    uint8_t get_causal() const
    {
        return causal;
    }
    void set_causal(uint8_t value)
    {
        causal = value;
    }
    uint32_t get_sparseSize() const
    {
        return sparseSize;
    }
    void set_sparseSize(uint32_t value)
    {
        sparseSize = value;
    }
    uint32_t get_sparseMaskS1() const
    {
        return sparseMaskS1;
    }
    void set_sparseMaskS1(uint32_t value)
    {
        sparseMaskS1 = value;
    }
    uint32_t get_sparseMaskS2() const
    {
        return sparseMaskS2;
    }
    void set_sparseMaskS2(uint32_t value)
    {
        sparseMaskS2 = value;
    }
    uint32_t get_batchSize() const
    {
        return batchSize;
    }
    void set_batchSize(uint32_t value)
    {
        batchSize = value;
    }
    uint32_t get_headNumSize() const
    {
        return headNumSize;
    }
    void set_headNumSize(uint32_t value)
    {
        headNumSize = value;
    }
    uint32_t get_seqSize() const
    {
        return seqSize;
    }
    void set_seqSize(uint32_t value)
    {
        seqSize = value;
    }
    uint32_t get_headSize() const
    {
        return headSize;
    }
    void set_headSize(uint32_t value)
    {
        headSize = value;
    }
    float get_scaleValue() const
    {
        return scaleValue;
    }
    void set_scaleValue(float value)
    {
        scaleValue = value;
    }
    int32_t get_preTokens() const
    {
        return preTokens;
    }
    void set_preTokens(int32_t value)
    {
        preTokens = value;
    }
    int32_t get_nextTokens() const
    {
        return nextTokens;
    }
    void set_nextTokens(int32_t value)
    {
        nextTokens = value;
    }
    int32_t get_blockSize() const
    {
        return blockSize;
    }
    void set_blockSize(int32_t value)
    {
        blockSize = value;
    }
    int32_t get_blockTableDim2() const
    {
        return blockTableDim2;
    }
    void set_blockTableDim2(int32_t value)
    {
        blockTableDim2 = value;
    }
    int32_t get_PABlockNumSum() const
    {
        return PABlockNumSum;
    }
    void set_PABlockNumSum(int32_t value)
    {
        PABlockNumSum = value;
    }
    uint32_t get_dimNumOfseq() const
    {
        return dimNumOfseq;
    }
    void set_dimNumOfseq(uint32_t value)
    {
        dimNumOfseq = value;
    }
    uint32_t get_typeByteNum() const
    {
        return typeByteNum;
    }
    void set_typeByteNum(uint32_t value)
    {
        typeByteNum = value;
    }
    uint32_t get_seqInnerSize() const
    {
        return seqInnerSize;
    }
    void set_seqInnerSize(uint32_t value)
    {
        seqInnerSize = value;
    }
    uint32_t get_prefixSeqInnerSize() const
    {
        return prefixSeqInnerSize;
    }
    void set_prefixSeqInnerSize(uint32_t value)
    {
        prefixSeqInnerSize = value;
    }
    uint32_t get_usePseShift() const
    {
        return usePseShift;
    }
    void set_usePseShift(uint32_t value)
    {
        usePseShift = value;
    }
    uint32_t get_useMask() const
    {
        return useMask;
    }
    void set_useMask(uint32_t value)
    {
        useMask = value;
    }
    uint32_t get_headNumRatio() const
    {
        return headNumRatio;
    }
    void set_headNumRatio(uint32_t value)
    {
        headNumRatio = value;
    }
    uint32_t get_attenMaskElemType() const
    {
        return attenMaskElemType;
    }
    void set_attenMaskElemType(uint32_t value)
    {
        attenMaskElemType = value;
    }
    uint32_t get_pseShiftTypeByteNum() const
    {
        return pseShiftTypeByteNum;
    }
    void set_pseShiftTypeByteNum(uint32_t value)
    {
        pseShiftTypeByteNum = value;
    }
    uint32_t get_pseMaskMaxSize() const
    {
        return pseMaskMaxSize;
    }
    void set_pseMaskMaxSize(uint32_t value)
    {
        pseMaskMaxSize = value;
    }
    uint32_t get_maskTypeByteNum() const
    {
        return maskTypeByteNum;
    }
    void set_maskTypeByteNum(uint32_t value)
    {
        maskTypeByteNum = value;
    }
    uint32_t get_outputTypeByteNum() const
    {
        return outputTypeByteNum;
    }
    void set_outputTypeByteNum(uint32_t value)
    {
        outputTypeByteNum = value;
    }
    uint32_t get_softmaxTypeByteNum() const
    {
        return softmaxTypeByteNum;
    }
    void set_softmaxTypeByteNum(uint32_t value)
    {
        softmaxTypeByteNum = value;
    }
    uint32_t get_sparseMode() const
    {
        return sparseMode;
    }
    void set_sparseMode(uint32_t value)
    {
        sparseMode = value;
    }
    uint32_t get_alignedHeadSize() const
    {
        return alignedHeadSize;
    }
    void set_alignedHeadSize(uint32_t value)
    {
        alignedHeadSize = value;
    }
    uint32_t get_splitS2() const
    {
        return splitS2;
    }
    void set_splitS2(uint32_t value)
    {
        splitS2 = value;
    }
    uint32_t get_splitD() const
    {
        return splitD;
    }
    void set_splitD(uint32_t value)
    {
        splitD = value;
    }
    uint32_t get_layoutType() const
    {
        return layoutType;
    }
    void set_layoutType(uint32_t value)
    {
        layoutType = value;
    }
    uint32_t get_PAlayoutType() const
    {
        return PAlayoutType;
    }
    void set_PAlayoutType(uint32_t value)
    {
        PAlayoutType = value;
    }
    uint32_t get_pseShiftS1Size() const
    {
        return pseShiftS1Size;
    }
    void set_pseShiftS1Size(uint32_t value)
    {
        pseShiftS1Size = value;
    }
    uint32_t get_pseShiftS2Size() const
    {
        return pseShiftS2Size;
    }
    void set_pseShiftS2Size(uint32_t value)
    {
        pseShiftS2Size = value;
    }
    uint32_t get_maskKVsSize() const
    {
        return maskKVsSize;
    }
    void set_maskKVsSize(uint32_t value)
    {
        maskKVsSize = value;
    }
    uint32_t get_maskQsSize() const
    {
        return maskQsSize;
    }
    void set_maskQsSize(uint32_t value)
    {
        maskQsSize = value;
    }
    uint32_t get_isLayoutSH() const
    {
        return isLayoutSH;
    }
    void set_isLayoutSH(uint32_t value)
    {
        isLayoutSH = value;
    }
    uint32_t get_isActualSeqLengthsNull() const
    {
        return isActualSeqLengthsNull;
    }
    void set_isActualSeqLengthsNull(uint32_t value)
    {
        isActualSeqLengthsNull = value;
    }
    uint32_t get_isActualSeqLengthsKVNull() const
    {
        return isActualSeqLengthsKVNull;
    }
    void set_isActualSeqLengthsKVNull(uint32_t value)
    {
        isActualSeqLengthsKVNull = value;
    }
    uint32_t get_actualSeqLengthsSize() const
    {
        return actualSeqLengthsSize;
    }
    void set_actualSeqLengthsSize(uint32_t value)
    {
        actualSeqLengthsSize = value;
    }
    uint32_t get_actualSeqLengthsKVSize() const
    {
        return actualSeqLengthsKVSize;
    }
    void set_actualSeqLengthsKVSize(uint32_t value)
    {
        actualSeqLengthsKVSize = value;
    }
    uint32_t get_deqScaleFlag() const
    {
        return deqScaleFlag;
    }
    void set_deqScaleFlag(uint32_t value)
    {
        deqScaleFlag = value;
    }
    uint32_t get_deqScale2Flag() const
    {
        return deqScale2Flag;
    }
    void set_deqScale2Flag(uint32_t value)
    {
        deqScale2Flag = value;
    }
    uint32_t get_isAntiPerchannel() const
    {
        return isAntiPerchannel;
    }
    void set_isAntiPerchannel(uint32_t value)
    {
        isAntiPerchannel = value;
    }
    uint32_t get_isRowInvalid() const
    {
        return isRowInvalid;
    }
    void set_isRowInvalid(uint32_t value)
    {
        isRowInvalid = value;
    }
    uint32_t get_softmaxOuterSize() const
    {
        return softmaxOuterSize;
    }
    void set_softmaxOuterSize(uint32_t value)
    {
        softmaxOuterSize = value;
    }
    uint32_t get_isQuant2Perchannel() const
    {
        return isQuant2Perchannel;
    }
    void set_isQuant2Perchannel(uint32_t value)
    {
        isQuant2Perchannel = value;
    }
    uint32_t get_isQuant2BF16() const
    {
        return isQuant2BF16;
    }
    void set_isQuant2BF16(uint32_t value)
    {
        isQuant2BF16 = value;
    }
    uint32_t get_isKvContinuous() const
    {
        return isKvContinuous;
    }
    void set_isKvContinuous(uint32_t value)
    {
        isKvContinuous = value;
    }
    uint32_t get_fromFused() const
    {
        return fromFused;
    }
    void set_fromFused(uint32_t value)
    {
        fromFused = value;
    }
    uint32_t get_isBSNDOut() const
    {
        return isBSNDOut;
    }
    void set_isBSNDOut(uint32_t value)
    {
        isBSNDOut = value;
    }
    uint32_t get_isIFA() const
    {
        return isIFA;
    }
    void set_isIFA(uint32_t value)
    {
        isIFA = value;
    }
    uint32_t get_isSoftMaxLseEnable() const
    {
        return isSoftMaxLseEnable;
    }
    void set_isSoftMaxLseEnable(uint32_t value)
    {
        isSoftMaxLseEnable = value;
    }
    uint32_t get_isActualSharedPrefixLenNull() const
    {
        return isActualSharedPrefixLenNull;
    }
    void set_isActualSharedPrefixLenNull(uint32_t value)
    {
        isActualSharedPrefixLenNull = value;
    }
    uint32_t get_isQHasLeftPadding() const
    {
        return isQHasLeftPadding;
    }
    void set_isQHasLeftPadding(uint32_t value)
    {
        isQHasLeftPadding = value;
    }
    uint32_t get_isKVHasLeftPadding() const
    {
        return isKVHasLeftPadding;
    }
    void set_isKVHasLeftPadding(uint32_t value)
    {
        isKVHasLeftPadding = value;
    }
    int64_t get_keyAntiquantMode() const
    {
        return keyAntiquantMode;
    }
    void set_keyAntiquantMode(int64_t value)
    {
        keyAntiquantMode = value;
    }
    int64_t get_valueAntiquantMode() const
    {
        return valueAntiquantMode;
    }
    void set_valueAntiquantMode(int64_t value)
    {
        valueAntiquantMode = value;
    }
    uint32_t get_hasKeyAntiquantOffset() const
    {
        return hasKeyAntiquantOffset;
    }
    void set_hasKeyAntiquantOffset(uint32_t value)
    {
        hasKeyAntiquantOffset = value;
    }
    uint32_t get_isMsd() const
    {
        return isMsd;
    }
    void set_isMsd(uint32_t value)
    {
        isMsd = value;
    }
    uint32_t get_isQuant2FP16() const
    {
        return isQuant2FP16;
    }
    void set_isQuant2FP16(uint32_t value)
    {
        isQuant2FP16 = value;
    }
    uint32_t get_ropeHeadSize() const
    {
        return ropeHeadSize;
    }
    void set_ropeHeadSize(uint32_t value)
    {
        ropeHeadSize = value;
    }
    uint32_t get_qkHeadSize() const
    {
        return qkHeadSize;
    }
    void set_qkHeadSize(uint32_t value)
    {
        qkHeadSize = value;
    }
    uint32_t get_vHeadSize() const
    {
        return vHeadSize;
    }
    void set_vHeadSize(uint32_t value)
    {
        vHeadSize = value;
    }
    uint32_t get_gOfMla() const
    {
        return gOfMla;
    }
    void set_gOfMla(uint32_t value)
    {
        gOfMla = value;
    }
};

struct PromptAttentionBaseApiBaseParams {
    uint32_t batchSize;
    uint32_t headNumSize;
    uint32_t headSize;
    uint32_t maskTypeByteNum;
    uint32_t inputLayoutType;
    uint32_t kvHeadNumSize;
    uint32_t maxSeqLen;
    uint32_t maxKvSeqLen;
    uint32_t totalQBlkNum;
    uint32_t embeddingSizeV;
    uint32_t quantType;
    uint32_t dataShapeType;
    uint32_t scaleType;
    uint64_t workSize;
    float tor;
    uint32_t headStride;
    uint32_t maskStride;
    uint32_t isTriuMask;
    uint32_t isClamp;
    uint32_t clampMin;
    uint32_t clampMax;
    uint32_t tilingHeadSize;
    uint32_t tilingParaSize;
    uint32_t isLongSeq;
    uint32_t isAlibiMaskSqrt;
    uint32_t maskType;
    uint32_t alibiCompressOffset;
    uint32_t alibiLeftAlign;
    uint32_t ppMScalar;
    uint32_t ppNScalar;
    uint32_t totalQBlkNumFirst;
    uint32_t get_batchSize() const
    {
        return batchSize;
    }
    void set_batchSize(uint32_t value)
    {
        batchSize = value;
    }
    uint32_t get_headNumSize() const
    {
        return headNumSize;
    }
    void set_headNumSize(uint32_t value)
    {
        headNumSize = value;
    }
    uint32_t get_headSize() const
    {
        return headSize;
    }
    void set_headSize(uint32_t value)
    {
        headSize = value;
    }
    uint32_t get_maskTypeByteNum() const
    {
        return maskTypeByteNum;
    }
    void set_maskTypeByteNum(uint32_t value)
    {
        maskTypeByteNum = value;
    }
    uint32_t get_inputLayoutType() const
    {
        return inputLayoutType;
    }
    void set_inputLayoutType(uint32_t value)
    {
        inputLayoutType = value;
    }
    uint32_t get_kvHeadNumSize() const
    {
        return kvHeadNumSize;
    }
    void set_kvHeadNumSize(uint32_t value)
    {
        kvHeadNumSize = value;
    }
    uint32_t get_maxSeqLen() const
    {
        return maxSeqLen;
    }
    void set_maxSeqLen(uint32_t value)
    {
        maxSeqLen = value;
    }
    uint32_t get_maxKvSeqLen() const
    {
        return maxKvSeqLen;
    }
    void set_maxKvSeqLen(uint32_t value)
    {
        maxKvSeqLen = value;
    }
    uint32_t get_totalQBlkNum() const
    {
        return totalQBlkNum;
    }
    void set_totalQBlkNum(uint32_t value)
    {
        totalQBlkNum = value;
    }
    uint32_t get_embeddingSizeV() const
    {
        return embeddingSizeV;
    }
    void set_embeddingSizeV(uint32_t value)
    {
        embeddingSizeV = value;
    }
    uint32_t get_quantType() const
    {
        return quantType;
    }
    void set_quantType(uint32_t value)
    {
        quantType = value;
    }
    uint32_t get_dataShapeType() const
    {
        return dataShapeType;
    }
    void set_dataShapeType(uint32_t value)
    {
        dataShapeType = value;
    }
    uint32_t get_scaleType() const
    {
        return scaleType;
    }
    void set_scaleType(uint32_t value)
    {
        scaleType = value;
    }
    uint64_t get_workSize() const
    {
        return workSize;
    }
    void set_workSize(uint64_t value)
    {
        workSize = value;
    }
    float get_tor() const
    {
        return tor;
    }
    void set_tor(float value)
    {
        tor = value;
    }
    uint32_t get_headStride() const
    {
        return headStride;
    }
    void set_headStride(uint32_t value)
    {
        headStride = value;
    }
    uint32_t get_maskStride() const
    {
        return maskStride;
    }
    void set_maskStride(uint32_t value)
    {
        maskStride = value;
    }
    uint32_t get_isTriuMask() const
    {
        return isTriuMask;
    }
    void set_isTriuMask(uint32_t value)
    {
        isTriuMask = value;
    }
    uint32_t get_isClamp() const
    {
        return isClamp;
    }
    void set_isClamp(uint32_t value)
    {
        isClamp = value;
    }
    uint32_t get_clampMin() const
    {
        return clampMin;
    }
    void set_clampMin(uint32_t value)
    {
        clampMin = value;
    }
    uint32_t get_clampMax() const
    {
        return clampMax;
    }
    void set_clampMax(uint32_t value)
    {
        clampMax = value;
    }
    uint32_t get_tilingHeadSize() const
    {
        return tilingHeadSize;
    }
    void set_tilingHeadSize(uint32_t value)
    {
        tilingHeadSize = value;
    }
    uint32_t get_tilingParaSize() const
    {
        return tilingParaSize;
    }
    void set_tilingParaSize(uint32_t value)
    {
        tilingParaSize = value;
    }
    uint32_t get_isLongSeq() const
    {
        return isLongSeq;
    }
    void set_isLongSeq(uint32_t value)
    {
        isLongSeq = value;
    }
    uint32_t get_isAlibiMaskSqrt() const
    {
        return isAlibiMaskSqrt;
    }
    void set_isAlibiMaskSqrt(uint32_t value)
    {
        isAlibiMaskSqrt = value;
    }
    uint32_t get_maskType() const
    {
        return maskType;
    }
    void set_maskType(uint32_t value)
    {
        maskType = value;
    }
    uint32_t get_alibiCompressOffset() const
    {
        return alibiCompressOffset;
    }
    void set_alibiCompressOffset(uint32_t value)
    {
        alibiCompressOffset = value;
    }
    uint32_t get_alibiLeftAlign() const
    {
        return alibiLeftAlign;
    }
    void set_alibiLeftAlign(uint32_t value)
    {
        alibiLeftAlign = value;
    }
    uint32_t get_ppMScalar() const
    {
        return ppMScalar;
    }
    void set_ppMScalar(uint32_t value)
    {
        ppMScalar = value;
    }
    uint32_t get_ppNScalar() const
    {
        return ppNScalar;
    }
    void set_ppNScalar(uint32_t value)
    {
        ppNScalar = value;
    }
    uint32_t get_totalQBlkNumFirst() const
    {
        return totalQBlkNumFirst;
    }
    void set_totalQBlkNumFirst(uint32_t value)
    {
        totalQBlkNumFirst = value;
    }
};

struct PromptAttentionSeqParams {
    uint32_t CoreHeadNumTail[64];
    uint32_t actualS1[64];
    uint32_t actualCoreNums[64];
    uint32_t singleCoreHeadNumSize[64];
    uint32_t coreSeqPosStart[64];
    uint32_t coreSeqPosEnd[64];
    uint32_t *get_CoreHeadNumTail()
    {
        return CoreHeadNumTail;
    }
    const uint32_t *get_CoreHeadNumTail() const
    {
        return CoreHeadNumTail;
    }
    void set_CoreHeadNumTail(const uint32_t *value)
    {
        std::memcpy(CoreHeadNumTail, value, sizeof(CoreHeadNumTail));
    }
    uint32_t *get_actualS1()
    {
        return actualS1;
    }
    const uint32_t *get_actualS1() const
    {
        return actualS1;
    }
    void set_actualS1(const uint32_t *value)
    {
        std::memcpy(actualS1, value, sizeof(actualS1));
    }
    uint32_t *get_actualCoreNums()
    {
        return actualCoreNums;
    }
    const uint32_t *get_actualCoreNums() const
    {
        return actualCoreNums;
    }
    void set_actualCoreNums(const uint32_t *value)
    {
        std::memcpy(actualCoreNums, value, sizeof(actualCoreNums));
    }
    uint32_t *get_singleCoreHeadNumSize()
    {
        return singleCoreHeadNumSize;
    }
    const uint32_t *get_singleCoreHeadNumSize() const
    {
        return singleCoreHeadNumSize;
    }
    void set_singleCoreHeadNumSize(const uint32_t *value)
    {
        std::memcpy(singleCoreHeadNumSize, value, sizeof(singleCoreHeadNumSize));
    }
    uint32_t *get_coreSeqPosStart()
    {
        return coreSeqPosStart;
    }
    const uint32_t *get_coreSeqPosStart() const
    {
        return coreSeqPosStart;
    }
    void set_coreSeqPosStart(const uint32_t *value)
    {
        std::memcpy(coreSeqPosStart, value, sizeof(coreSeqPosStart));
    }
    uint32_t *get_coreSeqPosEnd()
    {
        return coreSeqPosEnd;
    }
    const uint32_t *get_coreSeqPosEnd() const
    {
        return coreSeqPosEnd;
    }
    void set_coreSeqPosEnd(const uint32_t *value)
    {
        std::memcpy(coreSeqPosEnd, value, sizeof(coreSeqPosEnd));
    }
};

struct PromptAttentionSplitCoreParams {
    uint32_t startBlkArray[50];
    uint32_t endBlkArray[50];
    uint32_t *get_startBlkArray()
    {
        return startBlkArray;
    }
    const uint32_t *get_startBlkArray() const
    {
        return startBlkArray;
    }
    void set_startBlkArray(const uint32_t *value)
    {
        std::memcpy(startBlkArray, value, sizeof(startBlkArray));
    }
    uint32_t *get_endBlkArray()
    {
        return endBlkArray;
    }
    const uint32_t *get_endBlkArray() const
    {
        return endBlkArray;
    }
    void set_endBlkArray(const uint32_t *value)
    {
        std::memcpy(endBlkArray, value, sizeof(endBlkArray));
    }
};

struct PromptAttentionSingleCoreParams {
    uint32_t singleProcessSInnerSize;
    uint32_t singleProcessSOuterSize;
    uint32_t multiSmaxsInnerLoopTimes;
    uint32_t actualCoreNums;
    uint32_t pseShiftBatch;
    uint32_t attenMaskBatch;
    uint32_t kvAntiquantSInnerSize;
    uint32_t get_singleProcessSInnerSize() const
    {
        return singleProcessSInnerSize;
    }
    void set_singleProcessSInnerSize(uint32_t value)
    {
        singleProcessSInnerSize = value;
    }
    uint32_t get_singleProcessSOuterSize() const
    {
        return singleProcessSOuterSize;
    }
    void set_singleProcessSOuterSize(uint32_t value)
    {
        singleProcessSOuterSize = value;
    }
    uint32_t get_multiSmaxsInnerLoopTimes() const
    {
        return multiSmaxsInnerLoopTimes;
    }
    void set_multiSmaxsInnerLoopTimes(uint32_t value)
    {
        multiSmaxsInnerLoopTimes = value;
    }
    uint32_t get_actualCoreNums() const
    {
        return actualCoreNums;
    }
    void set_actualCoreNums(uint32_t value)
    {
        actualCoreNums = value;
    }
    uint32_t get_pseShiftBatch() const
    {
        return pseShiftBatch;
    }
    void set_pseShiftBatch(uint32_t value)
    {
        pseShiftBatch = value;
    }
    uint32_t get_attenMaskBatch() const
    {
        return attenMaskBatch;
    }
    void set_attenMaskBatch(uint32_t value)
    {
        attenMaskBatch = value;
    }
    uint32_t get_kvAntiquantSInnerSize() const
    {
        return kvAntiquantSInnerSize;
    }
    void set_kvAntiquantSInnerSize(uint32_t value)
    {
        kvAntiquantSInnerSize = value;
    }
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
    uint32_t get_mmResUbSize() const
    {
        return mmResUbSize;
    }
    void set_mmResUbSize(uint32_t value)
    {
        mmResUbSize = value;
    }
    uint32_t get_pseShiftUbSize() const
    {
        return pseShiftUbSize;
    }
    void set_pseShiftUbSize(uint32_t value)
    {
        pseShiftUbSize = value;
    }
    uint32_t get_attenMaskUbSize() const
    {
        return attenMaskUbSize;
    }
    void set_attenMaskUbSize(uint32_t value)
    {
        attenMaskUbSize = value;
    }
    uint32_t get_maskSize() const
    {
        return maskSize;
    }
    void set_maskSize(uint32_t value)
    {
        maskSize = value;
    }
    uint32_t get_softmaxMaxSize() const
    {
        return softmaxMaxSize;
    }
    void set_softmaxMaxSize(uint32_t value)
    {
        softmaxMaxSize = value;
    }
    uint32_t get_softmaxSumSize() const
    {
        return softmaxSumSize;
    }
    void set_softmaxSumSize(uint32_t value)
    {
        softmaxSumSize = value;
    }
    uint32_t get_softmaxExpSize() const
    {
        return softmaxExpSize;
    }
    void set_softmaxExpSize(uint32_t value)
    {
        softmaxExpSize = value;
    }
    uint32_t get_softmaxValueSize() const
    {
        return softmaxValueSize;
    }
    void set_softmaxValueSize(uint32_t value)
    {
        softmaxValueSize = value;
    }
    uint32_t get_spmTmpSize() const
    {
        return spmTmpSize;
    }
    void set_spmTmpSize(uint32_t value)
    {
        spmTmpSize = value;
    }
    uint32_t get_scmTmpSize() const
    {
        return scmTmpSize;
    }
    void set_scmTmpSize(uint32_t value)
    {
        scmTmpSize = value;
    }
    uint32_t get_bmm2ResUbSize() const
    {
        return bmm2ResUbSize;
    }
    void set_bmm2ResUbSize(uint32_t value)
    {
        bmm2ResUbSize = value;
    }
    uint32_t get_tmpMMResBmm2PreUbSize() const
    {
        return tmpMMResBmm2PreUbSize;
    }
    void set_tmpMMResBmm2PreUbSize(uint32_t value)
    {
        tmpMMResBmm2PreUbSize = value;
    }
    uint32_t get_tmpSoftmaxBmm2UbSize() const
    {
        return tmpSoftmaxBmm2UbSize;
    }
    void set_tmpSoftmaxBmm2UbSize(uint32_t value)
    {
        tmpSoftmaxBmm2UbSize = value;
    }
    uint32_t get_selectSpaceUbSize() const
    {
        return selectSpaceUbSize;
    }
    void set_selectSpaceUbSize(uint32_t value)
    {
        selectSpaceUbSize = value;
    }
    uint32_t get_tmpSoftMaxV2Size() const
    {
        return tmpSoftMaxV2Size;
    }
    void set_tmpSoftMaxV2Size(uint32_t value)
    {
        tmpSoftMaxV2Size = value;
    }
    uint32_t get_mm1TmpUbSize() const
    {
        return mm1TmpUbSize;
    }
    void set_mm1TmpUbSize(uint32_t value)
    {
        mm1TmpUbSize = value;
    }
    uint32_t get_mm2TmpUbSize() const
    {
        return mm2TmpUbSize;
    }
    void set_mm2TmpUbSize(uint32_t value)
    {
        mm2TmpUbSize = value;
    }
    uint32_t get_kvAntiquantUbSize() const
    {
        return kvAntiquantUbSize;
    }
    void set_kvAntiquantUbSize(uint32_t value)
    {
        kvAntiquantUbSize = value;
    }
    uint32_t get_bmm2ResUbMsdSize() const
    {
        return bmm2ResUbMsdSize;
    }
    void set_bmm2ResUbMsdSize(uint32_t value)
    {
        bmm2ResUbMsdSize = value;
    }
    uint32_t get_tempBmm2QueueMsdSize() const
    {
        return tempBmm2QueueMsdSize;
    }
    void set_tempBmm2QueueMsdSize(uint32_t value)
    {
        tempBmm2QueueMsdSize = value;
    }
    uint32_t get_msdInQueueSize() const
    {
        return msdInQueueSize;
    }
    void set_msdInQueueSize(uint32_t value)
    {
        msdInQueueSize = value;
    }
    uint32_t get_msdQRowSumBuffSize() const
    {
        return msdQRowSumBuffSize;
    }
    void set_msdQRowSumBuffSize(uint32_t value)
    {
        msdQRowSumBuffSize = value;
    }
    uint32_t get_msdAMaxTmpBuffSize() const
    {
        return msdAMaxTmpBuffSize;
    }
    void set_msdAMaxTmpBuffSize(uint32_t value)
    {
        msdAMaxTmpBuffSize = value;
    }
    uint32_t get_msdAMaxResBuffSize() const
    {
        return msdAMaxResBuffSize;
    }
    void set_msdAMaxResBuffSize(uint32_t value)
    {
        msdAMaxResBuffSize = value;
    }
    uint32_t get_msdSoftmaxResAmaxBuffSize() const
    {
        return msdSoftmaxResAmaxBuffSize;
    }
    void set_msdSoftmaxResAmaxBuffSize(uint32_t value)
    {
        msdSoftmaxResAmaxBuffSize = value;
    }
    uint32_t get_msdSoftmaxRowSumScaleBuffSize() const
    {
        return msdSoftmaxRowSumScaleBuffSize;
    }
    void set_msdSoftmaxRowSumScaleBuffSize(uint32_t value)
    {
        msdSoftmaxRowSumScaleBuffSize = value;
    }
    uint32_t get_msdScaleBuffSize() const
    {
        return msdScaleBuffSize;
    }
    void set_msdScaleBuffSize(uint32_t value)
    {
        msdScaleBuffSize = value;
    }
    uint32_t get_msdOffsetBuffSize() const
    {
        return msdOffsetBuffSize;
    }
    void set_msdOffsetBuffSize(uint32_t value)
    {
        msdOffsetBuffSize = value;
    }
    uint32_t get_msdTmpMm1BuffSize() const
    {
        return msdTmpMm1BuffSize;
    }
    void set_msdTmpMm1BuffSize(uint32_t value)
    {
        msdTmpMm1BuffSize = value;
    }
    uint32_t get_msdTmpMm2BuffSize() const
    {
        return msdTmpMm2BuffSize;
    }
    void set_msdTmpMm2BuffSize(uint32_t value)
    {
        msdTmpMm2BuffSize = value;
    }
    uint32_t get_msdOutQueueSize() const
    {
        return msdOutQueueSize;
    }
    void set_msdOutQueueSize(uint32_t value)
    {
        msdOutQueueSize = value;
    }
    uint32_t get_msdComputeLines() const
    {
        return msdComputeLines;
    }
    void set_msdComputeLines(uint32_t value)
    {
        msdComputeLines = value;
    }
};

struct PromptAttentionInitOutputParams {
    uint32_t singleCoreSize;
    int64_t totalOutputSize;
    int64_t totalSoftMaxLseOutputSize;
    uint32_t needInit;
    uint32_t isOneN;
    uint32_t get_singleCoreSize() const
    {
        return singleCoreSize;
    }
    void set_singleCoreSize(uint32_t value)
    {
        singleCoreSize = value;
    }
    int64_t get_totalOutputSize() const
    {
        return totalOutputSize;
    }
    void set_totalOutputSize(int64_t value)
    {
        totalOutputSize = value;
    }
    int64_t get_totalSoftMaxLseOutputSize() const
    {
        return totalSoftMaxLseOutputSize;
    }
    void set_totalSoftMaxLseOutputSize(int64_t value)
    {
        totalSoftMaxLseOutputSize = value;
    }
    uint32_t get_needInit() const
    {
        return needInit;
    }
    void set_needInit(uint32_t value)
    {
        needInit = value;
    }
    uint32_t get_isOneN() const
    {
        return isOneN;
    }
    void set_isOneN(uint32_t value)
    {
        isOneN = value;
    }
};

struct SglAdaBlockSparseAttentionTilingData {
    AscendC::tiling::TCubeTiling bmm1TilingDataRect;
    AscendC::tiling::TCubeTiling bmm2TilingDataRect;
    PromptAttentionBaseParams promptAttentionBaseParams;
    PromptAttentionSeqParams promptAttentionSeqParams;
    PromptAttentionSingleCoreParams promptAttentionSingleCoreParams;
    PromptAttentionSingleCoreTensorSize promptAttentionTensorSizeRect;
    PromptAttentionInitOutputParams promptAttentionInitOutputParams;
    AscendC::tiling::SoftMaxTiling softmaxTilingDataRect;
    AscendC::tiling::SoftMaxTiling softmaxFlashTilingDataRect;
    CopyTransposeTiling transposeTilingDataRect;
};

struct BSAInputParams {
    int64_t bSize;
    int64_t n2Size;
    int64_t gSize;
    int64_t s1Size;
    int64_t s2Size;
    int64_t alignedS2;
    int64_t dSize;
    int64_t valueDSize;
    float keepProb;
    float scaleValue;
    int64_t preTokens;
    int64_t nextTokens;
    int64_t pseS1Size;
    int64_t pseS2Size;
    uint32_t pseBSize;
    uint32_t bandIndex;
    uint8_t layoutType;
    uint8_t pseShapeType;
    uint8_t attenMaskShapeType;
    uint8_t attenMaskDataType;
    uint8_t attenMaskCompressMode;
    uint8_t implMode;
    uint8_t sparseType;
    uint8_t fromFused;
    uint8_t pseEncodeType;
    uint8_t isSoftMaxLseEnable;
    uint16_t remain;
    uint32_t attenMaskS2Size;
    uint32_t pseType;
    uint32_t rsv1;
    int64_t qStartIdx;
    int64_t kvStartIdx;
    int64_t get_bSize() const
    {
        return bSize;
    }
    void set_bSize(int64_t value)
    {
        bSize = value;
    }
    int64_t get_n2Size() const
    {
        return n2Size;
    }
    void set_n2Size(int64_t value)
    {
        n2Size = value;
    }
    int64_t get_gSize() const
    {
        return gSize;
    }
    void set_gSize(int64_t value)
    {
        gSize = value;
    }
    int64_t get_s1Size() const
    {
        return s1Size;
    }
    void set_s1Size(int64_t value)
    {
        s1Size = value;
    }
    int64_t get_s2Size() const
    {
        return s2Size;
    }
    void set_s2Size(int64_t value)
    {
        s2Size = value;
    }
    int64_t get_alignedS2() const
    {
        return alignedS2;
    }
    void set_alignedS2(int64_t value)
    {
        alignedS2 = value;
    }
    int64_t get_dSize() const
    {
        return dSize;
    }
    void set_dSize(int64_t value)
    {
        dSize = value;
    }
    int64_t get_valueDSize() const
    {
        return valueDSize;
    }
    void set_valueDSize(int64_t value)
    {
        valueDSize = value;
    }
    float get_keepProb() const
    {
        return keepProb;
    }
    void set_keepProb(float value)
    {
        keepProb = value;
    }
    float get_scaleValue() const
    {
        return scaleValue;
    }
    void set_scaleValue(float value)
    {
        scaleValue = value;
    }
    int64_t get_preTokens() const
    {
        return preTokens;
    }
    void set_preTokens(int64_t value)
    {
        preTokens = value;
    }
    int64_t get_nextTokens() const
    {
        return nextTokens;
    }
    void set_nextTokens(int64_t value)
    {
        nextTokens = value;
    }
    int64_t get_pseS1Size() const
    {
        return pseS1Size;
    }
    void set_pseS1Size(int64_t value)
    {
        pseS1Size = value;
    }
    int64_t get_pseS2Size() const
    {
        return pseS2Size;
    }
    void set_pseS2Size(int64_t value)
    {
        pseS2Size = value;
    }
    uint32_t get_pseBSize() const
    {
        return pseBSize;
    }
    void set_pseBSize(uint32_t value)
    {
        pseBSize = value;
    }
    uint32_t get_bandIndex() const
    {
        return bandIndex;
    }
    void set_bandIndex(uint32_t value)
    {
        bandIndex = value;
    }
    uint8_t get_layoutType() const
    {
        return layoutType;
    }
    void set_layoutType(uint8_t value)
    {
        layoutType = value;
    }
    uint8_t get_pseShapeType() const
    {
        return pseShapeType;
    }
    void set_pseShapeType(uint8_t value)
    {
        pseShapeType = value;
    }
    uint8_t get_attenMaskShapeType() const
    {
        return attenMaskShapeType;
    }
    void set_attenMaskShapeType(uint8_t value)
    {
        attenMaskShapeType = value;
    }
    uint8_t get_attenMaskDataType() const
    {
        return attenMaskDataType;
    }
    void set_attenMaskDataType(uint8_t value)
    {
        attenMaskDataType = value;
    }
    uint8_t get_attenMaskCompressMode() const
    {
        return attenMaskCompressMode;
    }
    void set_attenMaskCompressMode(uint8_t value)
    {
        attenMaskCompressMode = value;
    }
    uint8_t get_implMode() const
    {
        return implMode;
    }
    void set_implMode(uint8_t value)
    {
        implMode = value;
    }
    uint8_t get_sparseType() const
    {
        return sparseType;
    }
    void set_sparseType(uint8_t value)
    {
        sparseType = value;
    }
    uint8_t get_fromFused() const
    {
        return fromFused;
    }
    void set_fromFused(uint8_t value)
    {
        fromFused = value;
    }
    uint8_t get_pseEncodeType() const
    {
        return pseEncodeType;
    }
    void set_pseEncodeType(uint8_t value)
    {
        pseEncodeType = value;
    }
    uint8_t get_isSoftMaxLseEnable() const
    {
        return isSoftMaxLseEnable;
    }
    void set_isSoftMaxLseEnable(uint8_t value)
    {
        isSoftMaxLseEnable = value;
    }
    uint16_t get_remain() const
    {
        return remain;
    }
    void set_remain(uint16_t value)
    {
        remain = value;
    }
    uint32_t get_attenMaskS2Size() const
    {
        return attenMaskS2Size;
    }
    void set_attenMaskS2Size(uint32_t value)
    {
        attenMaskS2Size = value;
    }
    uint32_t get_pseType() const
    {
        return pseType;
    }
    void set_pseType(uint32_t value)
    {
        pseType = value;
    }
    uint32_t get_rsv1() const
    {
        return rsv1;
    }
    void set_rsv1(uint32_t value)
    {
        rsv1 = value;
    }
    int64_t get_qStartIdx() const
    {
        return qStartIdx;
    }
    void set_qStartIdx(int64_t value)
    {
        qStartIdx = value;
    }
    int64_t get_kvStartIdx() const
    {
        return kvStartIdx;
    }
    void set_kvStartIdx(int64_t value)
    {
        kvStartIdx = value;
    }
};

struct BSAMultiCoreParams {
    int32_t coreNum;
    int32_t reserve;
    int64_t totalSize;
    int64_t splitFactorSize;
    int64_t splitFactorTailSize;
    int64_t sparseStartIdx[48];
    int32_t get_coreNum() const
    {
        return coreNum;
    }
    void set_coreNum(int32_t value)
    {
        coreNum = value;
    }
    int32_t get_reserve() const
    {
        return reserve;
    }
    void set_reserve(int32_t value)
    {
        reserve = value;
    }
    int64_t get_totalSize() const
    {
        return totalSize;
    }
    void set_totalSize(int64_t value)
    {
        totalSize = value;
    }
    int64_t get_splitFactorSize() const
    {
        return splitFactorSize;
    }
    void set_splitFactorSize(int64_t value)
    {
        splitFactorSize = value;
    }
    int64_t get_splitFactorTailSize() const
    {
        return splitFactorTailSize;
    }
    void set_splitFactorTailSize(int64_t value)
    {
        splitFactorTailSize = value;
    }
    int64_t *get_sparseStartIdx()
    {
        return sparseStartIdx;
    }
    const int64_t *get_sparseStartIdx() const
    {
        return sparseStartIdx;
    }
    void set_sparseStartIdx(const int64_t *value)
    {
        std::memcpy(sparseStartIdx, value, sizeof(sparseStartIdx));
    }
};

struct BSACoreParams {
    int32_t s1BaseSize;
    int32_t s1BaseTailSize;
    int64_t s1OuterSize;
    int32_t s1Vec2BaseSize;
    int32_t s1Vec2BaseTailSize;
    int64_t s1Vec2OuterSize;
    int32_t s2BaseSize;
    int32_t s2BaseTailSize;
    int64_t s2OuterSize;
    int32_t dBaseSize;
    int32_t dBaseTailSize;
    int64_t dOuterSize;
    int32_t bBaseSize;
    int32_t bBaseTailSize;
    int64_t bOuterSize;
    int32_t n2BaseSize;
    int32_t n2BaseTailSize;
    int64_t n2OuterSize;
    int32_t gBaseSize;
    int32_t gBaseTailSize;
    int64_t gOuterSize;
    int32_t nRatio;
    int32_t rsvd;
    int64_t s1SparseValidSize;
    int64_t s2SparseValidSize;
    int64_t pseAlibiBaseS1;
    int64_t pseAlibiBaseS2;
    int32_t get_s1BaseSize() const
    {
        return s1BaseSize;
    }
    void set_s1BaseSize(int32_t value)
    {
        s1BaseSize = value;
    }
    int32_t get_s1BaseTailSize() const
    {
        return s1BaseTailSize;
    }
    void set_s1BaseTailSize(int32_t value)
    {
        s1BaseTailSize = value;
    }
    int64_t get_s1OuterSize() const
    {
        return s1OuterSize;
    }
    void set_s1OuterSize(int64_t value)
    {
        s1OuterSize = value;
    }
    int32_t get_s1Vec2BaseSize() const
    {
        return s1Vec2BaseSize;
    }
    void set_s1Vec2BaseSize(int32_t value)
    {
        s1Vec2BaseSize = value;
    }
    int32_t get_s1Vec2BaseTailSize() const
    {
        return s1Vec2BaseTailSize;
    }
    void set_s1Vec2BaseTailSize(int32_t value)
    {
        s1Vec2BaseTailSize = value;
    }
    int64_t get_s1Vec2OuterSize() const
    {
        return s1Vec2OuterSize;
    }
    void set_s1Vec2OuterSize(int64_t value)
    {
        s1Vec2OuterSize = value;
    }
    int32_t get_s2BaseSize() const
    {
        return s2BaseSize;
    }
    void set_s2BaseSize(int32_t value)
    {
        s2BaseSize = value;
    }
    int32_t get_s2BaseTailSize() const
    {
        return s2BaseTailSize;
    }
    void set_s2BaseTailSize(int32_t value)
    {
        s2BaseTailSize = value;
    }
    int64_t get_s2OuterSize() const
    {
        return s2OuterSize;
    }
    void set_s2OuterSize(int64_t value)
    {
        s2OuterSize = value;
    }
    int32_t get_dBaseSize() const
    {
        return dBaseSize;
    }
    void set_dBaseSize(int32_t value)
    {
        dBaseSize = value;
    }
    int32_t get_dBaseTailSize() const
    {
        return dBaseTailSize;
    }
    void set_dBaseTailSize(int32_t value)
    {
        dBaseTailSize = value;
    }
    int64_t get_dOuterSize() const
    {
        return dOuterSize;
    }
    void set_dOuterSize(int64_t value)
    {
        dOuterSize = value;
    }
    int32_t get_bBaseSize() const
    {
        return bBaseSize;
    }
    void set_bBaseSize(int32_t value)
    {
        bBaseSize = value;
    }
    int32_t get_bBaseTailSize() const
    {
        return bBaseTailSize;
    }
    void set_bBaseTailSize(int32_t value)
    {
        bBaseTailSize = value;
    }
    int64_t get_bOuterSize() const
    {
        return bOuterSize;
    }
    void set_bOuterSize(int64_t value)
    {
        bOuterSize = value;
    }
    int32_t get_n2BaseSize() const
    {
        return n2BaseSize;
    }
    void set_n2BaseSize(int32_t value)
    {
        n2BaseSize = value;
    }
    int32_t get_n2BaseTailSize() const
    {
        return n2BaseTailSize;
    }
    void set_n2BaseTailSize(int32_t value)
    {
        n2BaseTailSize = value;
    }
    int64_t get_n2OuterSize() const
    {
        return n2OuterSize;
    }
    void set_n2OuterSize(int64_t value)
    {
        n2OuterSize = value;
    }
    int32_t get_gBaseSize() const
    {
        return gBaseSize;
    }
    void set_gBaseSize(int32_t value)
    {
        gBaseSize = value;
    }
    int32_t get_gBaseTailSize() const
    {
        return gBaseTailSize;
    }
    void set_gBaseTailSize(int32_t value)
    {
        gBaseTailSize = value;
    }
    int64_t get_gOuterSize() const
    {
        return gOuterSize;
    }
    void set_gOuterSize(int64_t value)
    {
        gOuterSize = value;
    }
    int32_t get_nRatio() const
    {
        return nRatio;
    }
    void set_nRatio(int32_t value)
    {
        nRatio = value;
    }
    int32_t get_rsvd() const
    {
        return rsvd;
    }
    void set_rsvd(int32_t value)
    {
        rsvd = value;
    }
    int64_t get_s1SparseValidSize() const
    {
        return s1SparseValidSize;
    }
    void set_s1SparseValidSize(int64_t value)
    {
        s1SparseValidSize = value;
    }
    int64_t get_s2SparseValidSize() const
    {
        return s2SparseValidSize;
    }
    void set_s2SparseValidSize(int64_t value)
    {
        s2SparseValidSize = value;
    }
    int64_t get_pseAlibiBaseS1() const
    {
        return pseAlibiBaseS1;
    }
    void set_pseAlibiBaseS1(int64_t value)
    {
        pseAlibiBaseS1 = value;
    }
    int64_t get_pseAlibiBaseS2() const
    {
        return pseAlibiBaseS2;
    }
    void set_pseAlibiBaseS2(int64_t value)
    {
        pseAlibiBaseS2 = value;
    }
};

struct BSATensorSizeParams {
    int32_t bmm1ResUbSize;
    int32_t attenMaskUbSize;
    int32_t pseUbSize;
    int32_t dropMaskUbSize;
    int32_t castUbSize;
    int32_t softmaxMaxUbSize;
    int32_t softmaxSumUbSize;
    int32_t softmaxExpUbSize;
    int32_t apiTmpBufferBytes;
    int32_t bmm2ResUbSize;
    int32_t inputQueBytes;
    int32_t outputQueBytes;
    int32_t tmpBufBytes;
    int32_t softmaxMaxOffsetBytes;
    int32_t softmaxSumOffsetBytes;
    int32_t maxSumApiOffsetBytes;
    int32_t customSoftmaxApiOffsetBytes;
    int32_t pseTbufOffsetBytes;
    int32_t dropoutApiOffsetBytes;
    int32_t maxSumApiSize;
    int32_t customSoftmaxApiSize;
    int32_t dropoutApiSize;
    int32_t attenMaskApiSize;
    int32_t attenMaskApiOffsetBytes;
    int32_t bmm1ProcessTInStage2Size;
    int32_t bmm1ProcessTInStage2OffsetBytes;
    int32_t wkspSection1OffsetBytes;
    int32_t wkspSection2OffsetBytes;
    int32_t get_bmm1ResUbSize() const
    {
        return bmm1ResUbSize;
    }
    void set_bmm1ResUbSize(int32_t value)
    {
        bmm1ResUbSize = value;
    }
    int32_t get_attenMaskUbSize() const
    {
        return attenMaskUbSize;
    }
    void set_attenMaskUbSize(int32_t value)
    {
        attenMaskUbSize = value;
    }
    int32_t get_pseUbSize() const
    {
        return pseUbSize;
    }
    void set_pseUbSize(int32_t value)
    {
        pseUbSize = value;
    }
    int32_t get_dropMaskUbSize() const
    {
        return dropMaskUbSize;
    }
    void set_dropMaskUbSize(int32_t value)
    {
        dropMaskUbSize = value;
    }
    int32_t get_castUbSize() const
    {
        return castUbSize;
    }
    void set_castUbSize(int32_t value)
    {
        castUbSize = value;
    }
    int32_t get_softmaxMaxUbSize() const
    {
        return softmaxMaxUbSize;
    }
    void set_softmaxMaxUbSize(int32_t value)
    {
        softmaxMaxUbSize = value;
    }
    int32_t get_softmaxSumUbSize() const
    {
        return softmaxSumUbSize;
    }
    void set_softmaxSumUbSize(int32_t value)
    {
        softmaxSumUbSize = value;
    }
    int32_t get_softmaxExpUbSize() const
    {
        return softmaxExpUbSize;
    }
    void set_softmaxExpUbSize(int32_t value)
    {
        softmaxExpUbSize = value;
    }
    int32_t get_apiTmpBufferBytes() const
    {
        return apiTmpBufferBytes;
    }
    void set_apiTmpBufferBytes(int32_t value)
    {
        apiTmpBufferBytes = value;
    }
    int32_t get_bmm2ResUbSize() const
    {
        return bmm2ResUbSize;
    }
    void set_bmm2ResUbSize(int32_t value)
    {
        bmm2ResUbSize = value;
    }
    int32_t get_inputQueBytes() const
    {
        return inputQueBytes;
    }
    void set_inputQueBytes(int32_t value)
    {
        inputQueBytes = value;
    }
    int32_t get_outputQueBytes() const
    {
        return outputQueBytes;
    }
    void set_outputQueBytes(int32_t value)
    {
        outputQueBytes = value;
    }
    int32_t get_tmpBufBytes() const
    {
        return tmpBufBytes;
    }
    void set_tmpBufBytes(int32_t value)
    {
        tmpBufBytes = value;
    }
    int32_t get_softmaxMaxOffsetBytes() const
    {
        return softmaxMaxOffsetBytes;
    }
    void set_softmaxMaxOffsetBytes(int32_t value)
    {
        softmaxMaxOffsetBytes = value;
    }
    int32_t get_softmaxSumOffsetBytes() const
    {
        return softmaxSumOffsetBytes;
    }
    void set_softmaxSumOffsetBytes(int32_t value)
    {
        softmaxSumOffsetBytes = value;
    }
    int32_t get_maxSumApiOffsetBytes() const
    {
        return maxSumApiOffsetBytes;
    }
    void set_maxSumApiOffsetBytes(int32_t value)
    {
        maxSumApiOffsetBytes = value;
    }
    int32_t get_customSoftmaxApiOffsetBytes() const
    {
        return customSoftmaxApiOffsetBytes;
    }
    void set_customSoftmaxApiOffsetBytes(int32_t value)
    {
        customSoftmaxApiOffsetBytes = value;
    }
    int32_t get_pseTbufOffsetBytes() const
    {
        return pseTbufOffsetBytes;
    }
    void set_pseTbufOffsetBytes(int32_t value)
    {
        pseTbufOffsetBytes = value;
    }
    int32_t get_dropoutApiOffsetBytes() const
    {
        return dropoutApiOffsetBytes;
    }
    void set_dropoutApiOffsetBytes(int32_t value)
    {
        dropoutApiOffsetBytes = value;
    }
    int32_t get_maxSumApiSize() const
    {
        return maxSumApiSize;
    }
    void set_maxSumApiSize(int32_t value)
    {
        maxSumApiSize = value;
    }
    int32_t get_customSoftmaxApiSize() const
    {
        return customSoftmaxApiSize;
    }
    void set_customSoftmaxApiSize(int32_t value)
    {
        customSoftmaxApiSize = value;
    }
    int32_t get_dropoutApiSize() const
    {
        return dropoutApiSize;
    }
    void set_dropoutApiSize(int32_t value)
    {
        dropoutApiSize = value;
    }
    int32_t get_attenMaskApiSize() const
    {
        return attenMaskApiSize;
    }
    void set_attenMaskApiSize(int32_t value)
    {
        attenMaskApiSize = value;
    }
    int32_t get_attenMaskApiOffsetBytes() const
    {
        return attenMaskApiOffsetBytes;
    }
    void set_attenMaskApiOffsetBytes(int32_t value)
    {
        attenMaskApiOffsetBytes = value;
    }
    int32_t get_bmm1ProcessTInStage2Size() const
    {
        return bmm1ProcessTInStage2Size;
    }
    void set_bmm1ProcessTInStage2Size(int32_t value)
    {
        bmm1ProcessTInStage2Size = value;
    }
    int32_t get_bmm1ProcessTInStage2OffsetBytes() const
    {
        return bmm1ProcessTInStage2OffsetBytes;
    }
    void set_bmm1ProcessTInStage2OffsetBytes(int32_t value)
    {
        bmm1ProcessTInStage2OffsetBytes = value;
    }
    int32_t get_wkspSection1OffsetBytes() const
    {
        return wkspSection1OffsetBytes;
    }
    void set_wkspSection1OffsetBytes(int32_t value)
    {
        wkspSection1OffsetBytes = value;
    }
    int32_t get_wkspSection2OffsetBytes() const
    {
        return wkspSection2OffsetBytes;
    }
    void set_wkspSection2OffsetBytes(int32_t value)
    {
        wkspSection2OffsetBytes = value;
    }
};

struct MLAGeneralTilingData {
    BSAInputParams BSAinputParams;
    BSAMultiCoreParams BSAmultiCoreParams;
    BSACoreParams BSAcoreParams;
    BSATensorSizeParams BSAtensorSizeParams;
    AscendC::tiling::TCubeTiling bmm1TilingData;
    AscendC::tiling::TCubeTiling bmm2TilingData;
    AscendC::tiling::SoftMaxTiling softmaxFlashTilingData;
    CopyTransposeTiling transposeTilingData;
    CopyTransposeTiling transposeTilingDataTailCore;
};

struct AdaBlockSparseAttentionBaseApiTilingData {
    PromptAttentionBaseApiBaseParams promptAttentionBaseApiBaseParams;
    PromptAttentionSplitCoreParams promptAttentionSplitCoreParams;
};

struct InputParamsRegbase {
    int64_t bSize;
    int64_t n2Size;
    int64_t gSize;
    int64_t s1Size;
    int64_t s2Size;
    int64_t alignedS2;
    int64_t dSize;
    int64_t dSizeV;
    float keepProb;
    float scaleValue;
    int64_t preTokens;
    int64_t nextTokens;
    int64_t pseS1Size;
    int64_t pseS2Size;
    uint32_t pseBSize;
    uint32_t bandIndex;
    uint8_t layoutType;
    uint8_t pseShapeType;
    uint8_t attenMaskShapeType;
    uint8_t attenMaskDataType;
    uint8_t attenMaskCompressMode;
    uint8_t implMode;
    uint8_t sparseType;
    uint8_t needDropMaskOp;
    uint8_t dropMaskOuter;
    uint8_t pseEncodeType;
    uint16_t remain;
    uint32_t attenMaskS2Size;
    uint32_t pseType;
    uint32_t rsv1;
    int64_t qStartIdx;
    int64_t kvStartIdx;
    int64_t s1SparseValidSize;
    int64_t s2SparseValidSize;
    int64_t seed;
    int64_t offset;
    int64_t keepProbUint8;
    int64_t pseAlibiBaseS1;
    int64_t pseAlibiBaseS2;
    uint8_t deqScaleFlag;
    uint8_t deqScale2Flag;
    uint8_t isActualSeqLengthsNull;
    uint8_t isActualSeqLengthsKVNull;
    uint32_t actualSeqLengthsSize;
    uint32_t actualSeqLengthsKVSize;
    uint8_t isKvContinuous;
    uint8_t fromFused;
    uint8_t isBSNDOut;
    uint8_t isGqa;
    uint8_t isSoftMaxLseEnable;
    uint8_t isActualSharedPrefixLenNull;
    uint8_t isQHasLeftPadding;
    uint8_t isKVHasLeftPadding;
    uint32_t ropeHeadSize;
    uint32_t prefixSeqInnerSize;
    uint32_t headNumRatio;
    int32_t blockSize;
    int32_t blockTableDim2;
    int32_t paBlockNumSum;
    uint32_t paLayoutType;
    uint32_t attenMaskS1Size;
    uint32_t isRowInvalid;
    int64_t get_bSize() const
    {
        return bSize;
    }
    void set_bSize(int64_t value)
    {
        bSize = value;
    }
    int64_t get_n2Size() const
    {
        return n2Size;
    }
    void set_n2Size(int64_t value)
    {
        n2Size = value;
    }
    int64_t get_gSize() const
    {
        return gSize;
    }
    void set_gSize(int64_t value)
    {
        gSize = value;
    }
    int64_t get_s1Size() const
    {
        return s1Size;
    }
    void set_s1Size(int64_t value)
    {
        s1Size = value;
    }
    int64_t get_s2Size() const
    {
        return s2Size;
    }
    void set_s2Size(int64_t value)
    {
        s2Size = value;
    }
    int64_t get_alignedS2() const
    {
        return alignedS2;
    }
    void set_alignedS2(int64_t value)
    {
        alignedS2 = value;
    }
    int64_t get_dSize() const
    {
        return dSize;
    }
    void set_dSize(int64_t value)
    {
        dSize = value;
    }
    int64_t get_dSizeV() const
    {
        return dSizeV;
    }
    void set_dSizeV(int64_t value)
    {
        dSizeV = value;
    }
    float get_keepProb() const
    {
        return keepProb;
    }
    void set_keepProb(float value)
    {
        keepProb = value;
    }
    float get_scaleValue() const
    {
        return scaleValue;
    }
    void set_scaleValue(float value)
    {
        scaleValue = value;
    }
    int64_t get_preTokens() const
    {
        return preTokens;
    }
    void set_preTokens(int64_t value)
    {
        preTokens = value;
    }
    int64_t get_nextTokens() const
    {
        return nextTokens;
    }
    void set_nextTokens(int64_t value)
    {
        nextTokens = value;
    }
    int64_t get_pseS1Size() const
    {
        return pseS1Size;
    }
    void set_pseS1Size(int64_t value)
    {
        pseS1Size = value;
    }
    int64_t get_pseS2Size() const
    {
        return pseS2Size;
    }
    void set_pseS2Size(int64_t value)
    {
        pseS2Size = value;
    }
    uint32_t get_pseBSize() const
    {
        return pseBSize;
    }
    void set_pseBSize(uint32_t value)
    {
        pseBSize = value;
    }
    uint32_t get_bandIndex() const
    {
        return bandIndex;
    }
    void set_bandIndex(uint32_t value)
    {
        bandIndex = value;
    }
    uint8_t get_layoutType() const
    {
        return layoutType;
    }
    void set_layoutType(uint8_t value)
    {
        layoutType = value;
    }
    uint8_t get_pseShapeType() const
    {
        return pseShapeType;
    }
    void set_pseShapeType(uint8_t value)
    {
        pseShapeType = value;
    }
    uint8_t get_attenMaskShapeType() const
    {
        return attenMaskShapeType;
    }
    void set_attenMaskShapeType(uint8_t value)
    {
        attenMaskShapeType = value;
    }
    uint8_t get_attenMaskDataType() const
    {
        return attenMaskDataType;
    }
    void set_attenMaskDataType(uint8_t value)
    {
        attenMaskDataType = value;
    }
    uint8_t get_attenMaskCompressMode() const
    {
        return attenMaskCompressMode;
    }
    void set_attenMaskCompressMode(uint8_t value)
    {
        attenMaskCompressMode = value;
    }
    uint8_t get_implMode() const
    {
        return implMode;
    }
    void set_implMode(uint8_t value)
    {
        implMode = value;
    }
    uint8_t get_sparseType() const
    {
        return sparseType;
    }
    void set_sparseType(uint8_t value)
    {
        sparseType = value;
    }
    uint8_t get_needDropMaskOp() const
    {
        return needDropMaskOp;
    }
    void set_needDropMaskOp(uint8_t value)
    {
        needDropMaskOp = value;
    }
    uint8_t get_dropMaskOuter() const
    {
        return dropMaskOuter;
    }
    void set_dropMaskOuter(uint8_t value)
    {
        dropMaskOuter = value;
    }
    uint8_t get_pseEncodeType() const
    {
        return pseEncodeType;
    }
    void set_pseEncodeType(uint8_t value)
    {
        pseEncodeType = value;
    }
    uint16_t get_remain() const
    {
        return remain;
    }
    void set_remain(uint16_t value)
    {
        remain = value;
    }
    uint32_t get_attenMaskS2Size() const
    {
        return attenMaskS2Size;
    }
    void set_attenMaskS2Size(uint32_t value)
    {
        attenMaskS2Size = value;
    }
    uint32_t get_pseType() const
    {
        return pseType;
    }
    void set_pseType(uint32_t value)
    {
        pseType = value;
    }
    uint32_t get_rsv1() const
    {
        return rsv1;
    }
    void set_rsv1(uint32_t value)
    {
        rsv1 = value;
    }
    int64_t get_qStartIdx() const
    {
        return qStartIdx;
    }
    void set_qStartIdx(int64_t value)
    {
        qStartIdx = value;
    }
    int64_t get_kvStartIdx() const
    {
        return kvStartIdx;
    }
    void set_kvStartIdx(int64_t value)
    {
        kvStartIdx = value;
    }
    int64_t get_s1SparseValidSize() const
    {
        return s1SparseValidSize;
    }
    void set_s1SparseValidSize(int64_t value)
    {
        s1SparseValidSize = value;
    }
    int64_t get_s2SparseValidSize() const
    {
        return s2SparseValidSize;
    }
    void set_s2SparseValidSize(int64_t value)
    {
        s2SparseValidSize = value;
    }
    int64_t get_seed() const
    {
        return seed;
    }
    void set_seed(int64_t value)
    {
        seed = value;
    }
    int64_t get_offset() const
    {
        return offset;
    }
    void set_offset(int64_t value)
    {
        offset = value;
    }
    int64_t get_keepProbUint8() const
    {
        return keepProbUint8;
    }
    void set_keepProbUint8(int64_t value)
    {
        keepProbUint8 = value;
    }
    int64_t get_pseAlibiBaseS1() const
    {
        return pseAlibiBaseS1;
    }
    void set_pseAlibiBaseS1(int64_t value)
    {
        pseAlibiBaseS1 = value;
    }
    int64_t get_pseAlibiBaseS2() const
    {
        return pseAlibiBaseS2;
    }
    void set_pseAlibiBaseS2(int64_t value)
    {
        pseAlibiBaseS2 = value;
    }
    uint8_t get_deqScaleFlag() const
    {
        return deqScaleFlag;
    }
    void set_deqScaleFlag(uint8_t value)
    {
        deqScaleFlag = value;
    }
    uint8_t get_deqScale2Flag() const
    {
        return deqScale2Flag;
    }
    void set_deqScale2Flag(uint8_t value)
    {
        deqScale2Flag = value;
    }
    uint8_t get_isActualSeqLengthsNull() const
    {
        return isActualSeqLengthsNull;
    }
    void set_isActualSeqLengthsNull(uint8_t value)
    {
        isActualSeqLengthsNull = value;
    }
    uint8_t get_isActualSeqLengthsKVNull() const
    {
        return isActualSeqLengthsKVNull;
    }
    void set_isActualSeqLengthsKVNull(uint8_t value)
    {
        isActualSeqLengthsKVNull = value;
    }
    uint32_t get_actualSeqLengthsSize() const
    {
        return actualSeqLengthsSize;
    }
    void set_actualSeqLengthsSize(uint32_t value)
    {
        actualSeqLengthsSize = value;
    }
    uint32_t get_actualSeqLengthsKVSize() const
    {
        return actualSeqLengthsKVSize;
    }
    void set_actualSeqLengthsKVSize(uint32_t value)
    {
        actualSeqLengthsKVSize = value;
    }
    uint8_t get_isKvContinuous() const
    {
        return isKvContinuous;
    }
    void set_isKvContinuous(uint8_t value)
    {
        isKvContinuous = value;
    }
    uint8_t get_fromFused() const
    {
        return fromFused;
    }
    void set_fromFused(uint8_t value)
    {
        fromFused = value;
    }
    uint8_t get_isBSNDOut() const
    {
        return isBSNDOut;
    }
    void set_isBSNDOut(uint8_t value)
    {
        isBSNDOut = value;
    }
    uint8_t get_isGqa() const
    {
        return isGqa;
    }
    void set_isGqa(uint8_t value)
    {
        isGqa = value;
    }
    uint8_t get_isSoftMaxLseEnable() const
    {
        return isSoftMaxLseEnable;
    }
    void set_isSoftMaxLseEnable(uint8_t value)
    {
        isSoftMaxLseEnable = value;
    }
    uint8_t get_isActualSharedPrefixLenNull() const
    {
        return isActualSharedPrefixLenNull;
    }
    void set_isActualSharedPrefixLenNull(uint8_t value)
    {
        isActualSharedPrefixLenNull = value;
    }
    uint8_t get_isQHasLeftPadding() const
    {
        return isQHasLeftPadding;
    }
    void set_isQHasLeftPadding(uint8_t value)
    {
        isQHasLeftPadding = value;
    }
    uint8_t get_isKVHasLeftPadding() const
    {
        return isKVHasLeftPadding;
    }
    void set_isKVHasLeftPadding(uint8_t value)
    {
        isKVHasLeftPadding = value;
    }
    uint32_t get_ropeHeadSize() const
    {
        return ropeHeadSize;
    }
    void set_ropeHeadSize(uint32_t value)
    {
        ropeHeadSize = value;
    }
    uint32_t get_prefixSeqInnerSize() const
    {
        return prefixSeqInnerSize;
    }
    void set_prefixSeqInnerSize(uint32_t value)
    {
        prefixSeqInnerSize = value;
    }
    uint32_t get_headNumRatio() const
    {
        return headNumRatio;
    }
    void set_headNumRatio(uint32_t value)
    {
        headNumRatio = value;
    }
    int32_t get_blockSize() const
    {
        return blockSize;
    }
    void set_blockSize(int32_t value)
    {
        blockSize = value;
    }
    int32_t get_blockTableDim2() const
    {
        return blockTableDim2;
    }
    void set_blockTableDim2(int32_t value)
    {
        blockTableDim2 = value;
    }
    int32_t get_paBlockNumSum() const
    {
        return paBlockNumSum;
    }
    void set_paBlockNumSum(int32_t value)
    {
        paBlockNumSum = value;
    }
    uint32_t get_paLayoutType() const
    {
        return paLayoutType;
    }
    void set_paLayoutType(uint32_t value)
    {
        paLayoutType = value;
    }
    uint32_t get_attenMaskS1Size() const
    {
        return attenMaskS1Size;
    }
    void set_attenMaskS1Size(uint32_t value)
    {
        attenMaskS1Size = value;
    }
    uint32_t get_isRowInvalid() const
    {
        return isRowInvalid;
    }
    void set_isRowInvalid(uint32_t value)
    {
        isRowInvalid = value;
    }
};

struct MultiCoreParamsRegbase {
    int32_t coreNum;
    int64_t totalSize;
    int64_t s1OuterSize;
    int64_t splitFactorSize;
    int64_t splitFactorTailSize;
    uint32_t bnStartIdx[48];
    int64_t sparseStartIdx[48];
    int32_t get_coreNum() const
    {
        return coreNum;
    }
    void set_coreNum(int32_t value)
    {
        coreNum = value;
    }
    int64_t get_totalSize() const
    {
        return totalSize;
    }
    void set_totalSize(int64_t value)
    {
        totalSize = value;
    }
    int64_t get_s1OuterSize() const
    {
        return s1OuterSize;
    }
    void set_s1OuterSize(int64_t value)
    {
        s1OuterSize = value;
    }
    int64_t get_splitFactorSize() const
    {
        return splitFactorSize;
    }
    void set_splitFactorSize(int64_t value)
    {
        splitFactorSize = value;
    }
    int64_t get_splitFactorTailSize() const
    {
        return splitFactorTailSize;
    }
    void set_splitFactorTailSize(int64_t value)
    {
        splitFactorTailSize = value;
    }
    uint32_t *get_bnStartIdx()
    {
        return bnStartIdx;
    }
    const uint32_t *get_bnStartIdx() const
    {
        return bnStartIdx;
    }
    void set_bnStartIdx(const uint32_t *value)
    {
        std::memcpy(bnStartIdx, value, sizeof(bnStartIdx));
    }
    int64_t *get_sparseStartIdx()
    {
        return sparseStartIdx;
    }
    const int64_t *get_sparseStartIdx() const
    {
        return sparseStartIdx;
    }
    void set_sparseStartIdx(const int64_t *value)
    {
        std::memcpy(sparseStartIdx, value, sizeof(sparseStartIdx));
    }
};

struct DropmaskParamsRegbase {
    int32_t multiCoreFactorSize;
    int32_t baseUbCalSize;
    int64_t multiCoreTotalSize;
    int64_t shapeTotalSize;
    int32_t get_multiCoreFactorSize() const
    {
        return multiCoreFactorSize;
    }
    void set_multiCoreFactorSize(int32_t value)
    {
        multiCoreFactorSize = value;
    }
    int32_t get_baseUbCalSize() const
    {
        return baseUbCalSize;
    }
    void set_baseUbCalSize(int32_t value)
    {
        baseUbCalSize = value;
    }
    int64_t get_multiCoreTotalSize() const
    {
        return multiCoreTotalSize;
    }
    void set_multiCoreTotalSize(int64_t value)
    {
        multiCoreTotalSize = value;
    }
    int64_t get_shapeTotalSize() const
    {
        return shapeTotalSize;
    }
    void set_shapeTotalSize(int64_t value)
    {
        shapeTotalSize = value;
    }
};

struct InitOutputParams {
    uint32_t singleCoreSize;
    uint8_t needInit;
    uint8_t isOneN;
    uint8_t rsvd[2];
    int64_t totalOutputSize;
    int64_t totalSoftMaxLseOutputSize;
    uint32_t get_singleCoreSize() const
    {
        return singleCoreSize;
    }
    void set_singleCoreSize(uint32_t value)
    {
        singleCoreSize = value;
    }
    uint8_t get_needInit() const
    {
        return needInit;
    }
    void set_needInit(uint8_t value)
    {
        needInit = value;
    }
    uint8_t get_isOneN() const
    {
        return isOneN;
    }
    void set_isOneN(uint8_t value)
    {
        isOneN = value;
    }
    uint8_t *get_rsvd()
    {
        return rsvd;
    }
    const uint8_t *get_rsvd() const
    {
        return rsvd;
    }
    void set_rsvd(const uint8_t *value)
    {
        std::memcpy(rsvd, value, sizeof(rsvd));
    }
    int64_t get_totalOutputSize() const
    {
        return totalOutputSize;
    }
    void set_totalOutputSize(int64_t value)
    {
        totalOutputSize = value;
    }
    int64_t get_totalSoftMaxLseOutputSize() const
    {
        return totalSoftMaxLseOutputSize;
    }
    void set_totalSoftMaxLseOutputSize(int64_t value)
    {
        totalSoftMaxLseOutputSize = value;
    }
};

struct FlashAttentionScoreSimplifiedTilingData {
    InputParamsRegbase inputParamsRegbase;
    MultiCoreParamsRegbase multiCoreParamsRegbase;
    DropmaskParamsRegbase dropmaskParamsRegbase;
    InitOutputParams initOutputParams;
};

}  // namespace optiling

#endif  // ADA_BLOCK_SPARSE_ATTENTION_TILING_DATA_H
