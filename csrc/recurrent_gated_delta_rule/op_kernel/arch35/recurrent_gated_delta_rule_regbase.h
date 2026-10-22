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
 * \file recurrent_gated_delta_rule_regbase.h
 * \brief Register-based vector blocks used by recurrent_gated_delta_rule on Ascend 950 (arch35).
 *
 * The arch22 kernel relies on repeat-stride and mask-register block APIs. On arch35 the same math is
 * written with MicroAPI registers, following the arch35 recurrent_gated_delta_rule kernel in vllm-ascend.
 * Matrices are stored row-major with `rowStride` elements per row; only the first `cols` elements of each
 * row take part in the computation.
 */
#ifndef RECURRENT_GATED_DELTA_RULE_REGBASE_H
#define RECURRENT_GATED_DELTA_RULE_REGBASE_H

namespace RecurrentGatedDeltaRuleRegbase {
using namespace AscendC;
using namespace AscendC::MicroAPI;

constexpr uint16_t V_LENGTH = VECTOR_REG_WIDTH / sizeof(float);

// dst[r] = sum_c mat[r * rowStride + c] * vec[c]
__aicore__ inline void RowDotRegbase(LocalTensor<float> &dst, const LocalTensor<float> &mat,
                                     const LocalTensor<float> &vec, uint32_t rows, uint32_t rowStride, uint32_t cols)
{
    __ubuf__ float *dstAddr = (__ubuf__ float *)dst.GetPhyAddr();
    __ubuf__ float *matAddr = (__ubuf__ float *)mat.GetPhyAddr();
    __ubuf__ float *vecAddr = (__ubuf__ float *)vec.GetPhyAddr();

    uint16_t rowNum = static_cast<uint16_t>(rows);
    uint16_t colLoopTimes = static_cast<uint16_t>(Ceil(cols, V_LENGTH));
    __VEC_SCOPE__
    {
        RegTensor<float> matReg;
        RegTensor<float> vecReg;
        RegTensor<float> accReg;
        RegTensor<float> sumReg;
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregLoop;
        for (uint16_t i = 0; i < rowNum; i++) {
            uint32_t colLength = cols;
            Duplicate(accReg, 0.0f);
            for (uint16_t j = 0; j < colLoopTimes; j++) {
                pregLoop = UpdateMask<float>(colLength);
                DataCopy(matReg, matAddr + i * rowStride + j * V_LENGTH);
                DataCopy(vecReg, vecAddr + j * V_LENGTH);
                Mul(matReg, matReg, vecReg, pregLoop);
                Add<float, MaskMergeMode::MERGING>(accReg, accReg, matReg, pregLoop);
            }
            ReduceSum(sumReg, accReg, pregFull);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dstAddr + i, sumReg, pregFull);
        }
    }
}

// mat[r * rowStride + c] += rowScale[r] * vec[c]
__aicore__ inline void RankOneUpdateRegbase(LocalTensor<float> &mat, const LocalTensor<float> &rowScale,
                                            const LocalTensor<float> &vec, uint32_t rows, uint32_t rowStride,
                                            uint32_t cols)
{
    __ubuf__ float *matAddr = (__ubuf__ float *)mat.GetPhyAddr();
    __ubuf__ float *scaleAddr = (__ubuf__ float *)rowScale.GetPhyAddr();
    __ubuf__ float *vecAddr = (__ubuf__ float *)vec.GetPhyAddr();

    uint16_t rowNum = static_cast<uint16_t>(rows);
    uint16_t colLoopTimes = static_cast<uint16_t>(Ceil(cols, V_LENGTH));
    uint32_t colLength = cols;
    __VEC_SCOPE__
    {
        RegTensor<float> matReg;
        RegTensor<float> scaleReg;
        RegTensor<float> vecReg;
        MaskReg pregLoop;
        for (uint16_t j = 0; j < colLoopTimes; j++) {
            pregLoop = UpdateMask<float>(colLength);
            DataCopy(vecReg, vecAddr + j * V_LENGTH);
            for (uint16_t i = 0; i < rowNum; i++) {
                DataCopy<float, LoadDist::DIST_BRC_B32>(scaleReg, scaleAddr + i);
                DataCopy(matReg, matAddr + i * rowStride + j * V_LENGTH);
                Mul(scaleReg, scaleReg, vecReg, pregLoop);
                Add(matReg, matReg, scaleReg, pregLoop);
                DataCopy(matAddr + i * rowStride + j * V_LENGTH, matReg, pregLoop);
            }
        }
    }
}

// x[r * rowStride + c] /= sqrt(sum_c x[r * rowStride + c]^2 + eps)
__aicore__ inline void L2NormalizeRowsRegbase(LocalTensor<float> &x, uint32_t rows, uint32_t rowStride, uint32_t cols,
                                              float eps)
{
    __ubuf__ float *xAddr = (__ubuf__ float *)x.GetPhyAddr();

    uint16_t rowNum = static_cast<uint16_t>(rows);
    uint16_t colLoopTimes = static_cast<uint16_t>(Ceil(cols, V_LENGTH));
    __VEC_SCOPE__
    {
        RegTensor<float> xReg;
        RegTensor<float> accReg;
        RegTensor<float> sumReg;
        RegTensor<float> rootReg;
        RegTensor<float> normReg;
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregFirst = CreateMask<float, MaskPattern::VL1>();
        MaskReg pregLoop;
        for (uint16_t i = 0; i < rowNum; i++) {
            uint32_t colLength = cols;
            Duplicate(accReg, 0.0f);
            for (uint16_t j = 0; j < colLoopTimes; j++) {
                pregLoop = UpdateMask<float>(colLength);
                DataCopy(xReg, xAddr + i * rowStride + j * V_LENGTH);
                Mul(xReg, xReg, xReg, pregLoop);
                Add<float, MaskMergeMode::MERGING>(accReg, accReg, xReg, pregLoop);
            }
            ReduceSum(sumReg, accReg, pregFull);
            Adds(sumReg, sumReg, eps, pregFirst);
            Sqrt(rootReg, sumReg, pregFirst);
            MicroAPI::Duplicate<float, MicroAPI::HighLowPart::LOWEST, MicroAPI::MaskMergeMode::ZEROING>(
                normReg, rootReg, pregFull);

            colLength = cols;
            for (uint16_t j = 0; j < colLoopTimes; j++) {
                pregLoop = UpdateMask<float>(colLength);
                DataCopy(xReg, xAddr + i * rowStride + j * V_LENGTH);
                Div(xReg, xReg, normReg, pregLoop);
                DataCopy(xAddr + i * rowStride + j * V_LENGTH, xReg, pregLoop);
            }
        }
    }
}

}  // namespace RecurrentGatedDeltaRuleRegbase

#endif  // RECURRENT_GATED_DELTA_RULE_REGBASE_H
