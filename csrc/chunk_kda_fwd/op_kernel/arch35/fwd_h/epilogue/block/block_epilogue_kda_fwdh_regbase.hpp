/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BLOCK_EPILOGUE_KDA_FWDH_REGBASE_HPP
#define BLOCK_EPILOGUE_KDA_FWDH_REGBASE_HPP

#include "kernel_operator.h"
#ifndef FLA_NPU_REGBASE_HPP_INCLUDED
#define FLA_NPU_REGBASE_HPP_INCLUDED
#include "../../../../kernel_utils/vector/regbase.hpp"
#endif

namespace Catlass::Epilogue::Block::detail {

using namespace AscendC::MicroAPI;

constexpr CastTrait KDA_B16_TO_F32_ZERO = {
    RegLayout::ZERO,
    SatMode::UNKNOWN,
    MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

template <typename T>
__simd_callee__ inline void LoadKdaAsFloat(
    RegTensor<float> &dst, __ubuf__ T *src, MaskReg &mask)
{
    if constexpr (std::is_same<T, float>()) {
        DataCopy<float, LoadDist::DIST_NORM>(dst, src);
    } else if constexpr (std::is_same<T, half>() || std::is_same<T, bfloat16_t>()) {
        RegTensor<T> raw;
        DataCopy<T, LoadDist::DIST_UNPACK_B16>(raw, src);
        Cast<float, T, KDA_B16_TO_F32_ZERO>(dst, raw, mask);
    } else {
        static_assert(!std::is_same<T, T>::value, "KDA regbase only supports float/half/bfloat16_t");
    }
}

template <typename T>
__simd_callee__ inline void LoadKdaBroadcastAsFloat(
    RegTensor<float> &dst, __ubuf__ T *src, MaskReg &mask)
{
    if constexpr (std::is_same<T, float>()) {
        LoadAlign<float, LoadDist::DIST_BRC_B32>(dst, src);
    } else if constexpr (std::is_same<T, half>() || std::is_same<T, bfloat16_t>()) {
        RegTensor<T> raw;
        LoadAlign<T, LoadDist::DIST_BRC_B16>(raw, src);
        Cast<float, T, KDA_B16_TO_F32_ZERO>(dst, raw, mask);
    } else {
        static_assert(!std::is_same<T, T>::value, "KDA regbase only supports float/half/bfloat16_t");
    }
}

__simd_callee__ inline void LoadKdaFloat(RegTensor<float> &dst, __ubuf__ float *src)
{
    DataCopy<float, LoadDist::DIST_NORM>(dst, src);
}

__simd_callee__ inline void StoreKdaFloat(
    __ubuf__ float *dst, RegTensor<float> &src, MaskReg &mask)
{
    DataCopy<float, StoreDist::DIST_NORM_B32>(dst, src, mask);
}

template <typename T, bool USE_EXP2>
static __simd_vf__ inline void PrepareKGateRegbase(
    __ubuf__ float *gateOutput, __ubuf__ T *gateInput, uint16_t count)
{
    constexpr uint16_t FP32_PER_REG = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    constexpr float LN2 = 0.6931471805599453f;
    RegTensor<float> gateReg;
    MaskReg mask;
    uint32_t remaining = count;
    uint32_t offset = 0;
    while (remaining > 0) {
        mask = UpdateMask<float>(remaining);
        LoadKdaAsFloat<T>(gateReg, gateInput + offset, mask);
        if constexpr (USE_EXP2) {
            Muls(gateReg, gateReg, LN2, mask);
        }
        Exp(gateReg, gateReg, mask);
        StoreKdaFloat(gateOutput + offset, gateReg, mask);
        offset += FP32_PER_REG;
    }
}

template <typename T>
static __simd_vf__ inline void ComputeVNewRegbaseDualIssue(
    __ubuf__ float *workspace, __ubuf__ T *uInput, uint32_t count)
{
    constexpr uint32_t FP32_PER_REG = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    RegTensor<float> uReg0;
    RegTensor<float> uReg1;
    RegTensor<float> wsReg0;
    RegTensor<float> wsReg1;
    MaskReg mask0;
    MaskReg mask1;

    uint32_t remaining = count;
    uint32_t offset = 0;
    while (remaining > FP32_PER_REG) {
        mask0 = UpdateMask<float>(remaining);
        mask1 = UpdateMask<float>(remaining);
        LoadKdaAsFloat<T>(uReg0, uInput + offset, mask0);
        LoadKdaAsFloat<T>(uReg1, uInput + offset + FP32_PER_REG, mask1);
        LoadKdaFloat(wsReg0, workspace + offset);
        LoadKdaFloat(wsReg1, workspace + offset + FP32_PER_REG);
        Sub(wsReg0, uReg0, wsReg0, mask0);
        Sub(wsReg1, uReg1, wsReg1, mask1);
        StoreKdaFloat(workspace + offset, wsReg0, mask0);
        StoreKdaFloat(workspace + offset + FP32_PER_REG, wsReg1, mask1);
        offset += 2 * FP32_PER_REG;
    }
    if (remaining > 0) {
        mask0 = UpdateMask<float>(remaining);
        LoadKdaAsFloat<T>(uReg0, uInput + offset, mask0);
        LoadKdaFloat(wsReg0, workspace + offset);
        Sub(wsReg0, uReg0, wsReg0, mask0);
        StoreKdaFloat(workspace + offset, wsReg0, mask0);
    }
}

template <typename T>
static __simd_vf__ inline void ApplyKGateUpdateRegbaseDualIssue(
    __ubuf__ float *update, __ubuf__ T *state, __ubuf__ float *rowScale,
    uint16_t rows, uint16_t cols)
{
    constexpr uint16_t FP32_PER_REG = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    RegTensor<float> stateReg0;
    RegTensor<float> stateReg1;
    RegTensor<float> updateReg0;
    RegTensor<float> updateReg1;
    RegTensor<float> scaleReg0;
    RegTensor<float> scaleReg1;
    MaskReg mask0;
    MaskReg mask1;

    uint16_t row = 0;
    for (; row + 1 < rows; row += 2) {
        LoadAlign<float, LoadDist::DIST_BRC_B32>(scaleReg0, rowScale + row);
        LoadAlign<float, LoadDist::DIST_BRC_B32>(scaleReg1, rowScale + row + 1);
        uint32_t remaining0 = cols;
        uint32_t remaining1 = cols;
        uint16_t colLoops = static_cast<uint16_t>((cols + FP32_PER_REG - 1) / FP32_PER_REG);
        for (uint16_t colLoop = 0; colLoop < colLoops; ++colLoop) {
            uint32_t col = static_cast<uint32_t>(colLoop) * FP32_PER_REG;
            mask0 = UpdateMask<float>(remaining0);
            mask1 = UpdateMask<float>(remaining1);
            LoadKdaAsFloat<T>(stateReg0, state + static_cast<uint32_t>(row) * cols + col, mask0);
            LoadKdaAsFloat<T>(stateReg1, state + static_cast<uint32_t>(row + 1) * cols + col, mask1);
            LoadKdaFloat(updateReg0, update + static_cast<uint32_t>(row) * cols + col);
            LoadKdaFloat(updateReg1, update + static_cast<uint32_t>(row + 1) * cols + col);
            Mul(stateReg0, stateReg0, scaleReg0, mask0);
            Mul(stateReg1, stateReg1, scaleReg1, mask1);
            Add(updateReg0, stateReg0, updateReg0, mask0);
            Add(updateReg1, stateReg1, updateReg1, mask1);
            StoreKdaFloat(update + static_cast<uint32_t>(row) * cols + col, updateReg0, mask0);
            StoreKdaFloat(update + static_cast<uint32_t>(row + 1) * cols + col, updateReg1, mask1);
        }
    }

    if (row < rows) {
        LoadAlign<float, LoadDist::DIST_BRC_B32>(scaleReg0, rowScale + row);
        uint32_t remaining = cols;
        uint16_t colLoops = static_cast<uint16_t>((cols + FP32_PER_REG - 1) / FP32_PER_REG);
        for (uint16_t colLoop = 0; colLoop < colLoops; ++colLoop) {
            uint32_t col = static_cast<uint32_t>(colLoop) * FP32_PER_REG;
            mask0 = UpdateMask<float>(remaining);
            LoadKdaAsFloat<T>(stateReg0, state + static_cast<uint32_t>(row) * cols + col, mask0);
            LoadKdaFloat(updateReg0, update + static_cast<uint32_t>(row) * cols + col);
            Mul(stateReg0, stateReg0, scaleReg0, mask0);
            Add(updateReg0, stateReg0, updateReg0, mask0);
            StoreKdaFloat(update + static_cast<uint32_t>(row) * cols + col, updateReg0, mask0);
        }
    }
}

template <typename KType, typename VType>
static __simd_vf__ inline void ComputeKdaTailFinalStateRegbaseDualIssue(
    __ubuf__ float *state, __ubuf__ KType *kInput, __ubuf__ VType *vInput,
    __ubuf__ float *rowScale, uint16_t rows, uint16_t cols,
    uint16_t tokens, uint16_t kStride, uint16_t kRowOffset)
{
    constexpr uint16_t FP32_PER_REG = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    RegTensor<float> stateReg0;
    RegTensor<float> stateReg1;
    RegTensor<float> valueReg;
    RegTensor<float> productReg0;
    RegTensor<float> productReg1;
    RegTensor<float> weightReg0;
    RegTensor<float> weightReg1;
    RegTensor<float> scaleReg0;
    RegTensor<float> scaleReg1;
    MaskReg mask0;
    MaskReg mask1;

    uint16_t row = 0;
    for (; row + 1 < rows; row += 2) {
        LoadAlign<float, LoadDist::DIST_BRC_B32>(scaleReg0, rowScale + row);
        LoadAlign<float, LoadDist::DIST_BRC_B32>(scaleReg1, rowScale + row + 1);
        uint32_t remaining0 = cols;
        uint32_t remaining1 = cols;
        uint16_t colLoops = static_cast<uint16_t>((cols + FP32_PER_REG - 1) / FP32_PER_REG);
        for (uint16_t colLoop = 0; colLoop < colLoops; ++colLoop) {
            uint32_t col = static_cast<uint32_t>(colLoop) * FP32_PER_REG;
            mask0 = UpdateMask<float>(remaining0);
            mask1 = UpdateMask<float>(remaining1);
            LoadKdaFloat(stateReg0, state + static_cast<uint32_t>(row) * cols + col);
            LoadKdaFloat(stateReg1, state + static_cast<uint32_t>(row + 1) * cols + col);
            Mul(stateReg0, stateReg0, scaleReg0, mask0);
            Mul(stateReg1, stateReg1, scaleReg1, mask1);
            for (uint16_t token = 0; token < tokens; ++token) {
                uint32_t kOffset = static_cast<uint32_t>(token) * kStride + kRowOffset + row;
                LoadKdaBroadcastAsFloat<KType>(weightReg0, kInput + kOffset, mask0);
                LoadKdaBroadcastAsFloat<KType>(weightReg1, kInput + kOffset + 1, mask1);
                LoadKdaAsFloat<VType>(
                    valueReg, vInput + static_cast<uint32_t>(token) * cols + col, mask0);
                Mul(productReg0, valueReg, weightReg0, mask0);
                Mul(productReg1, valueReg, weightReg1, mask1);
                Add(stateReg0, stateReg0, productReg0, mask0);
                Add(stateReg1, stateReg1, productReg1, mask1);
            }
            StoreKdaFloat(state + static_cast<uint32_t>(row) * cols + col, stateReg0, mask0);
            StoreKdaFloat(state + static_cast<uint32_t>(row + 1) * cols + col, stateReg1, mask1);
        }
    }

    if (row < rows) {
        LoadAlign<float, LoadDist::DIST_BRC_B32>(scaleReg0, rowScale + row);
        uint32_t remaining = cols;
        uint16_t colLoops = static_cast<uint16_t>((cols + FP32_PER_REG - 1) / FP32_PER_REG);
        for (uint16_t colLoop = 0; colLoop < colLoops; ++colLoop) {
            uint32_t col = static_cast<uint32_t>(colLoop) * FP32_PER_REG;
            mask0 = UpdateMask<float>(remaining);
            LoadKdaFloat(stateReg0, state + static_cast<uint32_t>(row) * cols + col);
            Mul(stateReg0, stateReg0, scaleReg0, mask0);
            for (uint16_t token = 0; token < tokens; ++token) {
                uint32_t kOffset = static_cast<uint32_t>(token) * kStride + kRowOffset + row;
                LoadKdaBroadcastAsFloat<KType>(weightReg0, kInput + kOffset, mask0);
                LoadKdaAsFloat<VType>(
                    valueReg, vInput + static_cast<uint32_t>(token) * cols + col, mask0);
                Mul(productReg0, valueReg, weightReg0, mask0);
                Add(stateReg0, stateReg0, productReg0, mask0);
            }
            StoreKdaFloat(state + static_cast<uint32_t>(row) * cols + col, stateReg0, mask0);
        }
    }
}

template <typename WType, typename HType>
static __simd_vf__ inline void ComputeKdaTailVWorkspaceRegbaseDualIssue(
    __ubuf__ float *output, __ubuf__ WType *wInput, __ubuf__ HType *hInput,
    uint16_t rows, uint16_t cols, uint16_t reduction)
{
    constexpr uint16_t FP32_PER_REG = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    RegTensor<float> accumReg0;
    RegTensor<float> accumReg1;
    RegTensor<float> valueReg;
    RegTensor<float> productReg0;
    RegTensor<float> productReg1;
    RegTensor<float> weightReg0;
    RegTensor<float> weightReg1;
    MaskReg mask0;
    MaskReg mask1;

    uint16_t row = 0;
    for (; row + 1 < rows; row += 2) {
        uint32_t remaining0 = cols;
        uint32_t remaining1 = cols;
        uint16_t colLoops = static_cast<uint16_t>((cols + FP32_PER_REG - 1) / FP32_PER_REG);
        for (uint16_t colLoop = 0; colLoop < colLoops; ++colLoop) {
            uint32_t col = static_cast<uint32_t>(colLoop) * FP32_PER_REG;
            mask0 = UpdateMask<float>(remaining0);
            mask1 = UpdateMask<float>(remaining1);
            Duplicate(accumReg0, 0.0f, mask0);
            Duplicate(accumReg1, 0.0f, mask1);
            for (uint16_t reductionIdx = 0; reductionIdx < reduction; ++reductionIdx) {
                LoadKdaBroadcastAsFloat<WType>(
                    weightReg0,
                    wInput + static_cast<uint32_t>(row) * reduction + reductionIdx,
                    mask0);
                LoadKdaBroadcastAsFloat<WType>(
                    weightReg1,
                    wInput + static_cast<uint32_t>(row + 1) * reduction + reductionIdx,
                    mask1);
                LoadKdaAsFloat<HType>(
                    valueReg,
                    hInput + static_cast<uint32_t>(reductionIdx) * cols + col,
                    mask0);
                Mul(productReg0, valueReg, weightReg0, mask0);
                Mul(productReg1, valueReg, weightReg1, mask1);
                Add(accumReg0, accumReg0, productReg0, mask0);
                Add(accumReg1, accumReg1, productReg1, mask1);
            }
            StoreKdaFloat(output + static_cast<uint32_t>(row) * cols + col, accumReg0, mask0);
            StoreKdaFloat(output + static_cast<uint32_t>(row + 1) * cols + col, accumReg1, mask1);
        }
    }

    if (row < rows) {
        uint32_t remaining = cols;
        uint16_t colLoops = static_cast<uint16_t>((cols + FP32_PER_REG - 1) / FP32_PER_REG);
        for (uint16_t colLoop = 0; colLoop < colLoops; ++colLoop) {
            uint32_t col = static_cast<uint32_t>(colLoop) * FP32_PER_REG;
            mask0 = UpdateMask<float>(remaining);
            Duplicate(accumReg0, 0.0f, mask0);
            for (uint16_t reductionIdx = 0; reductionIdx < reduction; ++reductionIdx) {
                LoadKdaBroadcastAsFloat<WType>(
                    weightReg0,
                    wInput + static_cast<uint32_t>(row) * reduction + reductionIdx,
                    mask0);
                LoadKdaAsFloat<HType>(
                    valueReg,
                    hInput + static_cast<uint32_t>(reductionIdx) * cols + col,
                    mask0);
                Mul(productReg0, valueReg, weightReg0, mask0);
                Add(accumReg0, accumReg0, productReg0, mask0);
            }
            StoreKdaFloat(output + static_cast<uint32_t>(row) * cols + col, accumReg0, mask0);
        }
    }
}

static __simd_vf__ inline void ApplyRowScaleDualIssue(
    __ubuf__ float *matrix, __ubuf__ float *rowScale, uint32_t rowScaleOffset,
    uint16_t rows, uint16_t cols)
{
    using namespace AscendC::MicroAPI;
    constexpr uint16_t FP32_PER_REG = AscendC::VECTOR_REG_WIDTH / sizeof(float);

    RegTensor<float> matrixReg0;
    RegTensor<float> matrixReg1;
    RegTensor<float> scaleReg0;
    RegTensor<float> scaleReg1;
    MaskReg mask0;
    MaskReg mask1;

    uint16_t row = 0;
    for (; row + 1 < rows; row += 2) {
        LoadAlign<float, LoadDist::DIST_BRC_B32>(scaleReg0, rowScale + rowScaleOffset + row);
        LoadAlign<float, LoadDist::DIST_BRC_B32>(scaleReg1, rowScale + rowScaleOffset + row + 1);
        uint32_t remaining0 = cols;
        uint32_t remaining1 = cols;
        uint16_t colLoops = static_cast<uint16_t>((cols + FP32_PER_REG - 1) / FP32_PER_REG);
        for (uint16_t colLoop = 0; colLoop < colLoops; ++colLoop) {
            uint32_t col = static_cast<uint32_t>(colLoop) * FP32_PER_REG;
            mask0 = UpdateMask<float>(remaining0);
            mask1 = UpdateMask<float>(remaining1);
            LoadAlign(matrixReg0, matrix + static_cast<uint32_t>(row) * cols + col);
            LoadAlign(matrixReg1, matrix + static_cast<uint32_t>(row + 1) * cols + col);
            Mul(matrixReg0, matrixReg0, scaleReg0, mask0);
            Mul(matrixReg1, matrixReg1, scaleReg1, mask1);
            StoreAlign(matrix + static_cast<uint32_t>(row) * cols + col, matrixReg0, mask0);
            StoreAlign(matrix + static_cast<uint32_t>(row + 1) * cols + col, matrixReg1, mask1);
        }
    }

    if (row < rows) {
        LoadAlign<float, LoadDist::DIST_BRC_B32>(scaleReg0, rowScale + rowScaleOffset + row);
        uint32_t remaining = cols;
        uint16_t colLoops = static_cast<uint16_t>((cols + FP32_PER_REG - 1) / FP32_PER_REG);
        for (uint16_t colLoop = 0; colLoop < colLoops; ++colLoop) {
            uint32_t col = static_cast<uint32_t>(colLoop) * FP32_PER_REG;
            mask0 = UpdateMask<float>(remaining);
            LoadAlign(matrixReg0, matrix + static_cast<uint32_t>(row) * cols + col);
            Mul(matrixReg0, matrixReg0, scaleReg0, mask0);
            StoreAlign(matrix + static_cast<uint32_t>(row) * cols + col, matrixReg0, mask0);
        }
    }
}

} // namespace Catlass::Epilogue::Block::detail

#endif
