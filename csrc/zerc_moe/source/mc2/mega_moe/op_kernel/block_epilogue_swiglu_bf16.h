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
 * \file block_epilogue_swiglu_bf16.h
 * \brief GMM1 epilogue for the BF16 no-quant path: SwiGLU only.
 *
 * BlockEpilogueSwigluMxQuant 的 BF16 对应版本 (dispatch_quant_mode=0)。
 * GMM1 的 AIC 侧仍按 MX 路径的布局把两个 weight half 的 BF16 结果写到 UB
 * (first @0, second @MAX_SINGLE_MN*2B)，本类只做 SwiGLU 并把 BF16 结果写回 GM
 * (ptrA2)，不再做 MX 量化，也不产生 scale。
 *
 * - SwiGLU 结果原地写回 first half 的 UB 行: 第 r 行第 j 段在读完后才被覆盖,
 *   且每段只写本行有效的 nSize 个元素, 不会越界覆盖尚未读取的后续行。
 * - 向量化沿用 BlockEpilogueSwigluMxQuant::VfSwigluAndMaxExp256 已验证的
 *   lane 约定: Cast 的 mask 按源 dtype 解释, BF16->FP32 用 B16 mask 分
 *   ONE/ZERO 两次取奇偶 lane, 结果窄化回 BF16 后用整数加合并 lane。
 * - 发布语义 (MTE3_S + AtomicAdd flagSwigluToGmm2) 与 MX 版本一致。
 */

#ifndef BLOCK_EPILOGUE_SWIGLU_BF16_H
#define BLOCK_EPILOGUE_SWIGLU_BF16_H

#if defined(__DAV_C310__)
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "mega_moe_base.h"
#include "block_epilogue_swiglu_mx_quant.h"

namespace {
constexpr uint32_t SWIGLU_BF16_VF_ELEMS = AscendC::VECTOR_REG_WIDTH / sizeof(bfloat16_t);
constexpr uint32_t SWIGLU_BF16_UB_ROW_ALIGN = AscendC::ONE_BLK_SIZE / sizeof(bfloat16_t);
}  // namespace

template <typename L0TileShape_, typename DataTypeIn_, bool IsTensorList_, bool PublishReady_ = true>
class BlockEpilogueSwigluBf16
{
public:
    __aicore__ inline BlockEpilogueSwigluBf16() {}

    // Same layout as BlockEpilogueSwigluMxQuant::Arguments so MegaMoe can
    // initialize either epilogue with one brace list; scale fields are unused.
    struct Arguments {
        GM_ADDR yGmAddr{nullptr};
        GM_ADDR yScaleGmAddr{nullptr};
        GM_ADDR groupFlagListGmAddr{nullptr};
        GM_ADDR x2ScaleGmAddr{nullptr};
        GM_ADDR x1ScaleGmAddr{nullptr};
        GM_ADDR biasGmAddr{nullptr};
        uint32_t baseM;
        uint32_t baseN;
        float activationClamp = -1.0f;
        Arguments() = default;
    };

    using Params = Arguments;
    using DataTypeIn = DataTypeIn_;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;

    static_assert(AscendC::IsSameType<DataTypeIn, bfloat16_t>::value,
                  "The BF16 SwiGLU epilogue consumes the BF16 GMM1 output");

    __aicore__ inline void Init(Params const &params)
    {
        if constexpr (g_coreType == AscendC::AIC) {
            return;
        }
        params_ = params;
    }

    __aicore__ inline auto GetFirstL0c2UbTensor()
    {
        return l0cOutUbFirst_;
    }

    __aicore__ inline auto GetSecondL0c2UbTensor()
    {
        return l0cOutUbSecond_;
    }

    __aicore__ inline void UpdateGlobalAddr(const BlockCoord &baseOffset)
    {
        if constexpr (g_coreType == AscendC::AIV) {
            outputGlobal_.SetGlobalBuffer((__gm__ bfloat16_t *)params_.yGmAddr + Get<Y_IDX>(baseOffset));
            if constexpr (PublishReady_) {
                groupFlagListGmAddr_ =
                    (__gm__ int32_t *)params_.groupFlagListGmAddr + Get<GROUP_FLAG_IDX>(baseOffset) * INT_CACHELINE;
            }
        }
    }

    __aicore__ inline void UpdateNextProblem(const ProblemShape &problemShape)
    {
        n_ = Get<N_VALUE>(problemShape);  // n/2
    }

    __aicore__ inline void operator()(const BlockShape &blockShape, const BlockCoord &blockCoord,
                                      uint16_t pingpongIdx = 0)
    {
        (void)pingpongIdx;
        uint32_t singleM = static_cast<uint32_t>(Get<M_VALUE>(blockShape));
        uint32_t singleN = static_cast<uint32_t>(Get<N_VALUE>(blockShape));
        if (singleM == 0 || singleN == 0) {
            return;
        }
        // GMM1 CopyOut lays every UB row out on a 32B boundary.
        uint32_t rowStride = Ops::Base::CeilAlign(singleN, SWIGLU_BF16_UB_ROW_ALIGN);
        __ubuf__ bfloat16_t *firstSrc = (__ubuf__ bfloat16_t *)l0cOutUbFirst_.GetPhyAddr();
        __ubuf__ bfloat16_t *secondSrc = (__ubuf__ bfloat16_t *)l0cOutUbSecond_.GetPhyAddr();
        if (params_.activationClamp >= 0.0f) {
            VfSwigluInPlace<true>(firstSrc, secondSrc, static_cast<uint16_t>(singleM), static_cast<uint16_t>(singleN),
                                  static_cast<uint16_t>(rowStride), params_.activationClamp);
        } else {
            VfSwigluInPlace<false>(firstSrc, secondSrc, static_cast<uint16_t>(singleM), static_cast<uint16_t>(singleN),
                                   static_cast<uint16_t>(rowStride), 0.0f);
        }
        AscendC::SyncFunc<AscendC::HardEvent::V_MTE3>();
        CopyOutputFromUb2Gm(singleM, singleN, rowStride, static_cast<uint64_t>(Get<Y_IDX>(blockCoord)));
        // The next tile reuses the same UB rows; join the GM store first.
        AscendC::SyncFunc<AscendC::HardEvent::MTE3_V>();
        if constexpr (PublishReady_) {
            AscendC::SyncFunc<AscendC::HardEvent::MTE3_S>();
            AscendC::AtomicAdd(groupFlagListGmAddr_, 1);
        }
    }

private:
    template <bool HasClamp>
    __simd_vf__ inline void VfSwigluInPlace(__ubuf__ bfloat16_t *firstSrc, __ubuf__ bfloat16_t *secondSrc,
                                            uint16_t mSize, uint16_t nSize, uint16_t rowStride, float activationClamp)
    {
        AscendC::Reg::RegTensor<bfloat16_t> first, second, resultOddLanes, resultEvenLanes;
        AscendC::Reg::RegTensor<float> firstFp32, secondFp32, negative, exponent, denominator, swish, resultFp32;
        AscendC::Reg::MaskReg maskFp32 = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::MaskReg maskBf16 = AscendC::Reg::CreateMask<bfloat16_t, AscendC::Reg::MaskPattern::ALL>();
        // Full vector rows have no padding or masked lanes. Traverse the UB
        // half as one stream; keep row-local stores for partial-vector tails.
        const bool dense = rowStride == nSize && nSize % SWIGLU_BF16_VF_ELEMS == 0U;
        const uint32_t rowCount = dense ? 1U : mSize;
        const uint32_t columnCount = dense ? static_cast<uint32_t>(mSize) * nSize : nSize;
        for (uint32_t row = 0; row < rowCount; ++row) {
            __ubuf__ bfloat16_t *firstRow = firstSrc + static_cast<uint32_t>(row) * rowStride;
            __ubuf__ bfloat16_t *secondRow = secondSrc + static_cast<uint32_t>(row) * rowStride;
            __ubuf__ bfloat16_t *outRow = firstRow;
            uint32_t remaining = columnCount;
            for (uint32_t col = 0; col < columnCount; col += SWIGLU_BF16_VF_ELEMS) {
                // Reads may run past the row end inside the fixed UB half;
                // only the valid lanes of this row are ever stored.
                AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(first, firstRow,
                                                                                                SWIGLU_BF16_VF_ELEMS);
                AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(second, secondRow,
                                                                                                SWIGLU_BF16_VF_ELEMS);
                AscendC::Reg::MaskReg storeMask = AscendC::Reg::UpdateMask<bfloat16_t>(remaining);

                // ONE layout selects the odd BF16 lanes (1, 3, ..., 127).
                AscendC::Reg::Cast<float, bfloat16_t, CAST_ONE>(firstFp32, first, maskBf16);
                AscendC::Reg::Cast<float, bfloat16_t, CAST_ONE>(secondFp32, second, maskBf16);
                if constexpr (HasClamp) {
                    AscendC::Reg::Mins(firstFp32, firstFp32, activationClamp, maskFp32);
                    AscendC::Reg::Maxs(secondFp32, secondFp32, -activationClamp, maskFp32);
                    AscendC::Reg::Mins(secondFp32, secondFp32, activationClamp, maskFp32);
                }
                AscendC::Reg::Muls(negative, firstFp32, -1.0f, maskFp32);
                AscendC::Reg::Exp(exponent, negative, maskFp32);
                AscendC::Reg::Adds(denominator, exponent, 1.0f, maskFp32);
                AscendC::Reg::Div(swish, firstFp32, denominator, maskFp32);
                AscendC::Reg::Mul(resultFp32, swish, secondFp32, maskFp32);
                AscendC::Reg::Cast<bfloat16_t, float, CAST_SWIGLU_BF16_ODD>(resultOddLanes, resultFp32, maskFp32);

                // ZERO layout selects the even BF16 lanes (0, 2, ..., 126).
                AscendC::Reg::Cast<float, bfloat16_t, CAST_ZERO>(firstFp32, first, maskBf16);
                AscendC::Reg::Cast<float, bfloat16_t, CAST_ZERO>(secondFp32, second, maskBf16);
                if constexpr (HasClamp) {
                    AscendC::Reg::Mins(firstFp32, firstFp32, activationClamp, maskFp32);
                    AscendC::Reg::Maxs(secondFp32, secondFp32, -activationClamp, maskFp32);
                    AscendC::Reg::Mins(secondFp32, secondFp32, activationClamp, maskFp32);
                }
                AscendC::Reg::Muls(negative, firstFp32, -1.0f, maskFp32);
                AscendC::Reg::Exp(exponent, negative, maskFp32);
                AscendC::Reg::Adds(denominator, exponent, 1.0f, maskFp32);
                AscendC::Reg::Div(swish, firstFp32, denominator, maskFp32);
                AscendC::Reg::Mul(resultFp32, swish, secondFp32, maskFp32);
                AscendC::Reg::Cast<bfloat16_t, float, CAST_SWIGLU_BF16_EVEN>(resultEvenLanes, resultFp32, maskFp32);

                // Integer lane packing, NOT a BF16 sum: each lane has one
                // nonzero operand, so every rounded BF16 bit is preserved.
                AscendC::Reg::Add((AscendC::Reg::RegTensor<uint16_t> &)resultOddLanes,
                                  (AscendC::Reg::RegTensor<uint16_t> &)resultOddLanes,
                                  (AscendC::Reg::RegTensor<uint16_t> &)resultEvenLanes, maskBf16);
                AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(
                    outRow, resultOddLanes, SWIGLU_BF16_VF_ELEMS, storeMask);
            }
        }
    }

    __aicore__ inline void CopyOutputFromUb2Gm(uint32_t singleM, uint32_t singleN, uint32_t rowStride, uint64_t offset)
    {
        // blockLen/dstStride are bytes; the UB row advances to the next 32B
        // block after blockLen and srcStride counts any whole extra blocks.
        uint32_t blockBytes = singleN * sizeof(bfloat16_t);
        uint32_t rowBytes = rowStride * sizeof(bfloat16_t);
        AscendC::DataCopyExtParams ub2GmParams{1, 0, 0, 0, 0};
        ub2GmParams.blockCount = singleM;
        ub2GmParams.blockLen = blockBytes;
        ub2GmParams.srcStride =
            (rowBytes - Ops::Base::CeilAlign(blockBytes, static_cast<uint32_t>(AscendC::ONE_BLK_SIZE))) /
            AscendC::ONE_BLK_SIZE;
        ub2GmParams.dstStride = static_cast<uint32_t>((n_ - singleN) * sizeof(bfloat16_t));
        AscendC::DataCopyPad(outputGlobal_[offset], gluOut_, ub2GmParams);
    }

    AscendC::GlobalTensor<bfloat16_t> outputGlobal_;
    __gm__ int32_t *groupFlagListGmAddr_ = nullptr;

    // Same UB input layout as BlockEpilogueSwigluMxQuant: the AIC GMM1
    // CopyOut targets these offsets for the two weight halves.
    AscendC::LocalTensor<DataTypeIn> l0cOutUbFirst_{AscendC::TPosition::VECIN, 0, MAX_SINGLE_MN};
    static constexpr uint32_t kUbSecondOffset =
        (MAX_SINGLE_MN * sizeof(DataTypeIn) * 2U <= 256U * 1024U) ? (MAX_SINGLE_MN * sizeof(DataTypeIn)) : 0U;
    AscendC::LocalTensor<DataTypeIn> l0cOutUbSecond_{AscendC::TPosition::VECIN, kUbSecondOffset, MAX_SINGLE_MN};
    // SwiGLU result is written back over the first half.
    AscendC::LocalTensor<bfloat16_t> gluOut_{AscendC::TPosition::VECOUT, 0, MAX_SINGLE_MN};
    Params params_;
    int64_t n_ = 0;
};

// MegaMoe instantiates BlockEpilogueSwigluMxQuant from its dispatch element
// type. A BF16 output type (dispatch_quant_mode=0) routes to the SwiGLU-only
// epilogue, so GroupMatmulSwigluQuant and UpdateGlobalBuffer stay unchanged.
template <typename L0TileShape_, typename DataTypeIn_, typename DataTypeX2Scale_, typename DataTypeX1Scale_,
          bool IsTensorList_, bool IsInterleaved_, bool PublishReady_>
class BlockEpilogueSwigluMxQuant<L0TileShape_, bfloat16_t, DataTypeIn_, DataTypeX2Scale_, DataTypeX1Scale_,
                                 IsTensorList_, IsInterleaved_, PublishReady_>
    : public BlockEpilogueSwigluBf16<L0TileShape_, DataTypeIn_, IsTensorList_, PublishReady_>
{
    static_assert(!IsInterleaved_, "The BF16 no-quant epilogue has no interleaved producer");
};

#endif  // __DAV_C310__
#endif  // BLOCK_EPILOGUE_SWIGLU_BF16_H
