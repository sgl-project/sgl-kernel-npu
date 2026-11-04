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
 * \file block_epilogue_swiglu_mx_quant.h
 * \brief
 */

#ifndef BLOCK_EPILOGUE_SWIGLU_MX_QUANT_H
#define BLOCK_EPILOGUE_SWIGLU_MX_QUANT_H

#if defined(__DAV_C310__)
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "mega_moe_base.h"

namespace SwigluQuantMsg {
enum class QuantMode : uint32_t {
    DEFAULT = 0x0U,
    PERTENSOR_MODE = 0x1U,
    PERCHANNEL_MODE = 0x1U << 1,
    PERTOKEN_MODE = 0x1U << 2,
    MX_PERGROUP_MODE = 0x1U << 3,
    PERBLOCK_MODE = 0x1U << 4,
};

enum class QuantDtype : uint8_t {
    DEFAULT = 0x0U,
    FP8_E4M3FN = 0x1U,
    FP8_E5M2 = 0x1U << 1,
};
}  // namespace SwigluQuantMsg

namespace {
constexpr int64_t MEGAMOE_OUT_ELE_NUM_ONE_BLK = 64;
constexpr uint32_t Y_IDX = 0;
constexpr uint32_t Y_SCALE_IDX = 1;
constexpr uint32_t GROUP_FLAG_IDX = 2;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t MAX_SINGLE_MN = 256 * 256;
constexpr uint16_t MAX_EXP_FOR_BF16 = 0x7f80;
constexpr uint16_t MAX_EXP_FOR_FP8 = 0x00ff;
constexpr uint16_t BF16_EXP_BIAS = 0x7f00;
constexpr int16_t SHR_NUM_FOR_BF16 = 7;
constexpr uint16_t NAN_CUSTOMIZATION = 0x7f81;
constexpr uint16_t SPECIAL_EXP_THRESHOLD = 0x0040;
constexpr uint16_t FP8_E4M3_MAX_EXP = 0x0400;  // elem_emax右移7位(BF16E8M7)
constexpr uint16_t FP8_E5M2_MAX_EXP = 0x0780;
constexpr uint16_t FP4_E2M1_MAX_EXP = 0x0100;
constexpr uint16_t FP4_E1M2_MAX_EXP = 0x0000;
constexpr uint32_t FLAG_VALUE_ONE = 1;
constexpr int64_t QUANT_ONCE_NUM = 256;
constexpr int64_t QUANT_ONCE_NUM_FP4 = 128;
constexpr int64_t SCALE_ONCE_NUM = 8;
constexpr int64_t CONST_64 = 64;
constexpr uint32_t VF_LEN_FP32 = AscendC::VECTOR_REG_WIDTH / sizeof(float);
}  // namespace

constexpr AscendC::Reg::CastTrait ctInt322Fp32 = {AscendC::Reg::RegLayout::UNKNOWN, AscendC::Reg::SatMode::UNKNOWN,
                                                  AscendC::Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

constexpr AscendC::Reg::CastTrait ctFp322Half = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT,
                                                 AscendC::Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

constexpr AscendC::Reg::CastTrait ctHalf2Fp32Zero = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
                                                     AscendC::Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};

constexpr AscendC::Reg::CastTrait ctHalf2Fp32One = {AscendC::Reg::RegLayout::ONE, AscendC::Reg::SatMode::UNKNOWN,
                                                    AscendC::Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};

static constexpr AscendC::Reg::DivSpecificMode DIV_MODE = {
    AscendC::Reg::MaskMergeMode::ZEROING,
    true,
};
static constexpr AscendC::Reg::CastTrait CAST_ZERO = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
                                                      AscendC::Reg::MaskMergeMode::ZEROING,
                                                      AscendC::RoundMode::UNKNOWN};
static constexpr AscendC::Reg::CastTrait CAST_ONE = {AscendC::Reg::RegLayout::ONE, AscendC::Reg::SatMode::UNKNOWN,
                                                     AscendC::Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
static constexpr AscendC::Reg::CastTrait CAST_FP32_TO_FP16_BF16 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};
// Narrow each FP32 vector into a separate zero-filled BF16 register, then
// combine their disjoint lanes explicitly. This avoids interpreting the FP32
// source mask again at BF16 width in a MERGING implementation's masked move.
static constexpr AscendC::Reg::CastTrait CAST_SWIGLU_BF16_ODD = {
    AscendC::Reg::RegLayout::ONE, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};
static constexpr AscendC::Reg::CastTrait CAST_SWIGLU_BF16_EVEN = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};
static constexpr AscendC::Reg::CastTrait CAST_32_TO_80 = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::SAT,
                                                          AscendC::Reg::MaskMergeMode::ZEROING,
                                                          AscendC::RoundMode::CAST_RINT};
static constexpr AscendC::Reg::CastTrait CAST_32_TO_81 = {AscendC::Reg::RegLayout::ONE, AscendC::Reg::SatMode::SAT,
                                                          AscendC::Reg::MaskMergeMode::ZEROING,
                                                          AscendC::RoundMode::CAST_RINT};
static constexpr AscendC::Reg::CastTrait CAST_32_TO_82 = {AscendC::Reg::RegLayout::TWO, AscendC::Reg::SatMode::SAT,
                                                          AscendC::Reg::MaskMergeMode::ZEROING,
                                                          AscendC::RoundMode::CAST_RINT};
static constexpr AscendC::Reg::CastTrait CAST_32_TO_83 = {AscendC::Reg::RegLayout::THREE, AscendC::Reg::SatMode::SAT,
                                                          AscendC::Reg::MaskMergeMode::ZEROING,
                                                          AscendC::RoundMode::CAST_RINT};
#define BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS                                                       \
    template <typename L0TileShape_, typename DataTypeOut_, typename DataTypeIn_, typename DataTypeX2Scale_, \
              typename DataTypeX1Scale_, bool IsTensorList_, bool IsInterleaved_, bool PublishReady_>
#define BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS                                                                \
    L0TileShape_, DataTypeOut_, DataTypeIn_, DataTypeX2Scale_, DataTypeX1Scale_, IsTensorList_, IsInterleaved_, \
        PublishReady_

template <typename L0TileShape_, typename DataTypeOut_, typename DataTypeIn_, typename DataTypeX2Scale_,
          typename DataTypeX1Scale_, bool IsTensorList_, bool IsInterleaved_ = false, bool PublishReady_ = true>
class BlockEpilogueSwigluMxQuant
{
public:
    __aicore__ inline BlockEpilogueSwigluMxQuant() {}

    struct Arguments {
        GM_ADDR yGmAddr{nullptr};
        GM_ADDR yScaleGmAddr{nullptr};
        GM_ADDR groupFlagListGmAddr{nullptr};
        GM_ADDR x2ScaleGmAddr{nullptr};
        GM_ADDR x1ScaleGmAddr{nullptr};
        GM_ADDR biasGmAddr{nullptr};
        uint32_t baseM;
        uint32_t baseN;
        float activationClamp = 0.0f;
        Arguments() = default;
    };

    // params
    using Params = Arguments;

    using DataTypeOut = DataTypeOut_;
    using DataTypeIn = DataTypeIn_;
    using DataTypeX1Scale = DataTypeX1Scale_;
    using DataTypeX2Scale = DataTypeX2Scale_;

    // shape
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BaseOffset = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    // y, yScale, x2Scale, x1Scale, bias
    using ProblemShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;

public:
    __aicore__ inline void Init(Params const &params);
    __aicore__ inline auto GetFirstL0c2UbTensor();
    __aicore__ inline auto GetSecondL0c2UbTensor();
    __aicore__ inline void operator()(const BlockShape &blockShape, const BlockCoord &blockCoord,
                                      uint16_t pingpongIdx = 0);
    __aicore__ inline void UpdateGlobalAddr(const BlockCoord &baseOffset);
    __aicore__ inline void UpdateNextProblem(const ProblemShape &problemShape);

private:
    __aicore__ inline void VFDoSwigluForMX(uint16_t mSize, uint16_t pingpongIdx = 0);

    template <SwigluQuantMsg::QuantMode quantMode, bool IsInterleavedSrc = false>
    __aicore__ inline void VFDoSwigluAndQuantForMX(__ubuf__ int8_t *outputDst, __ubuf__ uint16_t *scaleDst,
                                                   __ubuf__ DataTypeIn *firstSrc, __ubuf__ DataTypeIn *secondSrc,
                                                   __ubuf__ bfloat16_t *gluResAddr, __ubuf__ uint16_t *maxExpAddr,
                                                   __ubuf__ uint16_t *halfScaleLocalAddr, uint16_t mSize,
                                                   uint16_t nSize);

    template <bool HasClamp>
    __simd_vf__ inline void VfSwiglu(__ubuf__ DataTypeIn *firstSrc, __ubuf__ DataTypeIn *secondSrc,
                                     __ubuf__ bfloat16_t *gluResAddr, __ubuf__ DataTypeIn *firstTailAddr,
                                     __ubuf__ DataTypeIn *secondTailAddr, __ubuf__ bfloat16_t *swigluTailAddr1,
                                     __ubuf__ bfloat16_t *swigluTailAddr2, uint32_t nSrcUbAligned,
                                     uint32_t nDstUbAligned, uint16_t dim0VfTimes, uint16_t dim1VfTimes,
                                     uint16_t dim1TailTimes, uint16_t dim1Tail2, uint32_t mask1Num, uint32_t mask2Num,
                                     uint32_t mask3Num, float activationClamp);

    __simd_vf__ inline void ComputeScale(__ubuf__ uint16_t *maxExpAddr, __ubuf__ uint16_t *mxScaleLocalAddr,
                                         __ubuf__ uint16_t *halfScaleLocalAddr, uint32_t totalScaleInUB,
                                         uint16_t loopNumScale);

    __simd_vf__ inline void ComputeMaxExp(__ubuf__ bfloat16_t *srcAddr, __ubuf__ uint16_t *maxExpAddr,
                                          uint32_t totalCountInUB, uint16_t loopNum);

    template <bool HasClamp>
    __simd_vf__ inline void VfSwigluAndMaxExp256(__ubuf__ bfloat16_t *firstSrc, __ubuf__ bfloat16_t *secondSrc,
                                                 __ubuf__ bfloat16_t *gluResAddr, __ubuf__ uint16_t *maxExpAddr,
                                                 uint16_t mSize, float activationClamp);

    __simd_vf__ inline void ComputeDataForQuantTargetFp8(__ubuf__ bfloat16_t *srcAddr,
                                                         __ubuf__ uint16_t *halfScaleLocalAddr,
                                                         __ubuf__ int8_t *outLocalAddr, uint32_t totalCountInUB,
                                                         uint16_t loopNum);

    __simd_vf__ inline void ComputeDataForQuantTargetFp4(__ubuf__ bfloat16_t *srcAddr,
                                                         __ubuf__ uint16_t *halfScaleLocalAddr,
                                                         __ubuf__ int8_t *outLocalAddr, uint32_t totalCountInUB,
                                                         uint16_t loopNum);

    __aicore__ inline void CopyOutputFromUb2Gm(uint64_t blockCount, uint64_t offset, AscendC::LocalTensor<int8_t> &src);

    __aicore__ inline void CopyScaleFromUb2GmCompact(uint64_t blockCount, uint64_t offset,
                                                     AscendC::LocalTensor<int8_t> &src);
    // GM ADDR
    AscendC::GlobalTensor<int8_t> quantOutputGlobal_;
    AscendC::GlobalTensor<int8_t> quantScaleGlobal_;
    __gm__ int32_t *groupFlagListGmAddr_;

    // UB ADDR
    AscendC::LocalTensor<DataTypeIn> l0cOutUbFirst_{AscendC::TPosition::VECIN, 0, MAX_SINGLE_MN};
    static constexpr uint32_t kUbSecondOffset =
        (MAX_SINGLE_MN * sizeof(DataTypeIn) * 2U <= 256U * 1024U) ? (MAX_SINGLE_MN * sizeof(DataTypeIn)) : 0U;
    AscendC::LocalTensor<DataTypeIn> l0cOutUbSecond_{AscendC::TPosition::VECIN, kUbSecondOffset, MAX_SINGLE_MN};
    AscendC::LocalTensor<int8_t> quantOutput_;
    AscendC::LocalTensor<int8_t> quantScaleOutput_;
    AscendC::LocalTensor<bfloat16_t> gluRes_;
    AscendC::LocalTensor<uint16_t> maxExp_;
    AscendC::LocalTensor<uint16_t> halfScale_;
    Params params_;

    int64_t n_;
    int64_t scaleN_;
    uint32_t subBlockIdx_ = AscendC::GetSubBlockIdx();
    uint32_t singleM_;  // cur singleShapeM
    uint32_t singleN_;
    bool isBiasEpilogue_ = false;
    int64_t UBBlockSize_ = 0;
    uint32_t vlForHalfNumber_ = 0;
    uint16_t elementAfterReduce_ = 0;
    uint16_t fpEmax_ = 0;

    BlockCoord blockCoord_{0, 0, 0, 0, 0, 0};
};

BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueSwigluMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::Init(Params const &params)
{
    if constexpr (g_coreType == AscendC::AIC) {
        return;
    }
    params_ = params;
    if constexpr (AscendC::IsSameType<DataTypeOut, fp8_e4m3fn_t>::value) {
        fpEmax_ = FP8_E4M3_MAX_EXP;
    } else if constexpr (AscendC::IsSameType<DataTypeOut, fp8_e5m2_t>::value) {
        fpEmax_ = FP8_E5M2_MAX_EXP;  // this
    } else if constexpr (AscendC::IsSameType<DataTypeOut, fp4x2_e2m1_t>::value) {
        fpEmax_ = FP4_E2M1_MAX_EXP;
    } else {
        fpEmax_ = FP4_E1M2_MAX_EXP;
    }

    // 重构UB内存
    // 当前master默认非interleaved，保留完整256x256空间；interleaved预留按half tile + pingpong偏移使用。
    constexpr uint32_t MAX_SINGLE_MN_ALIAS = IsInterleaved_ ? MAX_SINGLE_MN / SWIGLU_N_HALF : MAX_SINGLE_MN;
    constexpr uint32_t gluResOffset = 0;
    gluRes_ = AscendC::LocalTensor<bfloat16_t>(AscendC::TPosition::VECCALC, gluResOffset, MAX_SINGLE_MN_ALIAS);
    constexpr uint32_t quantOutputOffset = gluResOffset + MAX_SINGLE_MN_ALIAS * sizeof(bfloat16_t);
    quantOutput_ = AscendC::LocalTensor<int8_t>(AscendC::TPosition::VECOUT, quantOutputOffset, MAX_SINGLE_MN_ALIAS);
    constexpr uint32_t quantScaleOffset = quantOutputOffset + MAX_SINGLE_MN_ALIAS * sizeof(int8_t);
    quantScaleOutput_ = AscendC::LocalTensor<int8_t>(AscendC::TPosition::VECOUT, quantScaleOffset,
                                                     MAX_SINGLE_MN_ALIAS / AscendC::ONE_BLK_SIZE);
    constexpr uint32_t maxExpOffset = quantScaleOffset + MAX_SINGLE_MN_ALIAS / AscendC::ONE_BLK_SIZE * sizeof(int8_t);
    maxExp_ = AscendC::LocalTensor<uint16_t>(AscendC::TPosition::VECCALC, maxExpOffset,
                                             MAX_SINGLE_MN_ALIAS / AscendC::ONE_BLK_SIZE);
    constexpr uint32_t halfScaleOffset = maxExpOffset + MAX_SINGLE_MN_ALIAS / AscendC::ONE_BLK_SIZE * sizeof(uint16_t);
    halfScale_ = AscendC::LocalTensor<uint16_t>(AscendC::TPosition::VECCALC, halfScaleOffset,
                                                MAX_SINGLE_MN_ALIAS / AscendC::ONE_BLK_SIZE);
}

BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpilogueSwigluMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::UpdateGlobalAddr(const BlockCoord &baseOffset)
{
    if constexpr (g_coreType == AscendC::AIV) {
        quantOutputGlobal_.SetGlobalBuffer((__gm__ int8_t *)params_.yGmAddr + Get<Y_IDX>(baseOffset));
        quantScaleGlobal_.SetGlobalBuffer((__gm__ int8_t *)params_.yScaleGmAddr + Get<Y_SCALE_IDX>(baseOffset));
        if constexpr (PublishReady_) {
            groupFlagListGmAddr_ =
                (__gm__ int32_t *)params_.groupFlagListGmAddr + Get<GROUP_FLAG_IDX>(baseOffset) * INT_CACHELINE;
        }
    }
}

BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueSwigluMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::UpdateNextProblem(
    const ProblemShape &problemShape)
{
    n_ = Get<N_VALUE>(problemShape);  // n/2
    scaleN_ =
        Ops::Base::CeilDiv(static_cast<uint64_t>(n_), static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
}

BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueSwigluMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::CopyOutputFromUb2Gm(
    uint64_t blockCount, uint64_t offset, AscendC::LocalTensor<int8_t> &src)
{
    AscendC::DataCopyExtParams ub2GmParams{1, 0, 0, 0, 0};
    ub2GmParams.blockCount = blockCount;  // 128
    if constexpr (AscendC::IsSameType<DataTypeOut, fp4x2_e2m1_t>::value ||
                  AscendC::IsSameType<DataTypeOut, fp4x2_e1m2_t>::value) {
        ub2GmParams.blockLen = singleN_ >> 1;
        ub2GmParams.dstStride = (n_ - singleN_) >> 1;
        offset = offset >> 1;
    } else {
        uint64_t nDstUbAligned =
            Ops::Base::CeilAlign(static_cast<uint64_t>(singleN_), static_cast<uint64_t>(AscendC::ONE_BLK_SIZE));
        ub2GmParams.blockLen = singleN_;  // 256
        ub2GmParams.srcStride = (nDstUbAligned - singleN_) / AscendC::ONE_BLK_SIZE;
        ub2GmParams.dstStride = n_ - singleN_;
    }
    AscendC::DataCopyPad(quantOutputGlobal_[offset], src, ub2GmParams);
}

BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueSwigluMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::CopyScaleFromUb2GmCompact(
    uint64_t blockCount, uint64_t offset, AscendC::LocalTensor<int8_t> &src)
{
    AscendC::DataCopyExtParams ub2GmParams{0, 0, 0, 0, 0};
    auto blockScaleN = Ops::Base::CeilDiv(static_cast<uint64_t>(singleN_), static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) *
                       MXFP_MULTI_BASE_SIZE;  // 256 / 32 = 8
    // scale layout in UB is already compact: (mSize, blockScaleN). Compact copy avoids (mSize*8)->(mSize,32).
    ub2GmParams.blockCount = blockCount;  // 128
    ub2GmParams.blockLen = blockScaleN;   // 8
    ub2GmParams.srcStride = 0;
    ub2GmParams.dstStride = scaleN_ - blockScaleN;
    AscendC::DataCopyPad<int8_t, AscendC::PaddingMode::Compact>(quantScaleGlobal_[offset], src, ub2GmParams);
}

BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__simd_vf__ inline void BlockEpilogueSwigluMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::ComputeMaxExp(
    __ubuf__ bfloat16_t *srcAddr, __ubuf__ uint16_t *maxExpAddr, uint32_t totalCountInUB, uint16_t loopNum)
{
    int64_t onceNum = QUANT_ONCE_NUM;
    int64_t scaleNum = SCALE_ONCE_NUM;
    // __simd_vf__ 函数头标记取代了原先的 __VEC_SCOPE__ 包裹块(950/dav_3510 层现行写法,
    // 见 moe_init_routing_v3 的 vfComputeMaxExp 同源实现)。
    AscendC::Reg::RegTensor<bfloat16_t> vdExp0, vdExp1;
    AscendC::Reg::RegTensor<uint16_t> vdExpExtract0, vdExpExtract1;
    AscendC::Reg::RegTensor<uint16_t> expMaskBF16, vdMaxExp;
    AscendC::Reg::Duplicate(expMaskBF16, MAX_EXP_FOR_BF16);
    AscendC::Reg::MaskReg scaleMask1, scaleMask2;
    AscendC::Reg::UnalignReg u1;
    for (uint16_t i = 0; i < loopNum; i++) {
        scaleMask1 = AscendC::Reg::UpdateMask<bfloat16_t>(totalCountInUB);
        scaleMask2 = AscendC::Reg::UpdateMask<bfloat16_t>(totalCountInUB);
        AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                               AscendC::Reg::LoadDist::DIST_DINTLV_B16>(
            vdExp0, vdExp1, srcAddr, onceNum);  // copy two chunks from srcAddr to regbase
        AscendC::Reg::And(vdExpExtract0, (AscendC::Reg::RegTensor<uint16_t> &)vdExp0, expMaskBF16, scaleMask1);
        AscendC::Reg::And(vdExpExtract1, (AscendC::Reg::RegTensor<uint16_t> &)vdExp1, expMaskBF16, scaleMask1);
        AscendC::Reg::Max(vdMaxExp, vdExpExtract0, vdExpExtract1, scaleMask1);
        AscendC::Reg::ReduceMaxWithDataBlock(vdMaxExp, vdMaxExp, scaleMask1);
        AscendC::Reg::DataCopyUnAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(maxExpAddr, vdMaxExp, u1,
                                                                                             scaleNum);
    }
    AscendC::Reg::DataCopyUnAlignPost(maxExpAddr, u1, 0);
    return;
}

BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__simd_vf__ inline void BlockEpilogueSwigluMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::ComputeScale(
    __ubuf__ uint16_t *maxExpAddr, __ubuf__ uint16_t *mxScaleLocalAddr, __ubuf__ uint16_t *halfScaleLocalAddr,
    uint32_t totalScaleInUB, uint16_t loopNumScale)  // 128*8  8
{
    int64_t onceNum = QUANT_ONCE_NUM_FP4;
    int64_t onceNumMxScale = CONST_64;
    AscendC::Reg::RegTensor<uint16_t> expMask, vdMaxExp;
    AscendC::Reg::Duplicate(expMask, MAX_EXP_FOR_BF16);  // MAX_EXP_FOR_BF16表示bf16正无穷 大小：128
    AscendC::Reg::MaskReg cmpResult, zeroMask, cmpResultSub, preMaskScale;
    AscendC::Reg::RegTensor<uint16_t> maxExpValue, sharedExp, scaleValue, scaleBias, halfScale;
    AscendC::Reg::Duplicate(maxExpValue, fpEmax_);      // 0x0780 大小：128 对应bf16指数位后四位
    AscendC::Reg::Duplicate(scaleBias, BF16_EXP_BIAS);  // 0x7f00 大小：128
    AscendC::Reg::RegTensor<uint16_t> fp8NanRegTensor, zeroRegTensor, nanRegTensor;
    AscendC::Reg::Duplicate(fp8NanRegTensor, MAX_EXP_FOR_FP8);  // 0x00ff 大小：128
    AscendC::Reg::Duplicate(zeroRegTensor, 0);                  // 0 大小：128
    AscendC::Reg::Duplicate(nanRegTensor, NAN_CUSTOMIZATION);   // 0x7f81 大小：128
    AscendC::Reg::MaskReg invalidDataMask, specialDataMask;
    AscendC::Reg::RegTensor<uint16_t> specialExpRegTensor;
    AscendC::Reg::Duplicate(specialExpRegTensor, SPECIAL_EXP_THRESHOLD);    // 0x0040 大小：128
    for (uint16_t i = 0; i < loopNumScale; i++) {                           // 8
        preMaskScale = AscendC::Reg::UpdateMask<uint16_t>(totalScaleInUB);  // 128*8
        AscendC::Reg::DataCopy<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(
            vdMaxExp, maxExpAddr, onceNum);  // 每次搬运128个数到vdMaxExp
        // 得到不等于INF的结果掩码 cmpResult
        AscendC::Reg::Compare<uint16_t, AscendC::CMPMODE::NE>(cmpResult, vdMaxExp, expMask, preMaskScale);
        // 得到不等于0的结果掩码 zeroMask
        AscendC::Reg::Compare<uint16_t, AscendC::CMPMODE::NE>(zeroMask, vdMaxExp, zeroRegTensor, preMaskScale);
        // 得到小于或等于0x0780的结果掩码 invalidDataMask
        AscendC::Reg::Compare<uint16_t, AscendC::CMPMODE::LE>(invalidDataMask, vdMaxExp, maxExpValue, preMaskScale);
        // 将vdMaxExp中小于或等于0x0780的结果替换成0x0780
        AscendC::Reg::Select<uint16_t>(vdMaxExp, maxExpValue, vdMaxExp, invalidDataMask);
        AscendC::Reg::Sub(sharedExp, vdMaxExp, maxExpValue, preMaskScale);  // sharedExp = vdMaxExp - 0x0780
        // 逻辑右移7位 当前指数位在减去0x0780后，已移至最低位
        AscendC::Reg::ShiftRights(scaleValue, sharedExp, SHR_NUM_FOR_BF16, preMaskScale);
        // 将scaleValue中INF的结果替换成0x00ff
        AscendC::Reg::Select<uint16_t>(scaleValue, scaleValue, fp8NanRegTensor, cmpResult);
        // 将scaleValue中原来是0的结果替换成0
        AscendC::Reg::Select<uint16_t>(scaleValue, scaleValue, zeroRegTensor, zeroMask);
        // 将scaleValue中数取低半部分，搬运到mxScaleLocalAddr uint16--int8
        AscendC::Reg::DataCopy<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                               AscendC::Reg::StoreDist::DIST_PACK_B16>(mxScaleLocalAddr, scaleValue, onceNumMxScale,
                                                                       preMaskScale);
        // 得到sharedExp等于0x7f00的结果掩码 specialDataMask
        AscendC::Reg::Compare<uint16_t, AscendC::CMPMODE::EQ>(specialDataMask, sharedExp, scaleBias, preMaskScale);
        AscendC::Reg::Sub(halfScale, scaleBias, sharedExp, preMaskScale);  // halfScale = 0x7f00 - sharedExp
        // 将halfScale中原等于INF的数值替换成0x7f81
        AscendC::Reg::Select<uint16_t>(halfScale, halfScale, nanRegTensor, cmpResult);
        // 将halfScale中原等于0的数值替换成0
        AscendC::Reg::Select<uint16_t>(halfScale, halfScale, zeroRegTensor, zeroMask);
        // 将halfScale中原等于0x7f00的数值替换成0x0040
        AscendC::Reg::Select<uint16_t>(halfScale, specialExpRegTensor, halfScale, specialDataMask);
        // 将128个数搬运到halfScaleLocalAddr uint16--uint16
        AscendC::Reg::DataCopy<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(halfScaleLocalAddr, halfScale,
                                                                                      onceNum, preMaskScale);
    }
    return;
}

BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__simd_vf__ inline void
BlockEpilogueSwigluMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::ComputeDataForQuantTargetFp8(
    __ubuf__ bfloat16_t *srcAddr, __ubuf__ uint16_t *halfScaleLocalAddr, __ubuf__ int8_t *outLocalAddr,
    uint32_t totalCountInUB, uint16_t loopNum)
{
    using T = bfloat16_t;
    using U = DataTypeOut;
    (void)totalCountInUB;
    int64_t elementAfterReduce = SCALE_ONCE_NUM;
    int64_t onceXNum = QUANT_ONCE_NUM;
    AscendC::Reg::RegTensor<uint16_t> halfScaleForMul;
    AscendC::Reg::RegTensor<T> vdExp0, vdExp1;
    AscendC::Reg::RegTensor<float> vdExp0FP32Zero, vdExp0FP32One;
    AscendC::Reg::RegTensor<float> vdExp1FP32Zero, vdExp1FP32One;
    AscendC::Reg::RegTensor<U> vdExp0FP8Zero, vdExp0FP8One;
    AscendC::Reg::RegTensor<U> vdExp1FP8Zero, vdExp1FP8One;
    AscendC::Reg::MaskReg maskAll = AscendC::Reg::CreateMask<uint16_t, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg maskAllB8 = AscendC::Reg::CreateMask<uint8_t, AscendC::Reg::MaskPattern::ALL>();
    for (uint16_t i = 0; i < loopNum; i++) {
        // DIST_DINTLV_B16:双搬入模式，读取2*VL长度数据，将偶数索引的元素存入dst0，奇数索引的元素存入dst1
        AscendC::Reg::DataCopy<T, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_DINTLV_B16>(
            vdExp0, vdExp1, srcAddr, onceXNum);
        // 将halfScale中的8个数uint16广播到halfScaleForMul中，halfScale[0]*16 halfScale[1]*16...
        AscendC::Reg::DataCopy<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                               AscendC::Reg::LoadDist::DIST_E2B_B16>(halfScaleForMul, halfScaleLocalAddr,
                                                                     elementAfterReduce);
        // vdExp0/vdExp1乘以广播后的halfScale，得到量化前缩放值
        AscendC::Reg::Mul(vdExp0, vdExp0, (AscendC::Reg::RegTensor<T> &)halfScaleForMul, maskAll);
        AscendC::Reg::Mul(vdExp1, vdExp1, (AscendC::Reg::RegTensor<T> &)halfScaleForMul, maskAll);
        AscendC::Reg::Cast<float, T, CAST_ZERO>(vdExp0FP32Zero, vdExp0, maskAll);
        AscendC::Reg::Cast<float, T, CAST_ONE>(vdExp0FP32One, vdExp0, maskAll);
        AscendC::Reg::Cast<float, T, CAST_ZERO>(vdExp1FP32Zero, vdExp1, maskAll);
        AscendC::Reg::Cast<float, T, CAST_ONE>(vdExp1FP32One, vdExp1, maskAll);
        // CAST_32_TO_80/82/81/83把4路fp32 lane cast到fp8 lane，后续按uint8合并成连续fp8输出
        AscendC::Reg::Cast<U, float, CAST_32_TO_80>(vdExp0FP8Zero, vdExp0FP32Zero, maskAll);
        AscendC::Reg::Cast<U, float, CAST_32_TO_82>(vdExp0FP8One, vdExp0FP32One, maskAll);
        AscendC::Reg::Cast<U, float, CAST_32_TO_81>(vdExp1FP8Zero, vdExp1FP32Zero, maskAll);
        AscendC::Reg::Cast<U, float, CAST_32_TO_83>(vdExp1FP8One, vdExp1FP32One, maskAll);
        AscendC::Reg::Add((AscendC::Reg::RegTensor<uint8_t> &)vdExp0FP8Zero,
                          (AscendC::Reg::RegTensor<uint8_t> &)vdExp0FP8Zero,
                          (AscendC::Reg::RegTensor<uint8_t> &)vdExp0FP8One, maskAllB8);
        AscendC::Reg::Add((AscendC::Reg::RegTensor<uint8_t> &)vdExp0FP8Zero,
                          (AscendC::Reg::RegTensor<uint8_t> &)vdExp0FP8Zero,
                          (AscendC::Reg::RegTensor<uint8_t> &)vdExp1FP8Zero, maskAllB8);
        AscendC::Reg::Add((AscendC::Reg::RegTensor<uint8_t> &)vdExp0FP8Zero,
                          (AscendC::Reg::RegTensor<uint8_t> &)vdExp0FP8Zero,
                          (AscendC::Reg::RegTensor<uint8_t> &)vdExp1FP8One, maskAllB8);
        AscendC::Reg::DataCopy<int8_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                               AscendC::Reg::StoreDist::DIST_NORM_B8>(
            // 将src中有效元素的低8bit数据连续存储于dst中
            outLocalAddr, (AscendC::Reg::RegTensor<int8_t> &)vdExp0FP8Zero, onceXNum, maskAllB8);
    }
    return;
}

BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__simd_vf__ inline void
BlockEpilogueSwigluMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::ComputeDataForQuantTargetFp4(
    __ubuf__ bfloat16_t *srcAddr, __ubuf__ uint16_t *halfScaleLocalAddr, __ubuf__ int8_t *outLocalAddr,
    uint32_t totalCountInUB, uint16_t loopNum)
{
    using T = bfloat16_t;
    using U = DataTypeOut;
    int64_t elementAfterReduce = SCALE_ONCE_NUM;
    int64_t onceXNum = QUANT_ONCE_NUM;
    int64_t onceYNum = MEGAMOE_OUT_ELE_NUM_ONE_BLK;
    static constexpr AscendC::Reg::CastTrait castTrait = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
                                                          AscendC::Reg::MaskMergeMode::ZEROING,
                                                          AscendC::RoundMode::CAST_RINT};
    AscendC::Reg::RegTensor<uint16_t> halfScaleForMul;
    AscendC::Reg::RegTensor<T> vdExp0, vdExp1;
    AscendC::Reg::RegTensor<U> vdExp0FP4, vdExp1FP4;
    AscendC::Reg::MaskReg dataMask1;
    for (uint16_t i = 0; i < loopNum; i++) {
        dataMask1 = AscendC::Reg::UpdateMask<T>(totalCountInUB);
        AscendC::Reg::DataCopy<T, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::LoadDist::DIST_DINTLV_B16>(
            vdExp0, vdExp1, srcAddr, onceXNum);
        AscendC::Reg::DataCopy<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                               AscendC::Reg::LoadDist::DIST_E2B_B16>(halfScaleForMul, halfScaleLocalAddr,
                                                                     elementAfterReduce);
        AscendC::Reg::Mul(vdExp0, vdExp0, (AscendC::Reg::RegTensor<T> &)halfScaleForMul, dataMask1);
        AscendC::Reg::Mul(vdExp1, vdExp1, (AscendC::Reg::RegTensor<T> &)halfScaleForMul, dataMask1);
        AscendC::Reg::Interleave(vdExp0, vdExp1, vdExp0, vdExp1);
        AscendC::Reg::Cast<U, T, castTrait>(vdExp0FP4, vdExp0, dataMask1);
        AscendC::Reg::Cast<U, T, castTrait>(vdExp1FP4, vdExp1, dataMask1);
        AscendC::Reg::DataCopy<int8_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                               AscendC::Reg::StoreDist::DIST_PACK4_B32>(
            outLocalAddr, (AscendC::Reg::RegTensor<int8_t> &)vdExp0FP4, onceYNum, dataMask1);
        AscendC::Reg::DataCopy<int8_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                               AscendC::Reg::StoreDist::DIST_PACK4_B32>(
            outLocalAddr, (AscendC::Reg::RegTensor<int8_t> &)vdExp1FP4, onceYNum, dataMask1);
    }
    return;
}

BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
template <bool HasClamp>
__simd_vf__ inline void BlockEpilogueSwigluMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::VfSwiglu(
    __ubuf__ DataTypeIn *firstSrc, __ubuf__ DataTypeIn *secondSrc, __ubuf__ bfloat16_t *gluResAddr,
    __ubuf__ DataTypeIn *firstTailAddr, __ubuf__ DataTypeIn *secondTailAddr, __ubuf__ bfloat16_t *swigluTailAddr1,
    __ubuf__ bfloat16_t *swigluTailAddr2, uint32_t nSrcUbAligned, uint32_t nDstUbAligned, uint16_t dim0VfTimes,
    uint16_t dim1VfTimes, uint16_t dim1TailTimes, uint16_t dim1Tail2, uint32_t mask1Num, uint32_t mask2Num,
    uint32_t mask3Num, float activationClamp)
{
    const float scalarOne = 1.0f;
    const float negScalarOne = -1.0f;
    bfloat16_t numZero = 0;
    AscendC::Reg::RegTensor<DataTypeIn> vregX1;
    AscendC::Reg::RegTensor<DataTypeIn> vregX2;
    AscendC::Reg::RegTensor<float> vregX1F;
    AscendC::Reg::RegTensor<float> vregX2F;
    AscendC::Reg::RegTensor<float> negReg;
    AscendC::Reg::RegTensor<float> expReg;
    AscendC::Reg::RegTensor<float> addsReg;
    AscendC::Reg::RegTensor<float> sigmoidReg;
    AscendC::Reg::RegTensor<float> outFReg;
    AscendC::Reg::RegTensor<bfloat16_t> outTReg;
    AscendC::Reg::RegTensor<bfloat16_t> zeroReg;
    AscendC::Reg::MaskReg mask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg mask1 = AscendC::Reg::UpdateMask<float>(mask1Num);
    AscendC::Reg::MaskReg mask2 = AscendC::Reg::UpdateMask<float>(mask2Num);
    AscendC::Reg::MaskReg mask3 = AscendC::Reg::UpdateMask<bfloat16_t>(mask3Num);
    for (uint16_t dim0vfLoopIdx = 0; dim0vfLoopIdx < dim0VfTimes; dim0vfLoopIdx++) {
        for (uint16_t dim1vfLoopIdx = 0; dim1vfLoopIdx < dim1VfTimes; dim1vfLoopIdx++) {
            AscendC::Reg::AddrReg srcIdxOffset =
                AscendC::Reg::CreateAddrReg<DataTypeIn>(dim0vfLoopIdx, nSrcUbAligned, dim1vfLoopIdx, VF_LEN_FP32);
            // 每次计算m=1, n=64的数据大小
            AscendC::Reg::DataCopy<DataTypeIn, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
                vregX1, firstSrc, srcIdxOffset);  // swishInput:x bf16
            AscendC::Reg::DataCopy<DataTypeIn, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
                vregX2, secondSrc, srcIdxOffset);  // gateInput:y bf16
            // 数据类型转换
            AscendC::Reg::Cast<float, DataTypeIn, CAST_ZERO>(vregX1F, vregX1, mask);
            AscendC::Reg::Cast<float, DataTypeIn, CAST_ZERO>(vregX2F, vregX2, mask);
            if constexpr (HasClamp) {
                AscendC::Reg::Mins(vregX1F, vregX1F, activationClamp, mask);
                AscendC::Reg::Maxs(vregX2F, vregX2F, -activationClamp, mask);
                AscendC::Reg::Mins(vregX2F, vregX2F, activationClamp, mask);
            }

            // swish
            AscendC::Reg::Muls(negReg, vregX1F, negScalarOne, mask);  // -x
            AscendC::Reg::Exp(expReg, negReg, mask);                  // exp(-x)
            AscendC::Reg::Adds(addsReg, expReg, scalarOne, mask);     // exp(-x)+1
            AscendC::Reg::Div(sigmoidReg, vregX1F, addsReg, mask);    // swish(x)=x/(exp(-x)+1)
            AscendC::Reg::Mul(outFReg, sigmoidReg, vregX2F, mask);    // swish(x)*y

            AscendC::Reg::Cast<bfloat16_t, float, CAST_FP32_TO_FP16_BF16>(outTReg, outFReg, mask);
            AscendC::Reg::AddrReg outOffset =
                AscendC::Reg::CreateAddrReg<bfloat16_t>(dim0vfLoopIdx, nDstUbAligned, dim1vfLoopIdx, VF_LEN_FP32);
            AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(
                gluResAddr, outTReg, outOffset, mask);  // gluRes:swish(x)*y 搬运到目标地址
        }
        AscendC::Reg::AddrReg srcIdxOffset1 = AscendC::Reg::CreateAddrReg<DataTypeIn>(dim0vfLoopIdx, nSrcUbAligned);
        AscendC::Reg::AddrReg outOffset1 = AscendC::Reg::CreateAddrReg<bfloat16_t>(dim0vfLoopIdx, nDstUbAligned);
        for (uint16_t aa = 0; aa < dim1TailTimes; aa++) {
            AscendC::Reg::DataCopy<DataTypeIn, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(vregX1, firstTailAddr,
                                                                                        srcIdxOffset1);
            AscendC::Reg::DataCopy<DataTypeIn, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(vregX2, secondTailAddr,
                                                                                        srcIdxOffset1);
            AscendC::Reg::Cast<float, DataTypeIn, CAST_ZERO>(vregX1F, vregX1, mask1);
            AscendC::Reg::Cast<float, DataTypeIn, CAST_ZERO>(vregX2F, vregX2, mask1);
            if constexpr (HasClamp) {
                AscendC::Reg::Mins(vregX1F, vregX1F, activationClamp, mask1);
                AscendC::Reg::Maxs(vregX2F, vregX2F, -activationClamp, mask1);
                AscendC::Reg::Mins(vregX2F, vregX2F, activationClamp, mask1);
            }

            AscendC::Reg::Muls(negReg, vregX1F, negScalarOne, mask1);
            AscendC::Reg::Exp(expReg, negReg, mask1);
            AscendC::Reg::Adds(addsReg, expReg, scalarOne, mask1);
            AscendC::Reg::Div(sigmoidReg, vregX1F, addsReg, mask1);
            AscendC::Reg::Mul(outFReg, sigmoidReg, vregX2F, mask1);

            AscendC::Reg::Cast<bfloat16_t, float, CAST_FP32_TO_FP16_BF16>(outTReg, outFReg, mask1);
            AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(swigluTailAddr1, outTReg,
                                                                                       outOffset1, mask2);
        }
        for (uint16_t cc = 0; cc < dim1Tail2; cc++) {
            AscendC::Reg::Duplicate(zeroReg, numZero);
            // 在计算swiglu时需把pad 0做了
            AscendC::Reg::DataCopy<bfloat16_t>(swigluTailAddr2, zeroReg, outOffset1, mask3);
        }
    }
}

BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
template <bool HasClamp>
__simd_vf__ inline void BlockEpilogueSwigluMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::VfSwigluAndMaxExp256(
    __ubuf__ bfloat16_t *firstSrc, __ubuf__ bfloat16_t *secondSrc, __ubuf__ bfloat16_t *gluResAddr,
    __ubuf__ uint16_t *maxExpAddr, uint16_t mSize, float activationClamp)
{
    // Each full row has eight MX32 groups. Keep the BF16-rounded SwiGLU
    // result in the same even/odd layout that ComputeMaxExp normally loads.
    // This removes its BF16 reload without reducing ComputeScale's batch size.
    AscendC::Reg::RegTensor<bfloat16_t> firstEven, firstOdd, secondEven, secondOdd;
    AscendC::Reg::RegTensor<bfloat16_t> resultEven, resultOdd, resultEvenLanes;
    AscendC::Reg::RegTensor<float> firstFp32, secondFp32, negative, exponent, denominator, swish, resultFp32;
    AscendC::Reg::RegTensor<uint16_t> exponentMask, exponentEven, exponentOdd, maxExponent;
    AscendC::Reg::Duplicate(exponentMask, MAX_EXP_FOR_BF16);
    AscendC::Reg::MaskReg maskFp32 = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg maskBf16 = AscendC::Reg::CreateMask<bfloat16_t, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::UnalignReg maxExponentStore;
    for (uint16_t row = 0; row < mSize; ++row) {
        AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                               AscendC::Reg::LoadDist::DIST_DINTLV_B16>(firstEven, firstOdd, firstSrc, QUANT_ONCE_NUM);
        AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                               AscendC::Reg::LoadDist::DIST_DINTLV_B16>(secondEven, secondOdd, secondSrc,
                                                                        QUANT_ONCE_NUM);

        // Cast selects its mask bits using the SOURCE dtype. Both ZERO and
        // ONE widening therefore need the B16 mask, not the FP32 math mask.
        // DINTLV even + ONE selects original lanes 2, 6, ..., 254.
        AscendC::Reg::Cast<float, bfloat16_t, CAST_ONE>(firstFp32, firstEven, maskBf16);
        AscendC::Reg::Cast<float, bfloat16_t, CAST_ONE>(secondFp32, secondEven, maskBf16);
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
        AscendC::Reg::Cast<bfloat16_t, float, CAST_SWIGLU_BF16_ODD>(resultEven, resultFp32, maskFp32);

        // DINTLV even + ZERO selects original lanes 0, 4, ..., 252.
        AscendC::Reg::Cast<float, bfloat16_t, CAST_ZERO>(firstFp32, firstEven, maskBf16);
        AscendC::Reg::Cast<float, bfloat16_t, CAST_ZERO>(secondFp32, secondEven, maskBf16);
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
        // Integer lane packing, NOT a BF16 sum. In each lane at most one
        // operand is nonzero, so this preserves every rounded BF16 bit.
        AscendC::Reg::Add((AscendC::Reg::RegTensor<uint16_t> &)resultEven,
                          (AscendC::Reg::RegTensor<uint16_t> &)resultEven,
                          (AscendC::Reg::RegTensor<uint16_t> &)resultEvenLanes, maskBf16);

        // DINTLV odd + ONE selects original lanes 3, 7, ..., 255.
        AscendC::Reg::Cast<float, bfloat16_t, CAST_ONE>(firstFp32, firstOdd, maskBf16);
        AscendC::Reg::Cast<float, bfloat16_t, CAST_ONE>(secondFp32, secondOdd, maskBf16);
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
        AscendC::Reg::Cast<bfloat16_t, float, CAST_SWIGLU_BF16_ODD>(resultOdd, resultFp32, maskFp32);

        // DINTLV odd + ZERO selects original lanes 1, 5, ..., 253.
        AscendC::Reg::Cast<float, bfloat16_t, CAST_ZERO>(firstFp32, firstOdd, maskBf16);
        AscendC::Reg::Cast<float, bfloat16_t, CAST_ZERO>(secondFp32, secondOdd, maskBf16);
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
        AscendC::Reg::Add((AscendC::Reg::RegTensor<uint16_t> &)resultOdd,
                          (AscendC::Reg::RegTensor<uint16_t> &)resultOdd,
                          (AscendC::Reg::RegTensor<uint16_t> &)resultEvenLanes, maskBf16);

        // The inverse of DINTLV restores the original contiguous row. B16
        // INTLV stores have a 3510 precedent in moe_distribute_combine_v2_quant.h.
        AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::StoreDist::DIST_INTLV_B16>(
            gluResAddr + static_cast<uint32_t>(row) * QUANT_ONCE_NUM, resultEven, resultOdd, maskBf16);
        AscendC::Reg::And(exponentEven, (AscendC::Reg::RegTensor<uint16_t> &)resultEven, exponentMask, maskBf16);
        AscendC::Reg::And(exponentOdd, (AscendC::Reg::RegTensor<uint16_t> &)resultOdd, exponentMask, maskBf16);
        AscendC::Reg::Max(maxExponent, exponentEven, exponentOdd, maskBf16);
        AscendC::Reg::ReduceMaxWithDataBlock(maxExponent, maxExponent, maskBf16);
        AscendC::Reg::DataCopyUnAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(
            maxExpAddr, maxExponent, maxExponentStore, SCALE_ONCE_NUM);
    }
    AscendC::Reg::DataCopyUnAlignPost(maxExpAddr, maxExponentStore, 0);
}

BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
template <SwigluQuantMsg::QuantMode quantMode, bool IsInterleavedSrc>
__aicore__ inline void BlockEpilogueSwigluMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::VFDoSwigluAndQuantForMX(
    __ubuf__ int8_t *outputDst, __ubuf__ uint16_t *scaleDst, __ubuf__ DataTypeIn *firstSrc,
    __ubuf__ DataTypeIn *secondSrc, __ubuf__ bfloat16_t *gluResAddr, __ubuf__ uint16_t *maxExpAddr,
    __ubuf__ uint16_t *halfScaleLocalAddr, uint16_t mSize, uint16_t nSize)
{
    if constexpr (!IsInterleavedSrc && AscendC::IsSameType<DataTypeIn, bfloat16_t>::value &&
                  (AscendC::IsSameType<DataTypeOut, fp8_e4m3fn_t>::value ||
                   AscendC::IsSameType<DataTypeOut, fp8_e5m2_t>::value)) {
        if (nSize == QUANT_ONCE_NUM && mSize > 0 && mSize <= MAX_SINGLE_MN / QUANT_ONCE_NUM) {
            // The separate second input is consumed 512 bytes at a time. Its
            // prefix can then hold 16 bytes of MX32 exponents per row. Even
            // the final Unalign flush (rounded to 32 bytes) remains behind the
            // read frontier: 32*ceil(16*r/32) <= 512*r for every r >= 1.
            // For M=1 this is 32 <= 512; for M=256 it is 4096 <= 131072.
            // The usual maxExp UB region overlaps still-unread second-input
            // rows, so it cannot be used by this streaming producer.
            __ubuf__ uint16_t *streamedMaxExpAddr = (__ubuf__ uint16_t *)secondSrc;
            uint32_t totalDataInUb = static_cast<uint32_t>(mSize) * QUANT_ONCE_NUM;
            uint32_t totalScaleInUb = static_cast<uint32_t>(mSize) * SCALE_ONCE_NUM;
            uint16_t loopScaleNum = (totalScaleInUb + vlForHalfNumber_ - 1) / vlForHalfNumber_;
            if (params_.activationClamp > 0.0f) {
                VfSwigluAndMaxExp256<true>(firstSrc, secondSrc, gluResAddr, streamedMaxExpAddr, mSize,
                                           params_.activationClamp);
            } else {
                VfSwigluAndMaxExp256<false>(firstSrc, secondSrc, gluResAddr, streamedMaxExpAddr, mSize, 0.0f);
            }
            // All second-input reads finish before this batch scale pass.
            // It consumes the temporary exponents before quant output reuses
            // the same prefix; compact final scale/output layouts are unchanged.
            ComputeScale(streamedMaxExpAddr, scaleDst, halfScaleLocalAddr, totalScaleInUb, loopScaleNum);
            ComputeDataForQuantTargetFp8(gluResAddr, halfScaleLocalAddr, outputDst, totalDataInUb, mSize);
            return;
        }
    }
    uint32_t nSrcUbAligned;
    if constexpr (IsInterleavedSrc) {
        // interleaved源布局为[x1, x2]连续存放在同一行，下一行stride是2*nSize
        nSrcUbAligned = static_cast<uint32_t>(nSize) * 2U;
    } else {
        // 非interleaved源布局为两块独立UB，按原master逻辑只需要对齐单个nSize
        nSrcUbAligned = Ops::Base::CeilAlign(static_cast<uint32_t>(nSize),
                                             static_cast<uint32_t>(AscendC::ONE_BLK_SIZE / sizeof(DataTypeIn)));
    }
    uint32_t nDstUbAligned =
        Ops::Base::CeilAlign(static_cast<uint32_t>(nSize), static_cast<uint32_t>(AscendC::ONE_BLK_SIZE));
    uint16_t dim0VfTimes = mSize;
    uint16_t dim1VfTimes = nSize / VF_LEN_FP32;
    uint32_t dim1Tail = nSize % VF_LEN_FP32;
    uint16_t dim1TailTimes = 0;
    uint16_t dim1Tail2 = 0;
    uint32_t mask1Num = 0;
    uint32_t mask2Num = 0;
    uint32_t mask3Num = 0;
    __ubuf__ DataTypeIn *firstTailAddr = firstSrc;
    __ubuf__ DataTypeIn *secondTailAddr = secondSrc;
    __ubuf__ bfloat16_t *swigluTailAddr1 = gluResAddr;
    __ubuf__ bfloat16_t *swigluTailAddr2 = gluResAddr;
    if (dim1Tail > 0) {
        mask1Num = dim1Tail;
        dim1TailTimes = 1;
        uint32_t padNum = nDstUbAligned - dim1VfTimes * VF_LEN_FP32;
        if (padNum <= VF_LEN_FP32) {
            mask2Num = padNum;
        } else {
            dim1Tail2 = 1;
            mask2Num = VF_LEN_FP32;
            mask3Num = padNum - VF_LEN_FP32;
        }
        uint32_t offsetAlign = dim1VfTimes * VF_LEN_FP32;
        firstTailAddr = firstSrc + offsetAlign;
        secondTailAddr = secondSrc + offsetAlign;
        swigluTailAddr1 = gluResAddr + offsetAlign;
        swigluTailAddr2 = gluResAddr + offsetAlign + dim1TailTimes * VF_LEN_FP32;
    }
    // swiglu
    if (params_.activationClamp > 0.0f) {
        VfSwiglu<true>(firstSrc, secondSrc, gluResAddr, firstTailAddr, secondTailAddr, swigluTailAddr1, swigluTailAddr2,
                       nSrcUbAligned, nDstUbAligned, dim0VfTimes, dim1VfTimes, dim1TailTimes, dim1Tail2, mask1Num,
                       mask2Num, mask3Num, params_.activationClamp);
    } else {
        VfSwiglu<false>(firstSrc, secondSrc, gluResAddr, firstTailAddr, secondTailAddr, swigluTailAddr1,
                        swigluTailAddr2, nSrcUbAligned, nDstUbAligned, dim0VfTimes, dim1VfTimes, dim1TailTimes,
                        dim1Tail2, mask1Num, mask2Num, mask3Num, 0.0f);
    }

    // quant
    uint32_t totalDataInUb = mSize * nDstUbAligned;                   // 128*256
    uint32_t totalScaleInUb = totalDataInUb / AscendC::ONE_BLK_SIZE;  // 128*256 / 32 = 128 * 8
    uint16_t loopDataNum = (totalDataInUb + vlForHalfNumber_ * 2 - 1) / (vlForHalfNumber_ * 2);  // 128
    uint16_t loopScaleNum = (totalScaleInUb + vlForHalfNumber_ - 1) / vlForHalfNumber_;          // 8
    ComputeMaxExp(gluResAddr, maxExpAddr, totalDataInUb, loopDataNum);                           // 获取最大值
    ComputeScale(maxExpAddr, scaleDst, halfScaleLocalAddr, totalScaleInUb, loopScaleNum);  // 计算scale和halfScale
    if constexpr (AscendC::IsSameType<DataTypeOut, fp8_e4m3fn_t>::value ||
                  AscendC::IsSameType<DataTypeOut, fp8_e5m2_t>::value) {
        ComputeDataForQuantTargetFp8(gluResAddr, halfScaleLocalAddr, outputDst, totalDataInUb,
                                     loopDataNum);  // 计算量化后的值
    }
    if constexpr (AscendC::IsSameType<DataTypeOut, fp4x2_e2m1_t>::value ||
                  AscendC::IsSameType<DataTypeOut, fp4x2_e1m2_t>::value) {
        ComputeDataForQuantTargetFp4(gluResAddr, halfScaleLocalAddr, outputDst, totalDataInUb, loopDataNum);
    }
    return;
}

BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueSwigluMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::VFDoSwigluForMX(
    uint16_t mSize, uint16_t pingpongIdx)
{
    constexpr uint32_t pongElemOf_DataTypeIn = MAX_SINGLE_MN;
    constexpr uint32_t pongElemOf_bf16 = MAX_SINGLE_MN * sizeof(DataTypeIn) / sizeof(bfloat16_t);
    constexpr uint32_t pongElemOf_int8 = MAX_SINGLE_MN * sizeof(DataTypeIn);
    constexpr uint32_t pongElemOf_uint16 = MAX_SINGLE_MN * sizeof(DataTypeIn) / sizeof(uint16_t);
    // 当前master调用不传pingpongIdx，默认走非interleaved/idx0；interleaved接入后只在入口切当前half UB。
    const uint32_t pongMul = (IsInterleaved_ && pingpongIdx == 1U) ? 1U : 0U;

    __ubuf__ DataTypeIn *l0cOutUbBase =
        (__ubuf__ DataTypeIn *)l0cOutUbFirst_.GetPhyAddr() + pongMul * pongElemOf_DataTypeIn;
    __ubuf__ bfloat16_t *gluResAddr = (__ubuf__ bfloat16_t *)gluRes_.GetPhyAddr() + pongMul * pongElemOf_bf16;
    __ubuf__ int8_t *quantOutputInUbAddr = (__ubuf__ int8_t *)quantOutput_.GetPhyAddr() + pongMul * pongElemOf_int8;
    __ubuf__ uint16_t *quantScaleOutputInUbAddr =
        (__ubuf__ uint16_t *)quantScaleOutput_.GetPhyAddr() + pongMul * pongElemOf_uint16;
    __ubuf__ uint16_t *maxExpAddr = (__ubuf__ uint16_t *)maxExp_.GetPhyAddr() + pongMul * pongElemOf_uint16;
    __ubuf__ uint16_t *halfScaleAddr = (__ubuf__ uint16_t *)halfScale_.GetPhyAddr() + pongMul * pongElemOf_uint16;

    if constexpr (IsInterleaved_) {
        // interleaved布局中两半输入在同一行连续存放，第二半从singleN_之后开始。
        __ubuf__ DataTypeIn *l0cOutUbSecondAddr = l0cOutUbBase + singleN_;
        VFDoSwigluAndQuantForMX<SwigluQuantMsg::QuantMode::MX_PERGROUP_MODE, true>(
            quantOutputInUbAddr, quantScaleOutputInUbAddr, l0cOutUbBase, l0cOutUbSecondAddr, gluResAddr, maxExpAddr,
            halfScaleAddr, mSize, singleN_);
    } else {
        // 非interleaved保留upstream/master的两块UB输入布局。
        __ubuf__ DataTypeIn *l0cOutUbSecondAddr = (__ubuf__ DataTypeIn *)l0cOutUbSecond_.GetPhyAddr();
        VFDoSwigluAndQuantForMX<SwigluQuantMsg::QuantMode::MX_PERGROUP_MODE, false>(
            quantOutputInUbAddr, quantScaleOutputInUbAddr, l0cOutUbBase, l0cOutUbSecondAddr, gluResAddr, maxExpAddr,
            halfScaleAddr, mSize, singleN_);
    }
}

BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline auto BlockEpilogueSwigluMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::GetFirstL0c2UbTensor()
{
    return l0cOutUbFirst_;
}

BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline auto BlockEpilogueSwigluMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::GetSecondL0c2UbTensor()
{
    return l0cOutUbSecond_;
}

BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueSwigluMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::operator()(
    const BlockShape &blockShape, const BlockCoord &blockCoord, uint16_t pingpongIdx)
{
    singleM_ = Get<M_VALUE>(blockShape);  // 128
    singleN_ = Get<N_VALUE>(blockShape);  // 256
    blockCoord_ = blockCoord;

    if (singleM_ == 0) {
        return;
    }

    vlForHalfNumber_ = AscendC::VECTOR_REG_WIDTH / sizeof(bfloat16_t);  // 256 / 2 = 128
    UBBlockSize_ = BLOCK_SIZE;                                          // 32
    elementAfterReduce_ = AscendC::VECTOR_REG_WIDTH / UBBlockSize_;     // 256 / 32 = 8

    uint64_t yOffset = Get<Y_IDX>(blockCoord);
    uint64_t yScaleOffset = Get<Y_SCALE_IDX>(blockCoord);
    VFDoSwigluForMX(singleM_, pingpongIdx);  // switch(x)*y 计算quant quantScale
    AscendC::SyncFunc<AscendC::HardEvent::V_MTE3>();
    // scale已按compact布局生成，直接copy到GM，省掉原先TransMxScaleLayout重排scale。
    if constexpr (IsInterleaved_) {
        constexpr uint32_t PONG_INT8_ELEMS = MAX_SINGLE_MN * sizeof(DataTypeIn);
        if (pingpongIdx == 1U) {
            AscendC::LocalTensor<int8_t> quantOutputPong = quantOutput_[PONG_INT8_ELEMS];
            AscendC::LocalTensor<int8_t> quantScalePong = quantScaleOutput_[PONG_INT8_ELEMS];
            CopyOutputFromUb2Gm(singleM_, yOffset, quantOutputPong);
            CopyScaleFromUb2GmCompact(singleM_, yScaleOffset, quantScalePong);
        } else {
            CopyOutputFromUb2Gm(singleM_, yOffset, quantOutput_);
            CopyScaleFromUb2GmCompact(singleM_, yScaleOffset, quantScaleOutput_);
        }
    } else {
        CopyOutputFromUb2Gm(singleM_, yOffset, quantOutput_);
        CopyScaleFromUb2GmCompact(singleM_, yScaleOffset, quantScaleOutput_);
    }
    // 下一个 tile 会复用 quantOutput_/quantScaleOutput_，必须等待当前 GM 写入完成。
    AscendC::SyncFunc<AscendC::HardEvent::MTE3_V>();
    if constexpr (PublishReady_) {
        // Generic GMM2 overlaps GMM1 and consumes this per-expert tile count.
        // The staged core-wave specialization enters GMM2 only after a
        // mixed-core barrier, so it compiles out both this fence and the
        // otherwise unobserved shared AtomicAdd.
        AscendC::SyncFunc<AscendC::HardEvent::MTE3_S>();
        AscendC::AtomicAdd(groupFlagListGmAddr_, 1);
    }
    return;
}

#endif  // BLOCK_EPILOGUE_SWIGLU_QUANT_H
#endif
