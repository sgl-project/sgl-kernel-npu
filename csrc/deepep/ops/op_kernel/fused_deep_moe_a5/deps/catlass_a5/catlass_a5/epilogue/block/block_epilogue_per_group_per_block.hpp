/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_BLOCK_EPILOGUE_PER_GROUP_PER_BLOCK_HPP
#define CATLASS_EPILOGUE_BLOCK_EPILOGUE_PER_GROUP_PER_BLOCK_HPP

#include "catlass_a5/catlass.hpp"
#include "catlass_a5/arch/resource.hpp"
#include "catlass_a5/epilogue/dispatch_policy.hpp"
#include "catlass_a5/gemm_coord.hpp"
#include "catlass_a5/matrix_coord.hpp"
#include "tla_a5/tensor.hpp"
#include "tla_a5/layout.hpp"

namespace Catlass::Epilogue::Block {

#define QBMM_BLOCK_EPILOGUE_PERTILE_CLASS_LOCAL_PARAMS                                                                \
    template <class L0TileShape_, class DataTypeOut_, class DataTypeIn_, class DataTypeBias_, class DataTypeX1Scale_, \
              class DataTypeX2Scale_, class LayoutX1Scale_, class LayoutX2Scale_>
#define QBMM_BLOCK_EPILOGUE_PERTILE_FUNC_LOCAL_PARAMS                                                                 \
    BlockEpiloguePertile, L0TileShape_, DataTypeOut_, DataTypeIn_, DataTypeBias_, DataTypeX1Scale_, DataTypeX2Scale_, \
        LayoutX1Scale_, LayoutX2Scale_

template <class L0TileShape_, class DataTypeOut_, class DataTypeIn_, class DataTypeBias_, class DataTypeX1Scale_,
          class DataTypeX2Scale_, class LayoutX1Scale_, class LayoutX2Scale_>
class BlockEpilogue<BlockEpiloguePertile, L0TileShape_, DataTypeOut_, DataTypeIn_, DataTypeBias_, DataTypeX1Scale_,
                    DataTypeX2Scale_, LayoutX1Scale_, LayoutX2Scale_>
{
public:
    using DispatchPolicy = BlockEpiloguePertile;
    using ArchTag = typename DispatchPolicy::ArchTag;
    CATLASS_DEVICE BlockEpilogue()
    {
        if ASCEND_IS_AIV {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(1);
            if constexpr (!AscendC::IsSameType<CType, YType>::value) {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(1);
            }
        }
    }

    CATLASS_DEVICE ~BlockEpilogue()
    {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(1);
        if ASCEND_IS_AIV {
            if constexpr (!AscendC::IsSameType<CType, YType>::value) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(1);
            }
            if (isBias_) {
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(biasPingPongID_);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(biasPingPongID_ ^ 1);
            }
        }
    }

    struct Arguments {
        GM_ADDR outGmAddr{nullptr};
        GM_ADDR x2ScaleGmAddr{nullptr};
        GM_ADDR x1ScaleGmAddr{nullptr};
        GM_ADDR biasGmAddr{nullptr};
        uint32_t baseM;
        uint32_t baseN;
        uint32_t baseK;
        uint32_t groupSizeM = 1U;
        uint32_t groupSizeN = 128U;
        uint32_t groupSizeK = 128U;
        uint32_t isBias = 0;
        Arguments() = default;
    };

    // params
    using Params = Arguments;
    using YType = DataTypeOut_;
    using CType = DataTypeIn_;
    using BiasType = float;
    using X2ScaleType = DataTypeX2Scale_;
    using X1ScaleType = DataTypeX1Scale_;
    using LayoutX1Scale = LayoutX1Scale_;
    using LayoutX2Scale = LayoutX2Scale_;
    using CalcType = float;
    using BlockCoord = tla::Coord<int64_t, int64_t, int64_t, int64_t>;

    constexpr static uint32_t Y_IDX = 0;
    constexpr static uint32_t X2SCALE_IDX = 1;
    constexpr static uint32_t X1SCALE_IDX = 2;
    constexpr static uint32_t BIAS_IDX = 3;
    constexpr static uint32_t QBMM_BUFFER_NUM = 2;
    constexpr static uint16_t QBMM_AIV_SYNC_AIC_FLAG = 6;
    constexpr static uint16_t QBMM_AIC_SYNC_AIV_FLAG = 8;
    constexpr static uint8_t QBMM_AIC_SYNC_AIV_MODE = 4;
    constexpr static uint64_t QBMM_MAX_STEP_SCALEA_K = 16;
    constexpr static uint32_t QBMM_UB_ALIGN_SIZE = 32;
    constexpr static uint32_t UB_TWO_BANK_ELEMS_B32 = 128U;
    constexpr static int64_t PER_BLOCK_SIZE = 128LL;
    constexpr static uint32_t UB_SUB_BANK_NUM = 2U;
    constexpr static uint32_t UB_SUB_BANK_ELEMS_B32 = 64U;
    constexpr static uint32_t UB_ALIGN_SIZE = 32U;
    constexpr static uint32_t QBMM_BMM_BLOCK_NUM = 16;

    struct PertileUBParam {
        bool CopyOutWithSplitN = false;
        uint16_t ndNum;
        uint64_t singleM;
        uint64_t singleN;
        uint64_t validM;
        uint32_t validN[UB_SUB_BANK_NUM];
        uint64_t offsetScaleM;
        uint64_t offsetScaleN[UB_SUB_BANK_NUM];
        uint64_t offsetY[UB_SUB_BANK_NUM];
        uint64_t offsetBias[UB_SUB_BANK_NUM];
    };

public:
    CATLASS_DEVICE void Init(const Params *params);
    template <class TensorUb>
    CATLASS_DEVICE void operator()(const TensorUb &tensorUbPing);
    CATLASS_DEVICE void UpdateGlobalAddr(const BlockCoord &baseOffset);
    CATLASS_DEVICE void UpdateParamsForNextProblem(const GemmCoord &problemShape);
    CATLASS_DEVICE auto GetL0c2UbPingTensor();
    CATLASS_DEVICE auto GetL0c2UbPongTensor();

private:
    CATLASS_DEVICE void ProcessAivSingleKPertile(int64_t x1ScaleOffset,
                                                 __gm__ X2ScaleType *x2ScaleAddr[UB_SUB_BANK_NUM]);

    CATLASS_DEVICE void ProcessAivSingleKPerblock(int64_t x1ScaleOffset,
                                                  __gm__ X2ScaleType *x2ScaleAddr[UB_SUB_BANK_NUM]);
    CATLASS_DEVICE void CopyInBias();
    template <class T>
    CATLASS_DEVICE __ubuf__ T *CopyInX1Scale(uint64_t srcOffset, uint64_t m, uint64_t k);
    template <class T>
    CATLASS_DEVICE T CopyInX1ScalePertile(__gm__ T *src, uint64_t offset);
    template <class T>
    CATLASS_DEVICE void CopyInX2Scale(T x2Scale[UB_SUB_BANK_NUM], __gm__ T *src[UB_SUB_BANK_NUM], uint64_t offset);
    CATLASS_DEVICE int64_t CalcX1OffsetPerGroup();
    CATLASS_DEVICE void CalcX2OffsetPerGroup(int64_t x2ScaleOffset[UB_SUB_BANK_NUM]);
    template <class T>
    CATLASS_DEVICE __ubuf__ T *GetX1ScaleUbAddrPerGroup(int64_t x1ScaleOffset, uint64_t kOffset, uint64_t kElem);
    template <bool isFirstKLoop, uint32_t ndNum>
    __simd_vf__ void AivPerTensor(__ubuf__ CalcType *dst, __ubuf__ CType *l0cOut, __ubuf__ X1ScaleType *x1Scale,
                                  uint16_t mSize, uint32_t nSize0, uint32_t nSize1, uint16_t kSize,
                                  X2ScaleType x2Scale0, X2ScaleType x2Scale1, uint64_t x1ScaleKIdxInCache);
    template <bool isFirstKLoop, uint32_t ndNum>
    __simd_vf__ void AivPerTensor(__ubuf__ CType *dst, __ubuf__ CType *l0cOut, X1ScaleType x1Scale, uint16_t mSize,
                                  uint32_t nSize0, uint32_t nSize1, X2ScaleType x2Scale0, X2ScaleType x2Scale1);
    template <uint32_t ndNum>
    __simd_vf__ void AddBias(__ubuf__ CalcType *mmAdd, __ubuf__ BiasType *bias, uint16_t mSize, uint32_t nSize0,
                             uint32_t nSize1);
    CATLASS_DEVICE void AivPostProcess(const AscendC::LocalTensor<CalcType> &mmAddUb);
    CATLASS_DEVICE void CopyOut(const AscendC::LocalTensor<YType> &ubRes, uint16_t eventId, uint16_t blkCount,
                                uint32_t blkLen, uint32_t srcStride, uint32_t dstStride, uint64_t yOffset);
    CATLASS_DEVICE void CastAndCopyOut(const AscendC::LocalTensor<CalcType> &mmAddUb);
    CATLASS_DEVICE void UpdatePertileUBValidMN();
    CATLASS_DEVICE void UpdatePertileUBParam();
    CATLASS_DEVICE void WaitForCube(uint16_t crossPingPongID)
    {
        AscendC::CrossCoreWaitFlag<QBMM_AIC_SYNC_AIV_MODE, PIPE_V>(QBMM_AIV_SYNC_AIC_FLAG + crossPingPongID);
    }
    CATLASS_DEVICE void NotifyCube(uint16_t crossPingPongID)
    {
        AscendC::CrossCoreSetFlag<QBMM_AIC_SYNC_AIV_MODE, PIPE_V>(QBMM_AIC_SYNC_AIV_FLAG + crossPingPongID);
    }

    AscendC::GlobalTensor<BiasType> biasGlobal_;
    AscendC::GlobalTensor<YType> cGlobal_;
    AscendC::GlobalTensor<X1ScaleType> x1ScaleGlobal_;
    __gm__ X1ScaleType *x1ScaleGlobalPerblock_;
    __gm__ X2ScaleType *x2ScaleGlobal_;
    AscendC::LocalTensor<CType> mmResPing_;
    AscendC::LocalTensor<CType> mmResPong_;
    AscendC::LocalTensor<YType> ubResPing_;
    AscendC::LocalTensor<YType> ubResPong_;
    AscendC::LocalTensor<CalcType> mmAddUb_;
    AscendC::LocalTensor<X1ScaleType> x1ScaleUbPing_;
    AscendC::LocalTensor<X1ScaleType> x1ScaleUbPong_;
    AscendC::LocalTensor<BiasType> biasUbPing_;
    AscendC::LocalTensor<BiasType> biasUbPong_;

private:
    const Params *params_;
    PertileUBParam ubParams_;
    GemmCoord problemShape_{};
    GemmCoord actualSingleShape_{};
    BlockCoord baseOffset_{0, 0, 0, 0};
    BlockCoord blockCoord_{0, 0, 0, 0};

    uint64_t scaleM_ = 0;
    uint64_t scaleN_ = 0;
    uint64_t scaleK_ = 0;
    uint32_t subBlockIdx_;
    uint16_t crossPingPongID_ = 0;
    uint16_t x1ScalePingPongID_ = 0;
    uint16_t biasPingPongID_ = 2;
    bool isPergroup_ = false;
    bool isBias_ = false;
    bool needAivSet = false;
};

QBMM_BLOCK_EPILOGUE_PERTILE_CLASS_LOCAL_PARAMS
CATLASS_DEVICE void BlockEpilogue<QBMM_BLOCK_EPILOGUE_PERTILE_FUNC_LOCAL_PARAMS>::Init(const Params *params)
{
    if ASCEND_IS_AIC {
        return;
    }

    params_ = params;
    subBlockIdx_ = AscendC::GetSubBlockIdx();
    constexpr uint32_t elems = UB_TWO_BANK_ELEMS_B32 * PER_BLOCK_SIZE;
    constexpr uint32_t addUbOffset = elems * UB_SUB_BANK_NUM * sizeof(CType);  // l0c res 128KB
    mmAddUb_ = AscendC::LocalTensor<CalcType>(AscendC::TPosition::VECCALC, addUbOffset, elems);
    constexpr uint32_t afterAddOffset = addUbOffset + elems * sizeof(CalcType);  // ub add res 64KB
    if constexpr (!AscendC::IsSameType<CType, YType>::value) {
        ubResPing_ = AscendC::LocalTensor<YType>(AscendC::TPosition::VECCALC, afterAddOffset, elems);
        ubResPong_ = ubResPing_[elems / QBMM_BUFFER_NUM];
    }

    isPergroup_ = params_->groupSizeM == 1;
    if (isPergroup_) {
        constexpr uint32_t x1ScaleUbOffset =
            (AscendC::IsSameType<CType, YType>::value) ? afterAddOffset : afterAddOffset + elems * sizeof(YType);
        x1ScaleUbPing_ = AscendC::LocalTensor<X1ScaleType>(AscendC::TPosition::VECCALC, x1ScaleUbOffset,
                                                           PER_BLOCK_SIZE * QBMM_MAX_STEP_SCALEA_K);
        x1ScaleUbPong_ = AscendC::LocalTensor<X1ScaleType>(
            AscendC::TPosition::VECCALC,
            x1ScaleUbOffset + PER_BLOCK_SIZE * QBMM_MAX_STEP_SCALEA_K * sizeof(X1ScaleType),
            PER_BLOCK_SIZE * QBMM_MAX_STEP_SCALEA_K);
        if (params->isBias == 1) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(biasPingPongID_);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(biasPingPongID_ ^ 1);
            isBias_ = true;
            constexpr uint32_t afterx1ScaleUbOffset =
                x1ScaleUbOffset + 2 * PER_BLOCK_SIZE * QBMM_MAX_STEP_SCALEA_K * sizeof(X1ScaleType);
            biasUbPing_ =
                AscendC::LocalTensor<BiasType>(AscendC::TPosition::VECCALC, afterx1ScaleUbOffset, PER_BLOCK_SIZE);
            biasUbPong_ = AscendC::LocalTensor<BiasType>(
                AscendC::TPosition::VECCALC, afterx1ScaleUbOffset + PER_BLOCK_SIZE * sizeof(BiasType), PER_BLOCK_SIZE);
        }
    }
}

QBMM_BLOCK_EPILOGUE_PERTILE_CLASS_LOCAL_PARAMS
CATLASS_DEVICE void
BlockEpilogue<QBMM_BLOCK_EPILOGUE_PERTILE_FUNC_LOCAL_PARAMS>::UpdateParamsForNextProblem(const GemmCoord &problemShape)
{
    problemShape_ = problemShape;

    scaleM_ = CeilDiv(problemShape_.m(), params_->groupSizeM);
    scaleN_ = CeilDiv(problemShape_.n(), params_->groupSizeN);
    scaleK_ = CeilDiv(problemShape_.k(), params_->groupSizeK);
}

QBMM_BLOCK_EPILOGUE_PERTILE_CLASS_LOCAL_PARAMS
CATLASS_DEVICE void
BlockEpilogue<QBMM_BLOCK_EPILOGUE_PERTILE_FUNC_LOCAL_PARAMS>::UpdateGlobalAddr(const BlockCoord &baseOffset)
{
    if ASCEND_IS_AIV {
        x1ScaleGlobal_.SetGlobalBuffer((__gm__ X1ScaleType *)params_->x1ScaleGmAddr +
                                       tla::get<X1SCALE_IDX>(baseOffset));
        x1ScaleGlobalPerblock_ = (__gm__ X1ScaleType *)params_->x1ScaleGmAddr + tla::get<X1SCALE_IDX>(baseOffset);
        x2ScaleGlobal_ = (__gm__ X2ScaleType *)params_->x2ScaleGmAddr + tla::get<X2SCALE_IDX>(baseOffset);
        cGlobal_.SetGlobalBuffer((__gm__ YType *)params_->outGmAddr + tla::get<Y_IDX>(baseOffset));
        biasGlobal_.SetGlobalBuffer((__gm__ BiasType *)params_->biasGmAddr + tla::get<BIAS_IDX>(baseOffset));
    }
}

QBMM_BLOCK_EPILOGUE_PERTILE_CLASS_LOCAL_PARAMS
CATLASS_DEVICE int64_t BlockEpilogue<QBMM_BLOCK_EPILOGUE_PERTILE_FUNC_LOCAL_PARAMS>::CalcX1OffsetPerGroup()
{
    int64_t x1ScaleOffset = tla::get<X1SCALE_IDX>(blockCoord_) * scaleK_;
    if (subBlockIdx_ == 1) {
        x1ScaleOffset += (ubParams_.offsetScaleM * scaleK_);
    }
    return x1ScaleOffset;
}

QBMM_BLOCK_EPILOGUE_PERTILE_CLASS_LOCAL_PARAMS
CATLASS_DEVICE void BlockEpilogue<QBMM_BLOCK_EPILOGUE_PERTILE_FUNC_LOCAL_PARAMS>::CalcX2OffsetPerGroup(
    int64_t x2ScaleOffset[UB_SUB_BANK_NUM])
{
    x2ScaleOffset[0] = tla::get<X2SCALE_IDX>(blockCoord_) / params_->groupSizeN + ubParams_.offsetScaleN[0];
    x2ScaleOffset[1] = tla::get<X2SCALE_IDX>(blockCoord_) / params_->groupSizeN + ubParams_.offsetScaleN[1];
}

QBMM_BLOCK_EPILOGUE_PERTILE_CLASS_LOCAL_PARAMS
template <class T>
CATLASS_DEVICE __ubuf__ T *
BlockEpilogue<QBMM_BLOCK_EPILOGUE_PERTILE_FUNC_LOCAL_PARAMS>::CopyInX1Scale(uint64_t srcOffset, uint64_t m, uint64_t k)
{
    AscendC::DataCopyExtParams x1ScaleGm2UbParams;
    AscendC::DataCopyPadExtParams<X1ScaleType> padParams;
    x1ScaleGm2UbParams.blockCount = m;
    x1ScaleGm2UbParams.blockLen = k * sizeof(T);
    x1ScaleGm2UbParams.srcStride = (scaleK_ - k) * sizeof(T);

    auto x1ScaleUb = x1ScalePingPongID_ == 0 ? &x1ScaleUbPing_ : &x1ScaleUbPong_;
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(x1ScalePingPongID_);
    AscendC::DataCopyPad(*x1ScaleUb, x1ScaleGlobal_[srcOffset], x1ScaleGm2UbParams, padParams);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(x1ScalePingPongID_);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(x1ScalePingPongID_);
    return reinterpret_cast<__ubuf__ T *>(x1ScaleUb->GetPhyAddr());
}
QBMM_BLOCK_EPILOGUE_PERTILE_CLASS_LOCAL_PARAMS
CATLASS_DEVICE void BlockEpilogue<QBMM_BLOCK_EPILOGUE_PERTILE_FUNC_LOCAL_PARAMS>::CopyInBias()
{
    AscendC::DataCopyExtParams bias4N0Gm2UbParams;
    AscendC::DataCopyExtParams bias4N1Gm2UbParams;
    AscendC::DataCopyPadExtParams<BiasType> padParams;
    uint32_t validN0Align = ubParams_.validN[0] * sizeof(BiasType);
    bias4N0Gm2UbParams.blockCount = 1;
    bias4N0Gm2UbParams.blockLen = validN0Align;
    bias4N0Gm2UbParams.srcStride = 0;
    bias4N0Gm2UbParams.dstStride = 0;
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(biasPingPongID_);
    auto biasUb = biasPingPongID_ == 2 ? biasUbPing_ : biasUbPong_;
    AscendC::DataCopyPad(biasUb, biasGlobal_[ubParams_.offsetBias[0]], bias4N0Gm2UbParams, padParams);
    if (ubParams_.validN[1] != 0) {
        uint32_t validN1Align = ubParams_.validN[1] * sizeof(BiasType);
        bias4N1Gm2UbParams.blockCount = 1;
        bias4N1Gm2UbParams.blockLen = validN1Align;
        bias4N1Gm2UbParams.srcStride = 0;
        bias4N1Gm2UbParams.dstStride = 0;
        AscendC::DataCopyPad(biasUb[RoundUp(ubParams_.validN[0], UB_ALIGN_SIZE / sizeof(BiasType))],
                             biasGlobal_[ubParams_.offsetBias[1]], bias4N1Gm2UbParams, padParams);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(biasPingPongID_);
}
QBMM_BLOCK_EPILOGUE_PERTILE_CLASS_LOCAL_PARAMS
template <class T>
CATLASS_DEVICE T BlockEpilogue<QBMM_BLOCK_EPILOGUE_PERTILE_FUNC_LOCAL_PARAMS>::CopyInX1ScalePertile(__gm__ T *src,
                                                                                                    uint64_t offset)
{
    return src[offset];
}

QBMM_BLOCK_EPILOGUE_PERTILE_CLASS_LOCAL_PARAMS
template <class T>
CATLASS_DEVICE void BlockEpilogue<QBMM_BLOCK_EPILOGUE_PERTILE_FUNC_LOCAL_PARAMS>::CopyInX2Scale(
    T x2Scale[UB_SUB_BANK_NUM], __gm__ T *src[UB_SUB_BANK_NUM], uint64_t offset)
{
    x2Scale[0] = src[0][offset * scaleN_];
    x2Scale[1] = src[1][offset * scaleN_];
}

QBMM_BLOCK_EPILOGUE_PERTILE_CLASS_LOCAL_PARAMS
CATLASS_DEVICE void BlockEpilogue<QBMM_BLOCK_EPILOGUE_PERTILE_FUNC_LOCAL_PARAMS>::UpdatePertileUBValidMN()
{
    int64_t actualN = actualSingleShape_.n();
    if (AscendC::GetSubBlockIdx() == 0) {
        ubParams_.validM = ubParams_.singleM;
    } else {
        ubParams_.validM = actualSingleShape_.m() - ubParams_.singleM;
    }
    ubParams_.validN[0] = Min(ubParams_.singleN, static_cast<uint64_t>(actualN));
    ubParams_.validN[1] = actualN < ubParams_.singleN ? 0 : Min(ubParams_.singleN, actualN - ubParams_.singleN);
}

QBMM_BLOCK_EPILOGUE_PERTILE_CLASS_LOCAL_PARAMS
CATLASS_DEVICE void BlockEpilogue<QBMM_BLOCK_EPILOGUE_PERTILE_FUNC_LOCAL_PARAMS>::UpdatePertileUBParam()
{
    uint32_t fixpipeN = 0;
    // (m/2 * n) is written to 2 UB, m must be multiples of 2
    // | AIV0 singleN | AIV0 singleN |
    // | AIV1 singleN | AIV1 singleN |
    ubParams_.ndNum = actualSingleShape_.n() > UB_SUB_BANK_ELEMS_B32 ? 2 : 1;  // 2: 2 ND
    fixpipeN =
        RoundUp(actualSingleShape_.n(), static_cast<uint64_t>(AscendC::BLOCK_CUBE) * ubParams_.ndNum) / ubParams_.ndNum;
    ubParams_.singleN = fixpipeN;
    ubParams_.singleM = CeilDiv(actualSingleShape_.m(), AscendC::GetTaskRation());

    UpdatePertileUBValidMN();
    int64_t offsetM = 0;
    int64_t offsetN0 = 0;
    int64_t offsetN1 = 0;

    if (AscendC::GetSubBlockIdx() == 1) {
        offsetM += ubParams_.singleM;
    }
    offsetN1 = ubParams_.validN[1] == 0 ? 0 : ubParams_.singleN;

    ubParams_.offsetScaleM = offsetM / params_->groupSizeM;
    ubParams_.offsetScaleN[0] = offsetN0 / params_->groupSizeN;
    ubParams_.offsetScaleN[1] = offsetN1 / params_->groupSizeN;
    ubParams_.offsetY[0] = tla::get<Y_IDX>(blockCoord_) + offsetM * problemShape_.n() + offsetN0;
    ubParams_.offsetY[1] = tla::get<Y_IDX>(blockCoord_) + offsetM * problemShape_.n() + offsetN1;
    ubParams_.offsetBias[0] = tla::get<BIAS_IDX>(blockCoord_) + offsetN0;
    ubParams_.offsetBias[1] = tla::get<BIAS_IDX>(blockCoord_) + offsetN1;
}

QBMM_BLOCK_EPILOGUE_PERTILE_CLASS_LOCAL_PARAMS
template <class TensorUb>
CATLASS_DEVICE void BlockEpilogue<QBMM_BLOCK_EPILOGUE_PERTILE_FUNC_LOCAL_PARAMS>::operator()(const TensorUb &tensorUb)
{
    actualSingleShape_ = GemmCoord{tla::get<0>(tensorUb.shape()), tla::get<1>(tensorUb.shape()), problemShape_.k()};
    blockCoord_ = BlockCoord{tla::get<0>(tensorUb.coord()) * problemShape_.n() + tla::get<1>(tensorUb.coord()),
                             tla::get<1>(tensorUb.coord()), tla::get<0>(tensorUb.coord()), 0};

    constexpr uint32_t elems = UB_TWO_BANK_ELEMS_B32 * PER_BLOCK_SIZE;
    mmResPing_ = tensorUb.data();
    mmResPong_ = tensorUb.data()[elems];
    UpdatePertileUBParam();
    int64_t x1ScaleOffset = CalcX1OffsetPerGroup();  // one block same x1Scale
    int64_t x2ScaleOffset[UB_SUB_BANK_NUM] = {0};    // maybe 2 different x2Scale
    CalcX2OffsetPerGroup(x2ScaleOffset);
    __gm__ X2ScaleType *x2ScaleAddr[UB_SUB_BANK_NUM] = {x2ScaleGlobal_ + x2ScaleOffset[0],
                                                        x2ScaleGlobal_ + x2ScaleOffset[1]};

    if (isPergroup_) {
        ProcessAivSingleKPertile(x1ScaleOffset, x2ScaleAddr);
    } else {
        ProcessAivSingleKPerblock(x1ScaleOffset, x2ScaleAddr);
    }
}

QBMM_BLOCK_EPILOGUE_PERTILE_CLASS_LOCAL_PARAMS
template <class T>
CATLASS_DEVICE __ubuf__ T *BlockEpilogue<QBMM_BLOCK_EPILOGUE_PERTILE_FUNC_LOCAL_PARAMS>::GetX1ScaleUbAddrPerGroup(
    int64_t x1ScaleOffset, uint64_t kOffset, uint64_t kElem)
{
    uint64_t scaleX1GmOffset;
    scaleX1GmOffset = x1ScaleOffset + kOffset;
    return CopyInX1Scale<X1ScaleType>(scaleX1GmOffset, ubParams_.validM, kElem);
}

QBMM_BLOCK_EPILOGUE_PERTILE_CLASS_LOCAL_PARAMS
CATLASS_DEVICE void BlockEpilogue<QBMM_BLOCK_EPILOGUE_PERTILE_FUNC_LOCAL_PARAMS>::ProcessAivSingleKPertile(
    int64_t x1ScaleOffset, __gm__ X2ScaleType *x2ScaleAddr[UB_SUB_BANK_NUM])
{
    auto mmAddUbAddr = reinterpret_cast<__ubuf__ CType *>(mmAddUb_.GetPhyAddr());
    const uint16_t x1ScaleKElem = Min(QBMM_MAX_STEP_SCALEA_K, scaleK_);
    uint64_t kElem;
    __ubuf__ X1ScaleType *x1ScaleUbAddr;
    X2ScaleType x2Scale[UB_SUB_BANK_NUM];

    for (uint64_t kb = 0, kOffset = 0; kb < problemShape_.k(); kb += params_->baseK, kOffset++) {
        CopyInX2Scale<X2ScaleType>(x2Scale, x2ScaleAddr, kOffset);
        uint64_t x1ScaleKRem = kOffset % x1ScaleKElem;
        if (x1ScaleKRem == 0) {
            kElem = Min(static_cast<uint64_t>(x1ScaleKElem), scaleK_ - kOffset);
            x1ScaleUbAddr = GetX1ScaleUbAddrPerGroup<X1ScaleType>(x1ScaleOffset, kOffset, kElem);
        }

        WaitForCube(crossPingPongID_);
        auto mmUbInputAddr = crossPingPongID_ == 0 ? reinterpret_cast<uint64_t>(mmResPing_.GetPhyAddr())
                                                   : reinterpret_cast<uint64_t>(mmResPong_.GetPhyAddr());
        if (kb == 0) {
            if (ubParams_.ndNum == 1) {
                AivPerTensor<true, 1U>((__ubuf__ CalcType *)mmAddUbAddr, (__ubuf__ CType *)mmUbInputAddr, x1ScaleUbAddr,
                                       ubParams_.validM, ubParams_.validN[0], ubParams_.validN[1], kElem, x2Scale[0],
                                       x2Scale[1], x1ScaleKRem);
            } else {
                AivPerTensor<true, 2U>((__ubuf__ CalcType *)mmAddUbAddr, (__ubuf__ CType *)mmUbInputAddr, x1ScaleUbAddr,
                                       ubParams_.validM, ubParams_.validN[0], ubParams_.validN[1], kElem, x2Scale[0],
                                       x2Scale[1], x1ScaleKRem);
            }
        } else {
            if (ubParams_.ndNum == 1) {
                AivPerTensor<false, 1U>((__ubuf__ CalcType *)mmAddUbAddr, (__ubuf__ CType *)mmUbInputAddr,
                                        x1ScaleUbAddr, ubParams_.validM, ubParams_.validN[0], ubParams_.validN[1],
                                        kElem, x2Scale[0], x2Scale[1], x1ScaleKRem);
            } else {
                AivPerTensor<false, 2U>((__ubuf__ CalcType *)mmAddUbAddr, (__ubuf__ CType *)mmUbInputAddr,
                                        x1ScaleUbAddr, ubParams_.validM, ubParams_.validN[0], ubParams_.validN[1],
                                        kElem, x2Scale[0], x2Scale[1], x1ScaleKRem);
            }
        }
        if (x1ScaleKRem == x1ScaleKElem - 1 || kOffset == scaleK_ - 1) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(x1ScalePingPongID_);
            x1ScalePingPongID_ = (x1ScalePingPongID_ + 1) & 1;
        }
        NotifyCube(crossPingPongID_);
        needAivSet = needAivSet || crossPingPongID_ == 1;
        crossPingPongID_ = (crossPingPongID_ + 1) & 1;
    }
    if (isBias_) {
        CopyInBias();
    }
    AivPostProcess(mmAddUb_);
}

QBMM_BLOCK_EPILOGUE_PERTILE_CLASS_LOCAL_PARAMS
CATLASS_DEVICE void BlockEpilogue<QBMM_BLOCK_EPILOGUE_PERTILE_FUNC_LOCAL_PARAMS>::ProcessAivSingleKPerblock(
    int64_t x1ScaleOffset, __gm__ X2ScaleType *x2ScaleAddr[UB_SUB_BANK_NUM])
{
    auto mmAddUbAddr = reinterpret_cast<__ubuf__ CType *>(mmAddUb_.GetPhyAddr());
    auto x1ScaleAddr = x1ScaleGlobalPerblock_ + x1ScaleOffset;
    X2ScaleType x2Scale[UB_SUB_BANK_NUM];
    for (uint64_t kb = 0, kOffset = 0; kb < problemShape_.k(); kb += params_->baseK, kOffset++) {
        CopyInX2Scale<X2ScaleType>(x2Scale, x2ScaleAddr, kOffset);
        X1ScaleType x1Scale = CopyInX1ScalePertile(x1ScaleAddr, kOffset);
        WaitForCube(crossPingPongID_);
        auto mmUbInputAddr = crossPingPongID_ == 0 ? reinterpret_cast<uint64_t>(mmResPing_.GetPhyAddr())
                                                   : reinterpret_cast<uint64_t>(mmResPong_.GetPhyAddr());
        if (kb == 0) {
            if (ubParams_.ndNum == 1) {
                AivPerTensor<true, 1U>((__ubuf__ CType *)mmAddUbAddr, (__ubuf__ CType *)mmUbInputAddr, x1Scale,
                                       ubParams_.validM, ubParams_.validN[0], ubParams_.validN[1], x2Scale[0],
                                       x2Scale[1]);
            } else {
                AivPerTensor<true, 2U>((__ubuf__ CType *)mmAddUbAddr, (__ubuf__ CType *)mmUbInputAddr, x1Scale,
                                       ubParams_.validM, ubParams_.validN[0], ubParams_.validN[1], x2Scale[0],
                                       x2Scale[1]);
            }
        } else {
            if (ubParams_.ndNum == 1) {
                AivPerTensor<false, 1U>((__ubuf__ CType *)mmAddUbAddr, (__ubuf__ CType *)mmUbInputAddr, x1Scale,
                                        ubParams_.validM, ubParams_.validN[0], ubParams_.validN[1], x2Scale[0],
                                        x2Scale[1]);
            } else {
                AivPerTensor<false, 2U>((__ubuf__ CType *)mmAddUbAddr, (__ubuf__ CType *)mmUbInputAddr, x1Scale,
                                        ubParams_.validM, ubParams_.validN[0], ubParams_.validN[1], x2Scale[0],
                                        x2Scale[1]);
            }
        }
        NotifyCube(crossPingPongID_);
        crossPingPongID_ = (crossPingPongID_ + 1) & 1;
    }
    AivPostProcess(mmAddUb_);
}

QBMM_BLOCK_EPILOGUE_PERTILE_CLASS_LOCAL_PARAMS
template <bool isFirstKLoop, uint32_t ndNum>
__simd_vf__ void BlockEpilogue<QBMM_BLOCK_EPILOGUE_PERTILE_FUNC_LOCAL_PARAMS>::AivPerTensor(
    __ubuf__ CalcType *dst, __ubuf__ CType *l0cOut, __ubuf__ X1ScaleType *x1Scale, uint16_t mSize, uint32_t nSize0,
    uint32_t nSize1, uint16_t kSize, X2ScaleType x2Scale0, X2ScaleType x2Scale1, uint64_t x1ScaleKIdxInCache)
{
    uint16_t alignM = RoundUp(mSize, QBMM_UB_ALIGN_SIZE / sizeof(X1ScaleType));
    uint16_t alignK = RoundUp(kSize, QBMM_UB_ALIGN_SIZE / sizeof(X1ScaleType));
    AscendC::MicroAPI::RegTensor<X1ScaleType> x1ScaleReg, muledScaleReg;
    AscendC::MicroAPI::RegTensor<CType> l0cOutReg;
    AscendC::MicroAPI::RegTensor<CalcType> addReg;
    AscendC::MicroAPI::RegTensor<CalcType> ResReg, mulScaleOutReg;
    constexpr static AscendC::MicroAPI::CastTrait ctInt322Fp32 = {
        AscendC::MicroAPI::RegLayout::UNKNOWN, AscendC::MicroAPI::SatMode::UNKNOWN,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
    {
        for (uint16_t mIdx = 0; mIdx < mSize; mIdx++) {
            AscendC::MicroAPI::LoadAlign<X1ScaleType, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                x1ScaleReg, x1Scale + mIdx * alignK + x1ScaleKIdxInCache);

            // 1 ND
            uint32_t elementNum = nSize0;
            AscendC::MicroAPI::MaskReg maskN = AscendC::MicroAPI::UpdateMask<CType>(elementNum);
            // copy input from ub to register, addr of ub should align to 32B
            uint32_t offset = mIdx * UB_TWO_BANK_ELEMS_B32;
            uint32_t l0cOutOffset = offset;
            AscendC::MicroAPI::LoadAlign(l0cOutReg, l0cOut + offset);
            // l0c_out * scale
            AscendC::MicroAPI::Muls(muledScaleReg, x1ScaleReg, x2Scale0, maskN);
            if constexpr (AscendC::IsSameType<CType, int32_t>::value) {
                AscendC::MicroAPI::RegTensor<CalcType> l0cOutRegAfterCast{};
                AscendC::MicroAPI::Cast<CalcType, CType, ctInt322Fp32>(l0cOutRegAfterCast, l0cOutReg, maskN);
                AscendC::MicroAPI::Mul(mulScaleOutReg, l0cOutRegAfterCast, muledScaleReg, maskN);
            } else {
                AscendC::MicroAPI::Mul(mulScaleOutReg, l0cOutReg, muledScaleReg, maskN);
            }
            uint32_t dstUbOffset = offset;
            if constexpr (isFirstKLoop) {
                AscendC::MicroAPI::StoreAlign<CalcType, AscendC::MicroAPI::StoreDist::DIST_NORM_B32>(
                    dst + dstUbOffset, mulScaleOutReg, maskN);
            } else {
                AscendC::MicroAPI::LoadAlign(addReg, dst + dstUbOffset);
                AscendC::MicroAPI::Add(ResReg, mulScaleOutReg, addReg, maskN);
                // copy out from register to ub
                AscendC::MicroAPI::StoreAlign<CalcType, AscendC::MicroAPI::StoreDist::DIST_NORM_B32>(dst + dstUbOffset,
                                                                                                     ResReg, maskN);
            }
            // 2 ND
            if constexpr (ndNum == 1) {
                continue;
            }
            elementNum = nSize1;
            maskN = AscendC::MicroAPI::UpdateMask<CType>(elementNum);
            // copy input from ub to register, addr of ub should align to 32B
            l0cOutOffset = l0cOutOffset + nSize0;  // + distance of 2 NDs
            AscendC::MicroAPI::LoadAlign(l0cOutReg, l0cOut + l0cOutOffset);
            // l0c_out * scale
            AscendC::MicroAPI::Muls(muledScaleReg, x1ScaleReg, x2Scale1, maskN);
            if constexpr (AscendC::IsSameType<CType, int32_t>::value) {
                AscendC::MicroAPI::RegTensor<CalcType> l0cOutRegAfterCast{};
                AscendC::MicroAPI::Cast<CalcType, CType, ctInt322Fp32>(l0cOutRegAfterCast, l0cOutReg, maskN);
                AscendC::MicroAPI::Mul(mulScaleOutReg, l0cOutRegAfterCast, muledScaleReg, maskN);
            } else {
                AscendC::MicroAPI::Mul(mulScaleOutReg, l0cOutReg, muledScaleReg, maskN);
            }
            dstUbOffset = offset + nSize0;
            if constexpr (isFirstKLoop) {
                AscendC::MicroAPI::StoreAlign<CalcType, AscendC::MicroAPI::StoreDist::DIST_NORM_B32>(
                    dst + dstUbOffset, mulScaleOutReg, maskN);
            } else {
                AscendC::MicroAPI::LoadAlign(addReg, dst + dstUbOffset);
                AscendC::MicroAPI::Add(ResReg, mulScaleOutReg, addReg, maskN);
                // copy out from register to ub
                AscendC::MicroAPI::StoreAlign<CalcType, AscendC::MicroAPI::StoreDist::DIST_NORM_B32>(dst + dstUbOffset,
                                                                                                     ResReg, maskN);
            }
        }
    }
}

QBMM_BLOCK_EPILOGUE_PERTILE_CLASS_LOCAL_PARAMS
template <bool isFirstKLoop, uint32_t ndNum>
__simd_vf__ void BlockEpilogue<QBMM_BLOCK_EPILOGUE_PERTILE_FUNC_LOCAL_PARAMS>::AivPerTensor(
    __ubuf__ CType *dst, __ubuf__ CType *l0cOut, X1ScaleType x1Scale, uint16_t mSize, uint32_t nSize0, uint32_t nSize1,
    X2ScaleType x2Scale0, X2ScaleType x2Scale1)
{
    {
        for (uint16_t mIdx = 0; mIdx < mSize; mIdx++) {
            // 1 ND
            X2ScaleType scaleMul = x1Scale * x2Scale0;
            uint32_t elementNum = nSize0;
            AscendC::MicroAPI::MaskReg maskN = AscendC::MicroAPI::UpdateMask<CType>(elementNum);
            AscendC::MicroAPI::RegTensor<CType> l0cOutReg;
            AscendC::MicroAPI::RegTensor<CType> addReg;
            AscendC::MicroAPI::RegTensor<CType> ResReg, mulScaleOutReg;
            // copy input from ub to register, addr of ub should align to 32B
            uint32_t offset = mIdx * UB_TWO_BANK_ELEMS_B32;
            uint32_t l0cOutOffset = offset;
            AscendC::MicroAPI::LoadAlign(l0cOutReg, l0cOut + offset);
            // l0c_out * scale
            AscendC::MicroAPI::Muls(mulScaleOutReg, l0cOutReg, scaleMul, maskN);
            uint32_t dstUbOffset = offset;
            if constexpr (isFirstKLoop) {
                AscendC::MicroAPI::StoreAlign<CType, AscendC::MicroAPI::StoreDist::DIST_NORM_B32>(
                    dst + dstUbOffset, mulScaleOutReg, maskN);
            } else {
                AscendC::MicroAPI::LoadAlign(addReg, dst + dstUbOffset);
                AscendC::MicroAPI::Add(ResReg, mulScaleOutReg, addReg, maskN);
                // copy out from register to ub
                AscendC::MicroAPI::StoreAlign<CType, AscendC::MicroAPI::StoreDist::DIST_NORM_B32>(dst + dstUbOffset,
                                                                                                  ResReg, maskN);
            }
            // 2 ND
            if constexpr (ndNum == 1) {
                continue;
            }
            scaleMul = x1Scale * x2Scale1;
            elementNum = nSize1;
            maskN = AscendC::MicroAPI::UpdateMask<CType>(elementNum);
            // copy input from ub to register, addr of ub should align to 32B
            l0cOutOffset = offset + nSize0;  // + distance of 2 NDs
            AscendC::MicroAPI::LoadAlign(l0cOutReg, l0cOut + l0cOutOffset);
            // l0c_out * scale
            AscendC::MicroAPI::Muls(mulScaleOutReg, l0cOutReg, scaleMul, maskN);
            dstUbOffset = offset + nSize0;
            if constexpr (isFirstKLoop) {
                AscendC::MicroAPI::StoreAlign<CType, AscendC::MicroAPI::StoreDist::DIST_NORM_B32>(
                    dst + dstUbOffset, mulScaleOutReg, maskN);
            } else {
                AscendC::MicroAPI::LoadAlign(addReg, dst + dstUbOffset);
                AscendC::MicroAPI::Add(ResReg, mulScaleOutReg, addReg, maskN);
                // copy out from register to ub
                AscendC::MicroAPI::StoreAlign<CType, AscendC::MicroAPI::StoreDist::DIST_NORM_B32>(dst + dstUbOffset,
                                                                                                  ResReg, maskN);
            }
        }
    }
}

QBMM_BLOCK_EPILOGUE_PERTILE_CLASS_LOCAL_PARAMS
template <uint32_t ndNum>
__simd_vf__ void BlockEpilogue<QBMM_BLOCK_EPILOGUE_PERTILE_FUNC_LOCAL_PARAMS>::AddBias(__ubuf__ CalcType *mmAdd,
                                                                                       __ubuf__ BiasType *bias,
                                                                                       uint16_t mSize, uint32_t nSize0,
                                                                                       uint32_t nSize1)
{
    {
        uint32_t mmAddOffset{};
        for (uint16_t mIdx = 0; mIdx < mSize; mIdx++) {
            AscendC::MicroAPI::RegTensor<BiasType> biasReg{};
            AscendC::MicroAPI::RegTensor<CalcType> mmAddReg{};
            AscendC::MicroAPI::DataCopy(biasReg, bias);
            mmAddOffset = mIdx * PER_BLOCK_SIZE;
            uint32_t elementNum = nSize0;
            AscendC::MicroAPI::MaskReg maskN = AscendC::MicroAPI::UpdateMask<CalcType>(elementNum);
            AscendC::MicroAPI::DataCopy(mmAddReg, mmAdd + mmAddOffset);
            AscendC::MicroAPI::Add(mmAddReg, mmAddReg, biasReg, maskN);
            AscendC::MicroAPI::DataCopy<CalcType, AscendC::MicroAPI::StoreDist::DIST_NORM_B32>(mmAdd + mmAddOffset,
                                                                                               mmAddReg, maskN);
            if constexpr (ndNum == 1) {
                continue;
            }
            elementNum = nSize1;
            maskN = AscendC::MicroAPI::UpdateMask<CalcType>(elementNum);
            mmAddOffset += nSize0;
            AscendC::MicroAPI::DataCopy(biasReg, bias + nSize0);
            AscendC::MicroAPI::DataCopy(mmAddReg, mmAdd + mmAddOffset);
            AscendC::MicroAPI::Add(mmAddReg, mmAddReg, biasReg, maskN);
            AscendC::MicroAPI::DataCopy<CalcType, AscendC::MicroAPI::StoreDist::DIST_NORM_B32>(mmAdd + mmAddOffset,
                                                                                               mmAddReg, maskN);
        }
    }
}

QBMM_BLOCK_EPILOGUE_PERTILE_CLASS_LOCAL_PARAMS
CATLASS_DEVICE void BlockEpilogue<QBMM_BLOCK_EPILOGUE_PERTILE_FUNC_LOCAL_PARAMS>::AivPostProcess(
    const AscendC::LocalTensor<CalcType> &mmAddUb)
{
    if (ubParams_.validM == 0) {
        return;
    }
    if constexpr (AscendC::IsSameType<CalcType, BiasType>::value) {
        if (isBias_) {
            auto biasUb = biasPingPongID_ == 2 ? biasUbPing_ : biasUbPong_;
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(biasPingPongID_);
            auto mmAddUbAddr = reinterpret_cast<__ubuf__ CalcType *>(mmAddUb.GetPhyAddr());
            auto biasUbAddr = reinterpret_cast<__ubuf__ BiasType *>(biasUb.GetPhyAddr());
            if (ubParams_.ndNum == 1) {
                AddBias<1U>((__ubuf__ CalcType *)mmAddUbAddr, (__ubuf__ BiasType *)biasUbAddr, ubParams_.validM,
                            ubParams_.validN[0], ubParams_.validN[1]);
            } else if (ubParams_.ndNum == 2) {
                AddBias<2U>((__ubuf__ CalcType *)mmAddUbAddr, (__ubuf__ BiasType *)biasUbAddr, ubParams_.validM,
                            ubParams_.validN[0], ubParams_.validN[1]);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(biasPingPongID_);
            biasPingPongID_ = biasPingPongID_ ^ 1;
        }
    }
    if constexpr (AscendC::IsSameType<YType, CType>::value) {
        // mov optimize in splitM, 0~63 + 64 ~127 -> 0~127
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        if (ubParams_.ndNum == 2) {  // 2: 2ND, opt branch with splitM
            uint32_t sumN = ubParams_.validN[0] + ubParams_.validN[1];
            CopyOut(mmAddUb, 0, ubParams_.validM, sumN, UB_TWO_BANK_ELEMS_B32 - sumN, problemShape_.n() - sumN,
                    ubParams_.offsetY[0]);
        } else {
            for (uint64_t ndIdx = 0; ndIdx < ubParams_.ndNum; ndIdx++) {
                if (ubParams_.validN[ndIdx] > 0) {
                    CopyOut(mmAddUb[ndIdx * ubParams_.validN[0]], 0, ubParams_.validM, ubParams_.validN[ndIdx],
                            UB_TWO_BANK_ELEMS_B32 - ubParams_.validN[ndIdx],
                            problemShape_.n() - ubParams_.validN[ndIdx], ubParams_.offsetY[ndIdx]);
                }
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
    } else {
        AscendC::PipeBarrier<PIPE_V>();
        CastAndCopyOut(mmAddUb);
    }
}

QBMM_BLOCK_EPILOGUE_PERTILE_CLASS_LOCAL_PARAMS
CATLASS_DEVICE void BlockEpilogue<QBMM_BLOCK_EPILOGUE_PERTILE_FUNC_LOCAL_PARAMS>::CopyOut(
    const AscendC::LocalTensor<YType> &ubRes, uint16_t eventId, uint16_t blkCount, uint32_t blkLen, uint32_t srcStride,
    uint32_t dstStride, uint64_t yOffset)
{
    AscendC::DataCopyExtParams copyParams{blkCount, static_cast<uint32_t>(blkLen * sizeof(YType)),
                                          static_cast<uint32_t>(srcStride * sizeof(YType) / AscendC::ONE_BLK_SIZE),
                                          static_cast<uint32_t>(dstStride * sizeof(YType)), 0};
    AscendC::DataCopyPad<YType>(cGlobal_[yOffset], ubRes, copyParams);
}

QBMM_BLOCK_EPILOGUE_PERTILE_CLASS_LOCAL_PARAMS
CATLASS_DEVICE void BlockEpilogue<QBMM_BLOCK_EPILOGUE_PERTILE_FUNC_LOCAL_PARAMS>::CastAndCopyOut(
    const AscendC::LocalTensor<CalcType> &mmAddUb)
{
    if (ubParams_.ndNum == 2) {  // 2: 2ND, opt branch with splitM
        uint32_t sumN = ubParams_.validN[0] + ubParams_.validN[1];
        uint32_t mSizePing = CeilDiv(ubParams_.validM, static_cast<uint64_t>(QBMM_BUFFER_NUM));
        uint32_t mSize[QBMM_BUFFER_NUM] = {mSizePing, static_cast<uint32_t>(ubParams_.validM - mSizePing)};
        for (uint32_t mDbIdx = 0; mDbIdx < QBMM_BUFFER_NUM; ++mDbIdx) {
            if (mSize[mDbIdx] > 0 && sumN > 0) {
                auto ubRes = mDbIdx == 0 ? &ubResPing_ : &ubResPong_;
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mDbIdx);
                AscendC::Cast(*ubRes, mmAddUb[mDbIdx * mSizePing * UB_TWO_BANK_ELEMS_B32],
                              AscendC::RoundMode::CAST_RINT, mSize[mDbIdx] * UB_TWO_BANK_ELEMS_B32);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(mDbIdx);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(mDbIdx);
                CopyOut(*ubRes, mDbIdx, mSize[mDbIdx], sumN, UB_TWO_BANK_ELEMS_B32 - sumN, problemShape_.n() - sumN,
                        ubParams_.offsetY[0] + mDbIdx * mSizePing * problemShape_.n());
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mDbIdx);
            }
        }
    } else {
        for (uint64_t ndIdx = 0; ndIdx < ubParams_.ndNum; ndIdx++) {
            auto ubRes = ndIdx == 0 ? &ubResPing_ : &ubResPong_;
            if (ubParams_.validN[ndIdx] > 0) {
                AscendC::UnaryRepeatParams repeatParam;
                repeatParam.srcBlkStride = 1;
                repeatParam.dstBlkStride = 1;
                // write continuously
                repeatParam.dstRepStride = CeilDiv(ubParams_.singleN, AscendC::ONE_BLK_SIZE / sizeof(YType));
                // srcStride is 16(512B / 32B), subBank0 256B one repeat
                repeatParam.srcRepStride = QBMM_BMM_BLOCK_NUM;
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ndIdx);
                AscendC::Cast(*ubRes, mmAddUb[ubParams_.validN[0] * ndIdx], AscendC::RoundMode::CAST_RINT,
                              ubParams_.validN[ndIdx], ubParams_.validM, repeatParam);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ndIdx);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ndIdx);
                CopyOut(*ubRes, ndIdx, ubParams_.validM, ubParams_.validN[ndIdx],
                        ubParams_.singleN - ubParams_.validN[ndIdx], problemShape_.n() - ubParams_.validN[ndIdx],
                        ubParams_.offsetY[ndIdx]);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ndIdx);
            }
        }
    }
}

}  // namespace Catlass::Epilogue::Block

#endif  // CATLASS_EPILOGUE_BLOCK_EPILOGUE_PER_GROUP_PER_BLOCK_HPP
