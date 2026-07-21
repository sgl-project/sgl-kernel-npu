/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_BLOCK_EPILOGUE_DEQUANT_HPP
#define CATLASS_EPILOGUE_BLOCK_EPILOGUE_DEQUANT_HPP

#include "catlass_a5/catlass.hpp"
#include "catlass_a5/arch/resource.hpp"
#include "catlass_a5/epilogue/dispatch_policy.hpp"
#include "catlass_a5/gemm_coord.hpp"
#include "catlass_a5/matrix_coord.hpp"
#include "catlass_a5/epilogue/tile/tile_copy.hpp"
#include "catlass_a5/gemm/helper.hpp"
#include "catlass_a5/status.hpp"
#include "tla_a5/tensor.hpp"
#include "tla_a5/layout.hpp"

namespace Catlass::Epilogue::Block {

// CANN exposes dtype identifiers as global macros in some toolkit versions.
// Keep a namespaced fallback for versions that provide only AscendC constants.
#if defined(DT_FLOAT)
constexpr uint32_t kDTypeFloat = DT_FLOAT;
constexpr uint32_t kDTypeBf16 = DT_BF16;
constexpr uint32_t kDTypeFloat16 = DT_FLOAT16;
#else
constexpr uint32_t kDTypeFloat = AscendC::DT_FLOAT;
constexpr uint32_t kDTypeBf16 = AscendC::DT_BF16;
constexpr uint32_t kDTypeFloat16 = AscendC::DT_FLOAT16;
#endif

enum class QuantMode : uint32_t {
    DEFAULT = 0x0U,
    PERTENSOR_MODE = 0x1U,
    PERCHANNEL_MODE = 0x1U << 1,
    PERTOKEN_MODE = 0x1U << 2,
    MX_PERGROUP_MODE = 0x1U << 3,
    PERBLOCK_MODE = 0x1U << 4,
};

#define QMM_BLOCK_EPILOGUE_DEQUANT_CLASS_LOCAL_PARAMS                                                        \
    template <typename L0TileShape_, typename DataTypeOut_, typename DataTypeIn_, typename DataTypeX2Scale_, \
              typename DataTypeX1Scale_, typename DataTypeBias_>
#define QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS \
    BlockEpilogueDequant, L0TileShape_, DataTypeOut_, DataTypeIn_, DataTypeX2Scale_, DataTypeX1Scale_, DataTypeBias_

QMM_BLOCK_EPILOGUE_DEQUANT_CLASS_LOCAL_PARAMS
class BlockEpilogue<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>
{
public:
    CATLASS_DEVICE BlockEpilogue() {}

    CATLASS_DEVICE ~BlockEpilogue()
    {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
        for (uint32_t i = 0; i < DOUBLE_BUFFER_COUNT; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(i);
        }
    }

    struct DequantTiling {
        uint32_t baseM;
        uint32_t baseN;
        QuantMode x1QuantMode = QuantMode::DEFAULT;
        QuantMode x2QuantMode = QuantMode::DEFAULT;
        uint32_t biasDtype = kDTypeFloat;
        bool isBiasEpilogue = false;
    };

    struct Arguments {
        GM_ADDR yGmAddr{nullptr};
        GM_ADDR x2ScaleGmAddr{nullptr};
        GM_ADDR x1ScaleGmAddr{nullptr};
        GM_ADDR biasGmAddr{nullptr};
        DequantTiling dequantTiling;
    };

    struct Params {
        GM_ADDR yGmAddr{nullptr};
        GM_ADDR x2ScaleGmAddr{nullptr};
        GM_ADDR x1ScaleGmAddr{nullptr};
        GM_ADDR biasGmAddr{nullptr};
        DequantTiling dequantTiling;
    };

    using DispatchPolicy = BlockEpilogueDequant;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using DataTypeOut = DataTypeOut_;
    using DataTypeIn = DataTypeIn_;
    using DataTypeX1Scale = DataTypeX1Scale_;
    using DataTypeX2Scale = DataTypeX2Scale_;
    using DataTypeBias = DataTypeBias_;

    static constexpr int64_t l0M = tla::get<0>(L0TileShape_{});
    static constexpr int64_t l0N = tla::get<1>(L0TileShape_{});

    static constexpr uint32_t FP32_OUTPUT_TIMES = 4;
    static constexpr uint32_t Y_IDX = 0;
    static constexpr uint32_t X2SCALE_IDX = 1;
    static constexpr uint32_t X1SCALE_IDX = 2;
    static constexpr uint32_t BIAS_IDX = 3;
    static constexpr uint32_t M_IDX = 0;
    static constexpr uint32_t N_IDX = 1;
    static constexpr uint32_t UB_ALIGN_SIZE = 32;
    static constexpr int64_t DOUBLE_BUFFER_COUNT = 2;

    // shape
    using BlockShape = tla::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BaseOffset = tla::Coord<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = tla::Coord<int64_t, int64_t, int64_t, int64_t>;

public:
    CATLASS_DEVICE void Init(Arch::Resource<ArchTag> &resource, Params const &params, GemmCoord &problemShape);
    CATLASS_DEVICE void UpdateGlobalBuffer(Params const &params);
    CATLASS_DEVICE void UpdateGroupedParams(Params const &params, BaseOffset const &offset, uint32_t groupIdx);
    CATLASS_DEVICE auto GetL0c2UbTensor();
    CATLASS_DEVICE void SetOutL2CacheHint();
    template <class TensorUb>
    CATLASS_DEVICE void operator()(TensorUb &tensorUb);
    // static init
    CATLASS_HOST_DEVICE static Params InitParams(Arguments const &args)
    {
        Params params = {args.yGmAddr, args.x2ScaleGmAddr, args.x1ScaleGmAddr, args.biasGmAddr, args.dequantTiling};
        return params;
    }

    CATLASS_HOST_DEVICE static size_t GetWorkspaceSize(int64_t blockNum, int64_t l1M, int64_t l1N)
    {
        return 0;
    }

    CATLASS_HOST_DEVICE static Status CanImplement(Arguments const &args)
    {
        if (l0M * l0N * sizeof(DataTypeIn_) > ArchTag::UB_SIZE) {
            return Status::kInvalid;
        }
        return Status::kSuccess;
    }

private:
    CATLASS_DEVICE void UpdateTensorGlobalBuffer(Params const &params);
    CATLASS_DEVICE void CopyDataFromGm2Ub();
    CATLASS_DEVICE void CopyX1ScaleFromGm2Ub(AscendC::LocalTensor<DataTypeX1Scale> &dst, uint64_t blockLen,
                                             uint64_t offset);
    CATLASS_DEVICE void CopyX2ScaleFromGm2Ub(AscendC::LocalTensor<DataTypeX2Scale> &dst);
    template <class BiasDtype>
    CATLASS_DEVICE void CopyBiasFromGm2Ub(AscendC::LocalTensor<BiasDtype> &dst);
    CATLASS_DEVICE void CopyDequantResFromUb2Gm(uint64_t blockCount, uint64_t offset,
                                                AscendC::LocalTensor<DataTypeOut> &src);
    CATLASS_DEVICE void FreeUbTensor();
    CATLASS_DEVICE void VFDoDequantWithX1Pertoken(__ubuf__ DataTypeOut *dequantOutInUbAddr,
                                                  __ubuf__ DataTypeIn *l0cOutUbAddr, uint64_t offsetPtScale,
                                                  uint16_t mSize);
    CATLASS_DEVICE void VFDoDequantWithX1Pertensor(__ubuf__ DataTypeOut *dequantOutInUbAddr,
                                                   __ubuf__ DataTypeIn *l0cOutUbAddr, uint16_t mSize);
    CATLASS_DEVICE void VFDoDequantWithoutX1Scale(__ubuf__ DataTypeOut *dequantOutInUbAddr,
                                                  __ubuf__ DataTypeIn *l0cOutUbAddr, uint16_t mSize);
    template <bool isPertensor, QuantMode x1QuantMode, bool isBiasEpilogue, class BiasDtype>
    __simd_vf__ void VFDoDequant(__ubuf__ DataTypeOut *dst, __ubuf__ DataTypeIn *l0cOut,
                                 __ubuf__ DataTypeX2Scale *scale2, __ubuf__ DataTypeX1Scale *x1Scale,
                                 __ubuf__ BiasDtype *bias, uint16_t mSize, uint16_t nSize);

    // GM ADDR
    AscendC::GlobalTensor<DataTypeOut> yGlobal_;
    AscendC::GlobalTensor<float> biasGlobalFloat_;
    AscendC::GlobalTensor<bfloat16_t> biasGlobalB16_;
    AscendC::GlobalTensor<DataTypeX2Scale> x2ScaleGlobal_;
    AscendC::GlobalTensor<DataTypeX1Scale> x1ScaleGlobal_;

    // UB Tensor
    AscendC::LocalTensor<DataTypeIn> l0cOutUb_;
    AscendC::LocalTensor<DataTypeX2Scale> x2ScaleUb_;
    AscendC::LocalTensor<DataTypeX1Scale> x1ScaleUb_;
    AscendC::LocalTensor<bfloat16_t> biasUbB16_;
    AscendC::LocalTensor<float> biasUbFloat_;
    AscendC::LocalTensor<DataTypeOut> dequantOutInUB_[DOUBLE_BUFFER_COUNT];
    int32_t outInUBEventList_[DOUBLE_BUFFER_COUNT];
    float x2ScaleScalar_;
    float x1ScaleScalar_;
    const DequantTiling *dequantTiling_;
    GemmCoord problemShape_;
    uint32_t biasDtype_ = kDTypeFloat;
    uint32_t groupIdx_ = 0;
    uint32_t subBlockIdx_ = AscendC::GetSubBlockIdx();
    uint32_t singleM_;  // cur singleShapeM
    uint32_t singleN_;
    bool isBiasEpilogue_ = false;
    BaseOffset baseOffset_{0, 0, 0, 0};
    BlockCoord blockCoord_{0, 0, 0, 0};
};

QMM_BLOCK_EPILOGUE_DEQUANT_CLASS_LOCAL_PARAMS
CATLASS_DEVICE void
BlockEpilogue<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::UpdateTensorGlobalBuffer(Params const &params)
{
    if (dequantTiling_->x2QuantMode == QuantMode::PERTENSOR_MODE) {
        DataTypeX2Scale x2ScaleValue = *((__gm__ DataTypeX2Scale *)params.x2ScaleGmAddr + groupIdx_);
        if constexpr (AscendC::IsSameType<DataTypeX2Scale, bfloat16_t>::value) {
            x2ScaleScalar_ = AscendC::ToFloat(x2ScaleValue);
        } else {
            x2ScaleScalar_ = x2ScaleValue;
        }
    } else {
        x2ScaleGlobal_.SetGlobalBuffer((__gm__ DataTypeX2Scale *)params.x2ScaleGmAddr +
                                       tla::get<X2SCALE_IDX>(baseOffset_));
    }
    if (dequantTiling_->x1QuantMode == QuantMode::PERTENSOR_MODE) {
        x1ScaleScalar_ = *((__gm__ DataTypeX1Scale *)params.x1ScaleGmAddr + groupIdx_);
    } else if (dequantTiling_->x1QuantMode == QuantMode::PERTOKEN_MODE) {
        x1ScaleGlobal_.SetGlobalBuffer((__gm__ DataTypeX1Scale *)params.x1ScaleGmAddr +
                                       tla::get<X1SCALE_IDX>(baseOffset_));
    }
    // ub res + biasAdd
    if (isBiasEpilogue_) {
        if (biasDtype_ == kDTypeFloat) {
            biasGlobalFloat_.SetGlobalBuffer((__gm__ float *)params.biasGmAddr + tla::get<BIAS_IDX>(baseOffset_));
        } else {
            biasGlobalB16_.SetGlobalBuffer((__gm__ bfloat16_t *)params.biasGmAddr + tla::get<BIAS_IDX>(baseOffset_));
        }
    }
    yGlobal_.SetGlobalBuffer((__gm__ DataTypeOut *)params.yGmAddr + tla::get<Y_IDX>(baseOffset_));
}

QMM_BLOCK_EPILOGUE_DEQUANT_CLASS_LOCAL_PARAMS
CATLASS_DEVICE void
BlockEpilogue<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::UpdateGlobalBuffer(Params const &params)
{
    UpdateTensorGlobalBuffer(params);
}

QMM_BLOCK_EPILOGUE_DEQUANT_CLASS_LOCAL_PARAMS
CATLASS_DEVICE
void BlockEpilogue<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::Init(Arch::Resource<ArchTag> &resource,
                                                                       Params const &params, GemmCoord &problemShape)
{
    dequantTiling_ = &params.dequantTiling;
    uint64_t mForSingleVec = CeilDiv(dequantTiling_->baseM, AscendC::GetTaskRation());
    uint32_t ubOffset = 0;
    l0cOutUb_ = resource.ubBuf.template GetBufferByByte<DataTypeIn>(ubOffset);
    ubOffset += mForSingleVec * dequantTiling_->baseN * sizeof(DataTypeIn);
    if ASCEND_IS_AIV {
        isBiasEpilogue_ = dequantTiling_->isBiasEpilogue && params.biasGmAddr != nullptr;
        biasDtype_ = dequantTiling_->biasDtype;
        if (dequantTiling_->x2QuantMode == QuantMode::PERCHANNEL_MODE) {
            x2ScaleUb_ = resource.ubBuf.template GetBufferByByte<DataTypeX2Scale>(ubOffset);
            ubOffset += dequantTiling_->baseN * sizeof(DataTypeX2Scale);
        }
        if (dequantTiling_->x1QuantMode == QuantMode::PERTOKEN_MODE) {
            x1ScaleUb_ = resource.ubBuf.template GetBufferByByte<DataTypeX1Scale>(ubOffset);
            ubOffset += RoundUp(mForSingleVec * sizeof(DataTypeX1Scale), static_cast<uint64_t>(UB_ALIGN_SIZE));
        }
        if (isBiasEpilogue_) {
            if (biasDtype_ == kDTypeFloat) {
                biasUbFloat_ = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
                ubOffset += dequantTiling_->baseN * sizeof(float);
            } else {
                biasUbB16_ = resource.ubBuf.template GetBufferByByte<bfloat16_t>(ubOffset);
                ubOffset += dequantTiling_->baseN * sizeof(bfloat16_t);
            }
        }
        for (uint32_t i = 0; i < DOUBLE_BUFFER_COUNT; i++) {
            dequantOutInUB_[i] = resource.ubBuf.template GetBufferByByte<DataTypeOut>(ubOffset);
            ubOffset += CeilDiv(mForSingleVec, FP32_OUTPUT_TIMES) * dequantTiling_->baseN * sizeof(DataTypeOut);

            outInUBEventList_[i] = i;

            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outInUBEventList_[i]);
        }
        UpdateGlobalBuffer(params);
    }
    problemShape_ = problemShape;
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
}

QMM_BLOCK_EPILOGUE_DEQUANT_CLASS_LOCAL_PARAMS
CATLASS_DEVICE void BlockEpilogue<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::UpdateGroupedParams(
    Params const &params, BaseOffset const &offset, uint32_t groupIdx)
{
    baseOffset_ = offset;
    groupIdx_ = groupIdx;
    UpdateGlobalBuffer(params);
}

QMM_BLOCK_EPILOGUE_DEQUANT_CLASS_LOCAL_PARAMS
CATLASS_DEVICE
void BlockEpilogue<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::CopyDataFromGm2Ub()
{
    auto halfSingleM = CeilDiv(singleM_, AscendC::GetTaskRation());
    auto singleMInVec = subBlockIdx_ == 1 ? singleM_ - halfSingleM : halfSingleM;
    // scale2: GM -> UB
    if (dequantTiling_->x2QuantMode == QuantMode::PERCHANNEL_MODE) {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        CopyX2ScaleFromGm2Ub(x2ScaleUb_);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    }

    uint64_t mOffset = subBlockIdx_ * halfSingleM;
    // x1Scale: GM -> UB
    if (dequantTiling_->x1QuantMode == QuantMode::PERTOKEN_MODE) {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        CopyX1ScaleFromGm2Ub(x1ScaleUb_, singleMInVec, tla::get<X1SCALE_IDX>(blockCoord_) + mOffset);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
    }
    if (isBiasEpilogue_) {
        if (biasDtype_ == kDTypeFloat) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
            CopyBiasFromGm2Ub<float>(biasUbFloat_);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
        } else {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
            CopyBiasFromGm2Ub<bfloat16_t>(biasUbB16_);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID3);
        }
    }
}

QMM_BLOCK_EPILOGUE_DEQUANT_CLASS_LOCAL_PARAMS
CATLASS_DEVICE void BlockEpilogue<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::CopyX1ScaleFromGm2Ub(
    AscendC::LocalTensor<DataTypeX1Scale> &dst, uint64_t blockLen, uint64_t offset)
{
    auto x1Layout = tla::MakeLayout<DataTypeX1Scale, layout::VectorLayout>(1, blockLen);
    auto x1UbTensor = tla::MakeTensor(dst, x1Layout, Arch::PositionUB{});
    auto x1GmTensor = tla::MakeTensor(x1ScaleGlobal_[offset], x1Layout, Arch::PositionGM{});
    using CopyGmToUbX1 = Tile::CopyGm2UbTla<ArchTag, decltype(x1GmTensor), decltype(x1UbTensor)>;
    CopyGmToUbX1 copyGmToUbX1;
    copyGmToUbX1(x1UbTensor, x1GmTensor);
}

QMM_BLOCK_EPILOGUE_DEQUANT_CLASS_LOCAL_PARAMS
CATLASS_DEVICE void BlockEpilogue<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::CopyX2ScaleFromGm2Ub(
    AscendC::LocalTensor<DataTypeX2Scale> &dst)
{
    auto x2Layout = tla::MakeLayout<DataTypeX2Scale, layout::VectorLayout>(1, singleN_);
    auto x2UbTensor = tla::MakeTensor(dst, x2Layout, Arch::PositionUB{});
    auto x2GmTensor = tla::MakeTensor(x2ScaleGlobal_[tla::get<X2SCALE_IDX>(blockCoord_)], x2Layout, Arch::PositionGM{});
    using CopyGmToUbX2 = Tile::CopyGm2UbTla<ArchTag, decltype(x2GmTensor), decltype(x2UbTensor)>;
    CopyGmToUbX2 copyGmToUbX2;
    copyGmToUbX2(x2UbTensor, x2GmTensor);
}

QMM_BLOCK_EPILOGUE_DEQUANT_CLASS_LOCAL_PARAMS
template <class BiasDtype>
CATLASS_DEVICE void
BlockEpilogue<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::CopyBiasFromGm2Ub(AscendC::LocalTensor<BiasDtype> &dst)
{
    auto biasLayout = tla::MakeLayout<BiasDtype, layout::VectorLayout>(1, singleN_);
    auto biasUbTensor = tla::MakeTensor(dst, biasLayout, Arch::PositionUB{});
    if constexpr (AscendC::IsSameType<BiasDtype, float>::value) {
        auto biasGmTensor =
            tla::MakeTensor(biasGlobalFloat_[tla::get<BIAS_IDX>(blockCoord_)], biasLayout, Arch::PositionGM{});
        using CopyGmToUbBias = Tile::CopyGm2UbTla<ArchTag, decltype(biasGmTensor), decltype(biasUbTensor)>;
        CopyGmToUbBias copyGmToUbBias;
        copyGmToUbBias(biasUbTensor, biasGmTensor);
    } else {
        auto biasGmTensor =
            tla::MakeTensor(biasGlobalB16_[tla::get<BIAS_IDX>(blockCoord_)], biasLayout, Arch::PositionGM{});
        using CopyGmToUbBias = Tile::CopyGm2UbTla<ArchTag, decltype(biasGmTensor), decltype(biasUbTensor)>;
        CopyGmToUbBias copyGmToUbBias;
        copyGmToUbBias(biasUbTensor, biasGmTensor);
    }
}

QMM_BLOCK_EPILOGUE_DEQUANT_CLASS_LOCAL_PARAMS
CATLASS_DEVICE void BlockEpilogue<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::CopyDequantResFromUb2Gm(
    uint64_t blockCount, uint64_t offset, AscendC::LocalTensor<DataTypeOut> &src)
{
    AscendC::DataCopyExtParams ub2GmParams{1, 0, 0, 0, 0};
    ub2GmParams.blockLen = singleN_ * sizeof(DataTypeOut);
    ub2GmParams.blockCount = blockCount;
    ub2GmParams.dstStride = (problemShape_.n() - singleN_) * sizeof(DataTypeOut);
    AscendC::DataCopyPad(yGlobal_[tla::get<Y_IDX>(blockCoord_) + offset], src, ub2GmParams);
}

QMM_BLOCK_EPILOGUE_DEQUANT_CLASS_LOCAL_PARAMS
CATLASS_DEVICE
void BlockEpilogue<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::FreeUbTensor()
{
    if (dequantTiling_->x2QuantMode == QuantMode::PERCHANNEL_MODE) {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    }

    if (dequantTiling_->x1QuantMode == QuantMode::PERTOKEN_MODE) {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
    }

    if (isBiasEpilogue_) {
        if (biasDtype_ == kDTypeFloat) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        } else {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
        }
    }
}

QMM_BLOCK_EPILOGUE_DEQUANT_CLASS_LOCAL_PARAMS
CATLASS_DEVICE void BlockEpilogue<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::VFDoDequantWithX1Pertoken(
    __ubuf__ DataTypeOut *dequantOutInUbAddr, __ubuf__ DataTypeIn *l0cOutUbAddr, uint64_t offsetPtScale, uint16_t mSize)
{
    __ubuf__ DataTypeX1Scale *ptScaleUbAddr = (__ubuf__ DataTypeX1Scale *)x1ScaleUb_.GetPhyAddr();
    ptScaleUbAddr = ptScaleUbAddr + offsetPtScale;
    if (!isBiasEpilogue_) {
        if (dequantTiling_->x2QuantMode == QuantMode::PERTENSOR_MODE) {
            VFDoDequant<true, QuantMode::PERTOKEN_MODE, false, float>(dequantOutInUbAddr, l0cOutUbAddr, nullptr,
                                                                      ptScaleUbAddr, nullptr, mSize, singleN_);
        } else {
            VFDoDequant<false, QuantMode::PERTOKEN_MODE, false, float>(
                dequantOutInUbAddr, l0cOutUbAddr, (__ubuf__ DataTypeX2Scale *)x2ScaleUb_.GetPhyAddr(), ptScaleUbAddr,
                nullptr, mSize, singleN_);
        }
    } else {
        if (biasDtype_ == kDTypeFloat) {
            if (dequantTiling_->x2QuantMode == QuantMode::PERTENSOR_MODE) {
                VFDoDequant<true, QuantMode::PERTOKEN_MODE, true, float>(
                    dequantOutInUbAddr, l0cOutUbAddr, nullptr, ptScaleUbAddr,
                    (__ubuf__ float *)biasUbFloat_.GetPhyAddr(), mSize, singleN_);
            } else {
                VFDoDequant<false, QuantMode::PERTOKEN_MODE, true, float>(
                    dequantOutInUbAddr, l0cOutUbAddr, (__ubuf__ DataTypeX2Scale *)x2ScaleUb_.GetPhyAddr(),
                    ptScaleUbAddr, (__ubuf__ float *)biasUbFloat_.GetPhyAddr(), mSize, singleN_);
            }
        } else if (biasDtype_ == kDTypeBf16) {
            if (dequantTiling_->x2QuantMode == QuantMode::PERTENSOR_MODE) {
                VFDoDequant<true, QuantMode::PERTOKEN_MODE, true, bfloat16_t>(
                    dequantOutInUbAddr, l0cOutUbAddr, nullptr, ptScaleUbAddr,
                    (__ubuf__ bfloat16_t *)biasUbB16_.GetPhyAddr(), mSize, singleN_);
            } else {
                VFDoDequant<false, QuantMode::PERTOKEN_MODE, true, bfloat16_t>(
                    dequantOutInUbAddr, l0cOutUbAddr, (__ubuf__ DataTypeX2Scale *)x2ScaleUb_.GetPhyAddr(),
                    ptScaleUbAddr, (__ubuf__ bfloat16_t *)biasUbB16_.GetPhyAddr(), mSize, singleN_);
            }
        } else if (biasDtype_ == kDTypeFloat16) {
            if (dequantTiling_->x2QuantMode == QuantMode::PERTENSOR_MODE) {
                VFDoDequant<true, QuantMode::PERTOKEN_MODE, true, half>(
                    dequantOutInUbAddr, l0cOutUbAddr, nullptr, ptScaleUbAddr, (__ubuf__ half *)biasUbB16_.GetPhyAddr(),
                    mSize, singleN_);
            } else {
                VFDoDequant<false, QuantMode::PERTOKEN_MODE, true, half>(
                    dequantOutInUbAddr, l0cOutUbAddr, (__ubuf__ DataTypeX2Scale *)x2ScaleUb_.GetPhyAddr(),
                    ptScaleUbAddr, (__ubuf__ half *)biasUbB16_.GetPhyAddr(), mSize, singleN_);
            }
        }
    }
}

QMM_BLOCK_EPILOGUE_DEQUANT_CLASS_LOCAL_PARAMS
CATLASS_DEVICE void BlockEpilogue<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::VFDoDequantWithX1Pertensor(
    __ubuf__ DataTypeOut *dequantOutInUbAddr, __ubuf__ DataTypeIn *l0cOutUbAddr, uint16_t mSize)
{
    VFDoDequant<false, QuantMode::PERTENSOR_MODE, false, float>(dequantOutInUbAddr, l0cOutUbAddr,
                                                                (__ubuf__ DataTypeX2Scale *)x2ScaleUb_.GetPhyAddr(),
                                                                nullptr, nullptr, mSize, singleN_);
}

QMM_BLOCK_EPILOGUE_DEQUANT_CLASS_LOCAL_PARAMS
CATLASS_DEVICE void BlockEpilogue<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::VFDoDequantWithoutX1Scale(
    __ubuf__ DataTypeOut *dequantOutInUbAddr, __ubuf__ DataTypeIn *l0cOutUbAddr, uint16_t mSize)
{
    if (!isBiasEpilogue_) {
        VFDoDequant<false, QuantMode::DEFAULT, false, float>(dequantOutInUbAddr, l0cOutUbAddr,
                                                             (__ubuf__ DataTypeX2Scale *)x2ScaleUb_.GetPhyAddr(),
                                                             nullptr, nullptr, mSize, singleN_);
    } else {
        if (biasDtype_ == kDTypeFloat) {
            if (dequantTiling_->x2QuantMode == QuantMode::PERTENSOR_MODE) {
                VFDoDequant<true, QuantMode::DEFAULT, true, float>(dequantOutInUbAddr, l0cOutUbAddr, nullptr, nullptr,
                                                                   (__ubuf__ float *)biasUbFloat_.GetPhyAddr(), mSize,
                                                                   singleN_);
            } else {
                VFDoDequant<false, QuantMode::DEFAULT, true, float>(
                    dequantOutInUbAddr, l0cOutUbAddr, (__ubuf__ DataTypeX2Scale *)x2ScaleUb_.GetPhyAddr(), nullptr,
                    (__ubuf__ float *)biasUbFloat_.GetPhyAddr(), mSize, singleN_);
            }
        } else if (biasDtype_ == kDTypeBf16) {
            if (dequantTiling_->x2QuantMode == QuantMode::PERTENSOR_MODE) {
                VFDoDequant<true, QuantMode::DEFAULT, true, bfloat16_t>(
                    dequantOutInUbAddr, l0cOutUbAddr, nullptr, nullptr, (__ubuf__ bfloat16_t *)biasUbB16_.GetPhyAddr(),
                    mSize, singleN_);
            } else {
                VFDoDequant<false, QuantMode::DEFAULT, true, bfloat16_t>(
                    dequantOutInUbAddr, l0cOutUbAddr, (__ubuf__ DataTypeX2Scale *)x2ScaleUb_.GetPhyAddr(), nullptr,
                    (__ubuf__ bfloat16_t *)biasUbB16_.GetPhyAddr(), mSize, singleN_);
            }
        }
    }
}

QMM_BLOCK_EPILOGUE_DEQUANT_CLASS_LOCAL_PARAMS
template <bool isPertensor, QuantMode x1QuantMode, bool isBiasEpilogue, class BiasDtype>
__simd_vf__ void BlockEpilogue<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::VFDoDequant(
    __ubuf__ DataTypeOut *dst, __ubuf__ DataTypeIn *l0cOut, __ubuf__ DataTypeX2Scale *scale2,
    __ubuf__ DataTypeX1Scale *x1Scale, __ubuf__ BiasDtype *bias, uint16_t mSize, uint16_t nSize)
{
    uint32_t eleNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(DataTypeIn);
    uint32_t nSrcUbAligned = RoundUp(nSize, static_cast<uint16_t>(UB_ALIGN_SIZE / sizeof(DataTypeIn)));
    uint32_t nDstUbAligned = RoundUp(nSize, static_cast<uint16_t>(UB_ALIGN_SIZE / sizeof(DataTypeOut)));
    uint16_t nLoopCnt = (nSize + eleNumPerVf - 1) / eleNumPerVf;

    constexpr static AscendC::MicroAPI::CastTrait ctInt322Fp32 = {
        AscendC::MicroAPI::RegLayout::UNKNOWN, AscendC::MicroAPI::SatMode::UNKNOWN,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
    constexpr static AscendC::MicroAPI::CastTrait ctFp322Half = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
    constexpr static AscendC::MicroAPI::CastTrait ctHalf2Fp32Zero = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
    constexpr static AscendC::MicroAPI::CastTrait ctHalf2Fp32One = {
        AscendC::MicroAPI::RegLayout::ONE, AscendC::MicroAPI::SatMode::UNKNOWN,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};

    AscendC::MicroAPI::MaskReg maskN4B16 =
        AscendC::MicroAPI::CreateMask<bfloat16_t, AscendC::MicroAPI::MaskPattern::ALL>();
    for (uint16_t mIdx = 0; mIdx < mSize; mIdx++) {
        uint32_t elementNum = nSize;
        for (uint16_t vfBlockIdx = 0; vfBlockIdx < nLoopCnt; vfBlockIdx++) {
            AscendC::MicroAPI::RegTensor<DataTypeIn> l0cOutReg;
            AscendC::MicroAPI::RegTensor<DataTypeX2Scale> scaleReg;
            AscendC::MicroAPI::RegTensor<DataTypeX1Scale> perTokenScaleReg;
            AscendC::MicroAPI::RegTensor<BiasDtype> biasReg;
            AscendC::MicroAPI::RegTensor<float> castSrcOutReg, castScaleReg, castScaleOneReg, mulScaleOutReg,
                mulPtScaleOutReg, castBiasReg, castBiasOneReg, addBiasOutReg;
            AscendC::MicroAPI::RegTensor<DataTypeOut> castResultOutReg;
            AscendC::MicroAPI::MaskReg maskN = AscendC::MicroAPI::UpdateMask<DataTypeIn>(elementNum);
            // copy input from ub to register, addr of ub should align to 32B
            uint32_t l0cOutOffset = mIdx * nSrcUbAligned + vfBlockIdx * eleNumPerVf;
            AscendC::MicroAPI::DataCopy(l0cOutReg, l0cOut + l0cOutOffset);
            // cast l0cOut from int32 to float
            if constexpr (AscendC::IsSameType<DataTypeIn, int32_t>::value) {
                AscendC::MicroAPI::Cast<float, DataTypeIn, ctInt322Fp32>(castSrcOutReg, l0cOutReg, maskN);
            } else {
                castSrcOutReg = l0cOutReg;
            }
            // l0c_out * scale2
            if constexpr (isPertensor) {
                AscendC::MicroAPI::Muls(mulScaleOutReg, castSrcOutReg, x2ScaleScalar_, maskN);
            } else {
                AscendC::MicroAPI::DataCopy(scaleReg, scale2 + vfBlockIdx * eleNumPerVf);
                if constexpr (!AscendC::IsSameType<DataTypeX2Scale, float>::value) {  // cast scale2 from bf16 to float
                    AscendC::MicroAPI::Cast<float, DataTypeX2Scale, ctHalf2Fp32Zero>(castScaleReg, scaleReg, maskN);
                    AscendC::MicroAPI::Cast<float, DataTypeX2Scale, ctHalf2Fp32One>(castScaleOneReg, scaleReg,
                                                                                    maskN4B16);
                    AscendC::MicroAPI::Interleave(castScaleReg, castScaleOneReg, castScaleReg, castScaleOneReg);
                } else {
                    castScaleReg = scaleReg;
                }
                AscendC::MicroAPI::Mul(mulScaleOutReg, castSrcOutReg, castScaleReg, maskN);
            }
            // out * x1Scale
            if constexpr (x1QuantMode == QuantMode::PERTENSOR_MODE) {
                AscendC::MicroAPI::Muls(mulPtScaleOutReg, mulScaleOutReg, x1ScaleScalar_, maskN);
            } else if constexpr (x1QuantMode == QuantMode::PERTOKEN_MODE) {
                AscendC::MicroAPI::DataCopy<DataTypeX1Scale, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                    perTokenScaleReg, x1Scale + mIdx);
                AscendC::MicroAPI::Mul(mulPtScaleOutReg, mulScaleOutReg, perTokenScaleReg, maskN);
            } else {
                mulPtScaleOutReg = mulScaleOutReg;
            }
            // out + bias
            if constexpr (isBiasEpilogue) {
                AscendC::MicroAPI::DataCopy(biasReg, bias + vfBlockIdx * eleNumPerVf);
                // cast bias from bf16/fp16 to float
                if constexpr (AscendC::IsSameType<BiasDtype, bfloat16_t>::value ||
                              AscendC::IsSameType<BiasDtype, half>::value) {
                    AscendC::MicroAPI::Cast<float, BiasDtype, ctHalf2Fp32Zero>(castBiasReg, biasReg, maskN);
                    AscendC::MicroAPI::Cast<float, BiasDtype, ctHalf2Fp32One>(castBiasOneReg, biasReg, maskN4B16);
                    AscendC::MicroAPI::Interleave(castBiasReg, castBiasOneReg, castBiasReg, castBiasOneReg);
                } else {
                    castBiasReg = biasReg;
                }
                AscendC::MicroAPI::Add(addBiasOutReg, mulPtScaleOutReg, castBiasReg, maskN);
            } else {
                addBiasOutReg = mulPtScaleOutReg;
            }
            // cast dequant result from float to fp16/bf16
            if constexpr (!AscendC::IsSameType<DataTypeOut, float>::value) {
                AscendC::MicroAPI::Cast<DataTypeOut, float, ctFp322Half>(castResultOutReg, addBiasOutReg, maskN);
            } else {
                castResultOutReg = addBiasOutReg;
            }
            // copy out from register to ub
            uint32_t dstUbOffset = mIdx * nDstUbAligned + vfBlockIdx * eleNumPerVf;
            if constexpr (AscendC::IsSameType<DataTypeOut, float>::value) {
                AscendC::MicroAPI::DataCopy<DataTypeOut, AscendC::MicroAPI::StoreDist::DIST_NORM_B32>(
                    dst + dstUbOffset, castResultOutReg, maskN);
            } else {
                AscendC::MicroAPI::DataCopy<DataTypeOut, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
                    dst + dstUbOffset, castResultOutReg, maskN);
            }
        }
    }
}

QMM_BLOCK_EPILOGUE_DEQUANT_CLASS_LOCAL_PARAMS
CATLASS_DEVICE auto BlockEpilogue<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::GetL0c2UbTensor()
{
    return l0cOutUb_;
}

QMM_BLOCK_EPILOGUE_DEQUANT_CLASS_LOCAL_PARAMS
CATLASS_DEVICE void BlockEpilogue<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::SetOutL2CacheHint()
{
    yGlobal_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
}

QMM_BLOCK_EPILOGUE_DEQUANT_CLASS_LOCAL_PARAMS
template <class TensorUb>
CATLASS_DEVICE void BlockEpilogue<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::operator()(TensorUb &tensorUb)
{
    singleM_ = tla::get<M_IDX>(tensorUb.shape());
    singleN_ = tla::get<N_IDX>(tensorUb.shape());
    blockCoord_ =
        BlockCoord{tla::get<0>(tensorUb.coord()) * problemShape_.n() + tla::get<1>(tensorUb.coord()),
                   tla::get<1>(tensorUb.coord()), tla::get<0>(tensorUb.coord()), tla::get<1>(tensorUb.coord())};
    auto halfSingleM = CeilDiv(singleM_, AscendC::GetTaskRation());
    uint64_t singleMInVec = subBlockIdx_ == 1 ? singleM_ - halfSingleM : halfSingleM;
    if (singleMInVec == 0) {
        return;
    }
    uint64_t mOffset = subBlockIdx_ * halfSingleM;
    CopyDataFromGm2Ub();
    // 4 times out because of ub size
    uint16_t splitNumOfOut = singleMInVec >= 4 ? 4 : singleMInVec;
    auto mSizeForOnce = CeilDiv(singleMInVec, static_cast<uint64_t>(splitNumOfOut));
    uint32_t outInUBListId{0};
    for (uint16_t i = 0; i < splitNumOfOut; i++) {
        uint32_t outInUBListIdNext = (outInUBListId + 1 < DOUBLE_BUFFER_COUNT) ? (outInUBListId + 1) : 0;
        // do dequant in vector
        uint64_t offsetL0c =
            i * mSizeForOnce * RoundUp(singleN_, static_cast<uint64_t>(UB_ALIGN_SIZE / sizeof(DataTypeIn)));
        if (i * mSizeForOnce >= singleMInVec) {
            break;
        }
        auto mSize = singleMInVec - i * mSizeForOnce >= mSizeForOnce ? mSizeForOnce : singleMInVec - i * mSizeForOnce;
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outInUBEventList_[outInUBListId]);
        AscendC::LocalTensor<DataTypeOut> dequantOutInUB = dequantOutInUB_[outInUBListId];

        __ubuf__ DataTypeOut *dequantOutInUbAddr = (__ubuf__ DataTypeOut *)dequantOutInUB.GetPhyAddr();
        __ubuf__ DataTypeIn *l0cOutUbAddr = (__ubuf__ DataTypeIn *)l0cOutUb_.GetPhyAddr();
        l0cOutUbAddr = l0cOutUbAddr + offsetL0c;

        switch (dequantTiling_->x1QuantMode) {
            case (QuantMode::PERTOKEN_MODE): {
                uint64_t offsetPtScale = i * mSizeForOnce;
                VFDoDequantWithX1Pertoken(dequantOutInUbAddr, l0cOutUbAddr, offsetPtScale, mSize);
                break;
            }
            case (QuantMode::PERTENSOR_MODE): {
                VFDoDequantWithX1Pertensor(dequantOutInUbAddr, l0cOutUbAddr, mSize);
                break;
            }
            default: {
                VFDoDequantWithoutX1Scale(dequantOutInUbAddr, l0cOutUbAddr, mSize);
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outInUBEventList_[outInUBListId]);
        // mmDequant result: UB -> GM
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outInUBEventList_[outInUBListId]);
        CopyDequantResFromUb2Gm(mSize, (mOffset + i * mSizeForOnce) * problemShape_.n(), dequantOutInUB);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outInUBEventList_[outInUBListId]);
        outInUBListId = outInUBListIdNext;
    }
    FreeUbTensor();
}
}  // namespace Catlass::Epilogue::Block
#endif  // CATLASS_EPILOGUE_BLOCK_EPILOGUE_DEQUANT_HPP
