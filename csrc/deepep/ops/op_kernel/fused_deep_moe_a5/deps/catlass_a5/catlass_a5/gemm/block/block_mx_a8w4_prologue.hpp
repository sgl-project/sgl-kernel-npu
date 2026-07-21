/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_BLOCK_MX_A8W4_PROLOGUE_HPP
#define CATLASS_GEMM_BLOCK_MX_A8W4_PROLOGUE_HPP

#include "catlass_a5/arch/resource.hpp"
#include "catlass_a5/catlass.hpp"
#include "catlass_a5/coord.hpp"
#include "catlass_a5/numeric_size.hpp"
#include "catlass_a5/gemm/dispatch_policy.hpp"
#include "catlass_a5/gemm/helper.hpp"
#include "catlass_a5/gemm_coord.hpp"
#include "tla_a5/layout.hpp"
#include "tla_a5/tensor.hpp"

namespace Catlass::Gemm::Block {
template <class ArchTag, class InType_, class OutType_, class TileShapeL1_, class TileCopy_>
struct BlockPrologue<MxA8W4Prologue<ArchTag>, InType_, OutType_, TileShapeL1_, TileCopy_> {
public:
    using DispatchPolicy = MxA8W4Prologue<ArchTag>;
    using ElementIn = typename InType_::Element;
    using ElementOut = typename OutType_::Element;
    using LayoutIn = typename InType_::Layout;
    using LayoutOut = typename OutType_::Layout;
    using TileShapeL1 = TileShapeL1_;
    using TileCopy = TileCopy_;
    using LayoutB = typename TileCopy::LayoutB;
    template <int32_t t>
    using Int = AscendC::Std::integral_constant<int32_t, t>;
    using _0 = Int<0>;
    using _16 = Int<16>;
    using _32 = Int<32>;

    static constexpr uint32_t L1B_STAGES = DispatchPolicy::L1B_STAGES;

    struct Params {
        TileShapeL1 tileShapeL1;
        LayoutB layoutB;
        int32_t nUbSize;
        int32_t kUbSize;
        bool hasBias;
    };

    struct VfParamsNormal {
        uint16_t outExtend;
        uint16_t innerExtend;
        uint32_t dataBlockStride;
        uint32_t repeatStride;
        int32_t outDimOffset;
        uint32_t maskB8Tail0;
        uint32_t maskB8Tail1;
        __ubuf__ int8_t *weightInUbBaseAddr;
        __ubuf__ ElementOut *weightOutUbAddr;
        __ubuf__ ElementOut *weightOutUbAddr1;
    };

    struct VfParamsNz {
        uint16_t innerExtend;
        uint32_t innerDstExtend;
        uint32_t innerSrcExtend;
        uint32_t shiftLeftSize;
        uint32_t andMask;
        __ubuf__ int8_t *weightInUbBaseAddr;
        __ubuf__ ElementOut *weightOutUbAddr;
    };

    static_assert(std::is_same_v<LayoutIn, layout::RowMajor> || std::is_same_v<LayoutIn, layout::ColumnMajor> ||
                      std::is_same_v<LayoutIn, layout::zN>,
                  "Unsupported layout, only can be Rowmajor ColumnMajor or zN");

    CATLASS_DEVICE
    BlockPrologue(const Params &params)
    {
        nUbSize_ = params.nUbSize;
        kUbSize_ = params.kUbSize;
        nSize_ = tla::get<1>(params.layoutB.shape());
        kSize_ = tla::get<0>(params.layoutB.shape());
        nL1Size_ = tla::get<1>(params.tileShapeL1);
        kL1Size_ = tla::get<2>(params.tileShapeL1);  // 2 in order to obtain k
        bL1Size_ = nL1Size_ * RoundUp(kL1Size_, K_ALIGN_SIZE);
        aL1Size_ = tla::get<0>(params.tileShapeL1) * tla::get<2>(params.tileShapeL1);  // 2 in order to obtain k
        vecWeightInLen_ = (UB_STAGES * (nUbSize_ * RoundUp(kUbSize_, OFFSET_64))) >> INT4_DTYPE_PARAM;
        vecWeightOutLen_ =
            UB_STAGES * (RoundUp(nUbSize_, AscendC::BLOCK_CUBE) + 1) *
            RoundUp(RoundUp(kUbSize_, static_cast<int32_t>(AscendC::ONE_BLK_SIZE)), static_cast<int32_t>(K_ALIGN_SIZE));
        weightOutUb_ = AscendC::LocalTensor<ElementOut>(AscendC::TPosition::VECCALC, 0, vecWeightOutLen_);
        uint64_t ubOffset = vecWeightOutLen_ * sizeof(ElementOut);
        weightInUb_ = AscendC::LocalTensor<ElementIn>(AscendC::TPosition::VECCALC, ubOffset, vecWeightInLen_);
        l1Local_ = AscendC::LocalTensor<ElementOut>(AscendC::TPosition::B1, 0, ArchTag::L1_SIZE);

        for (uint32_t i = 0; i < L1B_STAGES; i++) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(i);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(i);
        }
    }

    CATLASS_DEVICE
    ~BlockPrologue()
    {
        for (uint32_t i = 0; i < L1B_STAGES; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(i);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(i);
        }
    }

    template <class TensorBIn, class TensorBOut, class ActualBlockShape>
    CATLASS_DEVICE void operator()(const TensorBIn &bGlobal, const TensorBOut &bLocal,
                                   const ActualBlockShape &actualBlockShape, const Params &params)
    {
        nL1Len_ = actualBlockShape.n();
        uint64_t kTileCount = CeilDiv(kSize_, static_cast<uint64_t>(tla::get<2>(params.tileShapeL1)));
        for (uint64_t kLoopIdx = 0; kLoopIdx < kTileCount; kLoopIdx++) {
            AscendC::CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE3>(AIV_SYNC_AIC_FLAG);
            kGmOffset_ = kLoopIdx * kL1Size_;
            kL1Len_ = Min(kSize_ - kGmOffset_, kL1Size_);
            auto tensorBlockB = GetTile(bGlobal, tla::MakeCoord(kGmOffset_, 0),
                                        tla::MakeShape(static_cast<uint64_t>(kL1Len_), static_cast<uint64_t>(nL1Len_)));
            nUbLen_ = nL1Len_;
            kUbLen_ = kL1Len_;
            if constexpr (L1B_STAGES == DOUBLE_BUFFER) {
                if (l1BufIdx_ == AscendC::GetSubBlockIdx()) {
                    ProcessL1NK(tensorBlockB, bLocal);
                }
            } else if (AscendC::GetSubBlockIdx() == 0) {
                ProcessL1NK(tensorBlockB, bLocal);
            }
            l1BufIdx_ = (l1BufIdx_ + 1) % L1B_STAGES;
            AscendC::CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(AIV_SYNC_AIC_FLAG);
        }
    }

    template <class TensorBIn, class TensorBOut>
    __aicore__ inline void ProcessL1NK(const TensorBIn &tensorBlockB, const TensorBOut &tensorL1B)
    {
        int32_t nFactor = CeilDiv(nL1Len_, nUbSize_);
        int32_t kFactor = CeilDiv(kL1Len_, kUbSize_);
        for (int32_t nLoopIdx = 0; nLoopIdx < nFactor; nLoopIdx++) {
            nL1Offset_ = nLoopIdx * nUbSize_;
            nUbLen_ = Min(nL1Len_ - static_cast<int32_t>(nL1Offset_), nUbSize_);
            for (int32_t kLoopIdx = 0; kLoopIdx < kFactor; kLoopIdx++) {
                kL1Offset_ = kLoopIdx * kUbSize_;
                kUbLen_ = Min(kL1Len_ - static_cast<int32_t>(kL1Offset_), kUbSize_);
                int64_t l1Offset = (l1BufIdx_ & 0x1) * L1_BUFFER_HALF_SIZE / sizeof(ElementOut) +
                                   RoundUp(nL1Size_, AscendC::BLOCK_CUBE) * kL1Offset_ +
                                   nL1Offset_ * AscendC::ONE_BLK_SIZE;
                ProcessL1(tensorBlockB, l1Offset, tensorL1B);
            }
        }
    }

    template <class TensorBIn, class TensorBOut>
    __aicore__ inline void ProcessL1(const TensorBIn &tensorBlockB, int64_t l1Offset, const TensorBOut &tensorL1B)
    {
        ubBufIdx_ = ubBufIdx_ % L1B_STAGES;
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(ubBufIdx_);
        CopyInTensorWeight(tensorBlockB);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ubBufIdx_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ubBufIdx_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(ubBufIdx_);
        AntiQuantCompute();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ubBufIdx_);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(ubBufIdx_);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ubBufIdx_);
        uint64_t weightOutUbOffset = ubBufIdx_ * (vecWeightOutLen_ / sizeof(ElementOut) / L1B_STAGES);
        CopyVecOut2L1(l1Offset, weightOutUb_[weightOutUbOffset], tensorL1B);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ubBufIdx_);
        ubBufIdx_++;
    }

    template <class TensorBIn>
    __aicore__ inline void CopyInTensorWeight(const TensorBIn &tensorBlockB)
    {
        AscendC::DataCopyExtParams intriParams;
        intriParams.dstStride = 0;
        AscendC::DataCopyPadExtParams<ElementIn> padParams;
        intriParams.blockCount = nUbLen_;
        intriParams.blockLen = CeilDiv(kUbLen_, 2);
        intriParams.srcStride = CeilDiv(kSize_, 2) - CeilDiv(kUbLen_, 2);
        uint64_t weightInOffset = ubBufIdx_ * (vecWeightInLen_ << INT4_DTYPE_PARAM) / L1B_STAGES;
        auto subTensorBlockB = tensorBlockB(tla::MakeCoord(kL1Offset_, nL1Offset_));
        DataCopyPad(weightInUb_[weightInOffset], subTensorBlockB, intriParams, padParams);
    }

    template <class TensorBOut>
    __aicore__ inline void CopyVecOut2L1(int64_t l1Offset, const AscendC::LocalTensor<ElementOut> &ubLocal,
                                         const TensorBOut &tensorL1B)
    {
        AscendC::DataCopyParams params;
        params.blockLen = nUbLen_;
        params.blockCount = CeilDiv(kUbLen_, static_cast<int32_t>(GROUP_SIZE));
        params.srcStride = 1 + RoundUp(nUbLen_, AscendC::BLOCK_CUBE) - nUbLen_;
        params.dstStride = RoundUp(nL1Size_, AscendC::BLOCK_CUBE) - nUbLen_;
        DataCopy(l1Local_[l1Offset], ubLocal, params);
    }

    __aicore__ inline void AntiQuantCompute()
    {
        uint64_t weightOutUbOffset;
        uint64_t weightInUbOffset;
        weightOutUbOffset = ubBufIdx_ * (vecWeightOutLen_ / sizeof(ElementOut) / L1B_STAGES);
        weightInUbOffset = ubBufIdx_ * (vecWeightInLen_ << INT4_DTYPE_PARAM) / L1B_STAGES;
        weightInUbBaseAddr_ = (__ubuf__ int8_t *)weightInUb_[weightInUbOffset].GetPhyAddr();
        weightOutUbAddr_ = (__ubuf__ ElementOut *)weightOutUb_[weightOutUbOffset].GetPhyAddr();

        uint16_t blockStride = RoundUp(nUbLen_, AscendC::BLOCK_CUBE) + 1;
        weightOutUbAddr1_ = weightOutUbAddr_ + VEC_MAX_ELEM_B8 * blockStride;
        AntiQuantComputeNormal();
    }

    __aicore__ inline void AntiQuantComputeNormal()
    {
        VfParamsNormal wParams;
        wParams.outExtend = static_cast<uint16_t>(nUbLen_);
        wParams.innerExtend = CeilDiv(RoundUp(kUbLen_, UB_ALIGN_SIZE_FOR_4BITS), VECTOR_REG_WIDTH_FOR_4BITS);
        wParams.dataBlockStride = RoundUp(nUbLen_, AscendC::BLOCK_CUBE) + 1;
        wParams.repeatStride = wParams.dataBlockStride * AscendC::BLOCK_CUBE;
        wParams.outDimOffset =
            AscendC::ONE_BLOCK_SIZE - wParams.innerExtend * wParams.repeatStride * AscendC::ONE_BLOCK_SIZE;
        wParams.maskB8Tail0 =
            Min(kUbLen_ % VECTOR_REG_WIDTH_FOR_4BITS, static_cast<int32_t>(AscendC::VECTOR_REG_WIDTH)) +
            kUbLen_ / VECTOR_REG_WIDTH_FOR_4BITS * AscendC::VECTOR_REG_WIDTH;
        wParams.maskB8Tail1 =
            Max(kUbLen_ % VECTOR_REG_WIDTH_FOR_4BITS - static_cast<int32_t>(AscendC::VECTOR_REG_WIDTH), 0) +
            kUbLen_ / VECTOR_REG_WIDTH_FOR_4BITS * AscendC::VECTOR_REG_WIDTH;
        wParams.weightInUbBaseAddr = weightInUbBaseAddr_;
        wParams.weightOutUbAddr = weightOutUbAddr_;
        wParams.weightOutUbAddr1 = weightOutUbAddr1_;
        RegCompute(wParams);
    }

    __simd_vf__ inline void RegCompute(const VfParamsNormal wParams)
    {
        __ubuf__ ElementOut *weightOutUbAddr = wParams.weightOutUbAddr;
        __ubuf__ ElementOut *weightOutUbAddr1 = wParams.weightOutUbAddr1;
        AscendC::MicroAPI::RegTensor<uint8_t> wDIntlv0, wDIntlv1, wLoad0, sAnd0, sAnd1, wShr, wShl, s1, wOr0, wOr1,
            wdup1, wdup4;
        AscendC::MicroAPI::RegTensor<int8_t> wdup0, wdup2, wdup3;
        AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::Duplicate<int8_t, AscendC::MicroAPI::MaskMergeMode::ZEROING>(wdup0, DUP_CONFIG_2, preg);
        AscendC::MicroAPI::Duplicate<uint8_t, AscendC::MicroAPI::MaskMergeMode::ZEROING>(wdup1, DUP_CONFIG_MODE_1C,
                                                                                         preg);
        AscendC::MicroAPI::Duplicate<int8_t, AscendC::MicroAPI::MaskMergeMode::ZEROING>(wdup2, DUP_CONFIG_2, preg);
        AscendC::MicroAPI::Duplicate<int8_t, AscendC::MicroAPI::MaskMergeMode::ZEROING>(wdup3, DUP_CONFIG_4, preg);
        AscendC::MicroAPI::Duplicate<uint8_t, AscendC::MicroAPI::MaskMergeMode::ZEROING>(wdup4, DUP_FLAG_80, preg);
        // 一次处理一个N轴
        for (uint16_t outIdx = 0; outIdx < wParams.outExtend; ++outIdx) {
            uint32_t maskWeight0Tmp = wParams.maskB8Tail0;
            uint32_t maskWeight1Tmp = wParams.maskB8Tail1;
            for (uint16_t repeatIdx = 0; repeatIdx < wParams.innerExtend; ++repeatIdx) {
                AscendC::MicroAPI::MaskReg MaskRegB8Tail0 = AscendC::MicroAPI::UpdateMask<uint8_t>(maskWeight0Tmp);
                AscendC::MicroAPI::MaskReg MaskRegB8Tail1 = AscendC::MicroAPI::UpdateMask<uint8_t>(maskWeight1Tmp);
                AscendC::MicroAPI::AddrReg aregWeightB8 = AscendC::MicroAPI::CreateAddrReg<uint8_t>(
                    outIdx, RoundUp(kUbLen_, static_cast<int32_t>(K_ALIGN_SIZE)) >> 1, repeatIdx, VEC_MAX_ELEM_B8);
                AscendC::MicroAPI::LoadAlign(wLoad0, (__ubuf__ uint8_t *&)wParams.weightInUbBaseAddr, aregWeightB8);
                // 提取E/M
                AscendC::MicroAPI::ShiftRight(wShr, wLoad0, wdup0, preg);  // vr1
                AscendC::MicroAPI::And(wShr, wShr, wdup1, preg);           // vr1
                AscendC::MicroAPI::ShiftLeft(wShl, wLoad0, wdup2, preg);   // vr2
                AscendC::MicroAPI::And(wShl, wShl, wdup1, preg);           // vr2
                // 提取S
                AscendC::MicroAPI::ShiftLeft(s1, wLoad0, wdup3, preg);  // vr3
                AscendC::MicroAPI::And(sAnd0, s1, wdup4, preg);         // vr3
                AscendC::MicroAPI::And(sAnd1, wLoad0, wdup4, preg);     // vr4
                // 合并S/E/M
                AscendC::MicroAPI::Or(wOr0, wShr, sAnd1, preg);  // odd
                AscendC::MicroAPI::Or(wOr1, wShl, sAnd0, preg);  // even
                AscendC::MicroAPI::Interleave(wDIntlv0, wDIntlv1, wOr1, wOr0);
                AscendC::MicroAPI::StoreAlign<uint8_t, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                                              AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    (__ubuf__ uint8_t *&)weightOutUbAddr, wDIntlv0, wParams.dataBlockStride, wParams.repeatStride,
                    MaskRegB8Tail0);
                AscendC::MicroAPI::StoreAlign<uint8_t, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                                              AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    (__ubuf__ uint8_t *&)weightOutUbAddr1, wDIntlv1, wParams.dataBlockStride, wParams.repeatStride,
                    MaskRegB8Tail1);
            }
            weightOutUbAddr += wParams.outDimOffset;
            weightOutUbAddr1 += wParams.outDimOffset;
        }
    }

    __aicore__ inline void AntiQuantComputeNKMxNz()
    {
        static_assert(AscendC::Std::is_one_of_v<ElementIn, fp4x2_e2m1_t, fp4x2_e1m2_t>,
                      "only support fp4x2_e2m1_t and fp4x2_e1m2_t");
        VfParamsNz wParams;
        wParams.shiftLeftSize =
            AscendC::IsSameType<ElementIn, fp4x2_e2m1_t>::value ? E2M1_SHIFT_LEFT_SIZE : E1M2_SHIFT_LEFT_SIZE;
        wParams.andMask = AscendC::IsSameType<ElementIn, fp4x2_e2m1_t>::value ? E2M1_AND_MASK : E1M2_AND_MASK;
        wParams.innerExtend =
            CeilDiv(kUbLen_ * RoundUp(nUbLen_, AscendC::BLOCK_CUBE), static_cast<int32_t>(AscendC::VECTOR_REG_WIDTH));
        wParams.innerDstExtend = AscendC::VECTOR_REG_WIDTH * L1B_STAGES;
        wParams.innerSrcExtend = AscendC::VECTOR_REG_WIDTH >> 1;
        wParams.weightInUbBaseAddr = weightInUbBaseAddr_;
        wParams.weightOutUbAddr = weightOutUbAddr_;
        RegComputeNkNz(wParams);
    }

    __simd_vf__ inline void RegComputeNkNz(const VfParamsNz wParams)
    {
        AscendC::MicroAPI::RegTensor<int8_t> wdup0, wdup1, wdup2, wLoad0, wShl, wShr0, wShr1, wSel0, sAnd0;
        AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg pregVsel =
            AscendC::MicroAPI::CreateMask<uint16_t, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::Duplicate<int8_t, AscendC::MicroAPI::MaskMergeMode::ZEROING>(wdup0, wParams.shiftLeftSize,
                                                                                        preg);
        AscendC::MicroAPI::Duplicate<int8_t, AscendC::MicroAPI::MaskMergeMode::ZEROING>(wdup1, SHIFT_RIGHT_SIZE, preg);
        AscendC::MicroAPI::Duplicate<int8_t, AscendC::MicroAPI::MaskMergeMode::ZEROING>(wdup2, wParams.andMask, preg);
        for (uint16_t repeatIdx = 0; repeatIdx < wParams.innerExtend; ++repeatIdx) {
            AscendC::MicroAPI::AddrReg aregWeightB8In =
                AscendC::MicroAPI::CreateAddrReg<uint8_t>(repeatIdx, wParams.innerSrcExtend);
            AscendC::MicroAPI::AddrReg aregWeightB8Out =
                AscendC::MicroAPI::CreateAddrReg<uint8_t>(repeatIdx, wParams.innerDstExtend);
            AscendC::MicroAPI::LoadAlign<uint8_t, AscendC::MicroAPI::LoadDist::DIST_US_B8>(
                (AscendC::MicroAPI::RegTensor<uint8_t> &)wLoad0, (__ubuf__ uint8_t *&)wParams.weightInUbBaseAddr,
                aregWeightB8In);
            AscendC::MicroAPI::ShiftRight(wShr0, wLoad0, wdup0, preg);
            AscendC::MicroAPI::ShiftLeft(wShl, wLoad0, wdup1, preg);
            AscendC::MicroAPI::ShiftRight(wShr1, wShl, wdup0, preg);
            AscendC::MicroAPI::Select(wSel0, wShr1, wShr0, pregVsel);
            AscendC::MicroAPI::And(sAnd0, wSel0, wdup2, preg);
            AscendC::MicroAPI::StoreAlign<uint8_t, AscendC::MicroAPI::StoreDist::DIST_NORM_B8>(
                (__ubuf__ uint8_t *&)wParams.weightOutUbAddr, (AscendC::MicroAPI::RegTensor<uint8_t> &)sAnd0,
                aregWeightB8Out, preg);
        }
    }

    static constexpr int64_t DOUBLE_BUFFER = 2;
    static constexpr uint64_t SYNC_MODE4 = 4;
    static constexpr uint64_t L1_BUFFER_HALF_SIZE = 256 * 1024;
    static constexpr uint64_t K_ALIGN_SIZE = 64;
    static constexpr uint64_t INT4_DTYPE_PARAM = 1;
    static constexpr uint64_t BLOCK_NUM_REG = BYTE_PER_VECTOR_FRACTAL / BYTE_PER_BLK;
    static constexpr uint64_t SINGLE_BUFFER = 1;
    static constexpr uint64_t GROUP_SIZE = 32;
    static constexpr int32_t C0_SIZE_B8 = 32;
    static constexpr int32_t UB_ALIGN_SIZE_FOR_4BITS = 64;
    static constexpr uint32_t DUP_CONFIG_2 = 0x2;
    static constexpr uint32_t DUP_CONFIG_MODE_1C = 0x1C;
    static constexpr uint32_t DUP_CONFIG_4 = 0x4;
    static constexpr uint32_t DUP_FLAG_80 = 0x80;
    static constexpr uint32_t E1M2_SHIFT_LEFT_SIZE = 0x3;
    static constexpr uint32_t E1M2_AND_MASK = 0x8E;
    static constexpr uint32_t E2M1_SHIFT_LEFT_SIZE = 0x2;
    static constexpr uint32_t E2M1_AND_MASK = 0x9C;
    static constexpr uint32_t SHIFT_RIGHT_SIZE = 0x4;
    static constexpr int32_t VEC_MAX_ELEM_B8 = BYTE_PER_VECTOR_FRACTAL / sizeof(ElementOut);
    static constexpr int32_t VECTOR_REG_WIDTH_FOR_4BITS = 512;
    static constexpr int32_t OFFSET_64 = 64;
    static constexpr int32_t SYNC_MODE = 4;
    static constexpr uint16_t AIV_SYNC_AIC_FLAG = 0;

    uint64_t nSize_;
    uint64_t kSize_;
    int32_t nUbSize_;
    int32_t kUbSize_;
    int32_t nUbLen_;
    int32_t kUbLen_;
    int32_t nBiasUbLen_ = 0;
    uint64_t nL1Size_;
    uint64_t kL1Size_;
    uint64_t kGmOffset_;
    int32_t nL1Len_;
    int32_t kL1Len_;
    uint64_t aL1Size_;
    uint64_t bL1Size_;
    uint64_t vecWeightOutLen_;
    uint64_t vecWeightInLen_;
    uint64_t vecBiasLen_;
    uint64_t ubBufIdx_ = 0;
    int64_t l1BufIdx_ = 0;
    uint64_t nL1Offset_ = 0;
    uint64_t kL1Offset_ = 0;
    static constexpr int64_t UB_STAGES = L1B_STAGES;
    __ubuf__ ElementOut *weightOutUbAddr_;
    __ubuf__ ElementOut *weightOutUbAddr1_;
    __ubuf__ int8_t *weightInUbBaseAddr_;
    AscendC::LocalTensor<ElementIn> weightInUb_;
    AscendC::LocalTensor<ElementOut> weightOutUb_;
    AscendC::LocalTensor<ElementOut> l1Local_;
};
}  // namespace Catlass::Gemm::Block

#endif
