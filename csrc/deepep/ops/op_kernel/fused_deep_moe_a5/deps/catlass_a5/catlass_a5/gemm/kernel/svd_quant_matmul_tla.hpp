/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_SVD_QUANT_MATMUL_KERNEL_TLA_HPP
#define CATLASS_GEMM_SVD_QUANT_MATMUL_KERNEL_TLA_HPP

#include "catlass_a5/catlass.hpp"
#include "catlass_a5/arch/resource.hpp"
#include "catlass_a5/coord.hpp"
#include "catlass_a5/gemm_coord.hpp"
#include "catlass_a5/matrix_coord.hpp"
#include "catlass_a5/arch/cross_core_sync.hpp"
#include "tla_a5/layout.hpp"
#include "tla_a5/tensor.hpp"
#include "catlass_a5/detail/tag_to_layout.hpp"
#include "catlass_a5/gemm/tile/tile_copy.hpp"
#include "catlass_a5/epilogue/tile/tile_copy.hpp"

namespace Catlass::Gemm::Kernel {

#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)

template <
    class ArchTag_, class ElementIn_, class ElementSmoothScale_, class ElementQuantX_, class ElementMxScale_,
    class LayoutTag_, class L1TileShape_>
struct SmoothQuant {
    using ArchTag = ArchTag_;
    static_assert(std::is_same_v<ArchTag, Arch::Ascend950>, "unsupported ArchTag");
    using ElementIn = ElementIn_;
    using ElementSmoothScale = ElementSmoothScale_;
    using ElementMxScale = ElementMxScale_;
    using LayoutTagIn = LayoutTag_;
    using LayoutTagOut = LayoutTag_;
    using L1TileShape = L1TileShape_;

    using T = ElementIn;
    using U = ElementQuantX_;

    using LayoutIn = detail::TagToLayout_t<ElementIn, LayoutTagIn>;
    using TensorIn =
        tla::Tensor<AscendC::GlobalTensor<ElementIn>, LayoutIn, tla::Coord<tla::_0, tla::_0>, Arch::PositionGM{}>;
    using TensorUbIn =
        tla::Tensor<AscendC::LocalTensor<ElementIn>, LayoutIn, tla::Coord<tla::_0, tla::_0>, Arch::PositionUB{}>;

    using LayoutL1A = detail::TagToLayout_t<ElementIn, layout::zN>;
    using TensorL1A =
        tla::Tensor<AscendC::LocalTensor<ElementIn>, LayoutL1A, tla::Coord<tla::_0, tla::_0>, Arch::PositionL1{}>;
    using CopyUb2L1A = Epilogue::Tile::CopyUb2L1Tla<ArchTag, TensorUbIn, TensorL1A>;

    using LayoutSmoothScale = detail::TagToLayout_t<ElementIn, LayoutTagIn>;
    using LayoutQuantXV = detail::TagToLayout_t<int8_t, LayoutTagIn>;

    static constexpr bool HAS_SMOOTH = !std::is_void_v<ElementSmoothScale>;
    using GmSmoothType = std::conditional_t<HAS_SMOOTH, ElementSmoothScale, uint8_t>;

    static constexpr uint32_t STAGES = 2;
    static constexpr uint32_t AIV_NUM = 2;
    // 量化后的fp4_x2 shape为 row*(Cols/2), UB 上需要32B对齐，因此搬入需要64对齐
    static constexpr uint32_t UB_MX_SCALE_GROUP_NUM = 64;

    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});
    static constexpr auto L1A_LAYOUT = tla::MakeLayout<ElementIn, layout::zN>(L1_TILE_M, L1_TILE_K);

    CopyUb2L1A copyUb2L1A;

    CATLASS_DEVICE
    SmoothQuant(Arch::Resource<ArchTag>& resource)
    {
        static constexpr int8_t FLOAT_OVERFLOW_MODE_CTRL = 60; // 饱和模式控制位
        // <startBit, endBit>
        AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0); // 0: 单指令饱和 1: 全局饱和
    }

    CATLASS_DEVICE
    ~SmoothQuant()
    {}

    template <class TensorX, class TensorSmoothScale, class TensorQuantX, class TensorMxScale, class SmoothQuantParams>
    CATLASS_DEVICE void operator()(
        TensorX& gmTensorX, TensorSmoothScale& gmTensorSmoothScale, TensorQuantX& gmTensorQuantX,
        TensorMxScale& gmTensorMxScale, ElementIn qmaxInv, SmoothQuantParams& smoothQuantParams)
    {
        auto& l1ATensorList = smoothQuantParams.tensorLists.l1.l1A;
        auto& tensorLists = smoothQuantParams.tensorLists.ub;
        auto& eventLists = smoothQuantParams.eventLists;

        auto& xListId = smoothQuantParams.listId;

        uint32_t aivIdx = AscendC::GetSubBlockIdx();
        uint32_t aivNum = AscendC::GetSubBlockNum();

        uint32_t mBlockActual = tla::get<0>(gmTensorX.shape());
        uint32_t kBlockActual = tla::get<1>(gmTensorX.shape());
        uint32_t kL1Actual = min(kBlockActual, L1_TILE_K);
        uint32_t kAligned = RoundUp<UB_MX_SCALE_GROUP_NUM>(kL1Actual);
        bool isPad = (kL1Actual != kAligned);

        uint32_t mRowsPerCore = CeilDiv(mBlockActual, aivNum);
        uint32_t mL1Actual = (aivIdx < aivNum - 1) ? mRowsPerCore : (mBlockActual - (aivNum - 1) * mRowsPerCore);

        if (mL1Actual == 0) {
            return;
        }

        // Load first tileX
        auto gmTensorTileX =
            GetTile(gmTensorX, tla::MakeCoord(mRowsPerCore * aivIdx, 0), tla::MakeShape(mL1Actual, kL1Actual));
        auto layoutX = tla::MakeLayout<ElementIn, LayoutTagIn>(mL1Actual, kAligned);
        auto tensorX = tla::MakeTensor(tensorLists.x[xListId], layoutX, Arch::PositionUB{});
        auto tensorTileX = GetTile(tensorX, tla::MakeCoord(0, 0), tla::MakeShape(mL1Actual, kL1Actual));
        auto gmTensorSmoothTile = GetTile(gmTensorSmoothScale, tla::MakeCoord(0, 0), tla::MakeShape(1, kL1Actual));
        auto layoutSmooth = tla::MakeLayout<ElementIn, LayoutTagIn>(tla::Int<1>{}, kAligned);
        auto tensorSmoothScale = tla::MakeTensor(tensorLists.smoothScale[xListId], layoutSmooth, Arch::PositionUB{});
        auto tensorTileSmoothScale = GetTile(tensorSmoothScale, tla::MakeCoord(0, 0), tla::MakeShape(1, kL1Actual));

        if (isPad) {
            if constexpr (HAS_SMOOTH) {
                AscendC::Duplicate(tensorLists.x[xListId], static_cast<ElementIn>(0), mL1Actual * kAligned);
                AscendC::Duplicate(tensorLists.smoothScale[xListId], static_cast<ElementIn>(0), kAligned);
            } else {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventLists.pad);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventLists.pad);
                AscendC::Duplicate(tensorLists.x[xListId], static_cast<ElementIn>(0), mL1Actual * kAligned);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventLists.pad);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventLists.pad);
        }
        if constexpr (HAS_SMOOTH) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventLists.ubIn[xListId]);
            CopyGm2Ub(tensorTileX, gmTensorTileX, isPad);
            CopyGm2Ub(tensorTileSmoothScale, gmTensorSmoothTile, isPad);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventLists.ubIn[xListId]);
        } else {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventLists.ubIn[xListId]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventLists.ubIn[xListId]);
            CopyGm2Ub(tensorTileX, gmTensorTileX, isPad);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventLists.ubIn[xListId]);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventLists.ubIn[xListId]);
        }

        uint32_t kL1Loop = CeilDiv(kBlockActual, L1_TILE_K);
        for (uint32_t kL1Idx = 0; kL1Idx < kL1Loop; kL1Idx++) {
            uint32_t kL1Actual = (kL1Idx < kL1Loop - 1) ? L1_TILE_K : (kBlockActual - kL1Idx * L1_TILE_K);
            uint32_t kAligned = RoundUp<UB_MX_SCALE_GROUP_NUM>(kL1Actual);
            bool isPad = (kL1Actual != kAligned);

            uint32_t xListIdNext = (xListId + 1 < STAGES) ? (xListId + 1) : 0;
            if (kL1Idx < kL1Loop - 1) {
                uint32_t kL1IdxNext = kL1Idx + 1;
                uint32_t kL1ActualNext =
                    (kL1IdxNext < kL1Loop - 1) ? L1_TILE_K : (kBlockActual - kL1IdxNext * L1_TILE_K);
                uint32_t kAlignedNext = RoundUp<UB_MX_SCALE_GROUP_NUM>(kL1ActualNext);
                bool isPadNext = (kL1ActualNext != kAlignedNext);

                // X
                auto gmTensorTileXNext = GetTile(
                    gmTensorX, tla::MakeCoord(mRowsPerCore * aivIdx, L1_TILE_K * kL1IdxNext),
                    tla::MakeShape(mL1Actual, kL1ActualNext));
                auto layoutXNext = tla::MakeLayout<ElementIn, LayoutTagIn>(mL1Actual, kAlignedNext);
                auto tensorXNext = tla::MakeTensor(tensorLists.x[xListIdNext], layoutXNext, Arch::PositionUB{});
                auto tensorTileXNext =
                    GetTile(tensorXNext, tla::MakeCoord(0, 0), tla::MakeShape(mL1Actual, kL1ActualNext));
                // SmoothScale
                auto gmTensorSmoothTileNext = GetTile(
                    gmTensorSmoothScale, tla::MakeCoord(0, L1_TILE_K * kL1IdxNext), tla::MakeShape(1, kL1ActualNext));
                auto layoutSmoothNext = tla::MakeLayout<ElementIn, LayoutTagIn>(tla::Int<1>{}, kAlignedNext);
                auto tensorSmoothScaleNext =
                    tla::MakeTensor(tensorLists.smoothScale[xListIdNext], layoutSmoothNext, Arch::PositionUB{});
                auto tensorTileSmoothScaleNext =
                    GetTile(tensorSmoothScaleNext, tla::MakeCoord(0, 0), tla::MakeShape(1, kL1ActualNext));

                if (isPadNext) {
                    if constexpr (HAS_SMOOTH) {
                        AscendC::Duplicate(
                            tensorLists.x[xListIdNext], static_cast<ElementIn>(0), mL1Actual * kAlignedNext);
                        AscendC::Duplicate(
                            tensorLists.smoothScale[xListIdNext], static_cast<ElementIn>(0), kAlignedNext);
                    } else {
                        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventLists.pad);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventLists.pad);
                        AscendC::Duplicate(
                            tensorLists.x[xListIdNext], static_cast<ElementIn>(0), mL1Actual * kAlignedNext);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventLists.pad);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventLists.pad);
                }
                if constexpr (HAS_SMOOTH) {
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventLists.ubIn[xListIdNext]);
                    CopyGm2Ub(tensorTileXNext, gmTensorTileXNext, isPadNext);
                    CopyGm2Ub(tensorTileSmoothScaleNext, gmTensorSmoothTileNext, isPadNext);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventLists.ubIn[xListIdNext]);
                } else {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventLists.ubIn[xListIdNext]);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventLists.ubIn[xListIdNext]);
                    CopyGm2Ub(tensorTileXNext, gmTensorTileXNext, isPadNext);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventLists.ubIn[xListIdNext]);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventLists.ubIn[xListIdNext]);
                }
            }

            uint32_t realUbFactorDim0 = mL1Actual;
            uint32_t realUbFactorDim1 = CeilDiv<MX_SCALE_GROUP_NUM>(kAligned);
            uint32_t totalScaleInUB = realUbFactorDim0 * realUbFactorDim1;
            uint32_t totalCountInUB = realUbFactorDim0 * realUbFactorDim1 * MX_SCALE_GROUP_NUM;
            uint16_t loopNum = CeilDiv(totalCountInUB, ELE_NUM_REG_VF * DIGIT_TWO);
            uint16_t loopNumScale = CeilDiv<ELE_NUM_REG_VF>(totalScaleInUB);

            auto layoutX = tla::MakeLayout<ElementIn, LayoutTagIn>(mL1Actual, kAligned);
            auto tensorX = tla::MakeTensor(tensorLists.x[xListId], layoutX, Arch::PositionUB{});
            auto tensorTileX = GetTile(tensorX, tla::MakeCoord(0, 0), tla::MakeShape(mL1Actual, kL1Actual));
            auto l1ATensor = tla::MakeTensor(l1ATensorList[xListId], L1A_LAYOUT, Arch::PositionL1{});
            auto l1ATensorTile =
                GetTile(l1ATensor, tla::MakeCoord(mRowsPerCore * aivIdx, 0), tla::MakeShape(mL1Actual, kL1Actual));

            if constexpr (HAS_SMOOTH) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventLists.ubIn[xListId]);

                // compute X' = X * smooth
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventLists.ubIn[xListId]);
                ComputeSmoothX(
                    GetUbAddr(tensorLists.x[xListId]), GetUbAddr(tensorLists.smoothScale[xListId]),
                    GetUbAddr(tensorLists.smoothX[xListId]), realUbFactorDim0, kAligned);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventLists.ubIn[xListId]);

                auto tensorSmoothX = tla::MakeTensor(tensorLists.smoothX[xListId], layoutX, Arch::PositionUB{});
                auto tensorTileSmoothX =
                    GetTile(tensorSmoothX, tla::MakeCoord(0, 0), tla::MakeShape(mL1Actual, kL1Actual));

                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventLists.ubIn[xListId]);

                // copy ub to l1
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventLists.ubIn[xListId]);
                AscendC::CrossCoreWaitFlag<0x4, PIPE_MTE3>(eventLists.flagAicFinishMte1[0][xListId]);
                copyUb2L1A(l1ATensorTile, tensorTileSmoothX);
                AscendC::CrossCoreSetFlag<0x4, PIPE_MTE3>(eventLists.flagAivFinishMte3[0][xListId]);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventLists.ubIn[xListId]);

                // compute MaxExp
                ComputeMaxExp(
                    GetUbAddr(tensorLists.smoothX[xListId]), GetUbAddr(tensorLists.maxExp), totalCountInUB, loopNum,
                    qmaxInv);
            } else {
                // copy ub to l1
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventLists.ubIn[xListId]);
                AscendC::CrossCoreWaitFlag<0x4, PIPE_MTE3>(eventLists.flagAicFinishMte1[0][xListId]);
                copyUb2L1A(l1ATensorTile, tensorTileX);
                AscendC::CrossCoreSetFlag<0x4, PIPE_MTE3>(eventLists.flagAivFinishMte3[0][xListId]);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventLists.ubIn[xListId]);

                // compute MaxExp
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventLists.ubIn[xListId]);
                ComputeMaxExp(
                    GetUbAddr(tensorLists.x[xListId]), GetUbAddr(tensorLists.maxExp), totalCountInUB, loopNum, qmaxInv);
            }

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventLists.ubOutMxScale[xListId]);
            ComputeScale(
                GetUbAddr(tensorLists.maxExp), GetUbAddr<uint16_t>(tensorLists.mxScale[xListId]),
                GetUbAddr(tensorLists.halfScale), totalScaleInUB, loopNumScale);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventLists.ubOutMxScale[xListId]);

            // copyOut MxScale
            auto gmTensorTileMxScale = GetTile(
                gmTensorMxScale, tla::MakeCoord(mRowsPerCore * aivIdx, CeilDiv<MX_SCALE_GROUP_NUM>(kL1Idx * L1_TILE_K)),
                tla::MakeShape(mL1Actual, RoundUp<2>(realUbFactorDim1)));
            auto layoutMxScale = tla::MakeLayout<uint8_t, LayoutTagIn>(mL1Actual, realUbFactorDim1);
            auto tensorMxScaleX = tla::MakeTensor(tensorLists.mxScale[xListId], layoutMxScale, Arch::PositionUB{});
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventLists.ubOutMxScale[xListId]);
            CopyOutMxScale(gmTensorTileMxScale, tensorMxScaleX);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventLists.ubOutMxScale[xListId]);

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventLists.ubOutQuntX[xListId]);
            if constexpr (HAS_SMOOTH) {
                ComputeFp4x2<AscendC::RoundMode::CAST_TRUNC, AscendC::RoundMode::CAST_RINT>(
                    GetUbAddr(tensorLists.smoothX[xListId]), GetUbAddr(tensorLists.halfScale),
                    GetUbAddr(tensorLists.quantX[xListId]), totalCountInUB, loopNum);
            } else {
                ComputeFp4x2<AscendC::RoundMode::CAST_TRUNC, AscendC::RoundMode::CAST_RINT>(
                    GetUbAddr(tensorLists.x[xListId]), GetUbAddr(tensorLists.halfScale),
                    GetUbAddr(tensorLists.quantX[xListId]), totalCountInUB, loopNum);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventLists.ubIn[xListId]);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventLists.ubOutQuntX[xListId]);

            // CopyOut QuantX
            auto gmTensorTileQuantX = GetTile(
                gmTensorQuantX, tla::MakeCoord(mRowsPerCore * aivIdx, kL1Idx * L1_TILE_K / DIGIT_TWO),
                tla::MakeShape(mL1Actual, CeilDiv<2>(kL1Actual)));
            auto layoutQuantX = tla::MakeLayout<int8_t, LayoutTagIn>(mL1Actual, CeilDiv<2>(kAligned));
            auto tensorQuantX = tla::MakeTensor(tensorLists.quantX[xListId], layoutQuantX, Arch::PositionUB{});
            auto tensorTileQuantX =
                GetTile(tensorQuantX, tla::MakeCoord(0, 0), tla::MakeShape(mL1Actual, CeilDiv<2>(kL1Actual)));
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventLists.ubOutQuntX[xListId]);
            CopyOutFp4x2(gmTensorTileQuantX, tensorTileQuantX);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventLists.ubOutQuntX[xListId]);

            xListId = xListIdNext;
        }
    }

    template <class DstElement = void, class BuiltinTensor = EmptyClass>
    CATLASS_DEVICE auto GetUbAddr(BuiltinTensor const& tensor)
    {
        static_assert(!std::is_same_v<BuiltinTensor, EmptyClass>, "tensor is not a BuiltinTensor");
        using Element = std::conditional_t<std::is_void_v<DstElement>, typename BuiltinTensor::PrimType, DstElement>;
        return reinterpret_cast<__ubuf__ Element*>(tensor.GetPhyAddr());
    }

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void CopyGm2Ub(TensorDst const& dstTensor, TensorSrc const& srcTensor, const bool isPad = false)
    {
        using ElementSrc = ElementIn;
        static constexpr uint32_t ELE_NUM_PER_BLK = BYTE_PER_BLK / sizeof(ElementSrc);

        AscendC::DataCopyExtParams dataCopyParams(
            tla::get<0>(srcTensor.shape()), tla::get<1>(srcTensor.shape()) * sizeof(ElementSrc),
            (tla::get<0>(srcTensor.stride()) - tla::get<1>(srcTensor.shape())) * sizeof(ElementSrc),
            (tla::get<0>(dstTensor.stride()) - tla::get<1>(dstTensor.shape())) / ELE_NUM_PER_BLK, 0);
        AscendC::DataCopyPadExtParams<ElementSrc> padParams(false, 0, 0, 0);
        /*
        搬入在col方向需要64元素对齐，在非对齐的shape下需要padding 0。 在非对齐的场景下，在搬入之前已经对这块空间填充0，
        但UB上写入单位为一个datablock=32B，如果搬入不填0，datablock内剩下的空间仍然会出现脏数据，导致量化结果出错。
        padding规则要求 (leftPadding + rightPadding) <= 32B， 左右填充的单位是元素个数
        SmoothQuant申请的UB空间是32B对齐的，leftPadding=0，只需要考虑rihgtPadding
        对齐情况有以下几种(一个datablock中有16个fp16元素)：
        |    16    |    16    |    16    |    16    | cols=64, isPad=false
        |    16    |    16    |     0    |     0    | cols=48, isPad=true, unalignedLen=0, 实际不padding
        |    16    | 10 | pad |     0    |     0    | cols=26, isPad=true, unalignedLen=10, 需要padding，rightPadding=6
        */
        if (isPad) {
            uint32_t unalignedLen = tla::get<1>(srcTensor.shape()) % ELE_NUM_PER_BLK;
            if (unalignedLen > 0) {
                padParams.isPad = true;
                padParams.rightPadding = ELE_NUM_PER_BLK - unalignedLen;
                padParams.paddingValue = static_cast<ElementSrc>(0);
            }
        }
        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());
        AscendC::DataCopyPad(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], dataCopyParams, padParams);
    }

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void CopyOutMxScale(TensorDst const& dstTensor, TensorSrc const& srcTensor)
    {
        using ElementSrc = uint8_t;
        static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementSrc);
        // gmMxScale tensor layout is 3-dim
        AscendC::DataCopyExtParams dataCopyParams(
            tla::get<0>(dstTensor.shape()),
            tla::get<1, 0>(dstTensor.shape()) * tla::get<1, 1>(dstTensor.shape()) * sizeof(ElementSrc),
            (tla::get<0>(srcTensor.stride()) - tla::get<1>(srcTensor.shape())) / ELE_NUM_PER_C0,
            (tla::get<0>(dstTensor.stride()) -
             (tla::get<1, 0>(dstTensor.shape()) * tla::get<1, 1>(dstTensor.shape()))) *
                sizeof(ElementSrc),
            0);
        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());
        AscendC::DataCopyPad<ElementSrc, AscendC::PaddingMode::Compact>(
            dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], dataCopyParams);
    }

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void CopyOutFp4x2(TensorDst const& dstTensor, TensorSrc const& srcTensor)
    {
        using ElementSrc = int8_t;
        static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementSrc);

        AscendC::DataCopyExtParams dataCopyParams(
            tla::get<0>(dstTensor.shape()), tla::get<1>(dstTensor.shape()) * sizeof(ElementSrc),
            (tla::get<0>(srcTensor.stride()) - tla::get<1>(srcTensor.shape())) / ELE_NUM_PER_C0,
            (tla::get<0>(dstTensor.stride()) - tla::get<1>(dstTensor.shape())) * sizeof(ElementSrc), 0);
        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());
        AscendC::DataCopyPad(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], dataCopyParams);
    }

    __simd_vf__ inline void ComputeSmoothX(
        __ubuf__ T* srcAddr, __ubuf__ T* srcSmoothAddr, __ubuf__ T* smoothXAddr, uint16_t dim0, uint16_t dim1)
    {
        using namespace AscendC::MicroAPI;
        RegTensor<T> vdX;
        RegTensor<T> vdSmooth;
        RegTensor<T> vdXdst;
        MaskReg mulMask;
        uint16_t loopNum = CeilDiv<ELE_NUM_REG_VF>(dim1);
        for (uint16_t i = 0; i < dim0; i++) {
            uint32_t calNum = dim1;
            __ubuf__ T* srcAddr1 = srcAddr + i * dim1;
            __ubuf__ T* srcSmoothAddr1 = srcSmoothAddr;
            for (uint16_t j = 0; j < loopNum; j++) {
                mulMask = UpdateMask<T>(calNum);
                DataCopy<T, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_NORM>(vdX, srcAddr1, ELE_NUM_REG_VF);
                DataCopy<T, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_NORM>(
                    vdSmooth, srcSmoothAddr1, ELE_NUM_REG_VF);
                Mul(vdXdst, vdX, vdSmooth, mulMask);
                StoreAlign(smoothXAddr + i * dim1 + ELE_NUM_REG_VF * j, vdXdst, mulMask);
            }
        }
    }

    __simd_vf__ inline void ComputeMaxExp(
        __ubuf__ T* srcAddr, __ubuf__ uint16_t* maxExpAddr, uint32_t totalCountInUB, uint16_t loopNum, T qmaxInv)
    {
        using namespace AscendC::MicroAPI;
        RegTensor<T> vdExp0;
        RegTensor<T> vdExp1;
        RegTensor<bfloat16_t> vdExp0BF16;
        RegTensor<bfloat16_t> vdExp1BF16;
        RegTensor<uint16_t> vdExpExtract0;
        RegTensor<uint16_t> vdExpExtract1;
        RegTensor<uint16_t> vdExpSelect0;
        RegTensor<uint16_t> vdExpSelect1;
        RegTensor<uint16_t> expMaskBF16;
        Duplicate(expMaskBF16, MAX_EXP_FOR_BF16);
        RegTensor<uint16_t> invalidmaskfp16;
        Duplicate(invalidmaskfp16, INVALID_FLOAT16);
        RegTensor<uint16_t> vdMaxExp;
        MaskReg scaleMask1;
        // MaskReg scaleMask2;
        MaskReg invalidDataMask0;
        MaskReg invalidDataMask1;
        UnalignReg u1;
        static constexpr CastTrait castTraitHalf2Bf16 = {
            RegLayout::UNKNOWN, SatMode::UNKNOWN, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_TRUNC};

        RegTensor<uint16_t> mantissaMaskHalf;
        RegTensor<uint16_t> mantissaMaskBf16;
        RegTensor<uint16_t> mantissaZero;
        RegTensor<uint16_t> CeilAdd;
        Duplicate(mantissaMaskBf16, 0x007f); // bf16 类型的尾数掩码
        Duplicate(mantissaMaskHalf, 0x03ff); // half 类型的尾数掩码
        Duplicate(mantissaZero, 0);
        Duplicate(CeilAdd, 128);
        RegTensor<uint16_t> vdMantissa0;
        RegTensor<uint16_t> vdMantissa1;
        RegTensor<uint16_t> vdExpExtractCeil0;
        RegTensor<uint16_t> vdExpExtractCeil1;
        MaskReg hasMantissa0;
        MaskReg hasMantissa1;

        RegTensor<T> vdQmax;
        Duplicate(vdQmax, qmaxInv);

        for (uint16_t i = 0; i < loopNum; i++) {
            scaleMask1 = UpdateMask<T>(totalCountInUB);
            // scaleMask2 = UpdateMask<T>(totalCountInUB);
            DataCopy<T, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_DINTLV_B16>(
                vdExp0, vdExp1, srcAddr, ELE_NUM_REG_VF * DIGIT_TWO);

            // v = 2^(x-127)*(1+尾数/128)
            // log2(v) = x-127
            // x = 指数部分的二进制值
            if constexpr (AscendC::Std::IsSame<T, half>::value) {
                Mul(vdExp0, vdExp0, vdQmax, scaleMask1);
                Mul(vdExp1, vdExp1, vdQmax, scaleMask1);
                // 提取指数部分
                And(vdExpSelect0, (RegTensor<uint16_t>&)vdExp0, invalidmaskfp16, scaleMask1);
                And(vdExpSelect1, (RegTensor<uint16_t>&)vdExp1, invalidmaskfp16, scaleMask1);
                // 新增：提取尾数部分
                And(vdMantissa0, (RegTensor<uint16_t>&)vdExp0, mantissaMaskHalf, scaleMask1);
                And(vdMantissa1, (RegTensor<uint16_t>&)vdExp1, mantissaMaskHalf, scaleMask1);
                // 指数部分与无效值比较，不同的位取1
                Compare<uint16_t, AscendC::CMPMODE::NE>(invalidDataMask0, vdExpSelect0, invalidmaskfp16, scaleMask1);
                Compare<uint16_t, AscendC::CMPMODE::NE>(invalidDataMask1, vdExpSelect1, invalidmaskfp16, scaleMask1);
                Cast<bfloat16_t, T, castTraitHalf2Bf16>(vdExp0BF16, vdExp0, scaleMask1);
                Cast<bfloat16_t, T, castTraitHalf2Bf16>(vdExp1BF16, vdExp1, scaleMask1);
                // 提取bf16的指数部分
                And(vdExpExtract0, (RegTensor<uint16_t>&)vdExp0BF16, expMaskBF16, scaleMask1);
                And(vdExpExtract1, (RegTensor<uint16_t>&)vdExp1BF16, expMaskBF16, scaleMask1);
                Select<uint16_t>(vdExpExtract0, vdExpExtract0, expMaskBF16, invalidDataMask0);
                Select<uint16_t>(vdExpExtract1, vdExpExtract1, expMaskBF16, invalidDataMask1);
            } else {
                Mul(vdExp0, vdExp0, vdQmax, scaleMask1);
                Mul(vdExp1, vdExp1, vdQmax, scaleMask1);
                And(vdExpExtract0, (RegTensor<uint16_t>&)vdExp0, expMaskBF16, scaleMask1);
                And(vdExpExtract1, (RegTensor<uint16_t>&)vdExp1, expMaskBF16, scaleMask1);

                // 新增：提取尾数部分
                And(vdMantissa0, (RegTensor<uint16_t>&)vdExp0, mantissaMaskBf16, scaleMask1);
                And(vdMantissa1, (RegTensor<uint16_t>&)vdExp1, mantissaMaskBf16, scaleMask1);
            }
            // 尾数部分若不为0,为指数加一,实现向上取整
            Compare<uint16_t, AscendC::CMPMODE::NE>(hasMantissa0, vdMantissa0, mantissaZero, scaleMask1);
            Compare<uint16_t, AscendC::CMPMODE::NE>(hasMantissa1, vdMantissa1, mantissaZero, scaleMask1);
            Add(vdExpExtractCeil0, vdExpExtract0, CeilAdd, scaleMask1);
            Add(vdExpExtractCeil1, vdExpExtract1, CeilAdd, scaleMask1);
            Select<uint16_t>(vdExpExtract0, vdExpExtractCeil0, vdExpExtract0, hasMantissa0);
            Select<uint16_t>(vdExpExtract1, vdExpExtractCeil1, vdExpExtract1, hasMantissa1);

            Max(vdMaxExp, vdExpExtract0, vdExpExtract1, scaleMask1);
            ReduceMaxWithDataBlock(vdMaxExp, vdMaxExp, scaleMask1);

            DataCopyUnAlign<uint16_t, PostLiteral::POST_MODE_UPDATE>(maxExpAddr, vdMaxExp, u1, ELE_AFTER_REDUCE);
        }
        DataCopyUnAlignPost(maxExpAddr, u1, 0);
    }

    __simd_vf__ inline void ComputeScale(
        __ubuf__ uint16_t* maxExpAddr, __ubuf__ uint16_t* mxScaleLocalAddr, __ubuf__ uint16_t* halfScaleLocalAddr,
        uint32_t totalScaleInUB, uint16_t loopNumScale)
    {
        using namespace AscendC::MicroAPI;
        RegTensor<uint16_t> expMask;
        Duplicate(expMask, MAX_EXP_FOR_BF16);
        RegTensor<uint16_t> vdMaxExp;

        RegTensor<T> vdExp0;
        RegTensor<T> vdExp1;

        MaskReg cmpResult;
        MaskReg zeroMask;
        MaskReg cmpResultSub;
        MaskReg preMaskScale;
        RegTensor<uint16_t> maxExpValue;
        Duplicate(maxExpValue, FP4_EMAX);
        RegTensor<uint16_t> sharedExp;
        RegTensor<uint16_t> scaleValue;
        RegTensor<uint16_t> scaleBias;
        Duplicate(scaleBias, BF16_EXP_BIAS);
        RegTensor<uint16_t> halfScale;
        RegTensor<uint16_t> fp8NanRegTensor;
        Duplicate(fp8NanRegTensor, MAX_EXP_FOR_FP8);
        RegTensor<uint16_t> zeroRegTensor;
        Duplicate(zeroRegTensor, 0);
        RegTensor<uint16_t> nanRegTensor;
        Duplicate(nanRegTensor, NAN_CUSTOMIZATION);
        MaskReg invalidDataMask;
        MaskReg specialDataMask;
        RegTensor<uint16_t> specialExpRegTensor;
        Duplicate(specialExpRegTensor, SPECIAL_EXP_THRESHOLD);
        for (uint16_t i = 0; i < loopNumScale; i++) {
            preMaskScale = UpdateMask<uint16_t>(totalScaleInUB);
            DataCopy<uint16_t, PostLiteral::POST_MODE_UPDATE>(vdMaxExp, maxExpAddr, ELE_NUM_REG_VF);
            Compare<uint16_t, AscendC::CMPMODE::NE>(cmpResult, vdMaxExp, expMask, preMaskScale); // INF/NAN
            Compare<uint16_t, AscendC::CMPMODE::NE>(zeroMask, vdMaxExp, zeroRegTensor, preMaskScale);
            Compare<uint16_t, AscendC::CMPMODE::LE>(invalidDataMask, vdMaxExp, maxExpValue, preMaskScale);

            Select<uint16_t>(sharedExp, maxExpValue, vdMaxExp, invalidDataMask);

            ShiftRights(scaleValue, sharedExp, SHR_NUM_FOR_BF16, preMaskScale);

            Select<uint16_t>(scaleValue, scaleValue, fp8NanRegTensor, cmpResult);
            Select<uint16_t>(scaleValue, scaleValue, zeroRegTensor, zeroMask);

            DataCopy<uint16_t, PostLiteral::POST_MODE_UPDATE, StoreDist::DIST_PACK_B16>(
                mxScaleLocalAddr, scaleValue, ELE_NUM_REG_VF / DIGIT_TWO, preMaskScale);

            Compare<uint16_t, AscendC::CMPMODE::EQ>(specialDataMask, sharedExp, scaleBias, preMaskScale);
            Sub(halfScale, scaleBias, sharedExp, preMaskScale);
            Select<uint16_t>(halfScale, halfScale, nanRegTensor, cmpResult);
            Select<uint16_t>(halfScale, halfScale, zeroRegTensor, zeroMask);
            Select<uint16_t>(halfScale, specialExpRegTensor, halfScale, specialDataMask);

            DataCopy<uint16_t, PostLiteral::POST_MODE_UPDATE>(
                halfScaleLocalAddr, halfScale, ELE_NUM_REG_VF, preMaskScale);
        }
    }

    template <AscendC::RoundMode toBf16RoundMode, AscendC::RoundMode roundMode>
    __simd_vf__ inline void ComputeFp4x2(
        __ubuf__ T* srcAddr, __ubuf__ uint16_t* halfScaleLocalAddr, __ubuf__ int8_t* outLocalAddr,
        uint32_t totalCountInUB, uint16_t loopNum)
    {
        using namespace AscendC::MicroAPI;
        MaskReg dataMask1;
        RegTensor<uint16_t> halfScaleForMul;
        RegTensor<T> vdExp0;
        RegTensor<T> vdExp1;
        RegTensor<T> vdExp0Convert;
        RegTensor<T> vdExp1Convert;

        RegTensor<bfloat16_t> vdExp0BF16;
        RegTensor<bfloat16_t> vdExp1BF16;

        RegTensor<U> vdExp0FP4;
        RegTensor<U> vdExp1FP4;

        RegTensor<bfloat16_t> vdBF16Exp0FP4;
        RegTensor<bfloat16_t> vdBF16Exp1FP4;
        static constexpr CastTrait castTrait = {RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING, roundMode};
        static constexpr CastTrait castTraitHalf2Bf16 = {
            RegLayout::UNKNOWN, SatMode::UNKNOWN, MaskMergeMode::ZEROING, toBf16RoundMode};
        for (uint16_t i = 0; i < loopNum; i++) {
            dataMask1 = UpdateMask<T>(totalCountInUB);
            DataCopy<T, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_DINTLV_B16>(
                vdExp0, vdExp1, srcAddr, ELE_NUM_REG_VF * DIGIT_TWO);
            DataCopy<uint16_t, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_E2B_B16>(
                halfScaleForMul, halfScaleLocalAddr, ELE_AFTER_REDUCE);

            if constexpr (AscendC::Std::IsSame<T, half>::value) {
                if constexpr (roundMode == AscendC::RoundMode::CAST_RINT) {
                    FP16Convert(vdExp0, vdExp0, dataMask1);
                    FP16Convert(vdExp1, vdExp1, dataMask1);
                }
                Cast<bfloat16_t, T, castTraitHalf2Bf16>(vdExp0BF16, vdExp0, dataMask1);
                Cast<bfloat16_t, T, castTraitHalf2Bf16>(vdExp1BF16, vdExp1, dataMask1);
                Mul(vdExp0BF16, vdExp0BF16, (RegTensor<bfloat16_t>&)halfScaleForMul, dataMask1);
                Mul(vdExp1BF16, vdExp1BF16, (RegTensor<bfloat16_t>&)halfScaleForMul, dataMask1);
                Interleave(vdExp0BF16, vdExp1BF16, vdExp0BF16, vdExp1BF16);
                Cast<U, bfloat16_t, castTrait>(vdExp0FP4, vdExp0BF16, dataMask1);
                Cast<U, bfloat16_t, castTrait>(vdExp1FP4, vdExp1BF16, dataMask1);

            } else {
                Mul(vdExp0, vdExp0, (RegTensor<T>&)halfScaleForMul, dataMask1);
                Mul(vdExp1, vdExp1, (RegTensor<T>&)halfScaleForMul, dataMask1);
                Interleave(vdExp0, vdExp1, vdExp0, vdExp1);
                Cast<U, T, castTrait>(vdExp0FP4, vdExp0, dataMask1);
                Cast<U, T, castTrait>(vdExp1FP4, vdExp1, dataMask1);
            }
            DataCopy<int8_t, PostLiteral::POST_MODE_UPDATE, StoreDist::DIST_PACK4_B32>(
                outLocalAddr, (RegTensor<int8_t>&)vdExp0FP4, OUT_ELE_NUM_ONE_BLK, dataMask1);
            DataCopy<int8_t, PostLiteral::POST_MODE_UPDATE, StoreDist::DIST_PACK4_B32>(
                outLocalAddr, (RegTensor<int8_t>&)vdExp1FP4, OUT_ELE_NUM_ONE_BLK, dataMask1);
        }
    }

    template <class HalfReg, class MaskReg>
    __simd_callee__ inline void FP16Convert(HalfReg& output, HalfReg& input, MaskReg& mask)
    {
        using namespace AscendC::MicroAPI;
        RegTensor<uint16_t> specialValueTensor;
        RegTensor<uint16_t> newMantissa;
        RegTensor<uint16_t> andResult;
        RegTensor<uint16_t> newValue;
        MaskReg specialMask;
        MaskReg nonzeroMask;
        uint16_t specialValue = SPECIAL_VALUE_E1M2;
        if constexpr (AscendC::Std::IsSame<U, fp4x2_e2m1_t>::value) {
            specialValue = SPECIAL_VALUE_E2M1;
        }
        // specialValue= 0000 0000 1111 1111
        // NEW_MANTISSA = 0000 0000 0000 1000
        Duplicate(specialValueTensor, specialValue);
        Duplicate(newMantissa, NEW_MANTISSA);
        // andResult: half的后八位
        And(andResult, (RegTensor<uint16_t>&)input, specialValueTensor, mask);
        CompareScalar<uint16_t, AscendC::CMPMODE::GT>(nonzeroMask, andResult, 0, mask);
        CompareScalar<uint16_t, AscendC::CMPMODE::LT>(specialMask, andResult, NEW_MANTISSA, mask);
        MaskAnd(specialMask, specialMask, nonzeroMask, mask);
        Or(newValue, (RegTensor<uint16_t>&)input, newMantissa, mask);
        Select<uint16_t>((RegTensor<uint16_t>&)output, newValue, (RegTensor<uint16_t>&)input, specialMask);
    }

private:
    static constexpr uint32_t VECTOR_REG_WIDTH = 256;
    static constexpr uint32_t ELE_NUM_REG_VF = VECTOR_REG_WIDTH / sizeof(ElementIn);
    static constexpr uint16_t ELE_AFTER_REDUCE = VECTOR_REG_WIDTH / BYTE_PER_BLK;
    static constexpr int64_t ELE_NUM_PER_BLK = BYTE_PER_BLK / sizeof(ElementIn);

    static constexpr int64_t DIGIT_TWO = 2;
    static constexpr uint16_t MAX_EXP_FOR_BF16 = 0x7f80;
    static constexpr uint16_t INVALID_FLOAT16 = 0x7c00;
    static constexpr uint16_t BF16_EXP_BIAS = 0x7f00;
    static constexpr uint16_t MAX_EXP_FOR_FP8 = 0x00ff;
    static constexpr uint16_t NAN_CUSTOMIZATION = 0x7f81;
    static constexpr uint16_t SPECIAL_EXP_THRESHOLD = 0x0040;
    static constexpr int16_t SHR_NUM_FOR_BF16 = 7;
    static constexpr int64_t OUT_ELE_NUM_ONE_BLK = 64;
    static constexpr uint16_t FP4_E2M1_BF16_MAX_EXP = 0x0100;
    static constexpr uint16_t FP4_E1M2_MAX_EXP = 0x0000;
    static constexpr uint16_t SPECIAL_VALUE_E1M2 = 0x007f;
    static constexpr uint16_t SPECIAL_VALUE_E2M1 = 0x00ff;
    static constexpr uint16_t NEW_MANTISSA = 0x0008;
    static constexpr uint16_t FP4_EMAX = std::is_same_v<U, fp4x2_e2m1_t> ? FP4_E2M1_BF16_MAX_EXP : FP4_E1M2_MAX_EXP;
};

template <
    class SmoothQuant_, class BlockMmad1_, class BlockMmad2_, class BlockMmad3_, class BlockEpilogue_,
    class BlockScheduler_>
class SvdQuantMatmulTla {
public:
    using SmoothQuant = SmoothQuant_;
    using BlockMmad1 = BlockMmad1_;
    using BlockMmad2 = BlockMmad2_;
    using BlockMmad3 = BlockMmad3_;
    using BlockEpilogue = BlockEpilogue_;
    // Check ArchTag
    using ArchTag = typename BlockMmad1::ArchTag;
    static_assert(
        std::is_same_v<ArchTag, typename BlockMmad2::ArchTag>, "BlockMmad1 and BlockMmad2 is not the same ArchTag");
    static_assert(
        std::is_same_v<ArchTag, typename BlockMmad3::ArchTag>, "BlockMmad1 and BlockMmad3 is not the same ArchTag");

    using ElementX = typename BlockMmad1::ElementA;
    using LayoutX = typename BlockMmad1::LayoutA;
    using GmSmoothType = typename SmoothQuant::GmSmoothType;
    using LayoutSmoothScale = typename SmoothQuant::LayoutSmoothScale;
    using ElementSvd1 = typename BlockMmad1::ElementB;
    using LayoutSvd1 = typename BlockMmad1::LayoutB;
    using ElementSvd2 = typename BlockMmad2::ElementB;
    using LayoutSvd2 = typename BlockMmad2::LayoutB;
    using ElementW = typename BlockMmad3::ElementB;
    using LayoutW = typename BlockMmad3::LayoutB;
    using ElementMxScaleW = typename BlockMmad3::TileCopy::ElementMxScaleB;
    using LayoutMxScaleW = typename BlockMmad3::TileCopy::LayoutMxScaleB;

    static_assert(
        std::is_same_v<typename BlockMmad1::ElementC, typename BlockMmad2::ElementA>,
        "Element of C1 and A2 is not same");
    static_assert(
        std::is_same_v<typename BlockMmad1::LayoutC, typename BlockMmad2::LayoutA>, "Layout of C1 and A2 is not same");
    using ElementC1 = typename BlockMmad1::ElementC;
    using LayoutC1 = typename BlockMmad1::LayoutC;

    using ElementQuantX = typename BlockMmad3::ElementA;
    using LayoutQuantXC = typename BlockMmad3::LayoutA;
    using LayoutQuantXV = typename SmoothQuant::LayoutQuantXV;
    using ElementMxScaleX = uint8_t;
    using LayoutMxScaleX = typename BlockMmad3::TileCopy::LayoutMxScaleA;

    using ElementY = typename BlockMmad2::ElementC;
    using LayoutY = typename BlockMmad2::LayoutC;

    using BlockScheduler = BlockScheduler_;

    /// Parameters structure
    struct Params {
        // Data members
        GemmCoord problemShape;
        uint32_t problemRank;
        float qmax;
        GM_ADDR ptrX;
        LayoutX layoutX;
        GM_ADDR ptrSmoothScale;
        LayoutSmoothScale layoutSmoothScale;
        GM_ADDR ptrSvd1;
        LayoutSvd1 layoutSvd1;
        GM_ADDR ptrSvd2;
        LayoutSvd2 layoutSvd2;
        GM_ADDR ptrW;
        LayoutW layoutW;
        GM_ADDR ptrMxScaleW;
        LayoutMxScaleW layoutMxScaleW;
        GM_ADDR ptrBias;
        GM_ADDR ptrC1;
        LayoutC1 layoutC1;
        GM_ADDR ptrQuantX;
        LayoutQuantXV layoutQuantXV;
        LayoutQuantXC layoutQuantXC;
        GM_ADDR ptrMxScaleX;
        LayoutMxScaleX layoutMxScaleX;
        GM_ADDR ptrY;
        LayoutY layoutY;

        // Methods
        CATLASS_HOST_DEVICE
        Params()
        {}

        CATLASS_HOST_DEVICE
        Params(
            GemmCoord const& problemShape_, uint32_t problemRank_, float qmax_, GM_ADDR ptrX_, LayoutX layoutX_,
            GM_ADDR ptrSmoothScale_, LayoutSmoothScale layoutSmoothScale_, GM_ADDR ptrSvd1_, LayoutSvd1 layoutSvd1_,
            GM_ADDR ptrSvd2_, LayoutSvd2 layoutSvd2_, GM_ADDR ptrW_, LayoutW layoutW_, GM_ADDR ptrMxScaleW_,
            LayoutMxScaleW layoutMxScaleW_, GM_ADDR ptrBias_, GM_ADDR ptrC1_, LayoutC1 layoutC1_, GM_ADDR ptrQuantX_,
            LayoutQuantXV layoutQuantXV_, LayoutQuantXC layoutQuantXC_, GM_ADDR ptrMxScaleX_,
            LayoutMxScaleX layoutMxScaleX_, GM_ADDR ptrY_, LayoutY layoutY_)
            : problemShape(problemShape_),
              problemRank(problemRank_),
              qmax(qmax_),
              ptrX(ptrX_),
              layoutX(layoutX_),
              ptrSmoothScale(ptrSmoothScale_),
              layoutSmoothScale(layoutSmoothScale_),
              ptrSvd1(ptrSvd1_),
              layoutSvd1(layoutSvd1_),
              ptrSvd2(ptrSvd2_),
              layoutSvd2(layoutSvd2_),
              ptrW(ptrW_),
              layoutW(layoutW_),
              ptrMxScaleW(ptrMxScaleW_),
              layoutMxScaleW(layoutMxScaleW_),
              ptrBias(ptrBias_),
              ptrC1(ptrC1_),
              layoutC1(layoutC1_),
              ptrQuantX(ptrQuantX_),
              layoutQuantXV(layoutQuantXV_),
              layoutQuantXC(layoutQuantXC_),
              ptrMxScaleX(ptrMxScaleX_),
              layoutMxScaleX(layoutMxScaleX_),
              ptrY(ptrY_),
              layoutY(layoutY_)
        {}
    };

    struct Arguments {
        GemmCoord problemShape;
        uint32_t problemRank;
        float qmax;
        uint8_t* ptrX;
        LayoutX layoutX;
        uint8_t* ptrSvd1;
        LayoutSvd1 layoutSvd1;
        uint8_t* ptrSvd2;
        LayoutSvd1 layoutSvd2;
        uint8_t* ptrW;
        LayoutW layoutW;
        uint8_t* ptrMxScaleW;
        LayoutMxScaleW layoutMxScaleW;
        uint8_t* ptrSmoothScale;
        LayoutSmoothScale layoutSmoothScale;
        uint8_t* ptrBias;
        uint8_t* ptrY;
        LayoutY layoutY;
    };

    static bool CanImplement(const Arguments& args)
    {
        return args.problemRank <= BlockMmad1::L1_TILE_N;
    }

    static size_t GetWorkspaceSize(const Arguments& args)
    {
        size_t sizeC1 = args.problemShape.m() * args.problemRank * sizeof(typename SmoothQuant::ElementIn);
        size_t sizeQuantX = args.problemShape.m() * CeilDiv<2>(args.problemShape.k());
        size_t sizeMxScaleX = args.problemShape.m() * RoundUp<2>(CeilDiv<MX_SCALE_GROUP_NUM>(args.problemShape.k()));
        return sizeC1 + sizeQuantX + sizeMxScaleX;
    }

    static Params ToUnderlyingArguments(const Arguments& args, uint8_t* workspace)
    {
        uint32_t m = args.problemShape.m(), k = args.problemShape.k(), r = args.problemRank;
        uint32_t mxScaleK = CeilDiv<MX_SCALE_GROUP_NUM>(k);

        size_t sizeC1 = m * r * sizeof(typename SmoothQuant::ElementIn);
        size_t sizeQuantX = m * CeilDiv<2>(k);
        size_t sizeMxScaleX = m * RoundUp<2>(CeilDiv<MX_SCALE_GROUP_NUM>(k));
        uint8_t* gmC1 = workspace;
        uint8_t* gmQuantX = workspace + sizeC1;
        uint8_t* gmMxScaleX = workspace + sizeC1 + sizeQuantX;

        auto layoutC1 = tla::MakeLayout<typename BlockMmad1::ElementC, typename BlockMmad1::TileCopy::LayoutTagC>(m, r);
        auto layoutQuantXC = tla::MakeLayout<int8_t, typename BlockMmad3::TileCopy::LayoutTagA>(m, k);
        auto layoutQuantXV =
            tla::MakeLayout<typename SmoothQuant::U, typename BlockMmad3::TileCopy::LayoutTagA>(m, CeilDiv<2>(k));
        auto layoutMxScaleX =
            tla::MakeMxScaleLayout<float8_e8m0_t, typename BlockMmad3::TileCopy::LayoutTagA, false>(m, mxScaleK);

        Params params{
            args.problemShape,
            args.problemRank,
            args.qmax,
            args.ptrX,
            args.layoutX,
            args.ptrSmoothScale,
            args.layoutSmoothScale,
            args.ptrSvd1,
            args.layoutSvd1,
            args.ptrSvd2,
            args.layoutSvd2,
            args.ptrW,
            args.layoutW,
            args.ptrMxScaleW,
            args.layoutMxScaleW,
            args.ptrBias,
            gmC1,
            layoutC1,
            gmQuantX,
            layoutQuantXV,
            layoutQuantXC,
            gmMxScaleX,
            layoutMxScaleX,
            args.ptrY,
            args.layoutY,
        };
        return params;
    }

    CATLASS_DEVICE
    SvdQuantMatmulTla()
    {}

    template <class BlockMmad1, class SmoothQuant>
    struct SmoothQuantParams {
        static_assert(
            std::is_same_v<typename BlockMmad1::ElementA, typename SmoothQuant::ElementIn>,
            "Element of A1 and SmoothQuant's input is not same");
        static_assert(
            std::is_same_v<typename BlockMmad1::L1TileShape, typename SmoothQuant::L1TileShape>,
            "L1TileShape of Mmad1 and SmoothQuant is not same");
        static_assert(BlockMmad1::L1A_STAGES == SmoothQuant::STAGES, "");
        using ElementX = typename BlockMmad1::ElementA;
        static constexpr uint32_t TILE_ROWS = BlockMmad1::L1_TILE_M;
        static constexpr uint32_t TILE_COLS = RoundUp<SmoothQuant::UB_MX_SCALE_GROUP_NUM>(BlockMmad1::L1_TILE_K);
        static_assert(
            BlockMmad1::L1_TILE_K % SmoothQuant::UB_MX_SCALE_GROUP_NUM == 0, "Mmad1's L1_TILE_K must be 64 aligned");
        static constexpr uint32_t AIV_NUM = SmoothQuant::AIV_NUM;

        static constexpr uint32_t STAGES = BlockMmad1::L1A_STAGES;

        struct TensorList {
            struct L1TensorLists {
                AscendC::LocalTensor<typename BlockMmad1::ElementA> l1A[STAGES];
            } l1;
            struct UbTensorLists {
                AscendC::LocalTensor<typename SmoothQuant::ElementIn> x[STAGES];
                AscendC::LocalTensor<typename SmoothQuant::GmSmoothType> smoothScale[STAGES]; // optional
                AscendC::LocalTensor<typename SmoothQuant::ElementIn> smoothX[STAGES];        // optional
                AscendC::LocalTensor<uint16_t> maxExp;
                AscendC::LocalTensor<uint16_t> halfScale;
                AscendC::LocalTensor<int8_t> quantX[STAGES];
                AscendC::LocalTensor<ElementMxScaleX> mxScale[STAGES];
            } ub;
            struct GlobalMmadTensors {
                AscendC::GlobalTensor<typename BlockMmad1::ElementA> gmA;
                AscendC::GlobalTensor<typename BlockMmad1::ElementB> gmB;
                AscendC::GlobalTensor<typename BlockMmad1::ElementC> gmC;
                typename BlockMmad1::LayoutA layoutA;
                typename BlockMmad1::LayoutB layoutB;
                typename BlockMmad1::LayoutC layoutC;
            } gmMmad1;
            struct GlobalQuantTensors {
                AscendC::GlobalTensor<typename SmoothQuant::ElementIn> gmX;
                AscendC::GlobalTensor<typename SmoothQuant::GmSmoothType> gmSmooth; // optional
                AscendC::GlobalTensor<int8_t> gmQuantX;
                AscendC::GlobalTensor<ElementMxScaleX> gmMxScaleX;
            } gmQuant;
        } tensorLists;
        struct EventLists {
            Arch::FlagID flagAicFinishMte1[2][STAGES]; // 2 vectors
            Arch::FlagID flagAivFinishMte3[2][STAGES]; // 2 vectors
            uint32_t ubIn[STAGES];
            uint32_t ubOutQuntX[STAGES];
            uint32_t ubOutMxScale[STAGES];
            uint32_t pad = STAGES * 3;
        } eventLists;
        static constexpr uint32_t USED_IDS = SmoothQuant::HAS_SMOOTH ? (3 * STAGES + 1) : (3 * STAGES);
        static_assert(USED_IDS <= 8, "AIV used event ids can not exceed 8");
        uint32_t listId{0};

        struct TileSizes {
            static constexpr uint32_t l1A = BlockMmad1::L1A_TILE_SIZE;
            static constexpr uint32_t x = l1A / AIV_NUM;
            static constexpr uint32_t smoothScale = SmoothQuant::HAS_SMOOTH ? TILE_COLS * sizeof(ElementX) : 0;
            static constexpr uint32_t smoothX = SmoothQuant::HAS_SMOOTH ? x : 0;
            static constexpr uint32_t maxExp =
                TILE_ROWS / AIV_NUM * CeilDiv<MX_SCALE_GROUP_NUM>(TILE_COLS) * sizeof(uint16_t);
            static constexpr uint32_t halfScale = maxExp;
            static constexpr uint32_t quantX =
                RoundUp<BYTE_PER_BLK>(TILE_ROWS / AIV_NUM * CeilDiv<2>(TILE_COLS) * sizeof(ElementMxScaleX));
            // mxScale cols 方向需要对齐到偶数
            static constexpr uint32_t mxScale =
                TILE_ROWS / AIV_NUM * RoundUp<2>(CeilDiv<MX_SCALE_GROUP_NUM>(TILE_COLS)) * sizeof(int8_t);
            static constexpr uint32_t usedUbSize =
                (x + smoothScale + smoothX) * STAGES + maxExp + halfScale + (quantX + mxScale) * STAGES;
            static_assert(x % BYTE_PER_BLK == 0, "size of ub x is not 32B aligned");
            static_assert(smoothScale % BYTE_PER_BLK == 0, "size of ub smoothScale is not 32B aligned");
            static_assert(smoothX % BYTE_PER_BLK == 0, "size of ub smoothX is not 32B aligned");
            static_assert(maxExp % BYTE_PER_BLK == 0, "size of ub maxExp is not 32B aligned");
            static_assert(halfScale % BYTE_PER_BLK == 0, "size of ub halfScale is not 32B aligned");
            static_assert(quantX % BYTE_PER_BLK == 0, "size of ub quantX is not 32B aligned");
            static_assert(mxScale % BYTE_PER_BLK == 0, "size of ub mxScale is not 32B aligned");
            static_assert(usedUbSize <= ArchTag::UB_SIZE, "ub used size can not exceed ArchTag::UB_SIZE");
        } tileSizes;

        CATLASS_DEVICE
        SmoothQuantParams(Arch::Resource<ArchTag>& resource, Params const& params)
        {
            // init gm tensors
            tensorLists.gmMmad1.gmA.SetGlobalBuffer((__gm__ typename BlockMmad1::ElementA*)params.ptrX);
            tensorLists.gmMmad1.gmB.SetGlobalBuffer((__gm__ typename BlockMmad1::ElementB*)params.ptrSvd1);
            tensorLists.gmMmad1.gmC.SetGlobalBuffer((__gm__ typename BlockMmad1::ElementC*)params.ptrC1);
            tensorLists.gmMmad1.layoutA = params.layoutX;
            tensorLists.gmMmad1.layoutB = params.layoutSvd1;
            tensorLists.gmMmad1.layoutC = params.layoutC1;

            tensorLists.gmQuant.gmX.SetGlobalBuffer((__gm__ typename SmoothQuant::ElementIn*)params.ptrX);
            if constexpr (SmoothQuant::HAS_SMOOTH) {
                tensorLists.gmQuant.gmSmooth.SetGlobalBuffer(
                    (__gm__ typename SmoothQuant::GmSmoothType*)params.ptrSmoothScale);
            }
            tensorLists.gmQuant.gmQuantX.SetGlobalBuffer((__gm__ int8_t*)params.ptrQuantX);
            tensorLists.gmQuant.gmMxScaleX.SetGlobalBuffer((__gm__ ElementMxScaleX*)params.ptrMxScaleX);

            // --------------------------------------------------------------------------------------------------------
            // init ub/l1 tensors
            auto& ubTensorLists = tensorLists.ub;
            uint32_t ubBufAddrOffset = 0;
            // input tensors
            for (uint32_t i = 0; i < STAGES; i++) {
                eventLists.flagAicFinishMte1[0][i] = static_cast<Arch::FlagID>(i);
                eventLists.flagAicFinishMte1[1][i] = static_cast<Arch::FlagID>(i + 16);
                eventLists.flagAivFinishMte3[0][i] = static_cast<Arch::FlagID>(i + STAGES);
                eventLists.flagAivFinishMte3[1][i] = static_cast<Arch::FlagID>(i + STAGES + 16);
                eventLists.ubIn[i] = i;

                tensorLists.l1.l1A[i] = resource.l1Buf.template GetBufferByByte<ElementX>(tileSizes.l1A * i);
                ubTensorLists.x[i] = resource.ubBuf.template GetBufferByByte<ElementX>(ubBufAddrOffset);
                ubBufAddrOffset += tileSizes.x;
#ifdef __DAV_CUBE__
                AscendC::CrossCoreSetFlag<0x4, PIPE_MTE1>(eventLists.flagAicFinishMte1[0][i]);
                AscendC::CrossCoreSetFlag<0x4, PIPE_MTE1>(eventLists.flagAicFinishMte1[1][i]);
#elif defined __DAV_VEC__
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventLists.ubIn[i]);
#endif
            }

#ifdef __DAV_VEC__
            // then only ub tensors
            // smooth and smoothX
            for (uint32_t i = 0; i < STAGES; i++) {
                if constexpr (SmoothQuant::HAS_SMOOTH) {
                    ubTensorLists.smoothScale[i] =
                        resource.ubBuf.template GetBufferByByte<typename SmoothQuant::GmSmoothType>(ubBufAddrOffset);
                    ubBufAddrOffset += tileSizes.smoothScale;
                    ubTensorLists.smoothX[i] = resource.ubBuf.template GetBufferByByte<ElementX>(ubBufAddrOffset);
                    ubBufAddrOffset += tileSizes.smoothX;
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventLists.ubIn[i]);
                } else {
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventLists.ubIn[i]);
                }
            }

            // ub local tensors
            ubTensorLists.maxExp = resource.ubBuf.template GetBufferByByte<uint16_t>(ubBufAddrOffset);
            ubBufAddrOffset += tileSizes.maxExp;
            ubTensorLists.halfScale = resource.ubBuf.template GetBufferByByte<uint16_t>(ubBufAddrOffset);
            ubBufAddrOffset += tileSizes.halfScale;

            // output tensors
            for (uint32_t i = 0; i < STAGES; i++) {
                ubTensorLists.quantX[i] = resource.ubBuf.template GetBufferByByte<int8_t>(ubBufAddrOffset);
                eventLists.ubOutQuntX[i] = STAGES + i;
                ubBufAddrOffset += tileSizes.quantX;
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventLists.ubOutQuntX[i]);

                ubTensorLists.mxScale[i] = resource.ubBuf.template GetBufferByByte<ElementMxScaleX>(ubBufAddrOffset);
                ubBufAddrOffset += tileSizes.mxScale;
                eventLists.ubOutMxScale[i] = STAGES * 2 + i;
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventLists.ubOutMxScale[i]);
            }
#endif
        }

        CATLASS_DEVICE
        ~SmoothQuantParams()
        {
#ifdef __DAV_VEC__
            for (uint32_t i = 0; i < STAGES; i++) {
                AscendC::CrossCoreWaitFlag<0x4, PIPE_MTE3>(eventLists.flagAicFinishMte1[0][i]);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventLists.ubIn[i]);
                if constexpr (SmoothQuant::HAS_SMOOTH) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventLists.ubIn[i]);
                } else {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventLists.ubIn[i]);
                }
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventLists.ubOutQuntX[i]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventLists.ubOutMxScale[i]);
            }
#endif
        }
    };

    template <class BlockMmad2, class BlockMmad3>
    struct Mmad23Params {
        // Check DispatchPolicy
        static_assert(BlockMmad2::L1A_STAGES == BlockMmad3::L1A_STAGES, "L1A_STAGES of Mmad2 and Mmad3 is not same");
        static_assert(BlockMmad2::L1B_STAGES == BlockMmad3::L1B_STAGES, "L1B_STAGES of Mmad2 and Mmad3 is not same");
        static_assert(BlockMmad2::L0A_STAGES == BlockMmad3::L0A_STAGES, "L0A_STAGES of Mmad2 and Mmad3 is not same");
        static_assert(BlockMmad2::L0B_STAGES == BlockMmad3::L0B_STAGES, "L0B_STAGES of Mmad2 and Mmad3 is not same");
        static_assert(BlockMmad2::L0C_STAGES == BlockMmad3::L0C_STAGES, "L0C_STAGES of Mmad2 and Mmad3 is not same");
        static_assert(BlockMmad2::ENABLE_UNIT_FLAG == BlockMmad3::ENABLE_UNIT_FLAG, "");
        static constexpr uint32_t L1A_STAGES = BlockMmad3::L1A_STAGES;
        static constexpr uint32_t L1B_STAGES = BlockMmad3::L1B_STAGES;
        static constexpr uint32_t L0A_STAGES = BlockMmad3::L0A_STAGES;
        static constexpr uint32_t L0B_STAGES = BlockMmad3::L0B_STAGES;
        static constexpr uint32_t L0C_STAGES = BlockMmad3::L0C_STAGES;
        static constexpr bool ENABLE_UNIT_FLAG = BlockMmad3::ENABLE_UNIT_FLAG;

        // Check L1 tile shape
        using L1TileShape2 = typename BlockMmad2::L1TileShape;
        using L1TileShape3 = typename BlockMmad3::L1TileShape;
        static_assert(
            tla::get<0>(L1TileShape2{}) == tla::get<0>(L1TileShape3{}),
            "BlockMmad2 and BlockMmad3 L1Tile M is not same");
        static_assert(
            tla::get<1>(L1TileShape2{}) == tla::get<1>(L1TileShape3{}),
            "BlockMmad2 and BlockMmad3 L1Tile N is not same");
        static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape3{});
        static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape3{});
        static constexpr uint32_t L1_TILE_K2 = tla::get<2>(L1TileShape2{});
        static constexpr uint32_t L1_TILE_K3 = tla::get<2>(L1TileShape3{});

        // check C
        static_assert(
            std::is_same_v<typename BlockMmad2::ElementAccumulator, typename BlockMmad3::ElementAccumulator>,
            "ElementAccumulator of Mmad2 and Mmad3 is not same");
        static_assert(
            std::is_same_v<typename BlockMmad2::ElementC, typename BlockMmad3::ElementC>,
            "ElementC of Mmad2 and Mmad3 is not same");
        static_assert(
            std::is_same_v<typename BlockMmad2::LayoutC, typename BlockMmad3::LayoutC>,
            "LayoutC of Mmad2 and Mmad3 is not same");

        // half*half->(float)->half
        struct TensorLists2 {
            struct LocalTensors {
                AscendC::LocalTensor<typename BlockMmad2::ElementA> l1A[L1A_STAGES];
                AscendC::LocalTensor<typename BlockMmad2::ElementB> l1B[L1B_STAGES];
                AscendC::LocalTensor<typename BlockMmad2::ElementA> l0A[L0A_STAGES];
                AscendC::LocalTensor<typename BlockMmad2::ElementB> l0B[L0B_STAGES];
                AscendC::LocalTensor<typename BlockMmad2::ElementAccumulator> l0C[L0C_STAGES];
                AscendC::LocalTensor<typename BlockMmad2::GlobalTensorBiasType> l1Bias;
                AscendC::LocalTensor<typename BlockMmad2::ElementAccumulator> l0Bias;
            } local;
            struct GlobalTensors {
                AscendC::GlobalTensor<typename BlockMmad2::ElementA> gmA;
                AscendC::GlobalTensor<typename BlockMmad2::ElementB> gmB;
                AscendC::GlobalTensor<typename BlockMmad2::ElementC> gmC;
                AscendC::GlobalTensor<typename BlockMmad2::GlobalTensorBiasType> gmBias;
                typename BlockMmad2::LayoutA layoutA;
                typename BlockMmad2::LayoutB layoutB;
                typename BlockMmad2::LayoutC layoutC;
            } global;
        } tensorLists2;

        // fp4*fp4->(float)->half
        struct TensorLists3 {
            struct LocalTensors {
                AscendC::LocalTensor<typename BlockMmad3::ElementA> l1A[L1A_STAGES];
                AscendC::LocalTensor<typename BlockMmad3::ElementB> l1B[L1B_STAGES];
                AscendC::LocalTensor<typename BlockMmad3::ElementMxScaleA> l1MxScaleA[L1A_STAGES];
                AscendC::LocalTensor<typename BlockMmad3::ElementMxScaleB> l1MxScaleB[L1B_STAGES];
                AscendC::LocalTensor<typename BlockMmad3::ElementL0A> l0A[L0A_STAGES];
                AscendC::LocalTensor<typename BlockMmad3::ElementL0B> l0B[L0B_STAGES];
                AscendC::LocalTensor<typename BlockMmad3::ElementAccumulator> l0C[L0C_STAGES];
            } local;
            struct GlobalTensors {
                AscendC::GlobalTensor<typename BlockMmad3::ElementA> gmA;
                AscendC::GlobalTensor<typename BlockMmad3::ElementB> gmB;
                AscendC::GlobalTensor<typename BlockMmad3::ElementMxScaleA> gmMxScaleA;
                AscendC::GlobalTensor<typename BlockMmad3::ElementMxScaleB> gmMxScaleB;
                AscendC::GlobalTensor<typename BlockMmad3::ElementC> gmC;
                typename BlockMmad3::LayoutA layoutA;
                typename BlockMmad3::LayoutB layoutB;
                typename BlockMmad3::TileCopy::LayoutMxScaleA layoutMxScaleA;
                typename BlockMmad3::TileCopy::LayoutMxScaleB layoutMxScaleB;
                typename BlockMmad3::LayoutC layoutC;
            } global;
        } tensorLists3;

        struct TileSizes {
            static constexpr uint32_t l0A = Max(BlockMmad2::L0A_TILE_SIZE, BlockMmad3::L0A_TILE_SIZE);
            static constexpr uint32_t l0B = Max(BlockMmad2::L0B_TILE_SIZE, BlockMmad3::L0B_TILE_SIZE);
            static constexpr uint32_t l0C = Max(BlockMmad2::L0C_TILE_SIZE, BlockMmad3::L0C_TILE_SIZE);
            static constexpr uint32_t l1A = Max(BlockMmad2::L1A_TILE_SIZE, BlockMmad3::L1A_TILE_SIZE);
            static constexpr uint32_t l1B = Max(BlockMmad2::L1B_TILE_SIZE, BlockMmad3::L1B_TILE_SIZE);
            static constexpr uint32_t l1MxScaleA = BlockMmad3::L1SCALEA_TILE_SIZE;
            static constexpr uint32_t l1MxScaleB = BlockMmad3::L1SCALEB_TILE_SIZE;
            static constexpr uint32_t l1Bias =
                BlockMmad2::HAS_BIAS ? (BlockMmad2::L1_TILE_N * sizeof(typename BlockMmad2::GlobalTensorBiasType)) : 0;
            static constexpr uint32_t usedL1Size =
                l1A * L1A_STAGES + l1B * L1B_STAGES + l1MxScaleA * L1A_STAGES + l1MxScaleB * L1B_STAGES + l1Bias;
            static_assert(usedL1Size <= ArchTag::L1_SIZE, "Mmad2/Mmad3 used L1 size exceedding ArchTag::L1_SIZE");
        } tileSizes;
        struct EventLists {
            int32_t l1A[L1A_STAGES];
            int32_t l1B[L1B_STAGES];
            int32_t l0A[L0A_STAGES];
            int32_t l0B[L0B_STAGES];
            int32_t l0C[L0C_STAGES];
            int32_t l1Bias;
            int32_t l0Bias;
        } eventLists;
        struct ListIds {
            uint32_t l1A{0};
            uint32_t l1B{0};
            uint32_t l0A{0};
            uint32_t l0B{0};
            uint32_t l0C{0};
        } listIds;

        CATLASS_DEVICE
        Mmad23Params(Arch::Resource<ArchTag>& resource, Params const& params)
        {
#ifdef __DAV_VEC__
            return;
#endif
            if constexpr (ENABLE_UNIT_FLAG && tla::detail::isRowMajor<typename BlockMmad3::LayoutC>::value) {
                AscendC::SetMMLayoutTransform(true);
            }

            tensorLists2.global.gmA.SetGlobalBuffer((__gm__ typename BlockMmad2::ElementA*)params.ptrC1);
            tensorLists2.global.gmB.SetGlobalBuffer((__gm__ typename BlockMmad2::ElementB*)params.ptrSvd2);
            tensorLists2.global.gmC.SetGlobalBuffer((__gm__ typename BlockMmad2::ElementC*)params.ptrY);
            tensorLists2.global.layoutA = params.layoutC1;
            tensorLists2.global.layoutB = params.layoutSvd2;
            tensorLists2.global.layoutC = params.layoutY;
            if constexpr (BlockMmad2::HAS_BIAS) {
                tensorLists2.global.gmBias.SetGlobalBuffer(
                    (__gm__ typename BlockMmad2::GlobalTensorBiasType*)params.ptrBias);
            }

            tensorLists3.global.gmA.SetGlobalBuffer((__gm__ typename BlockMmad3::ElementA*)params.ptrQuantX);
            tensorLists3.global.gmB.SetGlobalBuffer((__gm__ typename BlockMmad3::ElementB*)params.ptrW);
            tensorLists3.global.gmMxScaleA.SetGlobalBuffer(
                (__gm__ typename BlockMmad3::ElementMxScaleA*)params.ptrMxScaleX);
            tensorLists3.global.gmMxScaleB.SetGlobalBuffer(
                (__gm__ typename BlockMmad3::ElementMxScaleB*)params.ptrMxScaleW);
            tensorLists3.global.gmC.SetGlobalBuffer((__gm__ typename BlockMmad3::ElementC*)params.ptrY);
            tensorLists3.global.layoutA = params.layoutQuantXC;
            tensorLists3.global.layoutB = params.layoutW;
            tensorLists3.global.layoutMxScaleA = params.layoutMxScaleX;
            tensorLists3.global.layoutMxScaleB = params.layoutMxScaleW;
            tensorLists3.global.layoutC = params.layoutY;

            // --------------------------------------------------------------------------------------------------------
            // initialize L1/L0 tensors
            uint32_t l1Offset = 0;
            for (uint32_t i = 0; i < L1A_STAGES; i++) {
                // Assign L1/L0A/L0B space for each stages
                tensorLists2.local.l1A[i] =
                    resource.l1Buf.template GetBufferByByte<typename BlockMmad2::ElementA>(l1Offset);
                tensorLists3.local.l1A[i] =
                    resource.l1Buf.template GetBufferByByte<typename BlockMmad3::ElementA>(l1Offset);
                l1Offset += tileSizes.l1A;
                // Assign event ID for each stages
                eventLists.l1A[i] = i;
                // The event id that needs to be set before the loop
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(eventLists.l1A[i]);
            }
            for (uint32_t i = 0; i < L1B_STAGES; i++) {
                // Assign L1/L0A/L0B space for each stages
                tensorLists2.local.l1B[i] =
                    resource.l1Buf.template GetBufferByByte<typename BlockMmad2::ElementB>(l1Offset);
                tensorLists3.local.l1B[i] =
                    resource.l1Buf.template GetBufferByByte<typename BlockMmad3::ElementB>(l1Offset);
                l1Offset += tileSizes.l1B;
                // Assign event ID for each stages
                eventLists.l1B[i] = i + L1A_STAGES;
                // The event id that needs to be set before the loop
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(eventLists.l1B[i]);
            }
            for (uint32_t i = 0; i < L0A_STAGES; i++) {
                // Assign L1/L0A/L0B space for each stages
                tensorLists2.local.l0A[i] =
                    resource.l0ABuf.template GetBufferByByte<typename BlockMmad2::ElementA>(tileSizes.l0A * i);
                tensorLists3.local.l0A[i] =
                    resource.l0ABuf.template GetBufferByByte<typename BlockMmad3::ElementL0A>(tileSizes.l0A * i);
                // Assign event ID for each stages
                eventLists.l0A[i] = i;
                // The event id that needs to be set before the loop
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(eventLists.l0A[i]);
            }
            for (uint32_t i = 0; i < L0B_STAGES; i++) {
                // Assign L1/L0A/L0B space for each stages
                tensorLists2.local.l0B[i] =
                    resource.l0BBuf.template GetBufferByByte<typename BlockMmad2::ElementB>(tileSizes.l0B * i);
                tensorLists3.local.l0B[i] =
                    resource.l0BBuf.template GetBufferByByte<typename BlockMmad3::ElementL0B>(tileSizes.l0B * i);
                // Assign event ID for each stages
                eventLists.l0B[i] = i + L0A_STAGES;
                // The event id that needs to be set before the loop
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(eventLists.l0B[i]);
            }
            if constexpr (!ENABLE_UNIT_FLAG) {
                for (uint32_t i = 0; i < L0C_STAGES; i++) {
                    tensorLists2.local.l0C[i] =
                        resource.l0CBuf.template GetBufferByByte<typename BlockMmad2::ElementAccumulator>(
                            tileSizes.l0C * i);
                    tensorLists3.local.l0C[i] =
                        resource.l0CBuf.template GetBufferByByte<typename BlockMmad3::ElementAccumulator>(
                            tileSizes.l0C * i);
                    eventLists.l0C[i] = i;
                    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(eventLists.l0C[i]);
                }
            } else {
                tensorLists2.local.l0C[0] =
                    resource.l0CBuf.template GetBufferByByte<typename BlockMmad2::ElementAccumulator>(0);
                tensorLists3.local.l0C[0] =
                    resource.l0CBuf.template GetBufferByByte<typename BlockMmad3::ElementAccumulator>(0);
            }

            for (uint32_t i = 0; i < L1A_STAGES; i++) {
                tensorLists3.local.l1MxScaleA[i] =
                    resource.l1Buf.template GetBufferByByte<typename BlockMmad3::ElementMxScaleA>(l1Offset);
                l1Offset += tileSizes.l1MxScaleA;
            }
            for (uint32_t i = 0; i < L1B_STAGES; i++) {
                tensorLists3.local.l1MxScaleB[i] =
                    resource.l1Buf.template GetBufferByByte<typename BlockMmad3::ElementMxScaleB>(l1Offset);
                l1Offset += tileSizes.l1MxScaleB;
            }

            if constexpr (BlockMmad2::HAS_BIAS) {
                tensorLists2.local.l1Bias =
                    resource.l1Buf.template GetBufferByByte<typename BlockMmad2::GlobalTensorBiasType>(l1Offset);
                l1Offset += tileSizes.l1Bias;
                tensorLists2.local.l0Bias =
                    resource.btBuf.template GetBufferByByte<typename BlockMmad2::ElementAccumulator>(0);
                eventLists.l1Bias = L1A_STAGES + L1B_STAGES;
                eventLists.l0Bias = L0A_STAGES + L0B_STAGES;
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(eventLists.l1Bias);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(eventLists.l0Bias);
            }
        }

        CATLASS_DEVICE
        ~Mmad23Params()
        {
#ifdef __DAV_VEC__
            return;
#endif

            if constexpr (ENABLE_UNIT_FLAG && tla::detail::isRowMajor<typename BlockMmad3::LayoutC>::value) {
                AscendC::SetMMLayoutTransform(false);
            }

            for (uint32_t i = 0; i < L1A_STAGES; i++) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(eventLists.l1A[i]);
            }
            for (uint32_t i = 0; i < L1B_STAGES; i++) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(eventLists.l1B[i]);
            }
            for (uint32_t i = 0; i < L0A_STAGES; i++) {
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(eventLists.l0A[i]);
            }
            for (uint32_t i = 0; i < L0B_STAGES; i++) {
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(eventLists.l0B[i]);
            }
            if constexpr (!ENABLE_UNIT_FLAG) {
                for (uint32_t i = 0; i < L0C_STAGES; i++) {
                    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(eventLists.l0C[i]);
                }
            }
            if constexpr (BlockMmad2::HAS_BIAS) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(eventLists.l1Bias);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(eventLists.l0Bias);
            }
        }
    };

    CATLASS_DEVICE void operator()(Params const& params)
    {
        Arch::Resource<ArchTag> resource;

        SmoothQuantMmad1(resource, params);

#ifdef __DAV_VEC__
        constexpr Arch::FlagID corssBarierFlagId = Arch::BarrierFlag<0x0, g_coreType>::ID;
        AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(corssBarierFlagId);
        AscendC::CrossCoreWaitFlag<0x0, PIPE_S>(corssBarierFlagId);

        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagFinishQuant);
#elif defined __DAV_CUBE__
        // wait AIC write C1(A2) finished
        constexpr Arch::FlagID corssBarierFlagId = Arch::BarrierFlag<0x0, g_coreType>::ID;
        AscendC::CrossCoreSetFlag<0x0, PIPE_FIX>(corssBarierFlagId);
        AscendC::CrossCoreWaitFlag<0x0, PIPE_S>(corssBarierFlagId);
        // wait AIV write quantX/quantX mxScale finished
        AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(flagFinishQuant);

        Mmad23(resource, params);
#endif

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    CATLASS_DEVICE void SmoothQuantMmad1(Arch::Resource<ArchTag>& resource, Params const& params)
    {
        SmoothQuantParams<BlockMmad1, SmoothQuant> smoothQuantParams(resource, params);

        auto& gmMmad1 = smoothQuantParams.tensorLists.gmMmad1;
        auto& gmQuant = smoothQuantParams.tensorLists.gmQuant;

        auto tensorA1 = tla::MakeTensor(gmMmad1.gmA, gmMmad1.layoutA, Arch::PositionGM{});
        auto tensorB1 = tla::MakeTensor(gmMmad1.gmB, gmMmad1.layoutB, Arch::PositionGM{});
        auto tensorC1 = tla::MakeTensor(gmMmad1.gmC, gmMmad1.layoutC, Arch::PositionGM{});

        // tensorX is tensorA1
        auto tensorSmooth = tla::MakeTensor(gmQuant.gmSmooth, params.layoutSmoothScale, Arch::PositionGM{});
        auto tensorQuantX = tla::MakeTensor(gmQuant.gmQuantX, params.layoutQuantXV, Arch::PositionGM{});
        auto tensorMxScale = tla::MakeTensor(gmQuant.gmMxScaleX, params.layoutMxScaleX, Arch::PositionGM{});

#ifdef __DAV_CUBE__
        uint32_t aiCoreIdx = AscendC::GetBlockIdx();
        uint32_t aiCoreNum = AscendC::GetBlockNum();
        BlockMmad1 blockMmad1(resource);
#elif defined __DAV_VEC__
        uint32_t aiCoreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t aiCoreNum = AscendC::GetBlockNum();
        SmoothQuant smoothQuant(resource);

        ElementX qmaxInv{1.0f}; // 1/qmax
        if constexpr (std::is_same_v<ElementX, half>) {
            qmaxInv = AscendC::Cast<float, half, AscendC::RoundMode::CAST_ODD>(1.0f / params.qmax);
        } else if constexpr (std::is_same_v<ElementX, bfloat16_t>) {
            qmaxInv = AscendC::Cast(1.0f / params.qmax);
        }
#endif

        static constexpr uint32_t L1_TILE_M = BlockMmad1::L1_TILE_M;
        GemmCoord mmadShape1{params.problemShape.m(), params.problemRank, params.problemShape.k()}; // m r k
        uint32_t normalBlockNum1 = RoundDown(mmadShape1.m(), aiCoreNum * L1_TILE_M) / L1_TILE_M;
        uint32_t mTailLen = mmadShape1.m() - normalBlockNum1 * L1_TILE_M;
        for (uint32_t normalLoopIdx = aiCoreIdx; normalLoopIdx < normalBlockNum1 + aiCoreNum;
             normalLoopIdx += aiCoreNum) {
            if (normalLoopIdx >= normalBlockNum1 && mTailLen == 0) {
                break;
            }
            GemmCoord coord{normalLoopIdx * L1_TILE_M, 0, 0};
            GemmCoord actualBlockShape{L1_TILE_M, mmadShape1.n(), mmadShape1.k()};
            // 尾轮切 M 实现负载均衡
            if (normalLoopIdx >= normalBlockNum1 && mTailLen > 0) {
                auto [mIdx, mLen] = SplitLengthByCores(mTailLen, aiCoreIdx, aiCoreNum);
                if (mLen == 0) {
                    break;
                }
                coord = GemmCoord{normalBlockNum1 * L1_TILE_M + mIdx, 0, 0};
                actualBlockShape = GemmCoord{mLen, mmadShape1.n(), mmadShape1.k()};
            }

            // shared tensor A
            auto tensorBlockA = GetTile(
                tensorA1, tla::MakeCoord(coord.m(), coord.k()),
                tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
#ifdef __DAV_CUBE__
            auto tensorBlockB = GetTile(
                tensorB1, tla::MakeCoord(coord.k(), coord.n()),
                tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
            auto tensorBlockC = GetTile(
                tensorC1, tla::MakeCoord(coord.m(), coord.n()),
                tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));
            blockMmad1(tensorBlockA, tensorBlockB, tensorBlockC, actualBlockShape, smoothQuantParams);
#elif defined __DAV_VEC__
            auto& tensorBlockX = tensorBlockA;
            auto tensorBlockSmooth =
                GetTile(tensorSmooth, tla::MakeCoord(0, coord.k()), tla::MakeShape(1, actualBlockShape.k()));
            auto tensorBlockQuantX = GetTile(
                tensorQuantX, tla::MakeCoord(coord.m(), coord.k() / 2),
                tla::MakeShape(actualBlockShape.m(), CeilDiv<2>(actualBlockShape.k())));
            auto tensorBlockMxScale = GetTile(
                tensorMxScale, tla::MakeCoord(coord.m(), coord.k() / MX_SCALE_GROUP_NUM),
                tla::MakeShape(actualBlockShape.m(), CeilDiv<MX_SCALE_GROUP_NUM>(actualBlockShape.k())));
            smoothQuant(
                tensorBlockX, tensorBlockSmooth, tensorBlockQuantX, tensorBlockMxScale, qmaxInv, smoothQuantParams);
#endif
        }
    }

    CATLASS_DEVICE void Mmad23(Arch::Resource<ArchTag>& resource, Params const& params)
    {
#ifdef __DAV_VEC__
        return;
#endif
        uint32_t aiCoreIdx = AscendC::GetBlockIdx();
        uint32_t aiCoreNum = AscendC::GetBlockNum();

        BlockMmad2 blockMmad2;
        BlockMmad3 blockMmad3;
        using Mmad23Params = Mmad23Params<BlockMmad2, BlockMmad3>;
        Mmad23Params mmad23Params(resource, params);
        auto& gmList2 = mmad23Params.tensorLists2.global;
        auto& gmList3 = mmad23Params.tensorLists3.global;

        BlockScheduler matmulBlockScheduler23(
            params.problemShape, MakeCoord(Mmad23Params::L1_TILE_M, Mmad23Params::L1_TILE_N));
        uint32_t coreLoops23 = matmulBlockScheduler23.GetCoreLoops();

        auto tensorA2 = tla::MakeTensor(gmList2.gmA, gmList2.layoutA, Arch::PositionGM{});
        auto tensorB2 = tla::MakeTensor(gmList2.gmB, gmList2.layoutB, Arch::PositionGM{});
        auto tensorA3 = tla::MakeTensor(gmList3.gmA, gmList3.layoutA, Arch::PositionGM{});
        auto tensorB3 = tla::MakeTensor(gmList3.gmB, gmList3.layoutB, Arch::PositionGM{});
        auto tensorMxScaleA3 = tla::MakeTensor(gmList3.gmMxScaleA, gmList3.layoutMxScaleA, Arch::PositionGM{});
        auto tensorMxScaleB3 = tla::MakeTensor(gmList3.gmMxScaleB, gmList3.layoutMxScaleB, Arch::PositionGM{});
        auto tensorC = tla::MakeTensor(gmList3.gmC, gmList3.layoutC, Arch::PositionGM{});

        auto layoutBias = tla::MakeLayout(params.problemShape.n());
        auto tensorBias = tla::MakeTensor(gmList2.gmBias, layoutBias, Arch::PositionGM{});

        uint32_t normalBlockNum23 = RoundDown(coreLoops23, aiCoreNum);
        uint32_t tailBlockNum23 = coreLoops23 - normalBlockNum23;

        for (uint32_t loopIdx = aiCoreIdx; loopIdx < normalBlockNum23 + aiCoreNum; loopIdx += aiCoreNum) {
            if (loopIdx >= normalBlockNum23 && tailBlockNum23 == 0) {
                break;
            }
            static constexpr uint32_t L1_TILE_M = Mmad23Params::L1_TILE_M;
            static constexpr uint32_t L1_TILE_N = Mmad23Params::L1_TILE_N;
            static constexpr uint32_t L1_TILE_K2 = Mmad23Params::L1_TILE_K2;
            static constexpr uint32_t L1_TILE_K3 = Mmad23Params::L1_TILE_K3;

            // Compute block location
            GemmCoord blockCoord, coord, actualBlockShape2, actualBlockShape3;
            if (loopIdx < normalBlockNum23) {
                blockCoord = matmulBlockScheduler23.GetBlockCoord(loopIdx);
                coord = GemmCoord{blockCoord.m() * L1_TILE_M, blockCoord.n() * L1_TILE_N, 0};
                actualBlockShape3 = matmulBlockScheduler23.GetActualBlockShape(blockCoord);   // m1 n1 k
                actualBlockShape2 =
                    GemmCoord{actualBlockShape3.m(), actualBlockShape3.n(), params.problemRank}; // m1 n1 r
            } else {
                uint32_t coreNumPerTailBlock = aiCoreNum / tailBlockNum23;         // 每个尾块分到的核数
                uint32_t tailBlockUsedCore = coreNumPerTailBlock * tailBlockNum23; // 参与尾轮计算的核数
                if (aiCoreIdx >= tailBlockUsedCore) {
                    break;
                }

                uint32_t tailLoopIdx = normalBlockNum23 + aiCoreIdx / coreNumPerTailBlock; // aiCore所在的尾块
                uint32_t coreIdxInTailBlock = aiCoreIdx % coreNumPerTailBlock;             // aiCore在尾块中的idx
                blockCoord = matmulBlockScheduler23.GetBlockCoord(tailLoopIdx);
                GemmCoord tailBlockShape3 = matmulBlockScheduler23.GetActualBlockShape(blockCoord); // m1 n1 k

                // 尾块自动切 M/N
                if (tailBlockShape3.m() >= tailBlockShape3.n()) {
                    auto [idx, len] = SplitLengthByCores(tailBlockShape3.m(), coreIdxInTailBlock, coreNumPerTailBlock);
                    if (len == 0) {
                        break;
                    }
                    coord = GemmCoord{blockCoord.m() * L1_TILE_M + idx, blockCoord.n() * L1_TILE_N, 0};
                    actualBlockShape2 = GemmCoord{len, tailBlockShape3.n(), params.problemRank};
                    actualBlockShape3 = GemmCoord{len, tailBlockShape3.n(), tailBlockShape3.k()};
                } else {
                    auto [idx, len] = SplitLengthByCores(tailBlockShape3.n(), coreIdxInTailBlock, coreNumPerTailBlock);
                    if (len == 0) {
                        break;
                    }
                    coord = GemmCoord{blockCoord.m() * L1_TILE_M, blockCoord.n() * L1_TILE_N + idx, 0};
                    actualBlockShape2 = GemmCoord{tailBlockShape3.m(), len, params.problemRank};
                    actualBlockShape3 = GemmCoord{tailBlockShape3.m(), len, tailBlockShape3.k()};
                }
            }

            auto tensorBlockA2 = GetTile(
                tensorA2, tla::MakeCoord(coord.m(), coord.k()),
                tla::MakeShape(actualBlockShape2.m(), actualBlockShape2.k()));
            auto tensorBlockB2 = GetTile(
                tensorB2, tla::MakeCoord(coord.k(), coord.n()),
                tla::MakeShape(actualBlockShape2.k(), actualBlockShape2.n()));

            auto tensorBlockA3 = GetTile(
                tensorA3, tla::MakeCoord(coord.m(), coord.k()),
                tla::MakeShape(actualBlockShape3.m(), actualBlockShape3.k()));
            auto tensorBlockB3 = GetTile(
                tensorB3, tla::MakeCoord(coord.k(), coord.n()),
                tla::MakeShape(actualBlockShape3.k(), actualBlockShape3.n()));
            auto tensorBlockMxScaleA3 = GetTile(
                tensorMxScaleA3, tla::MakeCoord(coord.m(), coord.k() / MX_SCALE_GROUP_NUM),
                tla::MakeShape(actualBlockShape3.m(), CeilDiv<MX_SCALE_GROUP_NUM>(actualBlockShape3.k())));
            auto tensorBlockMxScaleB3 = GetTile(
                tensorMxScaleB3, tla::MakeCoord(coord.k() / MX_SCALE_GROUP_NUM, coord.n()),
                tla::MakeShape(CeilDiv<MX_SCALE_GROUP_NUM>(actualBlockShape3.k()), actualBlockShape3.n()));

            auto tensorBlockC = GetTile(
                tensorC, tla::MakeCoord(coord.m(), coord.n()),
                tla::MakeShape(actualBlockShape2.m(), actualBlockShape2.n()));

            if constexpr (BlockMmad2::HAS_BIAS) {
                auto tensorBlockBias =
                    GetTile(tensorBias, tla::MakeCoord(coord.n()), tla::MakeShape(actualBlockShape2.n()));
                blockMmad2(
                    tensorBlockA2, tensorBlockB2, tensorBlockC, actualBlockShape2, mmad23Params.tensorLists2.local,
                    mmad23Params.eventLists, mmad23Params.listIds, false, tensorBlockBias);
            } else {
                blockMmad2(
                    tensorBlockA2, tensorBlockB2, tensorBlockC, actualBlockShape2, mmad23Params.tensorLists2.local,
                    mmad23Params.eventLists, mmad23Params.listIds, false);
            }
            blockMmad3(
                tensorBlockA3, tensorBlockB3, tensorBlockC, tensorBlockMxScaleA3, tensorBlockMxScaleB3,
                actualBlockShape3, mmad23Params.tensorLists3.local, mmad23Params.eventLists, mmad23Params.listIds,
                true);
        }
    }

    CATLASS_DEVICE
    auto SplitLengthByCores(uint32_t length, uint32_t coreIdx, uint32_t coreNum)
    {
        uint32_t lenPerCore = length / coreNum;
        uint32_t lenRemain = length % coreNum;
        if (coreIdx < lenRemain) {
            lenPerCore++;
        }
        uint32_t idx = coreIdx * lenPerCore;
        if (coreIdx >= lenRemain) {
            idx += lenRemain;
        }
        struct ret {
            uint32_t idx_;
            uint32_t lenPerCore_;
        };
        return ret{idx, lenPerCore};
    }

private:
    static constexpr Arch::FlagID flagFinishQuant{0};
};

#endif // (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)

} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_SVD_QUANT_MATMUL_KERNEL_TLA_HPP