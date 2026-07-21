/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_BLOCK_BLOCK_MMAD_DYNAMIC_PRELOAD_ASYNC_WITH_CALLBACK_HPP
#define CATLASS_GEMM_BLOCK_BLOCK_MMAD_DYNAMIC_PRELOAD_ASYNC_WITH_CALLBACK_HPP

#include "catlass_a5/catlass.hpp"
#include "catlass_a5/arch/resource.hpp"
#include "catlass_a5/coord.hpp"
#include "catlass_a5/detail/callback.hpp"
#include "catlass_a5/gemm_coord.hpp"
#include "catlass_a5/gemm/dispatch_policy.hpp"
#include "catlass_a5/gemm/helper.hpp"
#include "catlass_a5/gemm/tile/tile_copy.hpp"
#include "catlass_a5/gemm/tile/tile_mmad.hpp"

namespace Catlass::Gemm::Block {

template <uint32_t PRELOAD_STAGES_, uint32_t L1_STAGES_, uint32_t L0A_STAGES_, uint32_t L0B_STAGES_,
          uint32_t L0C_STAGES_, bool ENABLE_UNIT_FLAG_, bool ENABLE_SHUFFLE_K_, class L1TileShape_, class L0TileShape_,
          class AType_, class BType_, class CType_, class BiasType_, class TileCopy_, class TileMmad_>
struct BlockMmad<MmadAtlasA2DynamicPreloadAsyncWithCallback<PRELOAD_STAGES_, L1_STAGES_, L0A_STAGES_, L0B_STAGES_,
                                                            L0C_STAGES_, ENABLE_UNIT_FLAG_, ENABLE_SHUFFLE_K_>,
                 L1TileShape_, L0TileShape_, AType_, BType_, CType_, BiasType_, TileCopy_, TileMmad_> {
public:
    // Type Aliases
    using DispatchPolicy =
        MmadAtlasA2DynamicPreloadAsyncWithCallback<PRELOAD_STAGES_, L1_STAGES_, L0A_STAGES_, L0B_STAGES_, L0C_STAGES_,
                                                   ENABLE_UNIT_FLAG_, ENABLE_SHUFFLE_K_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementA = typename AType_::Element;
    using LayoutA = typename AType_::Layout;
    using ElementB = typename BType_::Element;
    using LayoutB = typename BType_::Layout;
    using ElementC = typename CType_::Element;
    using LayoutC = typename CType_::Layout;
    using TileMmad = TileMmad_;
    using CopyGmToL1A = typename TileCopy_::CopyGmToL1A;
    using CopyGmToL1B = typename TileCopy_::CopyGmToL1B;
    using CopyL1ToL0A = typename TileCopy_::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy_::CopyL1ToL0B;
    using CopyL0CToGm = typename TileCopy_::CopyL0CToGm;
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;
    using LayoutAInL1 = typename CopyL1ToL0A::LayoutSrc;
    using LayoutBInL1 = typename CopyL1ToL0B::LayoutSrc;
    using LayoutAInL0 = typename CopyL1ToL0A::LayoutDst;
    using LayoutBInL0 = typename CopyL1ToL0B::LayoutDst;
    using LayoutCInL0 = layout::zN;

    using L1AAlignHelper = Gemm::helper::L1AlignHelper<ElementA, LayoutA>;
    using L1BAlignHelper = Gemm::helper::L1AlignHelper<ElementB, LayoutB>;

    static constexpr uint32_t PRELOAD_STAGES = DispatchPolicy::PRELOAD_STAGES;
    static constexpr uint32_t L1_STAGES = DispatchPolicy::L1_STAGES;
    static constexpr uint32_t L0A_STAGES = DispatchPolicy::L0A_STAGES;
    static constexpr uint32_t L0B_STAGES = DispatchPolicy::L0B_STAGES;
    static constexpr uint32_t L0C_STAGES = DispatchPolicy::L0C_STAGES;

    static constexpr bool ENABLE_UNIT_FLAG = DispatchPolicy::ENABLE_UNIT_FLAG;
    static constexpr bool ENABLE_SHUFFLE_K = DispatchPolicy::ENABLE_SHUFFLE_K;

    // Check LayoutC
    static_assert(std::is_same_v<LayoutC, layout::RowMajor>, "LayoutC only support RowMajor yet!");

    CATLASS_DEVICE
    BlockMmad(GemmCoord const &l1TileShape_, GemmCoord const &l0TileShape_, Arch::Resource<ArchTag> &resource,
              uint32_t l1BufAddrStart = 0)
        : l1TileShape(l1TileShape_), l0TileShape(l0TileShape_)
    {
        if ASCEND_IS_AIV {
            return;
        }

        uint32_t l1ASize = l1TileShape.m() * l1TileShape.k() * sizeof(ElementA);
        uint32_t l1BSize = l1TileShape.k() * l1TileShape.n() * sizeof(ElementB);

        uint32_t l0ASize = l0TileShape.m() * l0TileShape.k() * sizeof(ElementA);
        uint32_t l0BSize = l0TileShape.k() * l0TileShape.n() * sizeof(ElementB);
        uint32_t l0CSize = l0TileShape.m() * l0TileShape.n() * sizeof(ElementAccumulator);

        uint32_t l1AOffset = l1BufAddrStart;
        uint32_t l1BOffset = l1BufAddrStart + l1ASize * L1_STAGES;
        for (uint32_t i = 0; i < L1_STAGES; ++i) {
            l1ATensorList[i] = resource.l1Buf.template GetBufferByByte<ElementA>(l1AOffset + l1ASize * i);
            l1BTensorList[i] = resource.l1Buf.template GetBufferByByte<ElementB>(l1BOffset + l1BSize * i);
            l1AEventList[i] = i;
            l1BEventList[i] = i + L1_STAGES;
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
        }

        for (uint32_t i = 0; i < L0A_STAGES; ++i) {
            l0ATensorList[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(l0ASize * i);
            l0AEventList[i] = i;
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
        }

        for (uint32_t i = 0; i < L0B_STAGES; ++i) {
            l0BTensorList[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(l0BSize * i);
            l0BEventList[i] = i + L0A_STAGES;
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
        }

        for (uint32_t i = 0; i < L0C_STAGES; ++i) {
            l0CTensorList[i] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(l0CSize * i);
            l0CEventList[i] = i;
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventList[i]);
        }
    }

    CATLASS_DEVICE
    ~BlockMmad()
    {
        if ASCEND_IS_AIV {
            return;
        }

        SynchronizeBlock();
        for (uint32_t i = 0; i < L1_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
        }
        for (uint32_t i = 0; i < L0A_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
        }
        for (uint32_t i = 0; i < L0B_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
        }
        for (uint32_t i = 0; i < L0C_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventList[i]);
        }
    }

    CATLASS_DEVICE
    void operator()(AscendC::GlobalTensor<ElementA> const &gmBlockA, LayoutA const &layoutA,
                    AscendC::GlobalTensor<ElementB> const &gmBlockB, LayoutB const &layoutB,
                    AscendC::GlobalTensor<ElementC> const &gmBlockC, LayoutC const &layoutC,
                    GemmCoord const &actualShape, Callback const &callbackBeforeFixpipe,
                    Callback const &callbackAfterFixpipe)
    {
        uint32_t kTileCount = CeilDiv(actualShape.k(), l1TileShape.k());

        uint32_t mRound = RoundUp<L1AAlignHelper::M_ALIGNED>(actualShape.m());
        uint32_t nRound = RoundUp<L1BAlignHelper::N_ALIGNED>(actualShape.n());

        uint32_t startTileIdx = 0;
        if constexpr (ENABLE_SHUFFLE_K) {
            startTileIdx = AscendC::GetBlockIdx() % kTileCount;
        }

        for (uint32_t kLoopIdx = 0; kLoopIdx < kTileCount; ++kLoopIdx) {
            uint32_t kTileIdx = (startTileIdx + kLoopIdx < kTileCount) ? (startTileIdx + kLoopIdx)
                                                                       : (startTileIdx + kLoopIdx - kTileCount);

            uint32_t kActual =
                (kTileIdx < kTileCount - 1) ? l1TileShape.k() : (actualShape.k() - kTileIdx * l1TileShape.k());

            // Emission load instruction from GM to L1
            MatrixCoord gmTileAOffset{0, kTileIdx * l1TileShape.k()};
            MatrixCoord gmTileBOffset{kTileIdx * l1TileShape.k(), 0};
            auto gmTileA = gmBlockA[layoutA.GetOffset(gmTileAOffset)];
            auto gmTileB = gmBlockB[layoutB.GetOffset(gmTileBOffset)];
            // Load first matrix A tile from GM to L1
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListId]);
            auto layoutTileA = layoutA.GetTileLayout(MakeCoord(actualShape.m(), kActual));
            auto l1a_layout = LayoutAInL1::template MakeLayout<ElementA>(l1TileShape.m(), l1TileShape.k());
            copyGmToL1A(l1ATensorList[l1ListId], gmTileA, l1a_layout, layoutTileA);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListId]);
            // Load first matrix B tile from GM to L1
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListId]);
            auto layoutTileB = layoutB.GetTileLayout(MakeCoord(kActual, actualShape.n()));
            auto l1b_layout = LayoutBInL1::template MakeLayout<ElementB>(l1TileShape.k(), l1TileShape.n());
            copyGmToL1B(l1BTensorList[l1ListId], gmTileB, l1b_layout, layoutTileB);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListId]);

            // If the number of preload instructions reaches the upper limit, perform an mmad calculation on L1 tile
            if (preloadCount == PRELOAD_STAGES) {
                L1TileMmad(l1TileMmadParamsList[l1TileMmadParamsId]);
            }

            // Store the current load status
            uint32_t preloadL1TileMmadParamsId = (l1TileMmadParamsId + preloadCount < PRELOAD_STAGES)
                                                     ? (l1TileMmadParamsId + preloadCount)
                                                     : (l1TileMmadParamsId + preloadCount - PRELOAD_STAGES);
            auto &l1TileMmadParams = l1TileMmadParamsList[preloadL1TileMmadParamsId];
            l1TileMmadParams.l1ListId = l1ListId;
            l1TileMmadParams.mRound = mRound;
            l1TileMmadParams.nRound = nRound;
            l1TileMmadParams.kActual = kActual;
            l1TileMmadParams.isKLoopFirst = (kLoopIdx == 0);
            l1TileMmadParams.isKLoopLast = (kLoopIdx == kTileCount - 1);
            if (kLoopIdx == kTileCount - 1) {
                l1TileMmadParams.gmBlockC = gmBlockC;
                l1TileMmadParams.layoutCInGm = layoutC.GetTileLayout(actualShape.GetCoordMN());
                l1TileMmadParams.callbackBeforeFixpipe = callbackBeforeFixpipe;
                l1TileMmadParams.callbackAfterFixpipe = callbackAfterFixpipe;
            }

            if (preloadCount < PRELOAD_STAGES) {
                ++preloadCount;
            } else {
                l1TileMmadParamsId = (l1TileMmadParamsId + 1 < PRELOAD_STAGES) ? (l1TileMmadParamsId + 1) : 0;
            }
            l1ListId = (l1ListId + 1 < L1_STAGES) ? (l1ListId + 1) : 0;
        }
    }

    CATLASS_DEVICE
    void SynchronizeBlock()
    {
        while (preloadCount > 0) {
            L1TileMmad(l1TileMmadParamsList[l1TileMmadParamsId]);
            l1TileMmadParamsId = (l1TileMmadParamsId + 1 < PRELOAD_STAGES) ? (l1TileMmadParamsId + 1) : 0;
            --preloadCount;
        }
    }

private:
    struct L1TileMmadParams {
        uint32_t l1ListId;
        uint32_t mRound;
        uint32_t nRound;
        uint32_t kActual;
        bool isKLoopFirst;
        bool isKLoopLast;
        AscendC::GlobalTensor<ElementC> gmBlockC;
        LayoutC layoutCInGm;
        Callback callbackBeforeFixpipe;
        Callback callbackAfterFixpipe;

        CATLASS_DEVICE
        L1TileMmadParams() = default;
    };

    CATLASS_DEVICE
    void InitL1(Arch::Resource<ArchTag> &resource, uint32_t l1BufAddrStart)
    {
        uint32_t l1ASize = l1TileShape.m() * l1TileShape.k() * sizeof(ElementA);
        uint32_t l1BSize = l1TileShape.k() * l1TileShape.n() * sizeof(ElementB);
        uint32_t l1AOffset = l1BufAddrStart;
        uint32_t l1BOffset = l1BufAddrStart + l1ASize * L1_STAGES;
        for (uint32_t i = 0; i < L1_STAGES; ++i) {
            l1ATensorList[i] = resource.l1Buf.template GetBufferByByte<ElementA>(l1AOffset + l1ASize * i);
            l1BTensorList[i] = resource.l1Buf.template GetBufferByByte<ElementB>(l1BOffset + l1BSize * i);
            l1AEventList[i] = i;
            l1BEventList[i] = i + L1_STAGES;
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
        }
    }

    CATLASS_DEVICE
    void L1TileMmad(L1TileMmadParams const &params)
    {
        uint32_t mPartLoop = CeilDiv(params.mRound, l0TileShape.m());
        uint32_t nPartLoop = CeilDiv(params.nRound, l0TileShape.n());
        uint32_t kPartLoop = CeilDiv(params.kActual, l0TileShape.k());

        auto &l1ATensor = l1ATensorList[params.l1ListId];
        auto &l1BTensor = l1BTensorList[params.l1ListId];

        auto &l0CTensor = l0CTensorList[l0CListId];
        LayoutCInL0 layoutCInL0 = LayoutCInL0::MakeLayoutInL0C(MakeCoord(params.mRound, params.nRound));

        if constexpr (!ENABLE_UNIT_FLAG) {
            if (params.isKLoopFirst) {
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventList[l0CListId]);
            }
        }

        for (uint32_t mPartIdx = 0; mPartIdx < mPartLoop; ++mPartIdx) {
            uint32_t mPartActual =
                (mPartIdx < mPartLoop - 1) ? l0TileShape.m() : (params.mRound - mPartIdx * l0TileShape.m());

            for (uint32_t kPartIdx = 0; kPartIdx < kPartLoop; ++kPartIdx) {
                uint32_t kPartActual =
                    (kPartIdx < kPartLoop - 1) ? l0TileShape.k() : (params.kActual - kPartIdx * l0TileShape.k());

                auto &l0ATile = l0ATensorList[l0AListId];
                auto layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(mPartActual, kPartActual);
                auto l1AOffset = MakeCoord(mPartIdx, kPartIdx) * MakeCoord(l0TileShape.m(), l0TileShape.k());
                auto l1a_layout = LayoutAInL1::template MakeLayout<ElementA>(l1TileShape.m(), l1TileShape.k());
                auto l1ATile = l1ATensor[l1a_layout.GetOffset(l1AOffset)];

                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                if ((mPartIdx == 0) && (kPartIdx == 0)) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[params.l1ListId]);
                }
                copyL1ToL0A(l0ATile, l1ATile, layoutAInL0, l1a_layout);
                if ((mPartIdx == mPartLoop - 1) && (kPartIdx == kPartLoop - 1)) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[params.l1ListId]);
                }

                for (uint32_t nPartIdx = 0; nPartIdx < nPartLoop; ++nPartIdx) {
                    uint32_t nPartActual =
                        (nPartIdx < nPartLoop - 1) ? l0TileShape.n() : (params.nRound - nPartIdx * l0TileShape.n());

                    auto &l0BTile = l0BTensorList[l0BListId];
                    auto layoutBInL0 = LayoutBInL0::template MakeLayout<ElementB>(kPartActual, nPartActual);
                    auto l1BOffset = MakeCoord(kPartIdx, nPartIdx) * MakeCoord(l0TileShape.k(), l0TileShape.n());
                    auto l1b_layout = LayoutBInL1::template MakeLayout<ElementB>(l1TileShape.k(), l1TileShape.n());
                    auto l1BTile = l1BTensor[l1b_layout.GetOffset(l1BOffset)];

                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                    if ((kPartIdx == 0) && (nPartIdx == 0)) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[params.l1ListId]);
                    }
                    copyL1ToL0B(l0BTile, l1BTile, layoutBInL0, l1b_layout);
                    if ((kPartIdx == kPartLoop - 1) && (nPartIdx == nPartLoop - 1)) {
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[params.l1ListId]);
                    }

                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);

                    auto l0COffset = MakeCoord(mPartIdx, nPartIdx) * MakeCoord(l0TileShape.m(), l0TileShape.n());
                    auto l0CTile = l0CTensor[layoutCInL0.GetOffset(l0COffset)];

                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
                    // If the current tile is the first tile on the k axis, the accumulator needs to be reset to 0
                    bool initC = (params.isKLoopFirst && (kPartIdx == 0));
                    // If the unit flag is enabled, the unit flag is set according to the calculation progress
                    uint8_t unitFlag = 0b00;
                    if constexpr (ENABLE_UNIT_FLAG) {
                        if (params.isKLoopLast && (mPartIdx == mPartLoop - 1) && (kPartIdx == kPartLoop - 1) &&
                            (nPartIdx == nPartLoop - 1)) {
                            unitFlag = 0b11;
                        } else {
                            unitFlag = 0b10;
                        }
                    }
                    tileMmad(l0CTile, l0ATile, l0BTile, mPartActual, nPartActual, kPartActual, initC, unitFlag);

                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                    l0BListId = (l0BListId + 1 < L0B_STAGES) ? (l0BListId + 1) : 0;
                }
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                l0AListId = (l0AListId + 1 < L0A_STAGES) ? (l0AListId + 1) : 0;
            }
        }

        if (params.isKLoopLast) {
            auto layoutCInGm = params.layoutCInGm;

            params.callbackBeforeFixpipe();

            if constexpr (!ENABLE_UNIT_FLAG) {
                AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CEventList[l0CListId]);
                AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CEventList[l0CListId]);
                copyL0CToGm(params.gmBlockC, l0CTensor, layoutCInGm, layoutCInL0);
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventList[l0CListId]);
            } else {
                copyL0CToGm(params.gmBlockC, l0CTensor, layoutCInGm, layoutCInL0, 0b11);
            }
            l0CListId = (l0CListId + 1 < L0C_STAGES) ? (l0CListId + 1) : 0;

            params.callbackAfterFixpipe();
        }
    }

protected:
    AscendC::LocalTensor<ElementA> l1ATensorList[L1_STAGES];
    AscendC::LocalTensor<ElementB> l1BTensorList[L1_STAGES];
    int32_t l1AEventList[L1_STAGES];
    int32_t l1BEventList[L1_STAGES];
    uint32_t l1ListId{0};

    AscendC::LocalTensor<ElementA> l0ATensorList[L0A_STAGES];
    int32_t l0AEventList[L0A_STAGES];
    uint32_t l0AListId{0};

    AscendC::LocalTensor<ElementB> l0BTensorList[L0B_STAGES];
    int32_t l0BEventList[L0B_STAGES];
    uint32_t l0BListId{0};

    AscendC::LocalTensor<ElementAccumulator> l0CTensorList[L0C_STAGES_];
    int32_t l0CEventList[L0C_STAGES_];
    uint32_t l0CListId{0};

    L1TileMmadParams l1TileMmadParamsList[PRELOAD_STAGES];
    uint32_t l1TileMmadParamsId{0};
    uint32_t preloadCount{0};
    GemmCoord l1TileShape;
    GemmCoord l0TileShape;

    TileMmad tileMmad;
    CopyGmToL1A copyGmToL1A;
    CopyGmToL1B copyGmToL1B;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;
    CopyL0CToGm copyL0CToGm;
};

}  // namespace Catlass::Gemm::Block

#endif  // CATLASS_GEMM_BLOCK_BLOCK_MMAD_DYNAMIC_PRELOAD_ASYNC_WITH_CALLBACK_HPP
