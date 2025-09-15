/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_BLOCK_BLOCK_MMAD_W8A16_HPP
#define CATLASS_GEMM_BLOCK_BLOCK_MMAD_W8A16_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"

namespace Catlass::Gemm::Block {

template <class ArchTag, class ElementIn_, class ElementOut_, class Layout_, class TileShape_, uint32_t STAGES = 2>
struct PrologueCast {
    using ElementIn = ElementIn_;
    using ElementOut = ElementOut_;
    using TileShape = TileShape_;
    using Layout = Layout_;

    static constexpr uint32_t ELE_NUM_PER_BLK_INT8 = BYTE_PER_BLK / sizeof(ElementIn);
    static constexpr uint32_t ELE_NUM_PER_BLK_HALF = BYTE_PER_BLK / sizeof(ElementOut);
    static constexpr uint32_t COMPUTE_LEN = 32 * 1024;
    static constexpr uint32_t TILES_PER_LOOP = 32;

    // Construct
    CATLASS_DEVICE
    PrologueCast(Arch::Resource<ArchTag> &resource, uint32_t ubBufAddrStart = 0) {
        if (g_coreType == AscendC::AIV) {
            uint32_t ubOffset = ubBufAddrStart;
            uint32_t ubInSize = COMPUTE_LEN * sizeof(ElementIn);
            uint32_t ubOutSize = COMPUTE_LEN * sizeof(ElementOut);
            // Init buffers
            for (uint32_t i = 0; i < STAGES; ++i) {
                ubInTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementIn>(ubOffset);
                ubOffset += ubInSize;
                ubOutTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementOut>(ubOffset);
                ubOffset += ubOutSize;

                ubEventList[i] = i;
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(ubEventList[i]);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ubEventList[i]);
            }
        }
    }

    CATLASS_DEVICE
    void operator()(AscendC::GlobalTensor<ElementOut> const &gmDst,
        AscendC::GlobalTensor<ElementIn> const &gmSrc,
        Layout const &layoutDst, Layout const &layoutSrc, half deqScalar, half deqZeroPoint)
    {
        uint32_t tileNum = layoutSrc.shape(0);
        uint32_t tileLen = layoutSrc.shape(1);
        uint32_t tileLenRoundInt8 = RoundUp(layoutSrc.shape(1), ELE_NUM_PER_BLK_INT8);
        uint64_t tileStrideSrc = layoutSrc.stride(0);
        uint64_t tileStrideDst = layoutDst.stride(0);
        if constexpr (std::is_same_v<Layout, layout::ColumnMajor>) {
            tileNum = layoutSrc.shape(1);
            tileLen = layoutSrc.shape(0);
            tileLenRoundInt8 = RoundUp(layoutSrc.shape(0), ELE_NUM_PER_BLK_INT8);
            tileStrideSrc = layoutSrc.stride(1);
            tileStrideDst = layoutDst.stride(1);
        }
        uint32_t tilesPerAiv = tileNum / AscendC::GetSubBlockNum();
        if (AscendC::GetSubBlockIdx() < (tileNum % AscendC::GetSubBlockNum())) {
            tilesPerAiv++;
        }
        uint64_t taskOffsetSrc = AscendC::GetSubBlockIdx() * tilesPerAiv * tileStrideSrc;
        uint64_t taskOffsetDst = AscendC::GetSubBlockIdx() * tilesPerAiv * tileStrideDst;
        if (AscendC::GetSubBlockIdx() >= (tileNum % AscendC::GetSubBlockNum())) {
            taskOffsetSrc += (tileNum % AscendC::GetSubBlockNum()) * tileStrideSrc;
            taskOffsetDst += (tileNum % AscendC::GetSubBlockNum()) * tileStrideDst;
        }
        uint32_t loops = CeilDiv(tilesPerAiv, TILES_PER_LOOP);
        uint32_t pingpong = 0;
        for (uint32_t loopIdx = 0; loopIdx < loops; ++loopIdx) {
            uint32_t actualTiles = TILES_PER_LOOP;
            if (loopIdx == loops - 1) {
                actualTiles = tilesPerAiv - loopIdx * TILES_PER_LOOP;
            }
            uint64_t tileOffsetSrc = loopIdx * TILES_PER_LOOP * tileStrideSrc;
            AscendC::DataCopyExtParams dataCopyParamsIn(
                actualTiles, tileLen * sizeof(ElementIn),
                (tileStrideSrc - tileLen) * sizeof(ElementIn),
                (tileLenRoundInt8 - tileLen) / ELE_NUM_PER_BLK_INT8, 0
            );
            AscendC::DataCopyPadExtParams<ElementIn> padParams(false, 0, 0, 0);

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(ubEventList[pingpong]);
            AscendC::DataCopyPad(ubInTensorList[pingpong],
                gmSrc[taskOffsetSrc + tileOffsetSrc], dataCopyParamsIn, padParams);

            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ubEventList[pingpong]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(ubEventList[pingpong]);

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ubEventList[pingpong]);

            AscendC::Cast(ubOutTensorList[pingpong], ubInTensorList[pingpong],
                AscendC::RoundMode::CAST_NONE, actualTiles * tileLenRoundInt8);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(ubEventList[pingpong]);

            AscendC::Adds(ubOutTensorList[pingpong], ubOutTensorList[pingpong],
                deqZeroPoint, actualTiles * tileLenRoundInt8);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Muls(ubOutTensorList[pingpong], ubOutTensorList[pingpong],
                deqScalar, actualTiles * tileLenRoundInt8);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ubEventList[pingpong]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ubEventList[pingpong]);

            uint64_t tileOffsetDst = loopIdx * TILES_PER_LOOP * tileStrideDst;
            AscendC::DataCopyExtParams dataCopyParamsOut(
                actualTiles, tileLen * sizeof(ElementOut),
                (tileLenRoundInt8 - tileLen) / ELE_NUM_PER_BLK_HALF,
                (tileStrideDst - tileLen) * sizeof(ElementOut), 0
            );
            AscendC::DataCopyPad(gmDst[taskOffsetDst + tileOffsetDst], ubOutTensorList[pingpong], dataCopyParamsOut);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ubEventList[pingpong]);

            pingpong = (pingpong + 1) % STAGES;
        }
    }

    /// Destructor
    CATLASS_DEVICE
    ~PrologueCast() {
        if (g_coreType == AscendC::AIV) {
            for (uint32_t i = 0; i < STAGES; ++i) {
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(ubEventList[i]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ubEventList[i]);
            }
        }
    }

protected:
    /// Data members
    AscendC::LocalTensor<ElementIn> ubInTensorList[STAGES];
    AscendC::LocalTensor<ElementOut> ubOutTensorList[STAGES];

    int32_t ubEventList[STAGES];
};

template <
    bool ENABLE_UNIT_FLAG_,
    bool ENABLE_SHUFFLE_K_,
    class L1TileShape_,
    class L0TileShape_,
    class AType_,
    class BType_,
    class CType_,
    class BiasType_,
    class TileCopy_,
    class TileMmad_
>
struct BlockMmad <
    MmadAtlasA2W8A16<ENABLE_UNIT_FLAG_, ENABLE_SHUFFLE_K_>,
    L1TileShape_,
    L0TileShape_,
    AType_,
    BType_,
    CType_,
    BiasType_,
    TileCopy_,
    TileMmad_
> {
public:
    // Type Aliases
    using DispatchPolicy = MmadAtlasA2Preload<ENABLE_UNIT_FLAG_, ENABLE_SHUFFLE_K_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
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

    using TileShapeB = MatrixShape<L1TileShape::K, L1TileShape::N>;
    using PrologueCastB = PrologueCast<ArchTag, int8_t, ElementB, LayoutB, TileShapeB>; // no use of TileShapeB

    static constexpr bool ENABLE_UNIT_FLAG = DispatchPolicy::ENABLE_UNIT_FLAG;
    static constexpr bool ENABLE_SHUFFLE_K = DispatchPolicy::ENABLE_SHUFFLE_K;
    static constexpr uint32_t STAGES = DispatchPolicy::STAGES;
    static constexpr uint32_t L1A_SIZE = L1TileShape::M * L1TileShape::K * sizeof(ElementA);
    static constexpr uint32_t L1B_SIZE = L1TileShape::N * L1TileShape::K * sizeof(ElementB);
    static constexpr uint32_t L0A_SIZE = ArchTag::L0A_SIZE;
    static constexpr uint32_t L0B_SIZE = ArchTag::L0B_SIZE;
    static constexpr uint32_t L0C_SIZE = ArchTag::L0C_SIZE;
    static constexpr uint32_t L0A_PINGPONG_BUF_SIZE = L0A_SIZE / STAGES;
    static constexpr uint32_t L0B_PINGPONG_BUF_SIZE = L0B_SIZE / STAGES;
    
    // Check LayoutA
    static_assert(std::is_same_v<LayoutA, layout::RowMajor> || std::is_same_v<LayoutA, layout::ColumnMajor>,
        "LayoutA only support RowMajor/ColumnMajor yet!");
    
    // Check LayoutB
    static_assert(std::is_same_v<LayoutB, layout::RowMajor> || std::is_same_v<LayoutB, layout::ColumnMajor>,
        "LayoutB only support RowMajor/ColumnMajor yet!");

    // Check LayoutC
    static_assert(std::is_same_v<LayoutC, layout::RowMajor>, "LayoutC only support RowMajor yet!");

    // Check L1TileShape
    static_assert((L1A_SIZE * STAGES + L1B_SIZE * STAGES) <= ArchTag::L1_SIZE, "L1TileShape exceeding the L1 space!");

    // Check L0TileShape
    static constexpr uint32_t L0A_TILE_SIZE = L0TileShape::M * L0TileShape::K * sizeof(ElementA);
    static constexpr uint32_t L0B_TILE_SIZE = L0TileShape::K * L0TileShape::N * sizeof(ElementB);
    static constexpr uint32_t L0C_TILE_SIZE = L0TileShape::M * L0TileShape::N * sizeof(ElementAccumulator);
    static_assert((L0A_TILE_SIZE * STAGES) <= L0A_SIZE, "L0TileShape exceeding the L0A space!");
    static_assert((L0B_TILE_SIZE * STAGES) <= L0B_SIZE, "L0TileShape exceeding the L0B space!");
    static_assert(L0C_TILE_SIZE <= L0C_SIZE, "L0TileShape exceeding the L0C space!");

    static_assert(L1TileShape::M == L0TileShape::M && L1TileShape::N == L0TileShape::N,
        "The situation where the basic blocks of L1 and L0 differ on the m and n axes is not supported yet");
    static_assert(L0TileShape::K <= L1TileShape::K, "L0TileShape::K cannot exceed L1TileShape::K");

    /// Construct
    CATLASS_DEVICE
    BlockMmad(Arch::Resource<ArchTag> &resource, uint32_t l1BufAddrStart = 0) : prologueCastB(resource)
    {
        if (g_coreType == AscendC::AIC) {
            uint32_t l1AOffset = l1BufAddrStart;
            uint32_t l1BOffset = l1BufAddrStart + L1A_SIZE * STAGES;
            // Init buffers
            for (uint32_t i = 0; i < STAGES; i++) {
                l1ATensorList[i] = resource.l1Buf.template GetBufferByByte<ElementA>(l1AOffset + L1A_SIZE * i);
                l1BTensorList[i] = resource.l1Buf.template GetBufferByByte<ElementB>(l1BOffset + L1B_SIZE * i);
                l0ATensorList[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(L0A_PINGPONG_BUF_SIZE * i);
                l0BTensorList[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_PINGPONG_BUF_SIZE * i);

                l1AEventList[i] = i;
                l1BEventList[i] = i + STAGES;
                l0AEventList[i] = i;
                l0BEventList[i] = i + STAGES;
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
            }
            l0CTensor = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(0);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);

            Arch::CrossCoreSetFlag<0x2, PIPE_MTE2>(notifyAiv[0]);
            Arch::CrossCoreSetFlag<0x2, PIPE_MTE2>(notifyAiv[1]);
        }
    }

    /// Destructor
    CATLASS_DEVICE
    ~BlockMmad()
    {
        if (g_coreType == AscendC::AIC) {
            for (uint32_t i = 0; i < STAGES; i++) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
            }
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        } else {
            Arch::CrossCoreWaitFlag(notifyAiv[0]);
            Arch::CrossCoreWaitFlag(notifyAiv[1]);
        }
    }

    /// Prologue: cast int8_t to half (w8a16)
    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<int8_t> const &gmBlockB, LayoutB const &layoutB,
        AscendC::GlobalTensor<int8_t> const &gmNextBlockB,
        AscendC::GlobalTensor<ElementB> const &gmBWksp,
        GemmCoord const &actualShape, GemmCoord const &actualShapeNext,
        bool isFirstBlock, bool hasNextBlock, half deqScalar, half deqZeroPoint)
    {
        uint32_t kTileCount = CeilDiv<L1TileShape::K>(actualShape.k());
        uint32_t kTileCountNext = CeilDiv<L1TileShape::K>(actualShapeNext.k());

        uint32_t wkspStrideB = L1TileShape::N;
        if (std::is_same_v<LayoutB, layout::ColumnMajor>) {
            wkspStrideB = L1TileShape::K;
        }

        uint32_t startTileIdx = 0;
        if constexpr (ENABLE_SHUFFLE_K) {
            startTileIdx = AscendC::GetBlockIdx() / 2;
        }
        uint32_t firstTileIdx = startTileIdx % kTileCount;
        uint32_t lastTileIdx = (startTileIdx + kTileCount - 1) % kTileCount;
        uint32_t kActual =
            (firstTileIdx < kTileCount - 1) ? L1TileShape::K : (actualShape.k() - firstTileIdx * L1TileShape::K);
        uint32_t firstTileIdxNext = startTileIdx % kTileCountNext;

        // k loop
        for (uint32_t kLoopIdx = 0; kLoopIdx < kTileCount; kLoopIdx++) {
            uint32_t shuffleKIdx = (startTileIdx + kLoopIdx) % kTileCount;
            // Load first matrix B tile in total kernel loop from GM to UB
            if (shuffleKIdx == firstTileIdx && isFirstBlock) {
                MatrixCoord gmTileBOffset{shuffleKIdx * L1TileShape::K, 0};
                auto gmTileB = gmBlockB[layoutB.GetOffset(gmTileBOffset)];
                // Load first matrix B tile from GM to UB
                auto layoutTileB = layoutB.GetTileLayout(MakeCoord(kActual, actualShape.n()));
                auto layoutWkspB = LayoutB{kActual, actualShape.n(), wkspStrideB};

                Arch::CrossCoreWaitFlag(notifyAiv[l1ListId]);
                prologueCastB(gmBWksp[l1ListId * L1TileShape::K * L1TileShape::N],
                    gmTileB, layoutWkspB, layoutTileB, deqScalar, deqZeroPoint);
                Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(notifyAic[l1ListId]);
            }

            uint32_t l1ListIdNext = (l1ListId + 1 < STAGES) ? (l1ListId + 1) : 0;
            uint32_t kActualNext{0};

            // preload next tile from GM to UB
            if (shuffleKIdx != lastTileIdx) {
                uint32_t shuffleKIdxNext = (startTileIdx + kLoopIdx + 1) % kTileCount;
                // Get GM tensor for next stage
                kActualNext = (shuffleKIdxNext < kTileCount - 1) ?
                    L1TileShape::K : (actualShape.k() - shuffleKIdxNext * L1TileShape::K);
                MatrixCoord gmTileBOffset{shuffleKIdxNext * L1TileShape::K, 0};
                auto gmTileB = gmBlockB[layoutB.GetOffset(gmTileBOffset)];
                // load next matrix B tile from GM to UB
                auto layoutTileB = layoutB.GetTileLayout(MakeCoord(kActualNext, actualShape.n()));
                auto layoutWkspB = LayoutB{kActualNext, actualShape.n(), wkspStrideB};

                Arch::CrossCoreWaitFlag(notifyAiv[l1ListIdNext]);
                prologueCastB(gmBWksp[l1ListIdNext * L1TileShape::K * L1TileShape::N],
                    gmTileB, layoutWkspB, layoutTileB, deqScalar, deqZeroPoint);
                Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(notifyAic[l1ListIdNext]);
            }
            if (shuffleKIdx == lastTileIdx && hasNextBlock) {
                // Get GM tensor for next stage
                kActualNext = (firstTileIdxNext < kTileCountNext - 1) ?
                    L1TileShape::K : (actualShapeNext.k() - firstTileIdxNext * L1TileShape::K);
                MatrixCoord gmTileBOffset{firstTileIdxNext * L1TileShape::K, 0};
                auto gmTileB = gmNextBlockB[layoutB.GetOffset(gmTileBOffset)];
                // load next matrix B tile from GM to UB
                auto layoutTileB = layoutB.GetTileLayout(MakeCoord(kActualNext, actualShapeNext.n()));
                auto layoutWkspB = LayoutB{kActualNext, actualShapeNext.n(), wkspStrideB};

                Arch::CrossCoreWaitFlag(notifyAiv[l1ListIdNext]);
                prologueCastB(gmBWksp[l1ListIdNext * L1TileShape::K * L1TileShape::N],
                    gmTileB, layoutWkspB, layoutTileB, deqScalar, deqZeroPoint);
                Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(notifyAic[l1ListIdNext]);
            }
            l1ListId = l1ListIdNext;
            kActual = kActualNext;
        }
    }

    /// Perform a block-scoped matrix multiply-accumulate
    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementA> const &gmBlockA, LayoutA const &layoutA,
        AscendC::GlobalTensor<ElementB> const &gmBlockB,
        AscendC::GlobalTensor<ElementC> const &gmBlockC, LayoutC const &layoutC,
        AscendC::GlobalTensor<ElementA> const &gmNextBlockA,
        GemmCoord const &actualShape, GemmCoord const &actualShapeNext,
        bool isFirstBlock, bool hasNextBlock)
    {
        uint32_t mRound = RoundUp<L1AAlignHelper::M_ALIGNED>(actualShape.m());
        uint32_t nRound = RoundUp<L1BAlignHelper::N_ALIGNED>(actualShape.n());

        auto layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(L1TileShape::M, L1TileShape::K);
        auto layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(L1TileShape::K, L1TileShape::N);
        auto layoutInL0C = LayoutCInL0::MakeLayoutInL0C(MakeCoord(mRound, nRound));

        uint32_t kTileCount = CeilDiv<L1TileShape::K>(actualShape.k());
        uint32_t kTileCountNext = CeilDiv<L1TileShape::K>(actualShapeNext.k());

        uint32_t wkspStrideB = L1TileShape::N;
        if (std::is_same_v<LayoutB, layout::ColumnMajor>) {
            wkspStrideB = L1TileShape::K;
        }

        if constexpr (!ENABLE_UNIT_FLAG) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        }
        uint32_t startTileIdx = 0;
        if constexpr (ENABLE_SHUFFLE_K) {
            startTileIdx = AscendC::GetBlockIdx();
        }
        uint32_t firstTileIdx = startTileIdx % kTileCount;
        uint32_t lastTileIdx = (startTileIdx + kTileCount - 1) % kTileCount;
        uint32_t kActual =
            (firstTileIdx < kTileCount - 1) ? L1TileShape::K : (actualShape.k() - firstTileIdx * L1TileShape::K);
        uint32_t firstTileIdxNext = startTileIdx % kTileCountNext;

        uint32_t mPartLoop = CeilDiv<L0TileShape::M>(mRound);
        uint32_t nPartLoop = CeilDiv<L0TileShape::N>(nRound);

        // k loop
        for (uint32_t kLoopIdx = 0; kLoopIdx < kTileCount; kLoopIdx++) {
            uint32_t shuffleKIdx = (startTileIdx + kLoopIdx) % kTileCount;
            // Load first matrix A tile in total kernel loop from GM to L1
            if (shuffleKIdx == firstTileIdx && isFirstBlock) {
                MatrixCoord gmTileAOffset{0, shuffleKIdx * L1TileShape::K};
                auto gmTileA = gmBlockA[layoutA.GetOffset(gmTileAOffset)];
                // Load first matrix A tile from GM to L1
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListId]);
                auto layoutTileA = layoutA.GetTileLayout(MakeCoord(actualShape.m(), kActual));
                copyGmToL1A(l1ATensorList[l1ListId], gmTileA, layoutAInL1, layoutTileA);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListId]);

                // Load first matrix B tile from GM to L1
                Arch::CrossCoreWaitFlag(notifyAic[l1ListId]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListId]);
                auto layoutTileB = LayoutB{kActual, actualShape.n(), wkspStrideB};
                copyGmToL1B(l1BTensorList[l1ListId], gmBlockB[l1ListId * L1TileShape::K * L1TileShape::N],
                    layoutBInL1, layoutTileB);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListId]);
                Arch::CrossCoreSetFlag<0x2, PIPE_MTE2>(notifyAiv[l1ListId]);
            }

            uint32_t l1ListIdNext = (l1ListId + 1 < STAGES) ? (l1ListId + 1) : 0;
            uint32_t kActualNext{0};

            // preload next tile from GM to L1
            if (shuffleKIdx != lastTileIdx) {
                uint32_t shuffleKIdxNext = (startTileIdx + kLoopIdx + 1) % kTileCount;
                // Get L1 tensor for next stage
                auto l1ATensor = l1ATensorList[l1ListIdNext];
                auto l1BTensor = l1BTensorList[l1ListIdNext];
                // Get GM tensor for next stage
                kActualNext = (shuffleKIdxNext < kTileCount - 1) ?
                    L1TileShape::K : (actualShape.k() - shuffleKIdxNext * L1TileShape::K);
                MatrixCoord gmTileAOffset{0, shuffleKIdxNext * L1TileShape::K};
                auto gmTileA = gmBlockA[layoutA.GetOffset(gmTileAOffset)];

                // load next matrix A tile from GM to L1
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListIdNext]);
                auto layoutTileA = layoutA.GetTileLayout(MakeCoord(actualShape.m(), kActualNext));
                copyGmToL1A(l1ATensor, gmTileA, layoutAInL1, layoutTileA);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListIdNext]);

                // load next matrix B tile from GM to L1
                Arch::CrossCoreWaitFlag(notifyAic[l1ListIdNext]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListIdNext]);
                auto layoutTileB = LayoutB{kActualNext, actualShape.n(), wkspStrideB};
                copyGmToL1B(l1BTensor, gmBlockB[l1ListIdNext * L1TileShape::K * L1TileShape::N],
                    layoutBInL1, layoutTileB);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListIdNext]);
                Arch::CrossCoreSetFlag<0x2, PIPE_MTE2>(notifyAiv[l1ListIdNext]);
            }
            if (shuffleKIdx == lastTileIdx && hasNextBlock) {
                // Get L1 tensor for next stage
                auto l1ATensor = l1ATensorList[l1ListIdNext];
                auto l1BTensor = l1BTensorList[l1ListIdNext];
                // Get GM tensor for next stage
                kActualNext = (firstTileIdxNext < kTileCountNext - 1) ?
                    L1TileShape::K : (actualShapeNext.k() - firstTileIdxNext * L1TileShape::K);
                MatrixCoord gmTileAOffset{0, firstTileIdxNext * L1TileShape::K};
                auto gmTileA = gmNextBlockA[layoutA.GetOffset(gmTileAOffset)];
                // load next matrix A tile from GM to L1
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListIdNext]);
                auto layoutTileA = layoutA.GetTileLayout(MakeCoord(actualShapeNext.m(), kActualNext));
                copyGmToL1A(l1ATensor, gmTileA, layoutAInL1, layoutTileA);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListIdNext]);

                // load next matrix B tile from GM to L1
                Arch::CrossCoreWaitFlag(notifyAic[l1ListIdNext]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListIdNext]);
                auto layoutTileB = LayoutB{kActualNext, actualShapeNext.n(), wkspStrideB};
                copyGmToL1B(l1BTensor, gmBlockB[l1ListIdNext * L1TileShape::K * L1TileShape::N],
                    layoutBInL1, layoutTileB);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListIdNext]);
                Arch::CrossCoreSetFlag<0x2, PIPE_MTE2>(notifyAiv[l1ListIdNext]);
            }

            // Get L1 tensor for current stage
            auto l1ATensor = l1ATensorList[l1ListId];
            auto l1BTensor = l1BTensorList[l1ListId];

            // Get the loop nums on L0
            uint32_t kPartLoop = CeilDiv<L0TileShape::K>(kActual);

            uint32_t l0ABufId = 0;
            uint32_t l0BBufId = 0;

            for (int mPartIdx = 0; mPartIdx < mPartLoop; mPartIdx++) {
                uint32_t mPartActual = (mPartIdx < mPartLoop - 1) ?
                    L0TileShape::M : (mRound - mPartIdx * L0TileShape::M);

                for (int kPartIdx = 0; kPartIdx < kPartLoop; kPartIdx++) {
                    uint32_t kPartActual = (kPartIdx < kPartLoop - 1) ?
                        L0TileShape::K : (kActual - kPartIdx * L0TileShape::K);

                    // Locate the current tile on L0A
                    auto l0ATile = l0ATensorList[l0ABufId];
                    LayoutAInL0 layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(mPartActual, kPartActual);
                    // Locate the current tile of matrix A on L1
                    MatrixCoord l1AOffset{mPartIdx * L0TileShape::M, kPartIdx * L0TileShape::K};
                    auto l1ATile = l1ATensor[layoutAInL1.GetOffset(l1AOffset)];

                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0ABufId]);
                    if (mPartIdx == 0 && kPartIdx == 0) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListId]);
                    }

                    copyL1ToL0A(l0ATile, l1ATile, layoutAInL0, layoutAInL1);

                    if (mPartIdx == mPartLoop - 1 && kPartIdx == kPartLoop - 1) {
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListId]);
                    }

                    for (int nPartIdx = 0; nPartIdx < nPartLoop; nPartIdx++) {
                        uint32_t nPartActual = (nPartIdx < nPartLoop - 1) ?
                            L0TileShape::N : (nRound - nPartIdx * L0TileShape::N);

                        // Locate the current tile on L0B
                        auto l0BTile = l0BTensorList[l0BBufId];
                        LayoutBInL0 layoutBInL0 = LayoutBInL0::template MakeLayout<ElementB>(kPartActual, nPartActual);
                        // Locate the current tile of matrix B on L1
                        MatrixCoord l1BOffset{kPartIdx * L0TileShape::K, nPartIdx * L0TileShape::N};
                        auto l1BTile = l1BTensor[layoutBInL1.GetOffset(l1BOffset)];

                        // Wait for mmad finished
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BBufId]);
                        // If the current tile is the first one on the k&n axis, wait for loading matrix B from GM to L1
                        if ((kPartIdx == 0) && (nPartIdx == 0)) {
                            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListId]);
                        }

                        // Load current tile from L1 to L0B
                        copyL1ToL0B(l0BTile, l1BTile, layoutBInL0, layoutBInL1);

                        // If the current tile is the last one on the k&n axis, notify to load matrix B from GM to L1
                        if (kPartIdx == kPartLoop - 1 && nPartIdx == nPartLoop - 1) {
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListId]);
                        }
                        // Notify to do mmad
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);

                        // Locate the current tile on L0C
                        MatrixCoord l0COffset{mPartIdx * L0TileShape::M, nPartIdx * L0TileShape::N};
                        auto l0CTile = l0CTensor[layoutInL0C.GetOffset(l0COffset)];

                        // Compute the matrix multiplication on L0A and L0B and write the result to the accumulator
                        // Wait for loading L0B
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);

                        // If the current tile is the first tile on the k axis, the accumulator needs to be reset to 0
                        bool initC = ((kLoopIdx == 0) && (kPartIdx == 0));
                        // If the unit flag is enabled, the unit flag is set according to the calculation progress
                        uint8_t unitFlag = 0b00;
                        if constexpr (ENABLE_UNIT_FLAG) {
                            if ((kLoopIdx == kTileCount - 1) && (mPartIdx == mPartLoop - 1) &&
                                (kPartIdx == kPartLoop - 1) && (nPartIdx == nPartLoop - 1)) {
                                unitFlag = 0b11;
                            } else {
                                unitFlag = 0b10;
                            }
                        }
                        // Perform calculation operations
                        tileMmad(l0CTile, l0ATile, l0BTile, mPartActual, nPartActual, kPartActual, initC, unitFlag);

                        // Notify to move the next L0B tile
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BBufId]);

                        l0BBufId = (l0BBufId + 1 < STAGES) ? (l0BBufId + 1) : 0;
                    }
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0ABufId]);
                    l0ABufId = (l0ABufId + 1 < STAGES) ? (l0ABufId + 1) : 0;
                }
            }

            l1ListId = l1ListIdNext;
            kActual = kActualNext;
        }

        // copy block out
        LayoutC layoutBlock = layoutC.GetTileLayout(actualShape.GetCoordMN());

        if constexpr (!ENABLE_UNIT_FLAG) {
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
            copyL0CToGm(gmBlockC, l0CTensor, layoutBlock, layoutInL0C);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        } else {
            copyL0CToGm(gmBlockC, l0CTensor, layoutBlock, layoutInL0C, 0b11);
        }
    }

protected:
    /// Data members
    AscendC::LocalTensor<ElementA> l1ATensorList[STAGES];
    AscendC::LocalTensor<ElementB> l1BTensorList[STAGES];
    AscendC::LocalTensor<ElementA> l0ATensorList[STAGES];
    AscendC::LocalTensor<ElementB> l0BTensorList[STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensor;

    int32_t l1AEventList[STAGES];
    int32_t l1BEventList[STAGES];
    int32_t l0AEventList[STAGES];
    int32_t l0BEventList[STAGES];

    Arch::CrossCoreFlag notifyAic[STAGES] = { EVENT_ID0, EVENT_ID1 };
    Arch::CrossCoreFlag notifyAiv[STAGES] = { EVENT_ID2, EVENT_ID3 };

    uint32_t l1ListId{0};

    TileMmad tileMmad;
    CopyGmToL1A copyGmToL1A;
    CopyGmToL1B copyGmToL1B;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;
    CopyL0CToGm copyL0CToGm;
    PrologueCastB prologueCastB;
};

} // namespace Catlass::Gemm::Block

#endif // CATLASS_GEMM_BLOCK_BLOCK_MMAD_W8A16_HPP
