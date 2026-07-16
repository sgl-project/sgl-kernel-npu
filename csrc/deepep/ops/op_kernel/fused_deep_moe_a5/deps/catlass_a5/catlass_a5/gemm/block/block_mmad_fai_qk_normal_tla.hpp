/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_BLOCK_MMAD_QK_TLA_HPP
#define CATLASS_GEMM_BLOCK_MMAD_QK_TLA_HPP

#include "catlass_a5/catlass.hpp"
#include "catlass_a5/arch/resource.hpp"
#include "catlass_a5/coord.hpp"
#include "catlass_a5/gemm/dispatch_policy.hpp"
#include "catlass_a5/gemm/helper.hpp"
#include "catlass_a5/gemm_coord.hpp"
#include "catlass_a5/gemm/tile/tile_copy.hpp"
#include "catlass_a5/gemm/tile/tile_mmad.hpp"
#include "tla_a5/layout.hpp"
#include "tla_a5/tensor.hpp"

////////////////////////////////////////////////////////////////////

namespace Catlass::Gemm::Block {
////////////////////////////////////////////////////////////////////

template <
    class ArchTag_,
    bool PAGED_CACHE_FLAG_,
    bool ENABLE_UNIT_FLAG_,
    class L1TileShape_,
    class L0TileShape_,
    class ElementA_,
    class ElementB_,
    class ElementC_,
    class ElementBias_,
    class TileCopy_,
    class TileMmad_>
struct BlockMmadTla<
    MmadFAIQK<ArchTag_, PAGED_CACHE_FLAG_, ENABLE_UNIT_FLAG_>,
    L1TileShape_,
    L0TileShape_,
    ElementA_,
    ElementB_,
    ElementC_,
    ElementBias_,
    TileCopy_,
    TileMmad_> {
public:
    // Type Aliases
    using DispatchPolicy = MmadFAIQK<ArchTag_, PAGED_CACHE_FLAG_, ENABLE_UNIT_FLAG_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    using ElementA = ElementA_;
    using LayoutA = typename TileCopy_::LayoutA;
    using ElementB = ElementB_;
    using LayoutB = typename TileCopy_::LayoutB;
    using ElementC = ElementC_;
    using LayoutC = typename TileCopy_::LayoutC;

    using TileMmad = TileMmad_;

    using CopyL1ToL0A = typename TileCopy_::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy_::CopyL1ToL0B;

    using ElementAccumulator = typename TileCopy_::ElementAccumulator;

    using LayoutTagL1A = typename TileCopy_::LayoutTagL1A;
    using LayoutTagL1B = typename TileCopy_::LayoutTagL1B;
    using LayoutTagL0A = typename TileCopy_::LayoutTagL0A;
    using LayoutTagL0B = typename TileCopy_::LayoutTagL0B;

    using L1AAlignHelper = typename TileCopy_::L1AAlignHelper;
    using L1BAlignHelper = typename TileCopy_::L1BAlignHelper;

    static_assert(tla::is_tuple<L1TileShape>::value && tla::is_static<L1TileShape>::value,
        "L1TileShape must be tla::tuple and static!");
    static_assert(tla::is_tuple<L0TileShape>::value && tla::is_static<L0TileShape>::value,
        "L0TileShape must be tla::tuple and static!");

    static constexpr bool ENABLE_UNIT_FLAG = DispatchPolicy::ENABLE_UNIT_FLAG;
    static constexpr uint32_t STAGES = DispatchPolicy::STAGES;
    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});
    static constexpr uint32_t L0_TILE_M = tla::get<0>(L0TileShape{});
    static constexpr uint32_t L0_TILE_N = tla::get<1>(L0TileShape{});
    static constexpr uint32_t L0_TILE_K = tla::get<2>(L0TileShape{});

    // L1 tile size
    static constexpr uint32_t L1A_TILE_SIZE = L1_TILE_M * L1_TILE_K * sizeof(ElementA);
    static constexpr uint32_t L1B_TILE_SIZE = L1_TILE_N * L1_TILE_K * sizeof(ElementB);
    // L0 tile size
    static constexpr uint32_t L0A_TILE_SIZE = L0_TILE_M * L0_TILE_K * sizeof(ElementA);
    static constexpr uint32_t L0B_TILE_SIZE = L0_TILE_K * L0_TILE_N * sizeof(ElementB);
    static constexpr uint32_t L0C_TILE_SIZE = L0_TILE_M * L0_TILE_N * sizeof(ElementAccumulator);
    static constexpr uint32_t L0A_PINGPONG_BUF_SIZE = ArchTag::L0A_SIZE / STAGES;
    static constexpr uint32_t L0B_PINGPONG_BUF_SIZE = ArchTag::L0B_SIZE / STAGES;
    static constexpr uint32_t L0C_PINGPONG_BUF_SIZE = ArchTag::L0C_SIZE / STAGES;

    // Check LayoutC
    static_assert(tla::detail::isRowMajor<LayoutC>::value ||
                      ((std::is_same_v<ElementC, half> || std::is_same_v<ElementC, bfloat16_t> ||
                          std::is_same_v<ElementC, float>) && tla::detail::iszN<ElementC, LayoutC>::value),
        "LayoutC only supports zN in half or bfloat16 or float, RowMajor in all dtype yet!");
    
    // Check L1TileShape
    static_assert(L1A_TILE_SIZE + L1B_TILE_SIZE * STAGES <= ArchTag::L1_SIZE,
        "L1TileShape exceeding the L1 space!");

    // Check L0TileShape
    static_assert(L0A_TILE_SIZE * STAGES <= ArchTag::L0A_SIZE, "L0TileShape exceeding the L0A space!");
    static_assert(L0B_TILE_SIZE * STAGES <= ArchTag::L0B_SIZE, "L0TileShape exceeding the L0B space!");
    static_assert(L0C_TILE_SIZE <= ArchTag::L0C_SIZE, "L0TileShape exceeding the L0C space!");

    static constexpr uint32_t BLOCK_SIZE = 16;
    static constexpr uint32_t EMBED_SPLIT_SIZE = 128;
    static constexpr uint32_t UNIT_BLOCK_STACK_NUM = 4;
    static constexpr uint32_t KV_BASE_BLOCK = 512;
    static constexpr uint32_t KV_SPLIT_SIZE = 128;

    CATLASS_DEVICE
    BlockMmadTla(Arch::Resource<ArchTag> &resource, uint32_t l1BufAddrStart = 0)
    {
        // Allocate L1 memory space
        l1ATensor = resource.l1Buf.template GetBufferByByte<ElementA>(l1BufAddrStart);
        for (uint32_t i = 0; i < STAGES; i++) {
            l1BTensor[i] =
                resource.l1Buf.template GetBufferByByte<ElementB>(l1BufAddrStart + L1A_TILE_SIZE + L1B_TILE_SIZE * i);
            l0ATensor[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(L0A_PINGPONG_BUF_SIZE * i);
            l0BTensor[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_PINGPONG_BUF_SIZE * i);
        }
        l0CTensor = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(0);
    }

    CATLASS_DEVICE
    ~BlockMmadTla() {}

    template <class TensorA>
    CATLASS_DEVICE
    void loadQGM(TensorA &tensorA){
        using CopyGmToL1A = Gemm::Tile::TileCopyFAQTla<ArchTag, TensorA, typename TileCopy_::TensorL1A>;
        CopyGmToL1A copyGmToL1A;

        uint32_t rowNum = tla::get<0>(tensorA.shape()) * tla::get<1>(tensorA.shape());
        uint32_t embed = tla::get<2>(tensorA.shape());
        auto L1A_LAYOUT = tla::MakeLayout<ElementA, LayoutTagL1A>(rowNum, embed);
        auto tensorL1A = tla::MakeTensor(l1ATensor, L1A_LAYOUT, Arch::PositionL1{});
        copyGmToL1A(tensorL1A, tensorA);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID3);
    }
    
    CATLASS_DEVICE
    void getBlockShape(GemmCoord &actualShape, uint32_t &nowNIdx, uint32_t &kIdx,
                       uint32_t &nLoop, uint32_t &kLoop, uint32_t &kvSeqlen, uint32_t &embed, bool firstBlock, uint32_t maskTailS = 0)
    {
        uint32_t nSplitSize = KV_SPLIT_SIZE;
        uint32_t embedSplitSize = EMBED_SPLIT_SIZE;
        if (nowNIdx == nLoop - 1) {
            nSplitSize = kvSeqlen - nowNIdx * KV_SPLIT_SIZE;
        }
        if (firstBlock && maskTailS != 0) {
            nSplitSize = nSplitSize - maskTailS;
        }
        // }
        if (kIdx == kLoop - 1) {
            embedSplitSize = embed - kIdx * EMBED_SPLIT_SIZE;
        }
        actualShape[1] = nSplitSize;
        actualShape[2] = embedSplitSize;
    }

    CATLASS_DEVICE
    void getKVOffset(AscendC::GlobalTensor<int32_t> &gBlockTable, uint32_t &kOffset, uint32_t &nowNIdx, uint32_t &kIdx,
                     uint32_t &nLoop, uint32_t &kLoop, uint32_t &strideKV, uint32_t &blockSize, uint32_t maskTailS = 0)
    {
        if (nowNIdx >= nLoop || kIdx >= kLoop) {
            kOffset = 0;
        }
        if constexpr (PAGED_CACHE_FLAG_) {
            uint32_t blockTableId = gBlockTable.GetValue(nowNIdx);
            kOffset = blockTableId * blockSize * strideKV + maskTailS * strideKV + kIdx * EMBED_SPLIT_SIZE;
        } else {
            kOffset = nowNIdx * KV_SPLIT_SIZE * strideKV + kIdx * EMBED_SPLIT_SIZE;
        }
    }

    template <class TensorA, class TensorB>
    CATLASS_DEVICE
    void operator()(TensorA &tensorA, TensorB &tensorB, AscendC::GlobalTensor<ElementC> gC, 
                    AscendC::GlobalTensor<int32_t> gBlockTable, GemmCoord actualOriShape,
                    uint32_t &nIdx, uint32_t &nLoop, uint32_t &blockSize, uint32_t kvSeqlen, uint32_t strideKV)
    {
        uint32_t rowNum = actualOriShape[0];
        uint32_t embed = actualOriShape[2];
        uint32_t kLoop = CeilDiv<L1_TILE_K>(embed);
        uint32_t nkBlockLoop = nLoop * kLoop;
        GemmCoord actualShape{rowNum, 0, 0};
        GemmCoord actualNextShape{rowNum, 0, 0};
        uint32_t nkBlockNextIdx = nIdx * kLoop + 1;
        uint32_t gBOffset = 0;
        uint32_t gBNextOffset = 0;
        uint32_t stackTile = 0;
        for (uint32_t blockStackIdx = 0; (blockStackIdx < UNIT_BLOCK_STACK_NUM) && ((nIdx + blockStackIdx) < nLoop);
            ++blockStackIdx) {
            for (uint32_t kIdx = 0; kIdx < kLoop; kIdx++) {
                uint32_t nowNIdx = nIdx + blockStackIdx;
                uint32_t nLoopNextIdx = nkBlockNextIdx / kLoop;
                uint32_t kLoopNextIdx = nkBlockNextIdx % kLoop;
                uint32_t gCOffset = blockStackIdx / 2 * 2 * KV_SPLIT_SIZE;
                getBlockShape(actualShape, nowNIdx, kIdx, nLoop, kLoop, kvSeqlen, embed, nowNIdx == nIdx);
                getBlockShape(actualNextShape, nLoopNextIdx, kLoopNextIdx, nLoop, kLoop, kvSeqlen, embed, nLoopNextIdx == nIdx);
                getKVOffset(gBlockTable, gBOffset, nowNIdx, kIdx, nLoop, kLoop, strideKV, blockSize);
                getKVOffset(gBlockTable, gBNextOffset, nLoopNextIdx, kLoopNextIdx, nLoop, kLoop, strideKV, blockSize);
                bool firstItr = ((blockStackIdx % 2) == 0) && (kIdx == 0);
                bool endItr = (((blockStackIdx % 2) == 1) || (nowNIdx == nLoop - 1)) && (kIdx == kLoop - 1);
                bool firstQtr = blockStackIdx == 0;
                bool endQItr = ((nowNIdx == nLoop - 1) || (blockStackIdx == UNIT_BLOCK_STACK_NUM - 1)) && (kIdx == kLoop - 1);
                int cc = 1;
                bool initMmad = kIdx == 0;
                stackTile += actualShape[1];
                auto layoutS = tla::MakeLayout(tla::MakeShape(rowNum, stackTile), tla::MakeStride(Int<512>{}, Int<1>{}));
                auto tensorCOffset = tla::MakeTensor(gC[gCOffset], layoutS, Arch::PositionGM{});
                computeQK(tensorA, tensorB, tensorCOffset, gBOffset, gBNextOffset, 
                          actualShape, actualNextShape, blockStackIdx, nkBlockNextIdx, nkBlockLoop, strideKV, firstItr, endItr, initMmad, firstQtr, endQItr);
                ++nkBlockNextIdx;
                if (endItr) {
                    stackTile = 0;
                }
            }
        }
    }

    template <class TensorA, class TensorB, class TensorC>
    CATLASS_DEVICE void computeQK(
        TensorA &tensorA, TensorB &tensorB, TensorC &tensorC, uint32_t gBOffset, uint32_t gBNextOffset, 
        GemmCoord actualShape, GemmCoord actualNextShape, uint32_t nowIdx, uint32_t &nkblockIdx,
        uint32_t &nkblockLoop, uint32_t strideKV, bool firstItr, bool endItr, bool initMmad, bool firstQItr, bool endQItr)
    {
        using CopyGmToL1B = typename TileCopy_::template CopyGmToL1B<TensorB>;
        using CopyL0CToGm = typename TileCopy_::template CopyL0CToGm<TensorC>;
        CopyGmToL1B copyGmToL1B;
        CopyL0CToGm copyL0CToGm;

        uint32_t mActual = actualShape.m();
        uint32_t kActual = actualShape.k();
        uint32_t nActual = actualShape.n();
        uint32_t mRound = RoundUp<L1AAlignHelper::M_ALIGNED>(mActual);
        uint32_t kRound = RoundUp<L1AAlignHelper::M_ALIGNED>(kActual);
        uint32_t nRound = RoundUp<L1AAlignHelper::M_ALIGNED>(nActual);
        auto layoutAInL1 = tla::MakeLayout<ElementA, LayoutTagL1A>(mRound, kActual);
        auto layoutAInL0 = tla::MakeLayout<ElementA, LayoutTagL0A>(mRound, kActual);
        auto layoutBInL1 = tla::MakeLayout<ElementB, LayoutTagL1B>(kActual, nActual);
        auto layoutBInL0 = tla::MakeLayout<ElementB, LayoutTagL0B>(kActual, nRound);
        uint32_t locPingPongFlag = nowIdx % 2;
        uint32_t l1KvPingPongFlag = nkblockIdx % 2;
        uint32_t l0ABPingPongFlag = nkblockIdx % 2;
        auto tensorL1A = tla::MakeTensor(l1ATensor, layoutAInL1, Arch::PositionL1{});
        auto tensorL0A = tla::MakeTensor(l0ATensor[0], layoutAInL0, Arch::PositionL0A{});
        auto tensorTileB = GetTile(tensorB, tla::MakeCoord(gBOffset % strideKV, gBOffset / strideKV), tla::MakeShape(kActual, nActual));
        auto tensorL1B = tla::MakeTensor(l1BTensor[l1KvPingPongFlag], layoutBInL1, Arch::PositionL1{});
        auto tensorL0B = tla::MakeTensor(l0BTensor[l0ABPingPongFlag], layoutBInL0, Arch::PositionL0B{});
        if (nkblockIdx == 1) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1KvPingPongFlag);
            copyGmToL1B(tensorL1B, tensorTileB);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1KvPingPongFlag);
        }
        if (nkblockIdx != nkblockLoop) {
            uint32_t nNextActual = actualNextShape.n();
            uint32_t kNextActual = actualNextShape.k();
            auto tensorTileBNext = GetTile(tensorB, tla::MakeCoord(gBNextOffset % strideKV, gBNextOffset / strideKV), tla::MakeShape(kNextActual, nNextActual));
            auto layoutBNextInL1 = tla::MakeLayout<ElementB, LayoutTagL1B>(kNextActual, nNextActual);
            auto tensorL1BNext = tla::MakeTensor(l1BTensor[1 - l1KvPingPongFlag], layoutBNextInL1, Arch::PositionL1{});
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(1 - l1KvPingPongFlag);
            copyGmToL1B(tensorL1BNext, tensorTileBNext);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(1 - l1KvPingPongFlag);
        }
        if(firstQItr) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(1);
            copyL1ToL0A(tensorL0A, tensorL1A);
        }
        
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1KvPingPongFlag);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag + 2);
        copyL1ToL0B(tensorL0B, tensorL1B);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1KvPingPongFlag);

        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0ABPingPongFlag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0ABPingPongFlag);
        uint8_t unitFlag = 0b00;
        if constexpr (!ENABLE_UNIT_FLAG_) {
            if (firstItr) {
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(0);
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(1);
            }
        } else {
            unitFlag = 0b11;
        }
        auto layoutCTileInL0 = tla::MakeLayoutL0C(mRound, nActual);
        auto tensorL0CTile = tla::MakeTensor(l0CTensor[locPingPongFlag * mRound * 128], layoutCTileInL0, Arch::PositionL0C{});
        tileMmad(
            tensorL0CTile, tensorL0A, tensorL0B, 
            mRound, nActual, kActual, initMmad, unitFlag);
        if (endQItr) {
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(1);
        }
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag + 2);
        auto layoutCInL0 = tla::MakeLayoutL0C(mActual, (uint32_t)256);
        auto tensorL0C = tla::MakeTensor(l0CTensor, layoutCInL0, Arch::PositionL0C{});
        if (endItr) {
            if constexpr (!ENABLE_UNIT_FLAG_) {
                AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l1KvPingPongFlag);
                AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l1KvPingPongFlag);
                copyL0CToGm(tensorC, tensorL0C);
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(1);
            } else {
                copyL0CToGm(tensorC, tensorL0C, unitFlag);
            }
        }
    }
 
protected:
    /// Data members
    AscendC::LocalTensor<ElementA> l1ATensor;
    AscendC::LocalTensor<ElementB> l1BTensor[STAGES];
    AscendC::LocalTensor<ElementA> l0ATensor[STAGES];
    AscendC::LocalTensor<ElementB> l0BTensor[STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensor;

    TileMmad tileMmad;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;
};

////////////////////////////////////////////////////////////////////

} // namespace Catlass::Gemm::Block

#endif // CATLASS_GEMM_BLOCK_MMAD_QK_TLA_HPP