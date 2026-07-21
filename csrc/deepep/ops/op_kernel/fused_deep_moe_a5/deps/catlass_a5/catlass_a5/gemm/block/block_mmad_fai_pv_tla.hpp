/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_BLOCK_BLOCK_MMAD_FAI_PV_TLA_HPP
#define CATLASS_GEMM_BLOCK_BLOCK_MMAD_FAI_PV_TLA_HPP

#include "catlass_a5/catlass.hpp"
#include "catlass_a5/arch/resource.hpp"
#include "catlass_a5/coord.hpp"
#include "catlass_a5/gemm/dispatch_policy.hpp"
#include "catlass_a5/gemm/helper.hpp"
#include "catlass_a5/gemm_coord.hpp"
#include "catlass_a5/gemm/tile/tile_copy.hpp"
#include "catlass_a5/gemm/tile/tile_mmad.hpp"
#include "tla_a5/tensor.hpp"
#include "tla_a5/layout.hpp"

////////////////////////////////////////////////////////////////////

namespace Catlass::Gemm::Block {
////////////////////////////////////////////////////////////////////

template <bool PAGED_CACHE_FLAG_, bool ENABLE_UNIT_FLAG_, class L1TileShape_, class L0TileShape_, class ElementA_,
          class ElementB_, class ElementC_, class ElementBias_, class TileCopy_, class TileMmad_>
struct BlockMmadTla<MmadFAIPV<Arch::Ascend950, PAGED_CACHE_FLAG_, ENABLE_UNIT_FLAG_>, L1TileShape_, L0TileShape_,
                    ElementA_, ElementB_, ElementC_, ElementBias_, TileCopy_, TileMmad_> {
public:
    using DispatchPolicy = MmadFAIPV<Arch::Ascend950, PAGED_CACHE_FLAG_, ENABLE_UNIT_FLAG_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    using TileCopy = TileCopy_;
    using TileMmad = TileMmad_;

    using ElementA = ElementA_;
    using ElementB = ElementB_;
    using ElementC = ElementC_;
    using ElementAccumulator = typename TileCopy::ElementAccumulator;

    using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;

    using LayoutTagA = typename TileCopy::LayoutTagA;
    using LayoutTagB = typename TileCopy::LayoutTagB;
    using LayoutTagC = typename TileCopy::LayoutTagC;

    using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
    using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
    using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
    using LayoutTagL0B = typename TileCopy::LayoutTagL0B;

    static constexpr uint32_t STAGES = DispatchPolicy::STAGES;
    // L1 tile shape
    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});  // s1
    static constexpr uint32_t L1_TILE_K = tla::get<1>(L1TileShape{});  // s2
    static constexpr uint32_t L1_TILE_N = tla::get<2>(L1TileShape{});  // d
    // L0 tile shape
    static constexpr uint32_t L0_TILE_M = tla::get<0>(L0TileShape{});  // s1
    static constexpr uint32_t L0_TILE_K = tla::get<1>(L0TileShape{});  // s2
    static constexpr uint32_t L0_TILE_N = tla::get<2>(L0TileShape{});  // d
    // L1 tile size
    static constexpr uint32_t L1A_TILE_SIZE = L1_TILE_M * L1_TILE_K * sizeof(ElementA);
    static constexpr uint32_t L1B_TILE_SIZE = L1_TILE_N * L1_TILE_K * sizeof(ElementB);
    // L0 tile size
    static constexpr uint32_t L0A_TILE_SIZE = L0_TILE_M * L0_TILE_K * sizeof(ElementA);
    static constexpr uint32_t L0B_TILE_SIZE = L0_TILE_K * L0_TILE_N * sizeof(ElementB);
    static constexpr uint32_t L0C_TILE_SIZE = L0_TILE_M * L0_TILE_N * sizeof(ElementAccumulator);

    static constexpr uint32_t BLOCK_L1_SIZE = L1B_TILE_SIZE * STAGES;
    static constexpr uint32_t BLOCK_L0C_SIZE = L0C_TILE_SIZE * STAGES;

    // Check L1/L0TileShape
    static_assert(L1_TILE_M == L0_TILE_M && L1_TILE_N == L0_TILE_N && L1_TILE_K == L0_TILE_K,
                  "The situation where the basic blocks of L1 and L0 differ on the m, n, k axes is not supported yet");
    static_assert(L0A_TILE_SIZE * STAGES <= ArchTag::L0A_SIZE, "L0TileShape exceeding the L0A space!");
    static_assert(L0B_TILE_SIZE * STAGES <= ArchTag::L0B_SIZE, "L0TileShape exceeding the L0B space!");

public:
    /// Construct
    CATLASS_DEVICE
    BlockMmadTla(Arch::Resource<ArchTag> &resource, uint32_t &l1BufAddrStart, uint32_t &l0CBufAddrStart)
    {
        uint32_t l1BOffset = l1BufAddrStart;
        uint32_t l0COffset = l0CBufAddrStart;

        for (uint32_t i = 0; i < STAGES; i++) {
            // Assign L1/L0A/L0B space for each stages
            l1BTensorList_[i] = resource.l1Buf.template GetBufferByByte<ElementB>(l1BOffset + L1B_TILE_SIZE * i);
            l0ATensorList_[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(L0A_TILE_SIZE * i);
            l0BTensorList_[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_TILE_SIZE * i);
            l0CTensorList_[i] =
                resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(l0COffset + L0C_TILE_SIZE * i);

            // Assign event ID for each stages
            l1BEventList_[i] = BLOCK_EVENT_ID + i + STAGES;
            l0AEventList_[i] = i;
            l0BEventList_[i] = i + STAGES;
            l0CEventList_[i] = BLOCK_EVENT_ID + i;

            // The event id that needs to be set before the loop
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList_[i]);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventList_[i]);
        }

        l1BufAddrStart += BLOCK_L1_SIZE;
        l0CBufAddrStart += BLOCK_L0C_SIZE;
    }

    /// Destructor
    CATLASS_DEVICE
    ~BlockMmadTla()
    {
        for (uint32_t i = 0; i < STAGES; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList_[i]);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventList_[i]);
        }
    }

public:
    template <class TensorA, class TensorB, class TensorC, class Shape>
    CATLASS_DEVICE void operator()(TensorA &tensorA, TensorB &tensorB, TensorC &tensorC,
                                   AscendC::GlobalTensor<int32_t> &blockTable, Shape &actualShape, int64_t taskId,
                                   int32_t blockSize)
    {
        int32_t blockM = tla::get<0>(actualShape);
        int32_t blockN = tla::get<1>(actualShape);
        int32_t blockK = tla::get<2>(actualShape);

        // load matrix B tile from GM to L1
        auto layoutBInL1 = tla::MakeLayout<ElementB, LayoutTagL1B>(blockK, blockN);
        auto tensorL1B = tla::MakeTensor(l1BTensorList_[l1BListId_], layoutBInL1, Arch::PositionL1{});
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList_[l1BListId_]);
        CopyInL1B(tensorL1B, tensorB, blockTable, blockK, blockN, blockSize);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList_[l1BListId_]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList_[l1BListId_]);

        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventList_[l0CListId_]);
        auto layoutInL0C = tla::MakeLayoutL0C(blockM, blockN);
        auto tensorL0C = tla::MakeTensor(l0CTensorList_[l0CListId_], layoutInL0C, Arch::PositionL0C{});

        uint32_t nLoops = (blockN + L0_TILE_N - 1) / L0_TILE_N;
        uint32_t tailSize = blockN % L0_TILE_N;
        uint32_t tailN = tailSize ? tailSize : L0_TILE_N;

        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList_[l0AListId_]);
        for (uint32_t nIdx = 0; nIdx < nLoops; nIdx++) {
            uint32_t tileN = (nIdx == (nLoops - 1)) ? tailN : L0_TILE_N;

            // load matrix A tile from L1 to L0A
            auto layoutAInL0 = tla::MakeLayout<ElementA, LayoutTagL0A>(blockM, blockK);
            auto tensorL0A = tla::MakeTensor(l0ATensorList_[l0AListId_], layoutAInL0, Arch::PositionL0A{});
            if (nIdx == 0) {
                auto tensorL1TileA = GetTile(tensorA, tla::MakeCoord(0, 0), tla::MakeShape(blockM, blockK));
                copyL1ToL0A(tensorL0A, tensorL1TileA);
            }

            // load matrix B tile from L1 to L0B
            auto layoutBInL0 = tla::MakeLayout<ElementB, LayoutTagL0B>(blockK, tileN);
            auto tensorL0B = tla::MakeTensor(l0BTensorList_[l0BListId_], layoutBInL0, Arch::PositionL0B{});
            auto tensorL1TileB = GetTile(tensorL1B, tla::MakeCoord(0, nIdx * L0_TILE_N), tla::MakeShape(blockK, tileN));
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList_[l0BListId_]);
            copyL1ToL0B(tensorL0B, tensorL1TileB);

            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0CEventList_[l0CListId_]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0CEventList_[l0CListId_]);

            // Mmad
            bool initC = true;
            uint32_t tileM = (blockM == 1 ? M_ALIGN : blockM);
            uint32_t tileK = blockK;
            auto tensorTileL0C = GetTile(tensorL0C, tla::MakeCoord(0, nIdx * L0_TILE_N), tla::MakeShape(tileM, tileN));
            tileMmad(tensorTileL0C, tensorL0A, tensorL0B, tileM, tileN, tileK, initC);

            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList_[l0BListId_]);
            l0BListId_ = (l0BListId_ + 1 < STAGES) ? (l0BListId_ + 1) : 0;
        }
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList_[l0AListId_]);
        l0AListId_ = (l0AListId_ + 1 < STAGES) ? (l0AListId_ + 1) : 0;

        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList_[l1BListId_]);
        l1BListId_ = (l1BListId_ + 1 < STAGES) ? (l1BListId_ + 1) : 0;

        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CEventList_[l0CListId_]);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CEventList_[l0CListId_]);

        AscendC::CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(MM2_RES_INTRA_EVENT[taskId]);
        AscendC::CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(16 + MM2_RES_INTRA_EVENT[taskId]);

        // copy block out
        using CopyL0CToDst = typename TileCopy::template CopyL0CToDst<TensorC>;
        CopyL0CToDst copyL0CToDst;
        copyL0CToDst(tensorC, tensorL0C);

        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventList_[l0CListId_]);
        l0CListId_ = (l0CListId_ + 1 < STAGES) ? (l0CListId_ + 1) : 0;
    }

private:
    template <class TensorL1B, class TensorB>
    CATLASS_DEVICE void CopyInL1B(TensorL1B &tensorL1B, TensorB &tensorB, AscendC::GlobalTensor<int32_t> &blockTable,
                                  int32_t tileK, int32_t tileN, int32_t blockSize)
    {
        using CopyGmToL1B = typename TileCopy::template CopyGmToL1B<TensorB>;
        CopyGmToL1B copyGmToL1B;

        if constexpr (PAGED_CACHE_FLAG_) {
            int32_t copyRowCnt = 0;
            int32_t blockLoops = (tileK + blockSize - 1) / blockSize;

            for (int32_t blockIdx = 0; blockIdx < blockLoops; ++blockIdx) {
                int32_t curCopyRow = blockIdx == blockLoops - 1 ? tileK - copyRowCnt : blockSize;
                int32_t idInBlockTable = blockTable.GetValue(blockIdx);

                auto tensorL1TileB =
                    GetTile(tensorL1B, tla::MakeCoord(copyRowCnt, 0), tla::MakeShape(curCopyRow, tileN));

                auto tensorTileB =
                    GetTile(tensorB, tla::MakeCoord(idInBlockTable * blockSize, 0), tla::MakeShape(curCopyRow, tileN));

                copyGmToL1B(tensorL1TileB, tensorTileB);
                copyRowCnt += curCopyRow;
            }
        } else {
            auto tensorTileB = GetTile(tensorB, tla::MakeCoord(0, 0), tla::MakeShape(tileK, tileN));

            copyGmToL1B(tensorL1B, tensorTileB);
        }
    }

private:
    static constexpr uint16_t M_ALIGN = 16;
    static constexpr uint32_t SYNC_MODE = 4;
    static constexpr uint32_t MM2_RES_INTRA_EVENT[2] = {7, 8};
    static constexpr uint32_t BLOCK_EVENT_ID = 4;

    // Multi-stage tensors list
    AscendC::LocalTensor<ElementA> l1ATensorList_[STAGES];
    AscendC::LocalTensor<ElementB> l1BTensorList_[STAGES];
    AscendC::LocalTensor<ElementA> l0ATensorList_[STAGES];
    AscendC::LocalTensor<ElementB> l0BTensorList_[STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensorList_[STAGES];

    // Multi-stage event id list
    int32_t l1AEventList_[STAGES];
    int32_t l1BEventList_[STAGES];
    int32_t l0AEventList_[STAGES];
    int32_t l0BEventList_[STAGES];
    int32_t l0CEventList_[STAGES];

    uint32_t l1AListId_{0};
    uint32_t l1BListId_{0};
    uint32_t l0AListId_{0};
    uint32_t l0BListId_{0};
    uint32_t l0CListId_{0};

    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;
    TileMmad tileMmad;
};
}  // namespace Catlass::Gemm::Block
#endif
