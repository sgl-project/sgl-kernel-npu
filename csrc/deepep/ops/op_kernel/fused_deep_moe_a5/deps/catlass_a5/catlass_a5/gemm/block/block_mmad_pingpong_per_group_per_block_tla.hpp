/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_BLOCK_BLOCK_MMAD_PINGPONG_PER_GROUP_PER_BLOCK_TLA_HPP
#define CATLASS_GEMM_BLOCK_BLOCK_MMAD_PINGPONG_PER_GROUP_PER_BLOCK_TLA_HPP

#include "catlass_a5/catlass.hpp"
#include "catlass_a5/arch/resource.hpp"
#include "catlass_a5/coord.hpp"
#include "catlass_a5/gemm_coord.hpp"
#include "catlass_a5/gemm/dispatch_policy.hpp"
#include "catlass_a5/gemm/helper.hpp"
#include "catlass_a5/gemm/tile/tile_copy.hpp"
#include "catlass_a5/gemm/tile/tile_mmad.hpp"
#include "tla_a5/layout.hpp"
#include "tla_a5/tensor.hpp"

namespace Catlass::Gemm::Block {

template <
    class ArchTag_,
    bool ENABLE_UNIT_FLAG_,
    class L1TileShape_,
    class L0TileShape_,
    class ElementA_,
    class ElementB_,
    class ElementC_,
    class ElementBias_,
    class TileCopy_,
    class TileMmad_
> 
struct BlockMmadTla <
    MmadPingpongPertile<ArchTag_, ENABLE_UNIT_FLAG_>,
    L1TileShape_,
    L0TileShape_,
    ElementA_,
    ElementB_,
    ElementC_,
    ElementBias_,
    TileCopy_,
    TileMmad_
> { 
public:
    struct Arguments {
        GM_ADDR aGmAddr{nullptr};
        GM_ADDR bGmAddr{nullptr};
        GM_ADDR cGmAddr{nullptr};
        GM_ADDR workspaceGmAddr{nullptr};
    };

    using Params = Arguments;

    using DispatchPolicy =  MmadPingpongPertile<ArchTag_, ENABLE_UNIT_FLAG_>;
    using ArchTag = typename DispatchPolicy::ArchTag;

    using TileCopy = TileCopy_;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    using ElementA = ElementA_;
    using LayoutA = typename TileCopy::LayoutA;
    using ElementB = ElementB_;
    using LayoutB = typename TileCopy::LayoutB;
    using ElementC = ElementC_;
    using LayoutC = typename TileCopy::LayoutC;
    using ElementBias = ElementBias_;

    using TileMmad = TileMmad_;

    using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;
    using CopyL1ToBT = typename TileCopy::CopyL1ToBT;

    using ElementAccumulator = typename TileCopy::ElementAccumulator;

    static constexpr bool HAS_BIAS = TileCopy::HAS_BIAS;

    using LayoutTagA = typename TileCopy::LayoutTagA;
    using LayoutTagB = typename TileCopy::LayoutTagB;
    using LayoutTagC = typename TileCopy::LayoutTagC;
    using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
    using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
    using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
    using LayoutTagL0B = typename TileCopy::LayoutTagL0B;

    uint64_t m_{1};
    uint64_t n_{1};
    uint64_t k_{1};

    constexpr static uint64_t STAGES = DispatchPolicy::STAGES; 

    constexpr static uint64_t ML1_ = tla::get<0>(L1TileShape{});
    constexpr static uint64_t NL1_ = tla::get<1>(L1TileShape{});
    constexpr static uint64_t KL1_ = tla::get<2>(L1TileShape{});

    constexpr static uint64_t ML0_ = tla::get<0>(L0TileShape{});
    constexpr static uint64_t NL0_ = tla::get<1>(L0TileShape{});
    constexpr static uint64_t KL0_ = tla::get<2>(L0TileShape{});

    constexpr static uint64_t L1A_TILE_SIZE = ML1_ * KL1_ * sizeof(ElementA);
    constexpr static uint64_t L1B_TILE_SIZE = NL1_ * KL1_ * sizeof(ElementB);

    constexpr static uint64_t L0A_TILE_SIZE = ML0_ * KL0_ * sizeof(ElementA);
    constexpr static uint64_t L0B_TILE_SIZE = NL0_ * KL0_ * sizeof(ElementB);
    constexpr static uint64_t L0C_TILE_SIZE = ML0_ * NL0_ * sizeof(ElementAccumulator);

    static_assert(ML1_ == ML0_ && NL1_ == NL0_ && KL1_ == KL0_,
        "The situation where the basic blocks of L1 and L0 differ is not supported yet");

#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)
    static_assert(TileCopy::CopyMode == Tile::CopyL0CToUBMode::SPLIT_M,
        "only supported SPLIT_M yet");
#endif

    CATLASS_DEVICE
    BlockMmadTla(const GemmCoord &shape)
    {
        uint32_t l1AOffset = 0;
        uint32_t l1BOffset = l1AOffset + L1A_TILE_SIZE * STAGES;
        // Init buffers
        for (uint32_t i = 0; i < STAGES; i++) {
            l1ATensorList_[i] = l1Buf_[l1AOffset + L1A_TILE_SIZE * i].template ReinterpretCast<ElementA>();
            l1BTensorList_[i] = l1Buf_[l1BOffset + L1B_TILE_SIZE * i].template ReinterpretCast<ElementB>();
            l0ATensorList_[i] = l0ABuf_[L0A_TILE_SIZE * i].template ReinterpretCast<ElementA>();
            l0BTensorList_[i] = l0BBuf_[L0B_TILE_SIZE * i].template ReinterpretCast<ElementB>();
            l0CTensorList_[i] = l0CBuf_[L0C_TILE_SIZE * i].template ReinterpretCast<ElementAccumulator>();

            l1AEventList_[i] = i;
            l1BEventList_[i] = i + STAGES;
            l0AEventList_[i] = i;
            l0BEventList_[i] = i + STAGES;
            l0CEventList_[i] = i;

            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList_[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList_[i]);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList_[i]);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList_[i]);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventList_[i]);
        }
        m_ = shape.m();
        n_ = shape.n();
        k_ = shape.k();
    }

    CATLASS_DEVICE
    ~BlockMmadTla()
    {
        for (uint32_t i = 0; i < STAGES; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList_[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList_[i]);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList_[i]);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList_[i]);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventList_[i]);
        }
    }

    CATLASS_DEVICE void WaitForVector(uint32_t crossPingPongID)
    {
        AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIV_SYNC_AIC_FLAG + (crossPingPongID_));
        AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIV_SYNC_AIC_FLAG + FLAG_ID_MAX + (crossPingPongID_));
    }

    CATLASS_DEVICE void NotifyVector(uint32_t crossPingPongID)
    {
        AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIC_SYNC_AIV_FLAG + (crossPingPongID_));
        AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIC_SYNC_AIV_FLAG + FLAG_ID_MAX + (crossPingPongID_));
    }

public:
    template <class DstTensor, class SrcATensor, class SrcBTensor>
    CATLASS_DEVICE
    void operator()(
        DstTensor& cTensor,
        SrcATensor& aTensor, 
        SrcBTensor& bTensor,
        GemmCoord tileShape)
    {
        using CopyL0CToOut = typename TileCopy::template CopyL0CToDst<DstTensor>;
        CopyL0CToOut copyL0CToOut;

        uint32_t curML0 = tileShape.m();
        uint32_t curNL0 = tileShape.n();
        auto l0CLayout = tla::MakeLayoutL0C(curML0, curNL0);

        uint32_t kL1Loops = CeilDiv(k_, KL1_);

        // GM -> L1
        uint32_t kL1OffsetLength = 0;
        for (uint32_t kL1Idx = 0; kL1Idx < kL1Loops; ++kL1Idx) {
            uint32_t curKL1 = (kL1Idx + 1 == kL1Loops) ? (k_ - kL1OffsetLength) : KL1_;

            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList_[l1ListId_]);
            auto l1ALayout = tla::MakeLayout<ElementA, LayoutTagL1A>(curML0, curKL1);
            auto l1ATensor = tla::MakeTensor(l1ATensorList_[l1ListId_], l1ALayout, Arch::PositionL1{});

            auto aTileTensor = GetTile(
                aTensor, 
                tla::MakeCoord(0, kL1OffsetLength),
                tla::MakeShape(curML0, curKL1)
            );

            using CopyGmToL1A = typename TileCopy::template CopyGmToL1A<SrcATensor>;
            CopyGmToL1A copyGmToL1A;
            copyGmToL1A(l1ATensor, aTileTensor);

            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList_[l1ListId_]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList_[l1ListId_]);

            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList_[l1ListId_]);

            auto l1BLayout = tla::MakeLayout<ElementB, LayoutTagL1B>(curKL1, curNL0);
            auto l1BTensor = tla::MakeTensor(l1BTensorList_[l1ListId_], l1BLayout, Arch::PositionL1{});
            auto bTileTensor = GetTile(
                bTensor, 
                tla::MakeCoord(kL1OffsetLength, 0),
                tla::MakeShape(curKL1, curNL0)
            );

            using CopyGmToL1B = typename TileCopy::template CopyGmToL1B<SrcBTensor>;
            CopyGmToL1B copyGmToL1B;
            copyGmToL1B(l1BTensor, bTileTensor);

            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList_[l1ListId_]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList_[l1ListId_]);

            kL1OffsetLength += curKL1;

            // L1 -> L0
            uint32_t kL0Loops = CeilDiv(curKL1, KL0_);
            for (uint32_t kL0Idx = 0; kL0Idx < kL0Loops; ++kL0Idx) {
                uint32_t curKL0 = (kL0Idx + 1 == kL0Loops) ? (curKL1 - kL0Idx * KL0_) : KL0_;

                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList_[l0ListId_]);

                auto l0ALayout = tla::MakeLayout<ElementA, LayoutTagL0A>(curML0, curKL0);
                auto l0ATensor = tla::MakeTensor(l0ATensorList_[l1ListId_], l0ALayout, Arch::PositionL0A{});
                auto l1ATileTensor = GetTile(
                    l1ATensor,
                    tla::MakeCoord(0, kL0Idx * KL0_),
                    tla::MakeShape(curML0, curKL0)
                );

                copyL1ToL0A(l0ATensor, l1ATileTensor);

                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList_[l0ListId_]);
                
                auto l0BLayout = tla::MakeLayout<ElementB, LayoutTagL0B>(curKL0, curNL0);
                auto l0BTensor = tla::MakeTensor(l0BTensorList_[l0ListId_], l0BLayout, Arch::PositionL0B{});
                auto l1BTileTensor = GetTile(
                    l1BTensor,
                    tla::MakeCoord(kL0Idx * KL0_, 0),
                    tla::MakeShape(curKL0, curNL0)
                );

                copyL1ToL0B(l0BTensor, l1BTileTensor); 

                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventList_[l0CListId_]);
                auto l0CTensor = tla::MakeTensor(l0CTensorList_[l0CListId_], l0CLayout, Arch::PositionL0C{});

                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0CEventList_[l0CListId_]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0CEventList_[l0CListId_]);

                // Mmad
                bool initC = (kL1Idx == 0) && (kL0Idx == 0);
                curML0 = (curML0 == 1 ? M_ALIGN : curML0);
                tileMmad(l0CTensor, l0ATensor, l0BTensor, curML0, curNL0, curKL0, true);
                
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList_[l0ListId_]);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList_[l0ListId_]);
                l0ListId_ = (l0ListId_ + 1 < STAGES) ? (l0ListId_ + 1) : 0;

                AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CEventList_[l0CListId_]);
                AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CEventList_[l0CListId_]);

                WaitForVector(crossPingPongID_);
                auto outTensor = tla::MakeTensor(crossPingPongID_ == 0 ? cTensor.data() : cTensor.data()[UB_OFFSET],
                                                 cTensor.layout(), Arch::PositionUB{});
                copyL0CToOut(outTensor, l0CTensor);
                NotifyVector(crossPingPongID_);
                
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventList_[l0CListId_]);
                l0CListId_ = (l0CListId_ + 1 < STAGES) ? (l0CListId_ + 1) : 0;
                crossPingPongID_ = (crossPingPongID_ + 1 < STAGES) ? (crossPingPongID_ + 1) : 0;

            }
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList_[l1ListId_]);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList_[l1ListId_]);
            l1ListId_ = (l1ListId_ + 1 < STAGES) ? (l1ListId_ + 1) : 0;
        }
    }

private:
    constexpr static uint16_t M_ALIGN = 16;
    constexpr static uint16_t AIC_SYNC_AIV_MODE_4 = 4;
    constexpr static uint16_t AIV_SYNC_AIC_FLAG = 8;
    constexpr static uint16_t AIC_SYNC_AIV_FLAG = 6;
    constexpr static uint16_t FLAG_ID_MAX = 16;
    constexpr static uint32_t UB_TWO_BANK_ELEMS_B32 = 128U;
    constexpr static int64_t PER_BLOCK_SIZE = 128LL;
    constexpr static uint64_t UB_OFFSET = UB_TWO_BANK_ELEMS_B32 * PER_BLOCK_SIZE;

    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;
    TileMmad tileMmad;
    
    AscendC::LocalTensor<uint8_t> l1Buf_{AscendC::LocalTensor<uint8_t>(AscendC::TPosition::A1, 0, ArchTag::L1_SIZE)};
    AscendC::LocalTensor<uint8_t> l0ABuf_{AscendC::LocalTensor<uint8_t>(AscendC::TPosition::A2, 0, ArchTag::L0A_SIZE)};
    AscendC::LocalTensor<uint8_t> l0BBuf_{AscendC::LocalTensor<uint8_t>(AscendC::TPosition::B2, 0, ArchTag::L0B_SIZE)};
    AscendC::LocalTensor<uint8_t> l0CBuf_{AscendC::LocalTensor<uint8_t>(AscendC::TPosition::CO1, 0, ArchTag::L0C_SIZE)};

    AscendC::LocalTensor<ElementA> l1ATensorList_[STAGES];
    AscendC::LocalTensor<ElementB> l1BTensorList_[STAGES];
    AscendC::LocalTensor<ElementA> l0ATensorList_[STAGES];
    AscendC::LocalTensor<ElementB> l0BTensorList_[STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensorList_[STAGES];

    int32_t l1AEventList_[STAGES];
    int32_t l1BEventList_[STAGES];
    int32_t l0AEventList_[STAGES];
    int32_t l0BEventList_[STAGES];
    int32_t l0CEventList_[STAGES];

    uint32_t l1ListId_{0};
    uint32_t l0ListId_{0};
    uint32_t l0CListId_{0};
    uint32_t crossPingPongID_{0};
};
}  // namespace Catlass::Gemm::Block

#endif  // CATLASS_GEMM_BLOCK_BLOCK_MMAD_PINGPONG_PER_GROUP_PER_BLOCK_TLA_HPP