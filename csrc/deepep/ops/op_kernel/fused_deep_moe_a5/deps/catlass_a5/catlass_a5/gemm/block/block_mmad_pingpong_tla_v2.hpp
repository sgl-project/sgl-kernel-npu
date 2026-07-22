/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_BLOCK_BLOCK_MMAD_PINGPONG_TLA_V2_HPP
#define CATLASS_GEMM_BLOCK_BLOCK_MMAD_PINGPONG_TLA_V2_HPP

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

template <class ArchTag_, bool ENABLE_UNIT_FLAG_, class L1TileShape_, class L0TileShape_, class ElementA_,
          class ElementB_, class ElementC_, class ElementBias_, class TileCopy_, class TileMmad_>
struct BlockMmadTla<MmadPingpongTlaV2<ArchTag_, ENABLE_UNIT_FLAG_>, L1TileShape_, L0TileShape_, ElementA_, ElementB_,
                    ElementC_, ElementBias_, TileCopy_, TileMmad_> {
public:
    // Type Aliases
    using DispatchPolicy = MmadPingpongTlaV2<ArchTag_, ENABLE_UNIT_FLAG_>;
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
    using LayoutTagL0C = typename TileCopy_::LayoutTagL0C;

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
    static constexpr uint32_t L0C_TILE_SIZE = L1_TILE_M * L1_TILE_N * sizeof(ElementAccumulator);

    // Check LayoutC
    static_assert(tla::detail::isRowMajor<LayoutC>::value ||
                      ((std::is_same_v<ElementC, half> || std::is_same_v<ElementC, bfloat16_t> ||
                        std::is_same_v<ElementC, float>) &&
                       tla::detail::iszN<ElementC, LayoutC>::value),
                  "LayoutC only supports zN in half or bfloat16 or float, RowMajor in all dtype yet!");

    // Check L1TileShape
    static_assert((L1A_TILE_SIZE + L1B_TILE_SIZE) * STAGES <= ArchTag::L1_SIZE, "L1TileShape exceeding the L1 space!");

    // Check L0TileShape
    static_assert(L0A_TILE_SIZE * STAGES <= ArchTag::L0A_SIZE, "L0TileShape exceeding the L0A space!");
    static_assert(L0B_TILE_SIZE * STAGES <= ArchTag::L0B_SIZE, "L0TileShape exceeding the L0B space!");
    static_assert(L0C_TILE_SIZE <= ArchTag::L0C_SIZE, "L0TileShape exceeding the L0C space!");

    static constexpr uint32_t _32B = 32 * 8;  // in bits
    static_assert(L1_TILE_M == L0_TILE_M && L1_TILE_N == L0_TILE_N,
                  "The situation where the basic blocks of L1 and L0 differ on the m and n axes is not supported yet");
    static_assert(L0_TILE_K <= L1_TILE_K, "L0TileShape::K cannot exceed L1TileShape::K");
    static_assert(L1_TILE_M * SizeOfBits<ElementA>::value % _32B == 0, "L1TileShape::M must be 32B aligned.");
    static_assert(L1_TILE_K * SizeOfBits<ElementA>::value % _32B == 0, "L1TileShape::K must be 32B aligned.");
    static_assert(L1_TILE_K * SizeOfBits<ElementB>::value % _32B == 0, "L1TileShape::K must be 32B aligned.");
    static_assert(L1_TILE_N * SizeOfBits<ElementB>::value % _32B == 0, "L1TileShape::N must be 32B aligned.");
    static_assert(L0_TILE_K * SizeOfBits<ElementB>::value % _32B == 0, "L0TileShape::K must be 32B aligned.");

    static constexpr auto L1A_LAYOUT =
        tla::MakeLayout<ElementA, LayoutTagL1A>(tla::Int<L1_TILE_M>{}, tla::Int<L1_TILE_K>{});
    static constexpr auto L1B_LAYOUT =
        tla::MakeLayout<ElementB, LayoutTagL1B>(tla::Int<L1_TILE_K>{}, tla::Int<L1_TILE_N>{});

    /// Construct
    CATLASS_DEVICE
    BlockMmadTla(Arch::Resource<ArchTag> &resource, uint32_t l1BufAddrStart = 0)
    {
        uint32_t l1AOffset = l1BufAddrStart;
        uint32_t l1BOffset = l1BufAddrStart + L1A_TILE_SIZE * STAGES;
        // Init buffers
        for (uint32_t i = 0; i < STAGES; i++) {
            // Assign L1/L0A/L0B space for each stages
            l1ATensorList[i] = resource.l1Buf.template GetBufferByByte<ElementA>(l1AOffset + L1A_TILE_SIZE * i);
            l1BTensorList[i] = resource.l1Buf.template GetBufferByByte<ElementB>(l1BOffset + L1B_TILE_SIZE * i);
            l0ATensorList[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(L0A_TILE_SIZE * i);
            l0BTensorList[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_TILE_SIZE * i);

            // Assign event ID for each stages
            l1AEventList[i] = i;
            l1BEventList[i] = i + STAGES;
            l0AEventList[i] = i;
            l0BEventList[i] = i + STAGES;

            // The event id that needs to be set before the loop
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
        }
        l0CTensor = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
    }

    /// Destructor
    CATLASS_DEVICE
    ~BlockMmadTla()
    {
        for (uint32_t i = 0; i < STAGES; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
        }
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
    }

    /// Perform a block-scoped matrix multiply-accumulate
    template <class TensorA, class TensorB, class TensorC>
    CATLASS_DEVICE void operator()(TensorA &tensorA, TensorB &tensorB, TensorC &tensorC)
    {
        using CopyGmToL1A = typename TileCopy_::template CopyGmToL1A<TensorA>;
        using CopyGmToL1B = typename TileCopy_::template CopyGmToL1B<TensorB>;
        CopyGmToL1A copyGmToL1A;
        CopyGmToL1B copyGmToL1B;
#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 2201)
        using CopyL0CToGm = typename TileCopy_::template CopyL0CToGm<TensorC>;
        CopyL0CToGm copyL0CToDst;
#endif
#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)
        using CopyL0CToDst = typename TileCopy_::template CopyL0CToDst<TensorC>;
        CopyL0CToDst copyL0CToDst;
#endif

        // Create an accumulator tensor view on L0C buffer.
        // - Logical size comes from tensorC.layout().originShape() (tail-aware)
        // - Layout is constructed from LayoutTagL0C
        // - coord is initialized to (0, 0) (new buffer view)
        auto tensorL0C = tla::MakeTensorLike<LayoutTagL0C, ElementAccumulator>(l0CTensor, tensorC, Arch::PositionL0C{});

        // load first matrix A tile from GM to L1
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListId]);
        // TileView: tileCoord is in tile units (not element units).
        // It internally converts tileCoord to elementOffset = tileCoord ⊙ tileShape and handles tail tiles via
        // originShape.
        auto tensorTileA = tla::TileView(tensorA, tla::MakeCoord(0u, 0u),  // (m_tile, k_tile)
                                         tla::MakeShape(tla::Int<L1_TILE_M>{}, tla::Int<L1_TILE_K>{}));
        auto tensorL1A =
            tla::MakeTensorLike<LayoutTagL1A>(l1ATensorList[l1ListId], tensorTileA, Arch::PositionL1{}, L1A_LAYOUT);
        copyGmToL1A(tensorL1A, tensorTileA);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListId]);

        // load first matrix B tile from GM to L1
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListId]);
        auto tensorTileB = tla::TileView(tensorB, tla::MakeCoord(0u, 0u),  // (k_tile, n_tile)
                                         tla::MakeShape(tla::Int<L1_TILE_K>{}, tla::Int<L1_TILE_N>{}));
        auto tensorL1B =
            tla::MakeTensorLike<LayoutTagL1B>(l1BTensorList[l1ListId], tensorTileB, Arch::PositionL1{}, L1B_LAYOUT);
        copyGmToL1B(tensorL1B, tensorTileB);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListId]);

        if constexpr (!ENABLE_UNIT_FLAG) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        }

        uint32_t mPartLoop = CeilDiv<L0_TILE_M>(tla::get<0>(tensorL0C.originShape()));  // dim 0 = M
        uint32_t nPartLoop = CeilDiv<L0_TILE_N>(tla::get<1>(tensorL0C.originShape()));  // dim 1 = N

        // main loop
        uint32_t kTileCount = CeilDiv<L1_TILE_K>(tla::get<1>(tensorA.originShape()));  // dim 1 = K
        for (uint32_t kLoopIdx = 0; kLoopIdx < kTileCount; kLoopIdx++) {
            uint32_t l1ListIdNext = (l1ListId + 1 < STAGES) ? (l1ListId + 1) : 0;
            // preload next tile from GM to L1
            if (kLoopIdx < kTileCount - 1) {
                uint32_t kLoopIdxNext = kLoopIdx + 1;

                // Get L1 tensor for next stage
                auto l1ATensor = l1ATensorList[l1ListIdNext];
                auto l1BTensor = l1BTensorList[l1ListIdNext];
                // Get GM tile for next stage
                auto tensorTileA = tla::TileView(tensorA, tla::MakeCoord(0u, kLoopIdxNext),  // (m_tile, k_tile)
                                                 tla::MakeShape(tla::Int<L1_TILE_M>{}, tla::Int<L1_TILE_K>{}));
                auto tensorTileB = tla::TileView(tensorB, tla::MakeCoord(kLoopIdxNext, 0u),  // (k_tile, n_tile)
                                                 tla::MakeShape(tla::Int<L1_TILE_K>{}, tla::Int<L1_TILE_N>{}));
                auto tensorL1A =
                    tla::MakeTensorLike<LayoutTagL1A>(l1ATensor, tensorTileA, Arch::PositionL1{}, L1A_LAYOUT);
                auto tensorL1B =
                    tla::MakeTensorLike<LayoutTagL1B>(l1BTensor, tensorTileB, Arch::PositionL1{}, L1B_LAYOUT);

                // load next matrix A tile from GM to L1
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListIdNext]);
                copyGmToL1A(tensorL1A, tensorTileA);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListIdNext]);

                // load next matrix B tile from GM to L1
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListIdNext]);
                copyGmToL1B(tensorL1B, tensorTileB);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListIdNext]);
            }

            // Get L1 tensor for current stage
            auto l1ATensor = l1ATensorList[l1ListId];
            auto l1BTensor = l1BTensorList[l1ListId];
            // Create tile view for current K iteration
            auto tensorTileA = tla::TileView(tensorA, tla::MakeCoord(0u, kLoopIdx),  // (m_tile, k_tile)
                                             tla::MakeShape(tla::Int<L1_TILE_M>{}, tla::Int<L1_TILE_K>{}));
            auto tensorTileB = tla::TileView(tensorB, tla::MakeCoord(kLoopIdx, 0u),  // (k_tile, n_tile)
                                             tla::MakeShape(tla::Int<L1_TILE_K>{}, tla::Int<L1_TILE_N>{}));
            auto tensorL1A = tla::MakeTensorLike<LayoutTagL1A>(l1ATensor, tensorTileA, Arch::PositionL1{}, L1A_LAYOUT);
            auto tensorL1B = tla::MakeTensorLike<LayoutTagL1B>(l1BTensor, tensorTileB, Arch::PositionL1{}, L1B_LAYOUT);
            // Get the loop nums on L0 based on current L1 tile's actual K size
            uint32_t kPartLoop = CeilDiv<L0_TILE_K>(tla::get<1>(tensorL1A.originShape()));  // dim 1 = K

            for (int mPartIdx = 0; mPartIdx < mPartLoop; mPartIdx++) {
                for (int kPartIdx = 0; kPartIdx < kPartLoop; kPartIdx++) {
                    // Locate the current tile on L0A
                    auto l0ATile = l0ATensorList[l0AListId];
                    // Locate the current tile of matrix A on L1
                    // Take a (L0_TILE_M, L0_TILE_K) tile from the current L1A tile (tile coordinates within L1 tile).
                    auto tensorTileL1A = tla::TileView(tensorL1A, tla::MakeCoord(mPartIdx, kPartIdx),
                                                       tla::MakeShape(tla::Int<L0_TILE_M>{}, tla::Int<L0_TILE_K>{}));
                    auto tensorL0A = tla::MakeTensorLike<LayoutTagL0A>(l0ATile, tensorTileL1A, Arch::PositionL0A{});

                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                    if ((mPartIdx == 0) && (kPartIdx == 0)) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListId]);
                    }

                    // Load current tile from L1 to L0A
                    copyL1ToL0A(tensorL0A, tensorTileL1A);

                    if ((mPartIdx == mPartLoop - 1) && (kPartIdx == kPartLoop - 1)) {
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListId]);
                    }

                    for (int nPartIdx = 0; nPartIdx < nPartLoop; nPartIdx++) {
                        // Locate the current tile on L0B
                        auto l0BTile = l0BTensorList[l0BListId];
                        // Locate the current tile of matrix B on L1
                        // Take a (L0_TILE_K, L0_TILE_N) tile from the current L1B tile (tile coordinates within L1
                        // tile).
                        auto tensorTileL1B =
                            tla::TileView(tensorL1B, tla::MakeCoord(kPartIdx, nPartIdx),
                                          tla::MakeShape(tla::Int<L0_TILE_K>{}, tla::Int<L0_TILE_N>{}));
                        auto tensorL0B = tla::MakeTensorLike<LayoutTagL0B>(l0BTile, tensorTileL1B, Arch::PositionL0B{});

                        // Wait for mmad finished
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                        // If the current tile is the first one on the k&n axis, wait for loading matrix B from GM to L1
                        if ((kPartIdx == 0) && (nPartIdx == 0)) {
                            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListId]);
                        }

                        // Load current tile from L1 to L0B
                        copyL1ToL0B(tensorL0B, tensorTileL1B);

                        // If the current tile is the last one on the k&n axis, notify to load matrix B from GM to L1
                        if ((kPartIdx == kPartLoop - 1) && (nPartIdx == nPartLoop - 1)) {
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListId]);
                        }
                        // Notify to do mmad
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);

                        // Locate the current tile on L0C
                        // View into L0C accumulator tile (tile coordinates in (m_part, n_part)).
                        auto tensorTileL0C =
                            tla::TileView(tensorL0C, tla::MakeCoord(mPartIdx, nPartIdx),
                                          tla::MakeShape(tla::Int<L0_TILE_M>{}, tla::Int<L0_TILE_N>{}));

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
                        tileMmad(tensorTileL0C, tensorL0A, tensorL0B, initC, unitFlag);

                        // Notify to move the next L0B tile
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                        l0BListId = (l0BListId + 1 < STAGES) ? (l0BListId + 1) : 0;
                    }
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                    l0AListId = (l0AListId + 1 < STAGES) ? (l0AListId + 1) : 0;
                }
            }
            l1ListId = l1ListIdNext;
        }

        // copy block out
        if constexpr (!ENABLE_UNIT_FLAG) {
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
            copyL0CToDst(tensorC, tensorL0C);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        } else {
            copyL0CToDst(tensorC, tensorL0C, 0b11);
        }
    }

protected:
    // Multi-stage tensors list
    AscendC::LocalTensor<ElementA> l1ATensorList[STAGES];
    AscendC::LocalTensor<ElementB> l1BTensorList[STAGES];
    AscendC::LocalTensor<ElementA> l0ATensorList[STAGES];
    AscendC::LocalTensor<ElementB> l0BTensorList[STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensor;

    // Multi-stage event id list
    int32_t l1AEventList[STAGES];
    int32_t l1BEventList[STAGES];
    int32_t l0AEventList[STAGES];
    int32_t l0BEventList[STAGES];

    // The id of current stage
    uint32_t l1ListId{0};
    uint32_t l0AListId{0};
    uint32_t l0BListId{0};

    TileMmad tileMmad;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;
};

}  // namespace Catlass::Gemm::Block

#endif  // CATLASS_GEMM_BLOCK_BLOCK_MMAD_PINGPONG_TLA_V2_HPP
