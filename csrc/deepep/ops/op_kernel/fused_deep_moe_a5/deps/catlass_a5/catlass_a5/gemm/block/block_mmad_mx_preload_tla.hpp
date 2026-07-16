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

#ifndef CATLASS_GEMM_BLOCK_BLOCK_MMAD_MX_PRELOAD_TLA_HPP
#define CATLASS_GEMM_BLOCK_BLOCK_MMAD_MX_PRELOAD_TLA_HPP

#include "catlass_a5/arch/resource.hpp"
#include "catlass_a5/catlass.hpp"
#include "catlass_a5/coord.hpp"
#include "catlass_a5/numeric_size.hpp"
#include "catlass_a5/gemm/dispatch_policy.hpp"
#include "catlass_a5/gemm/helper.hpp"
#include "catlass_a5/gemm/tile/tile_mmad.hpp"
#include "catlass_a5/gemm_coord.hpp"
#include "catlass_a5/matrix_coord.hpp"
#include "tla_a5/layout.hpp"
#include "tla_a5/tensor.hpp"

namespace Catlass::Gemm {

// Dispatch policy for the MX cross-block preload BlockMmad. ASYNC = true so the kernel drains the
// deferred-mmad queue via SynchronizeBlock() after the tile loop.
//
// The default config (PRELOAD_STAGES=1, L1A=L1B=2, scale stages = L1A/L1B, L0A=L0B=2, L0C=1) uses the
// SAME L1 budget as MmadMx (so it fits for both FP8 and FP4). PRELOAD_STAGES must stay < L1A_STAGES
// and < the per-tile K-stripe count (kL1Loop), and the whole tile's MX scale must fit one fetch
// (kBlockActual <= L1_TILE_K * L1_SCALE_FACTOR_K); both hold for example 55's shapes.
template <
    class ArchTag_,
    uint32_t PRELOAD_STAGES_ = 1,
    bool ENABLE_UNIT_FLAG_ = true,
    uint32_t L1_SCALE_FACTOR_K_ = 16,
    uint32_t L0C_STAGES_ = 1,
    bool ENABLE_L1_RESIDENT_ = false,
    uint32_t L1A_STAGES_ = 2,
    uint32_t L1B_STAGES_ = 2,
    uint32_t L0A_STAGES_ = 2,
    uint32_t L0B_STAGES_ = 2>
struct MmadMxPreload : public MmadBase<ArchTag_, true> {
    static constexpr uint32_t PRELOAD_STAGES = PRELOAD_STAGES_;
    static constexpr uint32_t L1A_STAGES = L1A_STAGES_;
    static constexpr uint32_t L1B_STAGES = L1B_STAGES_;
    static constexpr uint32_t L0A_STAGES = L0A_STAGES_;
    static constexpr uint32_t L0B_STAGES = L0B_STAGES_;
    static constexpr uint32_t L0C_STAGES = L0C_STAGES_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    static constexpr bool ENABLE_L1_RESIDENT = ENABLE_L1_RESIDENT_;
    static constexpr uint32_t L1_SCALE_FACTOR_K = L1_SCALE_FACTOR_K_;
};

namespace Block {

#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)

// Standalone (non BlockMmadTla-specialization) MX BlockMmad with cross-block preload. It is a faithful
// merge of:
//   * the deferred-mmad queue from block_mmad_preload_async_with_callback_tla.hpp (preloadCount /
//     l1TileMmadParamsList / SynchronizeBlock) — operator() issues the current tile's GM->L1 loads and
//     executes the mmad+fixpipe of an earlier (queued) K-stripe, so the next block's MTE2 overlaps the
//     current block's tail compute (fills the §8.5 block-boundary bubble); and
//   * the MX scale loading + scale-aware L1->L0 copy from block_mmad_mx_tla.hpp.
//
// Interface matches BlockMmadTla<MmadMx, ...> so the existing GroupedMxMatmulSliceMTla kernel drives it
// unchanged (it already calls SynchronizeBlock when DispatchPolicy::ASYNC).
//
// The whole tile's MX scale is loaded once per operator() into a per-tile scale buffer (one fetch),
// double-buffered across tiles with its own MTE1_MTE2/MTE2_MTE1 events so it survives the deferred
// mmads that consume it (which may run in the next operator()).
template <
    class DispatchPolicy_,
    class L1TileShape_,
    class L0TileShape_,
    class ElementA_,
    class ElementB_,
    class ElementC_,
    class ElementBias_,
    class TileCopy_,
    class TileMmad_ =
        Gemm::Tile::TileMmadTla<typename DispatchPolicy_::ArchTag, ElementA_, typename TileCopy_::LayoutTagL1A>>
struct BlockMmadMxPreloadTla {
public:
    using DispatchPolicy = DispatchPolicy_;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using TileCopy = TileCopy_;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    using ElementA = ElementA_;
    using LayoutA = typename TileCopy::LayoutA;
    using ElementB = ElementB_;
    using LayoutB = typename TileCopy::LayoutB;
    using ElementMxScaleA = typename TileCopy::ElementMxScaleA;
    using ElementMxScaleB = typename TileCopy::ElementMxScaleB;
    using ElementC = ElementC_;
    using LayoutC = typename TileCopy::LayoutC;
    using ElementBias = ElementBias_;
    using ElementL0A = typename helper::GetL0Element<ElementA, true>::Element;
    using ElementL0B = typename helper::GetL0Element<ElementB, true>::Element;

    using TileMmad = TileMmad_;
    using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;
    using CopyL1ToBT = typename TileCopy::CopyL1ToBT;
    using ElementAccumulator = typename TileCopy::ElementAccumulator;

    static constexpr bool HAS_BIAS = TileCopy::HAS_BIAS;
    static_assert(!HAS_BIAS, "BlockMmadMxPreloadTla does not support bias (event-id budget is full).");

    using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
    using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
    using LayoutTagL1MxScaleA = typename TileCopy::LayoutTagL1MxScaleA;
    using LayoutTagL1MxScaleB = typename TileCopy::LayoutTagL1MxScaleB;
    using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
    using LayoutTagL0B = typename TileCopy::LayoutTagL0B;

    static constexpr uint32_t PRELOAD_STAGES = DispatchPolicy::PRELOAD_STAGES;
    static constexpr bool ENABLE_UNIT_FLAG = DispatchPolicy::ENABLE_UNIT_FLAG;
    static constexpr bool ENABLE_L1_RESIDENT = DispatchPolicy::ENABLE_L1_RESIDENT;
    static constexpr uint32_t L1A_STAGES = DispatchPolicy::L1A_STAGES;
    static constexpr uint32_t L1B_STAGES = DispatchPolicy::L1B_STAGES;
    static constexpr uint32_t L0A_STAGES = DispatchPolicy::L0A_STAGES;
    static constexpr uint32_t L0B_STAGES = DispatchPolicy::L0B_STAGES;
    static constexpr uint32_t L0C_STAGES = DispatchPolicy::L0C_STAGES;
    static constexpr uint32_t L1_SCALE_FACTOR_K = DispatchPolicy::L1_SCALE_FACTOR_K;
    // Scale ring depth mirrors the A/B ring depth (one scale buffer per in-flight tile).
    static constexpr uint32_t SCALE_A_STAGES = L1A_STAGES;
    static constexpr uint32_t SCALE_B_STAGES = L1B_STAGES;

    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});
    static constexpr uint32_t L0_TILE_M = tla::get<0>(L0TileShape{});
    static constexpr uint32_t L0_TILE_N = tla::get<1>(L0TileShape{});
    static constexpr uint32_t L0_TILE_K = tla::get<2>(L0TileShape{});

    static constexpr uint32_t L1A_TILE_SIZE = L1_TILE_M * L1_TILE_K * SizeOfBits<ElementA>::value / 8;
    static constexpr uint32_t L1B_TILE_SIZE = L1_TILE_N * L1_TILE_K * SizeOfBits<ElementB>::value / 8;
    static constexpr uint32_t L1SCALEA_TILE_SIZE =
        L1_TILE_M * L1_TILE_K / MX_SCALE_GROUP_NUM * sizeof(ElementMxScaleA) * L1_SCALE_FACTOR_K;
    static constexpr uint32_t L1SCALEB_TILE_SIZE =
        L1_TILE_N * L1_TILE_K / MX_SCALE_GROUP_NUM * sizeof(ElementMxScaleB) * L1_SCALE_FACTOR_K;
    static constexpr uint32_t L1_USED_SIZE = L1A_TILE_SIZE * L1A_STAGES + L1B_TILE_SIZE * L1B_STAGES +
                                             L1SCALEA_TILE_SIZE * SCALE_A_STAGES + L1SCALEB_TILE_SIZE * SCALE_B_STAGES;
    static constexpr uint32_t L0A_TILE_SIZE = L0_TILE_M * L0_TILE_K * SizeOfBits<ElementL0A>::value / 8;
    static constexpr uint32_t L0B_TILE_SIZE = L0_TILE_K * L0_TILE_N * SizeOfBits<ElementL0B>::value / 8;
    static constexpr uint32_t L0C_TILE_SIZE = L1_TILE_M * L1_TILE_N * sizeof(ElementAccumulator);

    static_assert(!(ENABLE_UNIT_FLAG && L0C_STAGES != 1), "L0C_STAGES must be 1 when UnitFlag is true!");
    static_assert(
        L1_TILE_K % MX_BASEK_FACTOR == 0 && L0_TILE_K % MX_BASEK_FACTOR == 0,
        "L1TileShape::K and L0TileShape::K must be multiples of 64!");
    static_assert(L1_USED_SIZE <= ArchTag::L1_SIZE, "L1TileShape exceeding the L1 space!");
    static_assert(L0A_TILE_SIZE * L0A_STAGES <= ArchTag::L0A_SIZE, "L0TileShape exceeding the L0A space!");
    static_assert(L0B_TILE_SIZE * L0B_STAGES <= ArchTag::L0B_SIZE, "L0TileShape exceeding the L0B space!");
    static_assert(L0C_TILE_SIZE * L0C_STAGES <= ArchTag::L0C_SIZE, "L0TileShape exceeding the L0C space!");
    static_assert(L1_TILE_M == L0_TILE_M && L1_TILE_N == L0_TILE_N,
        "L1 and L0 basic blocks must match on m and n axes");
    static_assert(L0_TILE_K <= L1_TILE_K, "L0TileShape::K cannot exceed L1TileShape::K");
    // The deferred buffer just-loaded for stripe i must differ from the one consumed for stripe i-PRELOAD.
    static_assert(PRELOAD_STAGES >= 1 && PRELOAD_STAGES < L1A_STAGES && PRELOAD_STAGES < L1B_STAGES,
        "PRELOAD_STAGES must be in [1, min(L1A_STAGES, L1B_STAGES) - 1]");
    // MTE1_MTE2/MTE2_MTE1 event-id budget is 8 (0..7): A + B + scaleA + scaleB.
    static_assert(L1A_STAGES + L1B_STAGES + SCALE_A_STAGES + SCALE_B_STAGES <= 8,
        "L1 buffer event ids exceed 0..7");
    static_assert(L0A_STAGES + L0B_STAGES <= 8, "L0 buffer event ids exceed 0..7");

    static constexpr auto L1A_LAYOUT =
        tla::MakeLayout<ElementA, LayoutTagL1A>(tla::Int<L1_TILE_M>{}, tla::Int<L1_TILE_K>{});
    static constexpr auto L1B_LAYOUT =
        tla::MakeLayout<ElementB, LayoutTagL1B>(tla::Int<L1_TILE_K>{}, tla::Int<L1_TILE_N>{});
    static constexpr auto L1SCALEA_LAYOUT = tla::MakeMxScaleLayout<ElementMxScaleA, LayoutTagL1MxScaleA, false>(
        tla::Int<L1_TILE_M>{}, tla::Int<L1_TILE_K / MX_SCALE_GROUP_NUM * L1_SCALE_FACTOR_K>{});
    static constexpr auto L1SCALEB_LAYOUT = tla::MakeMxScaleLayout<ElementMxScaleB, LayoutTagL1MxScaleB, true>(
        tla::Int<L1_TILE_K / MX_SCALE_GROUP_NUM * L1_SCALE_FACTOR_K>{}, tla::Int<L1_TILE_N>{});

    CATLASS_DEVICE
    void RestoreStatus()
    {
        for (uint32_t i = 0; i < L1A_STAGES; ++i) {
            lastAddrA[i] = nullptr;
            lastCoordA[i] = MatrixCoord{0U, 0U};
        }
        for (uint32_t i = 0; i < L1B_STAGES; ++i) {
            lastAddrB[i] = nullptr;
            lastCoordB[i] = MatrixCoord{0U, 0U};
        }
    }

    CATLASS_DEVICE
    BlockMmadMxPreloadTla(Arch::Resource<ArchTag> &resource, uint32_t l1BufAddrStart = 0)
    {
        if constexpr (ENABLE_UNIT_FLAG && tla::detail::isRowMajor<LayoutC>::value) {
            AscendC::SetMMLayoutTransform(true);
        }
        uint32_t l1Offset = l1BufAddrStart;
        for (uint32_t i = 0; i < L1A_STAGES; ++i) {
            l1ATensorList[i] = resource.l1Buf.template GetBufferByByte<ElementA>(l1Offset);
            l1Offset += L1A_TILE_SIZE;
            l1AEventList[i] = i;
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
        }
        for (uint32_t i = 0; i < L1B_STAGES; ++i) {
            l1BTensorList[i] = resource.l1Buf.template GetBufferByByte<ElementB>(l1Offset);
            l1Offset += L1B_TILE_SIZE;
            l1BEventList[i] = i + L1A_STAGES;
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
        }
        for (uint32_t i = 0; i < L0A_STAGES; ++i) {
            l0ATensorList[i] = resource.l0ABuf.template GetBufferByByte<ElementL0A>(L0A_TILE_SIZE * i);
            l0AEventList[i] = i;
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
        }
        for (uint32_t i = 0; i < L0B_STAGES; ++i) {
            l0BTensorList[i] = resource.l0BBuf.template GetBufferByByte<ElementL0B>(L0B_TILE_SIZE * i);
            l0BEventList[i] = i + L0A_STAGES;
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
        }
        if constexpr (!ENABLE_UNIT_FLAG) {
            for (uint32_t i = 0; i < L0C_STAGES; ++i) {
                l0CTensorList[i] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(L0C_TILE_SIZE * i);
                l0CEventList[i] = i;
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventList[i]);
            }
        } else {
            l0CTensorList[0] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(0);
            l0CEventList[0] = 0;
        }
        for (uint32_t i = 0; i < SCALE_A_STAGES; ++i) {
            l1MxScaleATensorList[i] = resource.l1Buf.template GetBufferByByte<ElementMxScaleA>(l1Offset);
            l1Offset += L1SCALEA_TILE_SIZE;
            l1MxScaleAEventList[i] = i + L1A_STAGES + L1B_STAGES;
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1MxScaleAEventList[i]);
        }
        for (uint32_t i = 0; i < SCALE_B_STAGES; ++i) {
            l1MxScaleBTensorList[i] = resource.l1Buf.template GetBufferByByte<ElementMxScaleB>(l1Offset);
            l1Offset += L1SCALEB_TILE_SIZE;
            l1MxScaleBEventList[i] = i + L1A_STAGES + L1B_STAGES + SCALE_A_STAGES;
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1MxScaleBEventList[i]);
        }
        if constexpr (ENABLE_L1_RESIDENT) {
            RestoreStatus();
        }
    }

    CATLASS_DEVICE
    ~BlockMmadMxPreloadTla()
    {
        if constexpr (ENABLE_UNIT_FLAG && tla::detail::isRowMajor<LayoutC>::value) {
            AscendC::SetMMLayoutTransform(false);
        }
        for (uint32_t i = 0; i < L1A_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
        }
        for (uint32_t i = 0; i < L1B_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
        }
        for (uint32_t i = 0; i < SCALE_A_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1MxScaleAEventList[i]);
        }
        for (uint32_t i = 0; i < SCALE_B_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1MxScaleBEventList[i]);
        }
        for (uint32_t i = 0; i < L0A_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
        }
        for (uint32_t i = 0; i < L0B_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
        }
        if constexpr (!ENABLE_UNIT_FLAG) {
            for (uint32_t i = 0; i < L0C_STAGES; ++i) {
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventList[i]);
            }
        }
    }

    /// Issue this tile's GM->L1 loads (whole-tile MX scale once + per-stripe A/B), and execute the
    /// mmad+fixpipe of the K-stripe queued PRELOAD_STAGES ago (which may belong to an earlier tile).
    template <
        class TensorA,
        class TensorB,
        class TensorC,
        class TensorMxScaleA = EmptyClass,
        class TensorMxScaleB = EmptyClass,
        class TensorBias = EmptyClass>
    CATLASS_DEVICE void operator()(
        TensorA &tensorA,
        TensorB &tensorB,
        TensorC &tensorC,
        GemmCoord const &actualShape,
        TensorMxScaleA const &tensorMxScaleA = {},
        TensorMxScaleB const &tensorMxScaleB = {},
        TensorBias const &tensorBias = {})
    {
        using CopyGmToL1A = typename TileCopy::template CopyGmToL1A<TensorA>;
        using CopyGmToL1B = typename TileCopy::template CopyGmToL1B<TensorB>;
        using CopyGmToL1MxScaleA = typename TileCopy::template CopyGmToL1MxScaleA<TensorMxScaleA>;
        using CopyGmToL1MxScaleB = typename TileCopy::template CopyGmToL1MxScaleB<TensorMxScaleB>;
        CopyGmToL1A copyGmToL1A;
        CopyGmToL1B copyGmToL1B;
        CopyGmToL1MxScaleA copyGmToL1MxScaleA;
        CopyGmToL1MxScaleB copyGmToL1MxScaleB;

        uint32_t mBlockActual = actualShape.m();
        uint32_t kBlockActual = actualShape.k();
        uint32_t nBlockActual = actualShape.n();
        uint32_t mL1Actual = mBlockActual;
        uint32_t nL1Actual = nBlockActual;

        uint32_t kL1Loop = CeilDiv<L1_TILE_K>(kBlockActual);
        uint32_t kL1ScaleActual = (kBlockActual < L1_TILE_K * L1_SCALE_FACTOR_K)
            ? kBlockActual : (L1_TILE_K * L1_SCALE_FACTOR_K);

        // --- Tile-level MX scale load (one fetch covering the whole tile) ---
        uint32_t curScaleAId = l1MxScaleAListId;
        uint32_t curScaleBId = l1MxScaleBListId;
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1MxScaleAEventList[curScaleAId]);
        {
            auto tensorL1MxScaleA =
                tla::MakeTensor(l1MxScaleATensorList[curScaleAId], L1SCALEA_LAYOUT, Arch::PositionL1{});
            auto tensorTileMxScaleA = GetTile(
                tensorMxScaleA, tla::MakeCoord(0, 0),
                tla::MakeShape(mBlockActual, CeilDiv<MX_SCALE_GROUP_NUM>(kL1ScaleActual)));
            copyGmToL1MxScaleA(tensorL1MxScaleA, tensorTileMxScaleA);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1MxScaleAEventList[curScaleAId]);

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1MxScaleBEventList[curScaleBId]);
        {
            auto tensorL1MxScaleB =
                tla::MakeTensor(l1MxScaleBTensorList[curScaleBId], L1SCALEB_LAYOUT, Arch::PositionL1{});
            auto tensorTileMxScaleB = GetTile(
                tensorMxScaleB, tla::MakeCoord(0, 0),
                tla::MakeShape(CeilDiv<MX_SCALE_GROUP_NUM>(kL1ScaleActual), nBlockActual));
            copyGmToL1MxScaleB(tensorL1MxScaleB, tensorTileMxScaleB);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1MxScaleBEventList[curScaleBId]);

        l1MxScaleAListId = (l1MxScaleAListId + 1 < SCALE_A_STAGES) ? (l1MxScaleAListId + 1) : 0;
        l1MxScaleBListId = (l1MxScaleBListId + 1 < SCALE_B_STAGES) ? (l1MxScaleBListId + 1) : 0;

        // --- Per K-stripe load + deferred mmad ---
        for (uint32_t kL1Idx = 0; kL1Idx < kL1Loop; ++kL1Idx) {
            uint32_t kL1Actual = (kL1Idx < kL1Loop - 1) ? L1_TILE_K : (kBlockActual - kL1Idx * L1_TILE_K);

            // Load matrix A tile from GM to L1
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1AListId]);
            auto tensorL1A = tla::MakeTensor(l1ATensorList[l1AListId], L1A_LAYOUT, Arch::PositionL1{});
            auto tensorTileA =
                GetTile(tensorA, tla::MakeCoord(0, kL1Idx * L1_TILE_K), tla::MakeShape(mBlockActual, kL1Actual));
            if constexpr (ENABLE_L1_RESIDENT) {
                if (lastAddrA[l1AListId] != tensorTileA.data().GetPhyAddr() ||
                    tla::get<0>(tensorTileA.coord()) != lastCoordA[l1AListId].row() ||
                    tla::get<1>(tensorTileA.coord()) != lastCoordA[l1AListId].column()) {
                    copyGmToL1A(tensorL1A, tensorTileA);
                    lastCoordA[l1AListId] = MatrixCoord{
                        static_cast<MatrixCoord::Index>(tla::get<0>(tensorTileA.coord())),
                        static_cast<MatrixCoord::Index>(tla::get<1>(tensorTileA.coord()))};
                    lastAddrA[l1AListId] = const_cast<__gm__ typename AscendC::GlobalTensor<ElementA>::PrimType *>(
                        tensorTileA.data().GetPhyAddr());
                }
            } else {
                copyGmToL1A(tensorL1A, tensorTileA);
            }
            InitZeroInL1A(tensorL1A, tla::MakeShape(mL1Actual, kL1Actual));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1AListId]);

            // Load matrix B tile from GM to L1
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1BListId]);
            auto tensorL1B = tla::MakeTensor(l1BTensorList[l1BListId], L1B_LAYOUT, Arch::PositionL1{});
            auto tensorTileB =
                GetTile(tensorB, tla::MakeCoord(kL1Idx * L1_TILE_K, 0), tla::MakeShape(kL1Actual, nBlockActual));
            if constexpr (ENABLE_L1_RESIDENT) {
                if (lastAddrB[l1BListId] != tensorTileB.data().GetPhyAddr() ||
                    tla::get<0>(tensorTileB.coord()) != lastCoordB[l1BListId].row() ||
                    tla::get<1>(tensorTileB.coord()) != lastCoordB[l1BListId].column()) {
                    copyGmToL1B(tensorL1B, tensorTileB);
                    lastCoordB[l1BListId] = MatrixCoord{
                        static_cast<MatrixCoord::Index>(tla::get<0>(tensorTileB.coord())),
                        static_cast<MatrixCoord::Index>(tla::get<1>(tensorTileB.coord()))};
                    lastAddrB[l1BListId] = const_cast<__gm__ typename AscendC::GlobalTensor<ElementB>::PrimType *>(
                        tensorTileB.data().GetPhyAddr());
                }
            } else {
                copyGmToL1B(tensorL1B, tensorTileB);
            }
            InitZeroInL1B(tensorL1B, tla::MakeShape(kL1Actual, nL1Actual));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1BListId]);

            // Execute the oldest queued stripe's mmad once the preload window is full.
            if (preloadCount == PRELOAD_STAGES) {
                L1TileMmad(l1TileMmadParamsList[l1TileMmadParamsId]);
            }

            // Enqueue the current stripe.
            uint32_t slot = (l1TileMmadParamsId + preloadCount < PRELOAD_STAGES)
                ? (l1TileMmadParamsId + preloadCount) : (l1TileMmadParamsId + preloadCount - PRELOAD_STAGES);
            auto &p = l1TileMmadParamsList[slot];
            p.l1AListId = l1AListId;
            p.l1BListId = l1BListId;
            p.scaleAId = curScaleAId;
            p.scaleBId = curScaleBId;
            p.kL1Idx = kL1Idx;
            p.kL1Actual = kL1Actual;
            p.mL1Actual = mL1Actual;
            p.nL1Actual = nL1Actual;
            p.isKLoopFirst = (kL1Idx == 0);
            p.isKLoopLast = (kL1Idx == kL1Loop - 1);
            if (p.isKLoopLast) {
                p.gmBlockC = tensorC.data();
                p.layoutCInGm = tensorC.layout();
                p.coord = tla::MakeCoord(
                    static_cast<int64_t>(tla::get<0>(tensorC.coord())),
                    static_cast<uint32_t>(tla::get<1>(tensorC.coord())));
            }

            if (preloadCount < PRELOAD_STAGES) {
                ++preloadCount;
            } else {
                l1TileMmadParamsId = (l1TileMmadParamsId + 1 < PRELOAD_STAGES) ? (l1TileMmadParamsId + 1) : 0;
            }
            l1AListId = (l1AListId + 1 < L1A_STAGES) ? (l1AListId + 1) : 0;
            l1BListId = (l1BListId + 1 < L1B_STAGES) ? (l1BListId + 1) : 0;
        }
    }

    /// Drain the remaining queued K-stripe mmads (call once at the end of the kernel). The TensorC
    /// template parameter is accepted for interface compatibility with the kernel but is unused: the
    /// C tile is reconstructed from the stored (GM) params inside L1TileMmad.
    template <class TensorC = void>
    CATLASS_DEVICE void SynchronizeBlock()
    {
        while (preloadCount > 0) {
            L1TileMmad(l1TileMmadParamsList[l1TileMmadParamsId]);
            l1TileMmadParamsId = (l1TileMmadParamsId + 1 < PRELOAD_STAGES) ? (l1TileMmadParamsId + 1) : 0;
            --preloadCount;
        }
    }

protected:
    struct L1TileMmadParams {
        uint32_t l1AListId;
        uint32_t l1BListId;
        uint32_t scaleAId;
        uint32_t scaleBId;
        uint32_t kL1Idx;
        uint32_t kL1Actual;
        uint32_t mL1Actual;
        uint32_t nL1Actual;
        bool isKLoopFirst;
        bool isKLoopLast;
        AscendC::GlobalTensor<ElementC> gmBlockC;
        LayoutC layoutCInGm;
        tla::tuple<int64_t, uint32_t> coord;

        CATLASS_DEVICE
        L1TileMmadParams() = default;
    };

    /// Consume one queued K-stripe: L1->L0 (A/B with their MX scale), mmad into L0C, and on the last
    /// stripe of a tile, fixpipe L0C->GM and release the tile's scale buffers. The C tile is
    /// reconstructed from the stored GM params, so this is not templated on the caller's tensor type.
    CATLASS_DEVICE void L1TileMmad(L1TileMmadParams const &params)
    {
        auto tensorL1A = tla::MakeTensor(l1ATensorList[params.l1AListId], L1A_LAYOUT, Arch::PositionL1{});
        auto tensorL1B = tla::MakeTensor(l1BTensorList[params.l1BListId], L1B_LAYOUT, Arch::PositionL1{});
        auto tensorL1MxScaleA =
            tla::MakeTensor(l1MxScaleATensorList[params.scaleAId], L1SCALEA_LAYOUT, Arch::PositionL1{});
        auto tensorL1MxScaleB =
            tla::MakeTensor(l1MxScaleBTensorList[params.scaleBId], L1SCALEB_LAYOUT, Arch::PositionL1{});

        auto layoutInL0C = tla::MakeLayoutL0C(params.mL1Actual, params.nL1Actual);
        auto tensorL0C = tla::MakeTensor(l0CTensorList[l0CListId], layoutInL0C, Arch::PositionL0C{});

        // On the first stripe of a tile, wait the tile's scale buffers to be loaded, and (non-unitflag)
        // wait the L0C accumulator to be free.
        if (params.isKLoopFirst) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1MxScaleAEventList[params.scaleAId]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1MxScaleBEventList[params.scaleBId]);
            if constexpr (!ENABLE_UNIT_FLAG) {
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventList[l0CListId]);
            }
        }

        uint32_t const l0kOffset = L1_TILE_K * (params.kL1Idx % L1_SCALE_FACTOR_K);
        uint32_t kL0Loop = CeilDiv<L0_TILE_K>(params.kL1Actual);
        for (uint32_t kL0Idx = 0; kL0Idx < kL0Loop; ++kL0Idx) {
            uint32_t kL0Actual = (kL0Idx < kL0Loop - 1) ? L0_TILE_K : (params.kL1Actual - kL0Idx * L0_TILE_K);
            kL0Actual = RoundUp<MX_BASEK_FACTOR>(kL0Actual);

            auto layoutAInL0 = tla::MakeLayout<ElementA, LayoutTagL0A>(params.mL1Actual, kL0Actual);
            auto tensorL0A = tla::MakeTensor(l0ATensorList[l0AListId], layoutAInL0, Arch::PositionL0A{});
            auto tensorTileL1A =
                GetTile(tensorL1A, tla::MakeCoord(0, kL0Idx * L0_TILE_K), tla::MakeShape(params.mL1Actual, kL0Actual));
            auto tensorTileL1MxScaleA = GetTile(
                tensorL1MxScaleA,
                tla::MakeCoord(0, (l0kOffset + kL0Idx * L0_TILE_K) / MX_SCALE_GROUP_NUM),
                tla::MakeShape(params.mL1Actual, CeilDiv<MX_SCALE_GROUP_NUM>(kL0Actual)));

            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
            if (kL0Idx == 0) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[params.l1AListId]);
            }
            copyL1ToL0A(tensorL0A, tensorTileL1A, tensorTileL1MxScaleA);
            if (kL0Idx == kL0Loop - 1) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[params.l1AListId]);
            }

            auto layoutBInL0 = tla::MakeLayout<ElementB, LayoutTagL0B>(kL0Actual, params.nL1Actual);
            auto tensorL0B = tla::MakeTensor(l0BTensorList[l0BListId], layoutBInL0, Arch::PositionL0B{});
            auto tensorTileL1B =
                GetTile(tensorL1B, tla::MakeCoord(kL0Idx * L0_TILE_K, 0), tla::MakeShape(kL0Actual, params.nL1Actual));
            auto tensorTileL1MxScaleB = GetTile(
                tensorL1MxScaleB,
                tla::MakeCoord((l0kOffset + kL0Idx * L0_TILE_K) / MX_SCALE_GROUP_NUM, 0),
                tla::MakeShape(CeilDiv<MX_SCALE_GROUP_NUM>(kL0Actual), params.nL1Actual));

            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
            if (kL0Idx == 0) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[params.l1BListId]);
            }
            copyL1ToL0B(tensorL0B, tensorTileL1B, tensorTileL1MxScaleB);
            if (kL0Idx == kL0Loop - 1) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[params.l1BListId]);
            }

            bool initC = (params.isKLoopFirst && (kL0Idx == 0));

            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0CEventList[l0CListId]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0CEventList[l0CListId]);

            uint8_t unitFlag = 0b00;
            if constexpr (ENABLE_UNIT_FLAG) {
                if (params.isKLoopLast && (kL0Idx == kL0Loop - 1)) {
                    unitFlag = 0b11;
                } else {
                    unitFlag = 0b10;
                }
            }

            tileMmad(tensorL0C, tensorL0A, tensorL0B, params.mL1Actual, params.nL1Actual, kL0Actual, initC, unitFlag);

            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
            l0BListId = (l0BListId + 1 < L0B_STAGES) ? (l0BListId + 1) : 0;
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
            l0AListId = (l0AListId + 1 < L0A_STAGES) ? (l0AListId + 1) : 0;
        }

        if (params.isKLoopLast) {
            // The tile's scale buffers are now fully consumed -> release them for reuse.
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1MxScaleAEventList[params.scaleAId]);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1MxScaleBEventList[params.scaleBId]);

            auto tensorTileC =
                tla::MakeTensor(params.gmBlockC, params.layoutCInGm, params.coord, Arch::PositionGM{});
            using CopyL0CToDst = typename TileCopy::template CopyL0CToDst<decltype(tensorTileC)>;
            CopyL0CToDst copyL0CToDst;
            if constexpr (!ENABLE_UNIT_FLAG) {
                AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CEventList[l0CListId]);
                AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CEventList[l0CListId]);
                copyL0CToDst(tensorTileC, tensorL0C);
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventList[l0CListId]);
            } else {
                copyL0CToDst(tensorTileC, tensorL0C, 0b11);
            }
            l0CListId = (l0CListId + 1 < L0C_STAGES) ? (l0CListId + 1) : 0;
        }
    }

    template <class TensorL1, class Shape>
    CATLASS_DEVICE void InitZeroInL1A(TensorL1 &tensorL1, Shape actualShape)
    {
        constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<typename TensorL1::Element>::value;
        const uint32_t mL1Actual = tla::get<0>(actualShape);
        const uint32_t kL1Actual = tla::get<1>(actualShape);
        uint32_t alignKL1 = RoundUp<MX_BASEK_FACTOR>(kL1Actual);
        uint32_t validKL1 = kL1Actual;
        if constexpr (tla::detail::iszN<typename TensorL1::Element, typename TensorL1::Layout>::value) {
            validKL1 = RoundUp<ELE_NUM_PER_C0>(kL1Actual);
        }
        uint32_t padKL1 = alignKL1 - validKL1;
        if (padKL1 > 0) {
            AscendC::InitConstValueParams<uint16_t> initConstValueParams;
            if constexpr (tla::detail::iszN<typename TensorL1::Element, typename TensorL1::Layout>::value) {
                initConstValueParams.repeatTimes = 1;
                initConstValueParams.blockNum = mL1Actual;
                initConstValueParams.initValue = 0;
                initConstValueParams.dstGap = 0;
            } else {
                initConstValueParams.repeatTimes = CeilDiv<ELE_NUM_PER_C0>(mL1Actual);
                initConstValueParams.blockNum = padKL1;
                initConstValueParams.initValue = 0;
                initConstValueParams.dstGap = tla::get<0, 1>(tensorL1.stride()) / ELE_NUM_PER_C0 - padKL1;
            }
            auto offset = tensorL1.layout()(tla::MakeCoord(0, validKL1));
            auto srcUint16 = tensorL1.data()[offset].template ReinterpretCast<uint16_t>();
            InitConstValue(srcUint16, initConstValueParams);
        }
    }

    template <class TensorL1, class Shape>
    CATLASS_DEVICE void InitZeroInL1B(TensorL1 &tensorL1, Shape actualShape)
    {
        constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<typename TensorL1::Element>::value;
        const uint32_t kL1Actual = tla::get<0>(actualShape);
        const uint32_t nL1Actual = tla::get<1>(actualShape);
        uint32_t alignKL1 = RoundUp<MX_BASEK_FACTOR>(kL1Actual);
        uint32_t validKL1 = kL1Actual;
        if constexpr (tla::detail::isnZ<typename TensorL1::Element, typename TensorL1::Layout>::value) {
            validKL1 = RoundUp<ELE_NUM_PER_C0>(kL1Actual);
        }
        uint32_t padKL1 = alignKL1 - validKL1;
        if (padKL1 > 0) {
            AscendC::InitConstValueParams<uint16_t> initConstValueParams;
            if constexpr (tla::detail::iszN<typename TensorL1::Element, typename TensorL1::Layout>::value) {
                initConstValueParams.repeatTimes = CeilDiv<ELE_NUM_PER_C0>(nL1Actual);
                initConstValueParams.blockNum = padKL1;
                initConstValueParams.initValue = 0;
                initConstValueParams.dstGap = tla::get<1, 1>(tensorL1.stride()) / ELE_NUM_PER_C0 - padKL1;
            } else {
                initConstValueParams.repeatTimes = 1;
                initConstValueParams.blockNum = nL1Actual;
                initConstValueParams.initValue = 0;
                initConstValueParams.dstGap = 0;
            }
            auto offset = tensorL1.layout()(tla::MakeCoord(validKL1, 0));
            auto srcUint16 = tensorL1.data()[offset].template ReinterpretCast<uint16_t>();
            InitConstValue(srcUint16, initConstValueParams);
        }
    }

    // Multi-stage tensors and events
    AscendC::LocalTensor<ElementA> l1ATensorList[L1A_STAGES];
    AscendC::LocalTensor<ElementB> l1BTensorList[L1B_STAGES];
    AscendC::LocalTensor<ElementMxScaleA> l1MxScaleATensorList[SCALE_A_STAGES];
    AscendC::LocalTensor<ElementMxScaleB> l1MxScaleBTensorList[SCALE_B_STAGES];
    AscendC::LocalTensor<ElementL0A> l0ATensorList[L0A_STAGES];
    AscendC::LocalTensor<ElementL0B> l0BTensorList[L0B_STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensorList[L0C_STAGES];

    int32_t l1AEventList[L1A_STAGES];
    int32_t l1BEventList[L1B_STAGES];
    int32_t l1MxScaleAEventList[SCALE_A_STAGES];
    int32_t l1MxScaleBEventList[SCALE_B_STAGES];
    int32_t l0AEventList[L0A_STAGES];
    int32_t l0BEventList[L0B_STAGES];
    int32_t l0CEventList[L0C_STAGES];

    __gm__ typename AscendC::GlobalTensor<ElementA>::PrimType *lastAddrA[L1A_STAGES];
    __gm__ typename AscendC::GlobalTensor<ElementB>::PrimType *lastAddrB[L1B_STAGES];
    MatrixCoord lastCoordA[L1A_STAGES];
    MatrixCoord lastCoordB[L1B_STAGES];

    uint32_t l1AListId{0};
    uint32_t l1BListId{0};
    uint32_t l1MxScaleAListId{0};
    uint32_t l1MxScaleBListId{0};
    uint32_t l0AListId{0};
    uint32_t l0BListId{0};
    uint32_t l0CListId{0};

    L1TileMmadParams l1TileMmadParamsList[PRELOAD_STAGES];
    uint32_t l1TileMmadParamsId{0};
    uint32_t preloadCount{0};

    TileMmad tileMmad;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;
    CopyL1ToBT copyL1ToBT;
};

#endif // (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)

} // namespace Block
} // namespace Catlass::Gemm

#endif // CATLASS_GEMM_BLOCK_BLOCK_MMAD_MX_PRELOAD_TLA_HPP
