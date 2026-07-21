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

#ifndef CATLASS_CONV_BLOCK_BLOCK_CONV2D_PINGPONG_TLA_HPP
#define CATLASS_CONV_BLOCK_BLOCK_CONV2D_PINGPONG_TLA_HPP

#include "catlass_a5/arch/resource.hpp"
#include "catlass_a5/catlass.hpp"
#include "catlass_a5/conv/dispatch_policy.hpp"
#include "catlass_a5/conv_coord.hpp"
#include "catlass_a5/gemm/helper.hpp"

namespace Catlass::Conv::Block {

template <class ArchTag_, uint32_t L1A_STAGES_, uint32_t L1B_STAGES_, uint32_t L0A_STAGES_, uint32_t L0B_STAGES_,
          uint32_t L0C_STAGES_, bool ENABLE_UNIT_FLAG_,
          class FmapL1TileShape_,    // (hoBlock, woBlock, cin1BlockSmall)
          class FilterL1TileShape_,  // (coutBlock, cin1BlockBig)
          class L0TileShape_,        // (mL0, nL0, kL0)
          class ElementFmap_, class ElementFilter_, class ElementOutput_, class ElementBias_, class TileCopy_,
          class TileMmad_>
struct BlockConv2dTla<
    ConvPingpong<ArchTag_, L1A_STAGES_, L1B_STAGES_, L0A_STAGES_, L0B_STAGES_, L0C_STAGES_, ENABLE_UNIT_FLAG_>,
    FmapL1TileShape_, FilterL1TileShape_, L0TileShape_, ElementFmap_, ElementFilter_, ElementOutput_, ElementBias_,
    TileCopy_, TileMmad_> {
public:
    using DispatchPolicy =
        ConvPingpong<ArchTag_, L1A_STAGES_, L1B_STAGES_, L0A_STAGES_, L0B_STAGES_, L0C_STAGES_, ENABLE_UNIT_FLAG_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using FmapL1TileShape = FmapL1TileShape_;
    using FilterL1TileShape = FilterL1TileShape_;
    using L0TileShape = L0TileShape_;

    using TileCopy = TileCopy_;
    using TileMmad = TileMmad_;

    using ElementFmap = ElementFmap_;
    using LayoutFmap = typename TileCopy::LayoutFmap;
    using ElementFilter = ElementFilter_;
    using LayoutFilter = typename TileCopy::LayoutFilter;
    using ElementOutput = ElementOutput_;
    using LayoutOutput = typename TileCopy::LayoutOutput;

    using CopyGmToL1A = typename TileCopy::CopyGmToL1A;
    using CopyGmToL1B = typename TileCopy::CopyGmToL1B;
    using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;

    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementFmap, ElementFilter>::ElementAccumulator;

    using LayoutTagFmap = typename TileCopy::LayoutTagFmap;
    using LayoutTagFilter = typename TileCopy::LayoutTagFilter;
    using LayoutTagOutput = typename TileCopy::LayoutTagOutput;
    using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
    using LayoutTagL0B = typename TileCopy::LayoutTagL0B;

    static constexpr bool ENABLE_UNIT_FLAG = DispatchPolicy::ENABLE_UNIT_FLAG;
    static constexpr uint32_t MAX_STAGES = 2;
    static constexpr uint32_t L1A_STAGES =
        (DispatchPolicy::L1A_STAGES < MAX_STAGES) ? DispatchPolicy::L1A_STAGES : MAX_STAGES;
    static constexpr uint32_t L1B_STAGES =
        (DispatchPolicy::L1B_STAGES < MAX_STAGES) ? DispatchPolicy::L1B_STAGES : MAX_STAGES;
    static constexpr uint32_t L0A_STAGES =
        (DispatchPolicy::L0A_STAGES < MAX_STAGES) ? DispatchPolicy::L0A_STAGES : MAX_STAGES;
    static constexpr uint32_t L0B_STAGES =
        (DispatchPolicy::L0B_STAGES < MAX_STAGES) ? DispatchPolicy::L0B_STAGES : MAX_STAGES;

    static constexpr uint32_t L0_TILE_M = L0TileShape::M;
    static constexpr uint32_t L0_TILE_N = L0TileShape::N;
    static constexpr uint32_t L0_TILE_K = L0TileShape::K;

    static constexpr uint32_t L0A_PINGPONG_BUF_SIZE = ArchTag::L0A_SIZE / L0A_STAGES;
    static constexpr uint32_t L0B_PINGPONG_BUF_SIZE = ArchTag::L0B_SIZE / L0B_STAGES;

    // Check LayoutTagOutput
    static_assert(std::is_same_v<LayoutTagOutput, layout::NC1HWC0>, "LayoutTagOutput only support NC1HWC0 yet!");

    // Check UnitFlag
    static_assert(!ENABLE_UNIT_FLAG, "UnitFlag only support false yet!");

    // Check L0TileShape
    static constexpr uint32_t L0B_TILE_SIZE = L0TileShape::K * L0TileShape::N * sizeof(ElementFilter);
    static_assert((L0B_TILE_SIZE * L0B_STAGES) <= ArchTag::L0B_SIZE, "L0TileShape exceeding the L0B space!");

    static constexpr uint32_t ELE_NUM_A_PER_C0 = BYTE_PER_C0 / sizeof(ElementFmap);
    static constexpr uint32_t ELE_NUM_B_PER_C0 = BYTE_PER_C0 / sizeof(ElementFilter);

    // Check big and small L1TileShape::Cin1
    static_assert((FilterL1TileShape::Cin1 * ELE_NUM_B_PER_C0 >= FmapL1TileShape::Cin1 * ELE_NUM_A_PER_C0) &&
                      ((FilterL1TileShape::Cin1 * ELE_NUM_B_PER_C0) % (FmapL1TileShape::Cin1 * ELE_NUM_A_PER_C0) == 0),
                  "FilterL1TileShape::Cin must be a multiple of FmapL1TileShape::Cin");

    static constexpr uint32_t K_FMAP_PER_FILTER = FilterL1TileShape::Cin1 / FmapL1TileShape::Cin1;

    /// Construct
    CATLASS_DEVICE
    BlockConv2dTla(Arch::Resource<ArchTag> &resource, const Conv2dFilterParams &filterParams_,
                   uint32_t l1BufAddrStart = 0)
        : filterParams(filterParams_), copyL1ToL0A(filterParams_)
    {
        // 计算输出tile对应输入特征图上的区域大小
        hiBlock =
            (FmapL1TileShape::Ho - 1) * filterParams.strideH() + (filterParams.kh() - 1) * filterParams.dilationH() + 1;
        wiBlock =
            (FmapL1TileShape::Wo - 1) * filterParams.strideW() + (filterParams.kw() - 1) * filterParams.dilationW() + 1;
        l1A_size = FmapL1TileShape::Cin1 * hiBlock * wiBlock * BYTE_PER_C0;
        uint32_t coutRound = RoundUp(FilterL1TileShape::Cout, C0_NUM_PER_FRACTAL);
        l1B_size = FilterL1TileShape::Cin1 * filterParams.kh() * filterParams.kw() * coutRound * BYTE_PER_C0;

        uint32_t l1AOffset = l1BufAddrStart;
        uint32_t l1BOffset = l1BufAddrStart + l1A_size * L1A_STAGES;

        // Init buffers
        for (uint32_t i = 0; i < MAX_STAGES; i++) {
            if (i < L1A_STAGES) {
                l1ATensorList[i] = resource.l1Buf.template GetBufferByByte<ElementFmap>(l1AOffset + l1A_size * i);
                l1AEventList[i] = i;
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
            }
            if (i < L1B_STAGES) {
                l1BTensorList[i] = resource.l1Buf.template GetBufferByByte<ElementFilter>(l1BOffset + l1B_size * i);
                l1BEventList[i] = i + L1A_STAGES;
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
            }
            if (i < L0A_STAGES) {
                l0ATensorList[i] = resource.l0ABuf.template GetBufferByByte<ElementFmap>(L0A_PINGPONG_BUF_SIZE * i);
                l0AEventList[i] = i;
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
            }
            if (i < L0B_STAGES) {
                l0BTensorList[i] = resource.l0BBuf.template GetBufferByByte<ElementFilter>(L0B_PINGPONG_BUF_SIZE * i);
                l0BEventList[i] = i + L0A_STAGES;
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
            }
        }
        l0CTensor = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
    }

    /// Destructor
    CATLASS_DEVICE
    ~BlockConv2dTla()
    {
        for (uint32_t i = 0; i < MAX_STAGES; i++) {
            if (i < L1A_STAGES) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
            }
            if (i < L1B_STAGES) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
            }
            if (i < L0A_STAGES) {
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
            }
            if (i < L0B_STAGES) {
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
            }
        }
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
    }

    template <class TensorFmap, class TensorFilter, class TensorOutput>
    CATLASS_DEVICE void operator()(TensorFmap const &tensorFmap, TensorFilter const &tensorFilter,
                                   TensorOutput const &tensorOutput, Conv2dCoord const &actualShape,
                                   uint8_t *blockPadList)
    {
#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 2201)
        using CopyL0CToGm = typename TileCopy::template CopyL0CToGm<TensorOutput>;
        CopyL0CToGm copyL0CToDst;
#endif
#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)
        using CopyL0CToDst = typename TileCopy::template CopyL0CToDst<TensorOutput>;
        CopyL0CToDst copyL0CToDst;
#endif

        uint8_t blockPadLeft = blockPadList[0];
        uint8_t blockPadRight = blockPadList[1];
        uint8_t blockPadTop = blockPadList[2];
        uint8_t blockPadBottom = blockPadList[3];
        uint32_t wiActual = actualShape.w();
        uint32_t hiActual = actualShape.h();
        uint32_t coutActual = actualShape.cout();
        int32_t wiActualOrg = wiActual + blockPadLeft + blockPadRight;
        int32_t hiActualOrg = hiActual + blockPadTop + blockPadBottom;

        // 计算输出特征图大小
        uint32_t hoActual =
            (hiActualOrg - 1 - (filterParams.kh() - 1) * filterParams.dilationH()) / filterParams.strideH() + 1;
        uint32_t woActual =
            (wiActualOrg - 1 - (filterParams.kw() - 1) * filterParams.dilationW()) / filterParams.strideW() + 1;
        uint32_t howoActual = hoActual * woActual;

        uint32_t nPartLoop = CeilDiv(coutActual, L0_TILE_N);
        uint32_t cin1L0Tile = Max(L0_TILE_K / (filterParams.kh() * filterParams.kw() * ELE_NUM_A_PER_C0), 1U);

        // 特征图layout L1
        auto layoutFmapInL1 = tla::MakeLayoutFmap<ElementFmap>((uint32_t)1, FmapL1TileShape::Cin1, hiActual, wiActual);
        // 卷积核layout L1
        auto layoutFilterInL1 = tla::MakeLayoutFilter<ElementFilter, Arch::PositionL1>(
            FilterL1TileShape::Cin1, filterParams.kh(), filterParams.kw(), coutActual);
        // L0C layout： M: ho、wo合轴, N: cout
        auto layoutInL0C = tla::MakeLayoutL0C(howoActual, coutActual);
        auto tensorL0C = tla::MakeTensor(l0CTensor, layoutInL0C, Arch::PositionL0C{});

        uint32_t cin1Actual = min(actualShape.cin1(), FmapL1TileShape::Cin1);
        uint32_t cin1FilterActual = min(actualShape.cin1(), FilterL1TileShape::Cin1);

        // load first Fmap tile from GM to L1
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1AListId]);

        auto tensorL1A = tla::MakeTensor(l1ATensorList[l1AListId], layoutFmapInL1, Arch::PositionL1{});
        auto tensorTileFmap = GetTile(tensorFmap, tla::MakeCoord(0, 0, 0, 0, 0),
                                      tla::MakeShape((uint32_t)1, cin1Actual, hiActual, wiActual, ELE_NUM_A_PER_C0));

        copyGmToL1A(tensorL1A, tensorTileFmap);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1AListId]);

        // load first Filter tile from GM to L1
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1BListId]);

        auto tensorL1B = tla::MakeTensor(l1BTensorList[l1BListId], layoutFilterInL1, Arch::PositionL1{});
        auto tensorTileFilter =
            GetTile(tensorFilter, tla::MakeCoord(0, 0, 0, 0, 0),
                    tla::MakeShape(cin1FilterActual, (uint32_t)filterParams.kh(), (uint32_t)filterParams.kw(),
                                   actualShape.cout(), ELE_NUM_B_PER_C0));

        copyGmToL1B(tensorL1B, tensorTileFilter);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1BListId]);

        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);

        // main loop
        uint32_t cin1TileCnt = CeilDiv<FmapL1TileShape::Cin1>(actualShape.cin1());
        uint32_t cin1FilterTileCnt = CeilDiv<FilterL1TileShape::Cin1>(actualShape.cin1());
        for (uint32_t cin1LoopIdx = 0; cin1LoopIdx < cin1TileCnt; cin1LoopIdx++) {  // 切特征图输入通道数cin1  L1
            uint32_t cin1FmapIdx = cin1LoopIdx % K_FMAP_PER_FILTER;

            uint32_t l1AListIdNext = (l1AListId + 1 < L1A_STAGES) ? (l1AListId + 1) : 0;
            uint32_t l1BListIdNext = (l1BListId + 1 < L1B_STAGES) ? (l1BListId + 1) : 0;

            uint32_t cin1ActualNext{0};
            uint32_t cin1FilterActualNext{0};

            // preload next tile A from GM to L1
            if (cin1LoopIdx < cin1TileCnt - 1) {
                uint32_t cin1LoopIdxNext = cin1LoopIdx + 1;
                cin1ActualNext = (cin1LoopIdxNext < cin1TileCnt - 1)
                                     ? FmapL1TileShape::Cin1
                                     : (actualShape.cin1() - cin1LoopIdxNext * FmapL1TileShape::Cin1);

                // Get L1 tensor A for next stage
                auto l1ATensor = l1ATensorList[l1AListIdNext];
                auto tensorL1A = tla::MakeTensor(l1ATensor, layoutFmapInL1, Arch::PositionL1{});
                // Get GM tile A for next stage
                auto tensorTileFmap =
                    GetTile(tensorFmap, tla::MakeCoord(0, cin1LoopIdxNext * FmapL1TileShape::Cin1, 0, 0, 0),
                            tla::MakeShape((uint32_t)1, cin1ActualNext, hiActual, wiActual, ELE_NUM_A_PER_C0));

                // load next Fmap tile from GM to L1
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1AListIdNext]);

                copyGmToL1A(tensorL1A, tensorTileFmap);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1AListIdNext]);

                // preload next tile B from GM to L1
                if (cin1FmapIdx == K_FMAP_PER_FILTER - 1) {
                    uint32_t cin1FilterIdxNext = cin1LoopIdxNext / K_FMAP_PER_FILTER;
                    cin1FilterActualNext = (cin1FilterIdxNext < cin1FilterTileCnt - 1)
                                               ? FilterL1TileShape::Cin1
                                               : (actualShape.cin1() - cin1FilterIdxNext * FilterL1TileShape::Cin1);

                    // Get L1 tensor B for next stage
                    auto l1BTensor = l1BTensorList[l1BListIdNext];
                    auto tensorL1B = tla::MakeTensor(l1BTensor, layoutFilterInL1, Arch::PositionL1{});
                    // Get GM tile B for next stage
                    auto tensorTileFilter =
                        GetTile(tensorFilter, tla::MakeCoord(cin1FilterIdxNext * FilterL1TileShape::Cin1, 0, 0, 0, 0),
                                tla::MakeShape(cin1FilterActualNext, (uint32_t)filterParams.kh(),
                                               (uint32_t)filterParams.kw(), actualShape.cout(), ELE_NUM_B_PER_C0));
                    // load next Filter tile from GM to L1
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1BListIdNext]);
                    copyGmToL1B(tensorL1B, tensorTileFilter);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1BListIdNext]);
                }
            }

            // Get L1 tensor for current stage
            auto l1ATensor = l1ATensorList[l1AListId];
            auto l1BTensor = l1BTensorList[l1BListId];
            tensorL1A = tla::MakeTensor(l1ATensor, layoutFmapInL1, Arch::PositionL1{});
            tensorL1B = tla::MakeTensor(l1BTensor, layoutFilterInL1, Arch::PositionL1{});
            // Get the loop nums on L0
            uint32_t kPartLoop = CeilDiv(cin1Actual, cin1L0Tile);

            for (int kPartIdx = 0; kPartIdx < kPartLoop; kPartIdx++) {  // 切特征图输入通道数cin  L0
                uint32_t cin1PartActual =
                    (kPartIdx < kPartLoop - 1) ? cin1L0Tile : (cin1Actual - kPartIdx * cin1L0Tile);
                uint32_t kPartActual = cin1PartActual * filterParams.kh() * filterParams.kw() * ELE_NUM_A_PER_C0;

                // Locate the current tile on L0A
                auto l0ATile = l0ATensorList[l0AListId];
                auto layoutFmapInL0 = tla::MakeLayout<ElementFmap, LayoutTagL0A>(howoActual, kPartActual);
                auto tensorL0A = tla::MakeTensor(l0ATile, layoutFmapInL0, Arch::PositionL0A{});
                // Locate the current tile of matrix A on L1
                auto tensorTileL1A =
                    GetTile(tensorL1A, tla::MakeCoord(0, kPartIdx * cin1L0Tile, 0, 0, 0),
                            tla::MakeShape((uint32_t)1, cin1PartActual, hiActual, wiActual, ELE_NUM_A_PER_C0));

                // Wait for mmad finished
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                if (kPartIdx == 0) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1AListId]);
                }

                // Load current tile from L1 to L0A
                copyL1ToL0A(tensorL0A, tensorTileL1A, blockPadList);

                if (kPartIdx == kPartLoop - 1) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1AListId]);
                }

                for (int nPartIdx = 0; nPartIdx < nPartLoop; nPartIdx++) {  // 切卷积核cout  L0
                    uint32_t nPartActual = (nPartIdx < nPartLoop - 1) ? L0_TILE_N : (coutActual - nPartIdx * L0_TILE_N);

                    // Locate the current tile on L0B
                    auto l0BTile = l0BTensorList[l0BListId];
                    auto layoutFilterInL0 = tla::MakeLayout<ElementFilter, LayoutTagL0B>(kPartActual, nPartActual);
                    auto tensorL0B = tla::MakeTensor(l0BTile, layoutFilterInL0, Arch::PositionL0B{});
                    // Load current tile of matrix B on L1
                    auto tensorTileL1B =
                        GetTile(tensorL1B,
                                tla::MakeCoord(cin1FmapIdx * FmapL1TileShape::Cin1 + kPartIdx * cin1L0Tile, 0, 0,
                                               nPartIdx * L0_TILE_N, 0),
                                tla::MakeShape(cin1PartActual, (uint32_t)filterParams.kh(), (uint32_t)filterParams.kw(),
                                               nPartActual, ELE_NUM_B_PER_C0));

                    // Wait for mmad finished
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                    if ((cin1FmapIdx == 0) && (kPartIdx == 0) && (nPartIdx == 0)) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1BListId]);
                    }

                    // Load current tile from L1 to L0B
                    copyL1ToL0B(tensorL0B, tensorTileL1B);

                    if (((cin1FmapIdx == K_FMAP_PER_FILTER - 1) || (cin1LoopIdx == cin1TileCnt - 1)) &&
                        (kPartIdx == kPartLoop - 1) && (nPartIdx == nPartLoop - 1)) {
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1BListId]);
                    }

                    // Notify to do mmad
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);

                    // Locate the current tile on L0C
                    auto tensorTileL0C = GetTile(tensorL0C, tla::MakeCoord(0, nPartIdx * L0_TILE_N),
                                                 tla::MakeShape(howoActual, nPartActual));

                    // Compute the matrix multiplication on L0A and L0B and write the result to the accumulator
                    // Wait for loading L0B
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);

                    // If the current tile is the first tile on the k axis, the accumulator needs to be reset to 0
                    bool initC = (cin1LoopIdx == 0) && (kPartIdx == 0);

                    // Perform calculation operations
                    tileMmad(tensorTileL0C, tensorL0A, tensorL0B, howoActual, nPartActual, kPartActual, initC);

                    // Notify to move the next L0A, L0B tile
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                    l0BListId = (l0BListId + 1 < L0B_STAGES) ? (l0BListId + 1) : 0;
                }
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                l0AListId = (l0AListId + 1 < L0A_STAGES) ? (l0AListId + 1) : 0;
            }
            l1AListId = l1AListIdNext;
            cin1Actual = cin1ActualNext;
            if (cin1FmapIdx == K_FMAP_PER_FILTER - 1) {
                l1BListId = l1BListIdNext;
                cin1FilterActual = cin1FilterActualNext;
            }
        }

        // copy block out
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
        copyL0CToDst(tensorOutput, tensorL0C);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
    }

    static bool CanImplement(const Conv2dFilterParams &filterParams)
    {
        uint32_t hiBlock =
            (FmapL1TileShape::Ho - 1) * filterParams.strideH() + (filterParams.kh() - 1) * filterParams.dilationH() + 1;
        uint32_t wiBlock =
            (FmapL1TileShape::Wo - 1) * filterParams.strideW() + (filterParams.kw() - 1) * filterParams.dilationW() + 1;
        uint32_t l1A_size = FmapL1TileShape::Cin1 * hiBlock * wiBlock * BYTE_PER_C0;
        uint32_t coutRound = RoundUp(FilterL1TileShape::Cout, C0_NUM_PER_FRACTAL);
        uint32_t l1B_size = FilterL1TileShape::Cin1 * filterParams.kh() * filterParams.kw() * coutRound * BYTE_PER_C0;
        if (l1A_size * L1A_STAGES + l1B_size * L1B_STAGES > ArchTag::L1_SIZE) {
            return false;
        }
        return true;
    }

protected:
    Conv2dFilterParams filterParams;
    uint32_t hiBlock, wiBlock;
    uint32_t l1A_size, l1B_size;

    // Multi-stage tensors list
    AscendC::LocalTensor<ElementFmap> l1ATensorList[L1A_STAGES];
    AscendC::LocalTensor<ElementFilter> l1BTensorList[L1B_STAGES];
    AscendC::LocalTensor<ElementFmap> l0ATensorList[L0A_STAGES];
    AscendC::LocalTensor<ElementFilter> l0BTensorList[L0B_STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensor;

    // Multi-stage event id list
    int32_t l1AEventList[L1A_STAGES];
    int32_t l1BEventList[L1B_STAGES];
    int32_t l0AEventList[L0A_STAGES];
    int32_t l0BEventList[L0B_STAGES];

    // The id of current stage
    uint32_t l1AListId{0};
    uint32_t l1BListId{0};
    uint32_t l0AListId{0};
    uint32_t l0BListId{0};

    TileMmad tileMmad;
    CopyGmToL1A copyGmToL1A;
    CopyGmToL1B copyGmToL1B;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;
};

}  // namespace Catlass::Conv::Block

#endif  // CATLASS_CONV_BLOCK_BLOCK_CONV2D_PINGPONG_TLA_HPP
