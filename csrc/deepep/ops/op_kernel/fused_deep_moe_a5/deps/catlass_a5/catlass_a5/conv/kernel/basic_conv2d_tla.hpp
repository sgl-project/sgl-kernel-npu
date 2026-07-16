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

#ifndef CATLASS_CONV_KERNEL_BASIC_CONV2D_TLA_HPP
#define CATLASS_CONV_KERNEL_BASIC_CONV2D_TLA_HPP

#include "catlass_a5/arch/resource.hpp"
#include "catlass_a5/catlass.hpp"
#include "catlass_a5/conv_coord.hpp"
#include "tla_a5/layout.hpp"
#include "tla_a5/tensor.hpp"

namespace Catlass::Conv::Kernel {

// Template for Conv2d kernel. Compute output = fmap x filter
template <class BlockConv2d_, class BlockEpilogue_, class BlockScheduler_>
class BasicConv2dTla {
public:
    using BlockConv2d = BlockConv2d_;
    using ArchTag = typename BlockConv2d::ArchTag;
    using FmapL1TileShape = typename BlockConv2d::FmapL1TileShape;
    using FilterL1TileShape = typename BlockConv2d::FilterL1TileShape;
    using ElementFmap = typename BlockConv2d::ElementFmap;
    using LayoutFmap = typename BlockConv2d::LayoutFmap;
    using ElementFilter = typename BlockConv2d::ElementFilter;
    using LayoutFilter = typename BlockConv2d::LayoutFilter;
    using ElementOutput = typename BlockConv2d::ElementOutput;
    using LayoutOutput = typename BlockConv2d::LayoutOutput;
    using ElementAccumulator = typename BlockConv2d::ElementAccumulator;

    using BlockScheduler = BlockScheduler_;

    static constexpr uint16_t C0_FMAP = BYTE_PER_C0 / sizeof(ElementFmap);
    static constexpr uint16_t C0_FILTER = BYTE_PER_C0 / sizeof(ElementFilter);
    static constexpr uint16_t C0_OUT = BYTE_PER_C0 / sizeof(ElementOutput);

    /// Parameters structure
    struct Params {
        // Data members
        Conv2dParams problemShape;
        GM_ADDR ptrFmap;
        LayoutFmap layoutFmap;
        GM_ADDR ptrFilter;
        LayoutFilter layoutFilter;
        GM_ADDR ptrOutput;
        LayoutOutput layoutOutput;

        // Methods
        CATLASS_HOST_DEVICE
        Params()
        {
        }

        CATLASS_HOST_DEVICE
        Params(
            Conv2dParams const &problemShape_,
            GM_ADDR ptrFmap_,
            LayoutFmap layoutFmap_,
            GM_ADDR ptrFilter_,
            LayoutFilter layoutFilter_,
            GM_ADDR ptrOutput_,
            LayoutOutput layoutOutput_
        )
            : problemShape(problemShape_)
            , ptrFmap(ptrFmap_)
            , layoutFmap(layoutFmap_)
            , ptrFilter(ptrFilter_)
            , layoutFilter(layoutFilter_)
            , ptrOutput(ptrOutput_)
            , layoutOutput(layoutOutput_)
        {
        }
    };

    struct Arguments {
        Conv2dParams problemShape;
        GM_ADDR ptrFmap;
        GM_ADDR ptrFilter;
        GM_ADDR ptrOutput;
    };

    static bool CanImplement(const Arguments &args)
    {
        if (args.problemShape.strideH() == 0 || args.problemShape.strideW() == 0 ||
            args.problemShape.dilationH() == 0 || args.problemShape.dilationW() == 0) {
            return false; 
        }
        return BlockConv2d::CanImplement(args.problemShape.getFilterParams());
    }

    static size_t GetWorkspaceSize(const Arguments &args)
    {
        return 0;
    }

    static Params ToUnderlyingArguments(const Arguments &args, uint8_t *workspace)
    {
        auto layoutFmap = tla::MakeLayoutFmap<ElementFmap>(
            args.problemShape.batch(), args.problemShape.cin1(), args.problemShape.hi(), args.problemShape.wi());
        auto layoutFilter = tla::MakeLayoutFilter<ElementFilter, Arch::PositionGM>(
            args.problemShape.cin1(), args.problemShape.kh(), args.problemShape.kw(), args.problemShape.cout());
        auto layoutOutput = tla::MakeLayoutFmap<ElementOutput>(
            args.problemShape.batch(), args.problemShape.cout1(), args.problemShape.ho(), args.problemShape.wo());
        Params params{args.problemShape, args.ptrFmap,   layoutFmap,  args.ptrFilter,
                      layoutFilter,      args.ptrOutput, layoutOutput};
        return params;
    }

    // Methods
    CATLASS_DEVICE
    BasicConv2dTla()
    {
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const &params);

    /// Executes one Conv2d
    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const &params)
    {
        BlockScheduler conv2dBlockScheduler(
            params.problemShape.getPostIm2colShape(),
            MakeCoord(FmapL1TileShape::Ho, FmapL1TileShape::Wo, FilterL1TileShape::Cout)
        );
        uint32_t loops = conv2dBlockScheduler.GetLoops();
        Arch::Resource<ArchTag> resource;
        BlockConv2d blockConv2d(resource, params.problemShape.getFilterParams());

        // Represent the full gm
        AscendC::GlobalTensor<ElementFmap> gmFmap;
        gmFmap.SetGlobalBuffer((__gm__ ElementFmap *)params.ptrFmap);
        AscendC::GlobalTensor<ElementFilter> gmFilter;
        gmFilter.SetGlobalBuffer((__gm__ ElementFilter *)params.ptrFilter);
        AscendC::GlobalTensor<ElementOutput> gmOutput;
        gmOutput.SetGlobalBuffer((__gm__ ElementOutput *)params.ptrOutput);

        // Make tla::Tensor
        auto tensorFmap = tla::MakeTensor(gmFmap, params.layoutFmap, Arch::PositionGM{});
        auto tensorFilter = tla::MakeTensor(gmFilter, params.layoutFilter, Arch::PositionGM{});
        auto tensorOutput = tla::MakeTensor(gmOutput, params.layoutOutput, Arch::PositionGM{});

        for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < loops; loopIdx += AscendC::GetBlockNum()) {
            Conv2dCoord blockCoord = conv2dBlockScheduler.GetBlockCoord(loopIdx);
            Conv2dCoord actualBlockShape = conv2dBlockScheduler.GetActualBlockShape(blockCoord);

            uint8_t blockPadLeft = 0, blockPadRight = 0, blockPadTop = 0, blockPadBottom = 0;

            // Compute indices of hi 
            uint32_t hoStart = blockCoord.h() * FmapL1TileShape::Ho;
            int32_t hiStart = hoStart * params.problemShape.strideH() - params.problemShape.padTop();
            int32_t hiEnd = hiStart + (actualBlockShape.h() - 1) * params.problemShape.strideH()
                            + (params.problemShape.kh() - 1) * params.problemShape.dilationH();
            if (hiStart < 0) {
                blockPadTop = 0 - hiStart;
                hiStart = 0;
            }
            if (hiEnd > params.problemShape.hi() - 1) {
                blockPadBottom = hiEnd - (params.problemShape.hi() - 1);
                hiEnd = params.problemShape.hi() - 1;
            } 
            uint32_t hiActual = hiEnd - hiStart + 1;

            // Compute indexes of wi
            uint32_t woStart = blockCoord.w() * FmapL1TileShape::Wo;
            int32_t wiStart = woStart * params.problemShape.strideW() - params.problemShape.padLeft();
            int32_t wiEnd = wiStart + (actualBlockShape.w() - 1) * params.problemShape.strideW()
                            + (params.problemShape.kw() - 1) * params.problemShape.dilationW();
            if (wiStart < 0) {
                blockPadLeft = 0 - wiStart;
                wiStart = 0;
            }
            if (wiEnd > params.problemShape.wi() - 1) {
                blockPadRight = wiEnd - (params.problemShape.wi() - 1);
                wiEnd = params.problemShape.wi() - 1;
            }
            uint32_t wiActual = wiEnd - wiStart + 1;

            Conv2dCoord actualConv2dBlockShape(1, hiActual, wiActual, actualBlockShape.cout(), actualBlockShape.cin1());
            uint8_t blockPadList[4] = {blockPadLeft, blockPadRight, blockPadTop, blockPadBottom};

            auto tensorBlockFmap = GetTile(
                tensorFmap,
                tla::MakeCoord(blockCoord.batch(), 0, (uint32_t)hiStart, (uint32_t)wiStart, 0),
                tla::MakeShape(actualBlockShape.batch(), actualBlockShape.cin1(), hiActual, wiActual, C0_FMAP)
            );

            auto tensorBlockFilter = GetTile(
                tensorFilter,
                tla::MakeCoord(0, 0, 0, blockCoord.cout() * FilterL1TileShape::Cout, 0),
                tla::MakeShape(actualBlockShape.cin1(), params.problemShape.kh(), params.problemShape.kw(), actualBlockShape.cout(), C0_FILTER)
            );

            auto tensorBlockOutput = GetTile(
                tensorOutput,
                tla::MakeCoord(blockCoord.batch(), blockCoord.cout() * FilterL1TileShape::Cout / C0_OUT, hoStart, woStart, 0),
                tla::MakeShape(actualBlockShape.batch(), CeilDiv(actualBlockShape.cout(), C0_OUT), actualBlockShape.h(), actualBlockShape.w(), C0_OUT)
            );

            // Compute block-scoped matrix multiply-add
            blockConv2d(tensorBlockFmap, tensorBlockFilter, tensorBlockOutput, actualConv2dBlockShape, blockPadList);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const &params)
    {
    }
};

} // namespace Catlass::Conv::Kernel

#endif // CATLASS_CONV_KERNEL_BASIC_CONV2D_TLA_HPP
