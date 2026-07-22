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

#ifndef CATLASS_GEMM_KERNEL_QUANT_MULTI_CORE_SPLITK_MATMUL_TLA_HPP
#define CATLASS_GEMM_KERNEL_QUANT_MULTI_CORE_SPLITK_MATMUL_TLA_HPP

#include <algorithm>
#include <cstddef>
#include "catlass_a5/catlass.hpp"
#include "catlass_a5/coord.hpp"
#include "catlass_a5/gemm_coord.hpp"
#include "catlass_a5/matrix_coord.hpp"
#include "catlass_a5/arch/resource.hpp"
#include "catlass_a5/arch/cross_core_sync.hpp"
#include "catlass_a5/gemm/kernel/splitk_matmul.hpp"
#include "catlass_a5/gemm/block/block_swizzle.hpp"
#include "tla_a5/layout.hpp"
#include "tla_a5/tensor.hpp"

namespace Catlass::Gemm::Kernel {

template <class BlockMmad_, class BlockEpilogue_, class BlockScheduler_, class ReduceAdd_>
class QuantMultiCoreSplitkMatmulTla
{
public:
    using BlockMmad = BlockMmad_;
    using BlockEpilogue = BlockEpilogue_;
    using SplitkScheduler = BlockScheduler_;
    using ReduceAdd = ReduceAdd_;

    using EpilogueScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
    using L1TileShape = typename BlockMmad::L1TileShape;

    using ArchTag = typename BlockMmad::ArchTag;
    using ElementA = typename BlockMmad::ElementA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;

    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;
    using ElementAccumulator = typename BlockMmad::ElementAccumulator;

    using ElementScale = typename BlockEpilogue::ElementScale;
    using LayoutScale = typename BlockEpilogue::LayoutScale;
    using ElementPerTokenScale = typename BlockEpilogue::ElementPerTokenScale;
    using LayoutPerTokenScale = typename BlockEpilogue::LayoutPerTokenScale;
    using ElementD = typename BlockEpilogue::ElementD;
    using LayoutD = typename BlockEpilogue::LayoutD;

    using EpilogueParams = typename BlockEpilogue::Params;

    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});

    struct Params {
        GemmCoord problemShape;

        GM_ADDR ptrA;
        LayoutA layoutA;

        GM_ADDR ptrB;
        LayoutB layoutB;

        GM_ADDR ptrScale;
        LayoutScale layoutScale;

        GM_ADDR ptrPerTokenScale;
        LayoutPerTokenScale layoutPerTokenScale;

        GM_ADDR ptrD;
        LayoutD layoutD;

        GM_ADDR ptrWorkspace;
        uint32_t splitkFactor = 1;

        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(GemmCoord const &problemShape_, GM_ADDR ptrA_, LayoutA layoutA_, GM_ADDR ptrB_, LayoutB layoutB_,
               GM_ADDR ptrScale_, LayoutScale layoutScale_, GM_ADDR ptrPerTokenScale_,
               LayoutPerTokenScale layoutPerTokenScale_, GM_ADDR ptrD_, LayoutD layoutD_, GM_ADDR ptrWorkspace_,
               uint32_t splitkFactor_)
            : problemShape(problemShape_),
              ptrA(ptrA_),
              layoutA(layoutA_),
              ptrB(ptrB_),
              layoutB(layoutB_),
              ptrScale(ptrScale_),
              layoutScale(layoutScale_),
              ptrPerTokenScale(ptrPerTokenScale_),
              layoutPerTokenScale(layoutPerTokenScale_),
              ptrD(ptrD_),
              layoutD(layoutD_),
              ptrWorkspace(ptrWorkspace_),
              splitkFactor(splitkFactor_)
        {}
    };

    struct Arguments {
        GemmCoord problemShape;
        uint32_t aicCoreNum;

        GM_ADDR ptrA;
        GM_ADDR ptrB;
        GM_ADDR ptrScale;
        GM_ADDR ptrPerTokenScale;
        GM_ADDR ptrD;

        LayoutA layoutA;
        LayoutB layoutB;
        LayoutScale layoutScale;
        LayoutPerTokenScale layoutPerTokenScale;
        LayoutD layoutD;
    };

    static uint32_t GetSplitkFactor(uint32_t m, uint32_t n, uint32_t k, uint32_t aicCoreNum)
    {
        uint32_t splitkFactor = 2;
        uint32_t blockNum = CeilDiv(m, L1_TILE_M) * CeilDiv(n, L1_TILE_N);
        uint32_t kTileNum = CeilDiv(k, L1_TILE_K);

        if (blockNum > 0 && aicCoreNum / blockNum > 0) {
            splitkFactor = aicCoreNum / blockNum;
        }
        splitkFactor = std::min(splitkFactor, kTileNum);
        splitkFactor = std::max(splitkFactor, 1u);
        return splitkFactor;
    }

    static bool CanImplement(const Arguments &args)
    {
        return true;
    }

    static size_t GetWorkspaceSize(const Arguments &args)
    {
        uint32_t m = args.problemShape.m();
        uint32_t n = args.problemShape.n();
        uint32_t k = args.problemShape.k();
        uint32_t splitkFactor = GetSplitkFactor(m, n, k, args.aicCoreNum);

        size_t partialSize = static_cast<size_t>(m) * n * sizeof(ElementAccumulator) * splitkFactor;
        size_t reducedSize = static_cast<size_t>(m) * n * sizeof(ElementC);
        size_t totalSize = partialSize + reducedSize;

        return std::max(totalSize, static_cast<size_t>(2 * 1024 * 1024));
    }

    static Params ToUnderlyingArguments(const Arguments &args, uint8_t *workspace)
    {
        uint32_t splitkFactor =
            GetSplitkFactor(args.problemShape.m(), args.problemShape.n(), args.problemShape.k(), args.aicCoreNum);

        return Params{args.problemShape,
                      args.ptrA,
                      args.layoutA,
                      args.ptrB,
                      args.layoutB,
                      args.ptrScale,
                      args.layoutScale,
                      args.ptrPerTokenScale,
                      args.layoutPerTokenScale,
                      args.ptrD,
                      args.layoutD,
                      workspace,
                      splitkFactor};
    }

    CATLASS_DEVICE
    QuantMultiCoreSplitkMatmulTla() : flagAicFinish(0) {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const &params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const &params)
    {
        SplitkScheduler matmulBlockScheduler(params.problemShape, GemmCoord(L1_TILE_M, L1_TILE_N, L1_TILE_K),
                                             params.splitkFactor);
        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrA));

        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.ptrB));

        AscendC::GlobalTensor<ElementAccumulator> gmPartial;
        gmPartial.SetGlobalBuffer(reinterpret_cast<__gm__ ElementAccumulator *>(GetPartialWorkspace(params)));

        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();

        auto tensorA = tla::MakeTensor(gmA, params.layoutA, Arch::PositionGM{});
        auto tensorB = tla::MakeTensor(gmB, params.layoutB, Arch::PositionGM{});

        auto layoutPartial = tla::MakeLayout<ElementAccumulator, layout::RowMajor>(
            params.problemShape.m() * params.splitkFactor, params.problemShape.n());
        auto tensorPartial = tla::MakeTensor(gmPartial, layoutPartial, Arch::PositionGM{});

        BlockMmad blockMmad(resource);

        for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += coreNum) {
            GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
            uint32_t splitkSliceIdx = matmulBlockScheduler.GetSplitkSliceIdx(loopIdx);
            GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord, splitkSliceIdx);

            auto tensorBlockA = GetTile(tensorA, tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.k() * L1_TILE_K),
                                        tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));

            auto tensorBlockB = GetTile(tensorB, tla::MakeCoord(blockCoord.k() * L1_TILE_K, blockCoord.n() * L1_TILE_N),
                                        tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));

            auto tensorBlockPartial =
                GetTile(tensorPartial,
                        tla::MakeCoord(splitkSliceIdx * params.problemShape.m() + blockCoord.m() * L1_TILE_M,
                                       blockCoord.n() * L1_TILE_N),
                        tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));

            blockMmad(tensorBlockA, tensorBlockB, tensorBlockPartial, actualBlockShape);
        }

        if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
            blockMmad.SynchronizeBlock();
        }

        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(flagAicFinish);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const &params)
    {
        Catlass::Arch::CrossCoreWaitFlag(flagAicFinish);
        Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE2>();

        AscendC::GlobalTensor<ElementAccumulator> gmPartial;
        gmPartial.SetGlobalBuffer(reinterpret_cast<__gm__ ElementAccumulator *>(GetPartialWorkspace(params)));

        AscendC::GlobalTensor<ElementC> gmReduced;
        gmReduced.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(GetReducedWorkspace(params)));

        ReduceAdd reduceAdd(resource);
        reduceAdd(gmReduced, gmPartial,
                  static_cast<uint64_t>(params.problemShape.m()) * static_cast<uint64_t>(params.problemShape.n()),
                  params.splitkFactor);

        AscendC::PipeBarrier<PIPE_ALL>();
        Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE2>();

        EpilogueScheduler blockScheduler;
        BlockEpilogue blockEpilogue(resource);

        uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t coreNum = AscendC::GetBlockNum() / AscendC::GetSubBlockNum();

        auto layoutReduced =
            tla::MakeLayout<ElementC, layout::RowMajor>(params.problemShape.m(), params.problemShape.n());
        auto tensorC = tla::MakeTensor(gmReduced, layoutReduced, Arch::PositionGM{});

        AscendC::GlobalTensor<ElementScale> gmScale;
        gmScale.SetGlobalBuffer(reinterpret_cast<__gm__ ElementScale *>(params.ptrScale));
        auto tensorScale = tla::MakeTensor(gmScale, params.layoutScale, Arch::PositionGM{});

        AscendC::GlobalTensor<ElementPerTokenScale> gmPerTokenScale;
        gmPerTokenScale.SetGlobalBuffer(reinterpret_cast<__gm__ ElementPerTokenScale *>(params.ptrPerTokenScale));
        auto tensorPerTokenScale = tla::MakeTensor(gmPerTokenScale, params.layoutPerTokenScale, Arch::PositionGM{});

        AscendC::GlobalTensor<ElementD> gmD;
        gmD.SetGlobalBuffer(reinterpret_cast<__gm__ ElementD *>(params.ptrD));
        auto tensorD = tla::MakeTensor(gmD, params.layoutD, Arch::PositionGM{});

        blockScheduler.Update(params.problemShape, MakeCoord(L1_TILE_M, L1_TILE_N));

        blockEpilogue.UpdateParams(EpilogueParams{params.ptrScale, params.layoutScale, params.ptrPerTokenScale,
                                                  params.layoutPerTokenScale, params.ptrD, params.layoutD});
        uint32_t coreLoops = blockScheduler.GetCoreLoops();

        for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += coreNum) {
            GemmCoord blockCoord = blockScheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBlockShapeMNK = blockScheduler.GetActualBlockShape(blockCoord);

            auto tensorBlockC = GetTile(tensorC, tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.n() * L1_TILE_N),
                                        tla::MakeShape(actualBlockShapeMNK.m(), actualBlockShapeMNK.n()));

            auto tensorBlockScale = GetTile(tensorScale, tla::MakeCoord(0, blockCoord.n() * L1_TILE_N),
                                            tla::MakeShape(tla::Int<1>{}, actualBlockShapeMNK.n()));

            auto tensorBlockPerTokenScale = GetTile(tensorPerTokenScale, tla::MakeCoord(0, blockCoord.m() * L1_TILE_M),
                                                    tla::MakeShape(tla::Int<1>{}, actualBlockShapeMNK.m()));

            auto tensorBlockD = GetTile(tensorD, tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.n() * L1_TILE_N),
                                        tla::MakeShape(actualBlockShapeMNK.m(), actualBlockShapeMNK.n()));

            blockEpilogue(tensorBlockC, tensorBlockScale, tensorBlockPerTokenScale, tensorBlockD, actualBlockShapeMNK);
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    CATLASS_HOST_DEVICE
    static size_t GetLenMN(GemmCoord const &problemShape)
    {
        return static_cast<size_t>(problemShape.m()) * problemShape.n();
    }

    CATLASS_HOST_DEVICE
    static GM_ADDR GetPartialWorkspace(Params const &params)
    {
        return params.ptrWorkspace;
    }

    CATLASS_HOST_DEVICE
    static GM_ADDR GetReducedWorkspace(Params const &params)
    {
        size_t partialBytes = GetLenMN(params.problemShape) * params.splitkFactor * sizeof(ElementAccumulator);
        return params.ptrWorkspace + partialBytes;
    }

private:
    Arch::CrossCoreFlag flagAicFinish;
    Arch::Resource<ArchTag> resource;
};

}  // namespace Catlass::Gemm::Kernel

#endif  // CATLASS_GEMM_KERNEL_QUANT_MULTI_CORE_SPLITK_MATMUL_TLA_HPP
