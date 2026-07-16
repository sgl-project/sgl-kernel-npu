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

#ifndef CATLASS_GEMM_KERNEL_MX_BATCHED_MATMUL_TLA_HPP
#define CATLASS_GEMM_KERNEL_MX_BATCHED_MATMUL_TLA_HPP

#include "catlass_a5/arch/resource.hpp"
#include "catlass_a5/catlass.hpp"
#include "catlass_a5/coord.hpp"
#include "catlass_a5/gemm_coord.hpp"
#include "catlass_a5/matrix_coord.hpp"
#include "tla_a5/layout.hpp"
#include "tla_a5/tensor.hpp"

namespace Catlass::Gemm::Kernel {

#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)

template <
    class BlockMmad_,
    class BlockEpilogue_,
    class BlockScheduler_
>
class MxBatchedMatmulTla {
public:
    using BlockMmad = BlockMmad_;
    using MmadArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using LayoutA = typename BlockMmad::LayoutA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutB = typename BlockMmad::LayoutB;
    using ElementMxScaleA = typename BlockMmad::TileCopy::ElementMxScaleA;
    using LayoutMxScaleA = typename BlockMmad::TileCopy::LayoutMxScaleA;
    using ElementMxScaleB = typename BlockMmad::TileCopy::ElementMxScaleB;
    using LayoutMxScaleB = typename BlockMmad::TileCopy::LayoutMxScaleB;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;
    using ElementAccumulator = typename BlockMmad::ElementAccumulator;

    using BlockScheduler = BlockScheduler_;

    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});

    struct Params {
        uint32_t batchCount;
        GemmCoord problemShape;
        GM_ADDR ptrA;
        LayoutA layoutA;
        int64_t strideA;
        GM_ADDR ptrB;
        LayoutB layoutB;
        int64_t strideB;
        GM_ADDR ptrMxScaleA;
        LayoutMxScaleA layoutMxScaleA; 
        int64_t strideMxScaleA;
        GM_ADDR ptrMxScaleB;
        LayoutMxScaleB layoutMxScaleB;
        int64_t strideMxScaleB;
        GM_ADDR ptrC;
        LayoutC layoutC;
        int64_t strideC;

        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(
            uint32_t batchCount_,
            GemmCoord const &problemShape_,
            GM_ADDR ptrA_,
            LayoutA layoutA_,
            int64_t strideA_,
            GM_ADDR ptrB_,
            LayoutB layoutB_,
            int64_t strideB_,
            GM_ADDR ptrMxScaleA_,
            LayoutMxScaleA layoutMxScaleA_,
            int64_t strideMxScaleA_,
            GM_ADDR ptrMxScaleB_,
            LayoutMxScaleB layoutMxScaleB_,
            int64_t strideMxScaleB_,
            GM_ADDR ptrC_,
            LayoutC layoutC_,
            int64_t strideC_
        )
            : batchCount(batchCount_),
              problemShape(problemShape_),
              ptrA(ptrA_),
              layoutA(layoutA_),
              strideA(strideA_),
              ptrB(ptrB_),
              layoutB(layoutB_),
              strideB(strideB_),
              ptrMxScaleA(ptrMxScaleA_),
              layoutMxScaleA(layoutMxScaleA_),
              strideMxScaleA(strideMxScaleA_),
              ptrMxScaleB(ptrMxScaleB_),
              layoutMxScaleB(layoutMxScaleB_),
              strideMxScaleB(strideMxScaleB_),
              ptrC(ptrC_),
              layoutC(layoutC_),
              strideC(strideC_)
        {}
    };

    struct Arguments {
        uint32_t batchCount;
        GemmCoord problemShape;
        uint8_t *ptrA;
        LayoutA layoutA;
        uint8_t *ptrB;
        LayoutB layoutB;
        uint8_t *ptrMxScaleA;
        LayoutMxScaleA layoutMxScaleA; 
        uint8_t *ptrMxScaleB;
        LayoutMxScaleB layoutMxScaleB;
        uint8_t *ptrC;
        LayoutC layoutC;
    };

    static bool CanImplement(const Arguments &args)
    {
        return true;
    }

    static size_t GetWorkspaceSize(const Arguments &args)
    {
        return 0;
    }

    static Params ToUnderlyingArguments(const Arguments &args, uint8_t *workspace)
    {
        GemmCoord problemShape = args.problemShape;
        uint32_t m = problemShape.m();
        uint32_t n = problemShape.n();
        uint32_t k = problemShape.k();
        int64_t strideA = static_cast<int64_t>(m) * static_cast<int64_t>(k);
        int64_t strideB = static_cast<int64_t>(k) * static_cast<int64_t>(n);
        int64_t strideC = static_cast<int64_t>(m) * static_cast<int64_t>(n);
        int64_t strideMxScaleA = static_cast<int64_t>(m) * static_cast<int64_t>(CeilDiv<MX_BASEK_FACTOR>(k) * MX_SCALE_COPY_GROUP_NUM);
        int64_t strideMxScaleB = static_cast<int64_t>(CeilDiv<MX_BASEK_FACTOR>(k) * MX_SCALE_COPY_GROUP_NUM) * static_cast<int64_t>(n);
        Params params{
            args.batchCount,
            args.problemShape,
            args.ptrA,
            args.layoutA,
            strideA,
            args.ptrB,  
            args.layoutB,
            strideB,
            args.ptrMxScaleA,
            args.layoutMxScaleA,
            strideMxScaleA,
            args.ptrMxScaleB,
            args.layoutMxScaleB,
            strideMxScaleB,
            args.ptrC,
            args.layoutC,
            strideC
        };
        return params;
    }

    CATLASS_DEVICE
    MxBatchedMatmulTla() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params const &params);

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params const &params)
    {
        BlockScheduler matmulBlockScheduler(params.problemShape, MakeCoord(L1_TILE_M, L1_TILE_N));
        uint32_t coreLoops = params.batchCount * matmulBlockScheduler.GetCoreLoops();

        Arch::Resource<MmadArchTag> resource;
        BlockMmad blockMmad(resource);

        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrB);
        AscendC::GlobalTensor<ElementMxScaleA> gmMxScaleA;
        gmMxScaleA.SetGlobalBuffer((__gm__ ElementMxScaleA *)params.ptrMxScaleA);
        AscendC::GlobalTensor<ElementMxScaleB> gmMxScaleB;
        gmMxScaleB.SetGlobalBuffer((__gm__ ElementMxScaleB *)params.ptrMxScaleB);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrC);

        if (CeilDiv(params.problemShape.m(), L1_TILE_M) == 1) {
            gmB.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        }
        if (CeilDiv(params.problemShape.n(), L1_TILE_N) == 1) {
            gmA.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        }

        for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
            uint32_t batchIdx = matmulBlockScheduler.GetBatchIdx(loopIdx);
            GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);

            // batchOffset
            int64_t batchOffsetA = batchIdx * params.strideA;
            int64_t batchOffsetB = batchIdx * params.strideB;
            int64_t batchOffsetC = batchIdx * params.strideC;
            int64_t batchOffsetMxScaleA = batchIdx * params.strideMxScaleA;
            int64_t batchOffsetMxScaleB = batchIdx * params.strideMxScaleB;

            // Represent the full tensors
            auto tensorA = tla::MakeTensor(gmA[batchOffsetA], params.layoutA, Arch::PositionGM{});
            auto tensorB = tla::MakeTensor(gmB[batchOffsetB], params.layoutB, Arch::PositionGM{});
            auto tensorMxScaleA = tla::MakeTensor(gmMxScaleA[batchOffsetMxScaleA], params.layoutMxScaleA, Arch::PositionGM{});
            auto tensorMxScaleB = tla::MakeTensor(gmMxScaleB[batchOffsetMxScaleB], params.layoutMxScaleB, Arch::PositionGM{});
            auto tensorC = tla::MakeTensor(gmC[batchOffsetC], params.layoutC, Arch::PositionGM{});

            auto tensorBlockA = GetTile(
                tensorA, tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.k() * L1_TILE_K),
                tla::MakeShape(actualBlockShape.m(), actualBlockShape.k())
            );
            auto tensorBlockB = GetTile(
                tensorB, tla::MakeCoord(blockCoord.k() * L1_TILE_K, blockCoord.n() * L1_TILE_N),
                tla::MakeShape(actualBlockShape.k(), actualBlockShape.n())
            );
            auto tensorBlockMxScaleA = GetTile(
                tensorMxScaleA,
                tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.k() * L1_TILE_K / MX_SCALE_GROUP_NUM),
                tla::MakeShape(actualBlockShape.m(), CeilDiv<MX_SCALE_GROUP_NUM>(actualBlockShape.k()))
            );
            auto tensorBlockMxScaleB = GetTile(
                tensorMxScaleB,
                tla::MakeCoord(blockCoord.k() * L1_TILE_K / MX_SCALE_GROUP_NUM, blockCoord.n() * L1_TILE_N),
                tla::MakeShape(CeilDiv<MX_SCALE_GROUP_NUM>(actualBlockShape.k()), actualBlockShape.n())
            );
            auto tensorBlockC = GetTile(
                tensorC, tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.n() * L1_TILE_N),
                tla::MakeShape(actualBlockShape.m(), actualBlockShape.n())
            );

            blockMmad(
                tensorBlockA, tensorBlockB, tensorBlockC, actualBlockShape, tensorBlockMxScaleA, tensorBlockMxScaleB
            );
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const &params) {}
};

#endif // (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)

} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_MX_BATCHED_MATMUL_TLA_HPP