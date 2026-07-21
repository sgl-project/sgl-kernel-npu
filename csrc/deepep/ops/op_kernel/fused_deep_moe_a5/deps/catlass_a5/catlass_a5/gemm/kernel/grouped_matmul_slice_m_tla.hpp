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

#ifndef CATLASS_GEMM_KERNEL_GROUPED_MATMUL_SLICE_M_TLA_HPP
#define CATLASS_GEMM_KERNEL_GROUPED_MATMUL_SLICE_M_TLA_HPP

#include "catlass_a5/catlass.hpp"
#include "catlass_a5/arch/resource.hpp"
#include "catlass_a5/coord.hpp"
#include "catlass_a5/gemm_coord.hpp"
#include "catlass_a5/matrix_coord.hpp"
#include "tla_a5/layout.hpp"
#include "tla_a5/tensor.hpp"

namespace Catlass::Gemm::Kernel {

template <class BlockMmadTla_, class BlockEpilogue_, class BlockScheduler_, class ElementGroupList_>
class GroupedMatmulSliceMTla
{
public:
    using BlockMmadTla = BlockMmadTla_;
    using ArchTag = typename BlockMmadTla::ArchTag;
    using L1TileShape = typename BlockMmadTla::L1TileShape;
    using ElementA = typename BlockMmadTla::ElementA;
    using LayoutA = typename BlockMmadTla::LayoutA;
    using ElementB = typename BlockMmadTla::ElementB;
    using LayoutB = typename BlockMmadTla::LayoutB;
    using ElementC = typename BlockMmadTla::ElementC;
    using LayoutC = typename BlockMmadTla::LayoutC;
    using ElementAccumulator = typename BlockMmadTla::ElementAccumulator;

    using ElementGroupList = ElementGroupList_;

    using BlockScheduler = BlockScheduler_;
    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});

    struct Params {
        GemmCoord problemShape;
        uint32_t problemCount;
        __gm__ ElementGroupList *ptrGroupList;
        __gm__ ElementA *ptrA;
        LayoutA layoutA;
        __gm__ ElementB *ptrB;
        LayoutB layoutB;
        __gm__ ElementC *ptrC;
        LayoutC layoutC;

        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(GemmCoord const &problemShape_, uint32_t problemCount_, GM_ADDR ptrGroupList_, GM_ADDR ptrA_,
               LayoutA const &layoutA_, GM_ADDR ptrB_, LayoutB const &layoutB_, GM_ADDR ptrC_, LayoutC const &layoutC_)
            : problemShape(problemShape_),
              problemCount(problemCount_),
              ptrGroupList(reinterpret_cast<__gm__ ElementGroupList *>(ptrGroupList_)),
              ptrA(reinterpret_cast<__gm__ ElementA *>(ptrA_)),
              layoutA(layoutA_),
              ptrB(reinterpret_cast<__gm__ ElementB *>(ptrB_)),
              layoutB(layoutB_),
              ptrC(reinterpret_cast<__gm__ ElementC *>(ptrC_)),
              layoutC(layoutC_)
        {}
    };

    struct Arguments {
        GemmCoord problemShape;
        uint32_t problemCount;
        uint8_t *ptrGroupList;
        uint8_t *ptrA;
        LayoutA layoutA;
        uint8_t *ptrB;
        LayoutB layoutB;
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

    static Params ToUnderlyingArguments(const Arguments &args, void *workspace)
    {
        Params params{args.problemShape, args.problemCount, args.ptrGroupList, args.ptrA,   args.layoutA,
                      args.ptrB,         args.layoutB,      args.ptrC,         args.layoutC};
        return params;
    }

    CATLASS_HOST_DEVICE
    GroupedMatmulSliceMTla() {}

    CATLASS_HOST_DEVICE
    ~GroupedMatmulSliceMTla() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const &params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const &params)
    {
        BlockScheduler blockScheduler;
        Arch::Resource<ArchTag> resource;
        BlockMmadTla blockMmadTla(resource);

        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer(params.ptrA);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer(params.ptrC);
        AscendC::GlobalTensor<ElementGroupList> groupList;
        groupList.SetGlobalBuffer(params.ptrGroupList);

        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();
        int64_t gmGroupOffsetB = 0;
        int64_t totalM = 0;
        uint32_t startCoreIdx = 0;
        auto tensorA = tla::MakeTensor(gmA, params.layoutA, Arch::PositionGM{});
        auto tensorC = tla::MakeTensor(gmC, params.layoutC, Arch::PositionGM{});
        for (uint32_t groupIdx = 0; groupIdx < params.problemCount; ++groupIdx) {
            uint32_t currentM = (groupIdx == 0) ? groupList.GetValue(groupIdx)
                                                : (groupList.GetValue(groupIdx) - groupList.GetValue(groupIdx - 1));

            GemmCoord inGroupProblemShape{currentM, params.problemShape.n(), params.problemShape.k()};

            blockScheduler.Update(inGroupProblemShape, MakeCoord(L1_TILE_M, L1_TILE_N));
            uint32_t coreLoops = blockScheduler.GetCoreLoops();

            AscendC::GlobalTensor<ElementB> gmB;
            gmB.SetGlobalBuffer(params.ptrB + gmGroupOffsetB);
            if (CeilDiv(currentM, L1_TILE_M) == 1) {
                gmB.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            }

            uint32_t startLoopIdx;
            if (coreIdx < startCoreIdx) {
                startLoopIdx = coreIdx + coreNum - startCoreIdx;
            } else {
                startLoopIdx = coreIdx - startCoreIdx;
            }

            auto tensorB = tla::MakeTensor(gmB, params.layoutB, Arch::PositionGM{});
            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += coreNum) {
                GemmCoord blockCoord = blockScheduler.GetBlockCoord(loopIdx);
                GemmCoord actualBlockShape = blockScheduler.GetActualBlockShape(blockCoord);

                auto tensorBlockA =
                    GetTile(tensorA, tla::MakeCoord(totalM + blockCoord.m() * L1_TILE_M, blockCoord.k() * L1_TILE_K),
                            tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
                auto tensorBlockB =
                    GetTile(tensorB, tla::MakeCoord(blockCoord.k() * L1_TILE_K, blockCoord.n() * L1_TILE_N),
                            tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
                auto tensorBlockC =
                    GetTile(tensorC, tla::MakeCoord(totalM + blockCoord.m() * L1_TILE_M, blockCoord.n() * L1_TILE_N),
                            tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));

                blockMmadTla(tensorBlockA, tensorBlockB, tensorBlockC, actualBlockShape);
            }
            totalM += inGroupProblemShape.m();
            gmGroupOffsetB += static_cast<int64_t>(inGroupProblemShape.k()) * inGroupProblemShape.n();

            startCoreIdx = (startCoreIdx + coreLoops) % coreNum;
        }

        if constexpr (BlockMmadTla::DispatchPolicy::ASYNC) {
            blockMmadTla.template SynchronizeBlock<decltype(tensorC)>();
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const &params)
    {}
};

}  // namespace Catlass::Gemm::Kernel

#endif  // CATLASS_GEMM_KERNEL_GROUPED_MATMUL_SLICE_M_TLA_HPP
