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

#ifndef CATLASS_GEMM_KERNEL_GROUPED_MX_MATMUL_SLICE_M_TLA_HPP
#define CATLASS_GEMM_KERNEL_GROUPED_MX_MATMUL_SLICE_M_TLA_HPP

#include "catlass_a5/catlass.hpp"
#include "catlass_a5/arch/resource.hpp"
#include "catlass_a5/coord.hpp"
#include "catlass_a5/gemm_coord.hpp"
#include "catlass_a5/matrix_coord.hpp"
#include "tla_a5/layout.hpp"
#include "tla_a5/tensor.hpp"

namespace Catlass::Gemm::Kernel {

#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)

// Template for GroupedMxMatmulSliceM kernel
template <
    class BlockMmad_,
    class BlockEpilogue_,
    class BlockScheduler_,
    class ElementGroupList_
>
class GroupedMxMatmulSliceMTla {
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutTagA = typename BlockMmad::TileCopy::LayoutTagA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutB = typename BlockMmad::LayoutB;
    using LayoutTagB = typename BlockMmad::TileCopy::LayoutTagB;
    using ElementMxScaleA = typename BlockMmad::TileCopy::ElementMxScaleA;
    using LayoutMxScaleA = typename BlockMmad::TileCopy::LayoutMxScaleA;
    using ElementMxScaleB = typename BlockMmad::TileCopy::ElementMxScaleB;
    using LayoutMxScaleB = typename BlockMmad::TileCopy::LayoutMxScaleB;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;
    using ElementAccumulator = typename BlockMmad::ElementAccumulator;

    using ElementGroupList = ElementGroupList_;
    using BlockScheduler = BlockScheduler_;
    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});

    // Check given epilogue should be void
    static_assert(std::is_void_v<BlockEpilogue_>, "Current kernel: GroupedMxMatmulSliceMTla does not support epilogue.");

    /// Parameters structure
    struct Params {
        // Data members
        GemmCoord problemShape;
        uint32_t problemCount;
        __gm__ ElementGroupList *ptrGroupList; // int64_t
        __gm__ ElementA *ptrA; // fp8_e4m3 or fp8_e5m2 or fp4_e2m1
        LayoutA layoutA; // {m, k}
        __gm__ ElementB *ptrB; // fp8_e4m3 or fp8_e5m2 or fp4_e2m1
        LayoutB layoutB; // {k, n}
        __gm__ ElementMxScaleA *ptrMxScaleA; // fp8_e8m0
        LayoutMxScaleA layoutMxScaleA; // {m // groups, k // 32}
        __gm__ ElementMxScaleB *ptrMxScaleB; // fp8_e8m0
        LayoutMxScaleB layoutMxScaleB; // {groups, k // 32, n}
        __gm__ ElementC *ptrC;
        LayoutC layoutC;

        // Methods
        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(
            GemmCoord const &problemShape_, uint32_t problemCount_, GM_ADDR ptrGroupList_,
            GM_ADDR ptrA_, LayoutA const &layoutA_,
            GM_ADDR ptrB_, LayoutB const &layoutB_,
            GM_ADDR ptrMxScaleA_, LayoutMxScaleA layoutMxScaleA_,
            GM_ADDR ptrMxScaleB_, LayoutMxScaleB layoutMxScaleB_,
            GM_ADDR ptrC_, LayoutC const &layoutC_
        ) : problemShape(problemShape_),
            problemCount(problemCount_), ptrGroupList(reinterpret_cast<__gm__ ElementGroupList *>(ptrGroupList_)),
            ptrA(reinterpret_cast<__gm__ ElementA *>(ptrA_)), layoutA(layoutA_),
            ptrB(reinterpret_cast<__gm__ ElementB *>(ptrB_)), layoutB(layoutB_),
            ptrMxScaleA(reinterpret_cast<__gm__ ElementMxScaleA *>(ptrMxScaleA_)), layoutMxScaleA(layoutMxScaleA_),
            ptrMxScaleB(reinterpret_cast<__gm__ ElementMxScaleB *>(ptrMxScaleB_)), layoutMxScaleB(layoutMxScaleB_),
            ptrC(reinterpret_cast<__gm__ ElementC *>(ptrC_)), layoutC(layoutC_)
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
        uint8_t *ptrMxScaleA;
        LayoutMxScaleA layoutMxScaleA;
        uint8_t *ptrMxScaleB;
        LayoutMxScaleB layoutMxScaleB;
        uint8_t *ptrC;
        LayoutC layoutC;
    };

    static bool CanImplement(const Arguments &args)
    {
        return AscendC::Std::is_one_of_v<ElementA, float8_e4m3_t, float8_e5m2_t, float4_e2m1x2_t, float4_e1m2x2_t> &&
               AscendC::Std::is_one_of_v<ElementB, float8_e4m3_t, float8_e5m2_t, float4_e2m1x2_t, float4_e1m2x2_t> &&
               std::is_same_v<ElementMxScaleA, float8_e8m0_t> &&
               std::is_same_v<ElementMxScaleB, float8_e8m0_t> &&
               std::is_same_v<LayoutTagA, layout::RowMajor>;
    }
    static size_t GetWorkspaceSize(const Arguments &args)
    {
        return 0;
    }
    static Params ToUnderlyingArguments(const Arguments &args, [[maybe_unused]] uint8_t* workspace)
    {
        Params params{args.problemShape, args.problemCount, args.ptrGroupList,
            args.ptrA, args.layoutA,
            args.ptrB, args.layoutB,
            args.ptrMxScaleA, args.layoutMxScaleA,
            args.ptrMxScaleB, args.layoutMxScaleB,
            args.ptrC, args.layoutC};
        return params;
    }
    // Methods
    CATLASS_DEVICE
    GroupedMxMatmulSliceMTla() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params const &params);

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params const &params)
    {
        AscendC::ICachePreLoad(1);
        BlockMmad blockMmad(resource);

        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();

        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);

        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrC);

        AscendC::GlobalTensor<ElementGroupList> groupList;
        groupList.SetGlobalBuffer(params.ptrGroupList);

        int64_t gmGroupOffsetB = 0;
        int64_t gmGroupOffsetMxScaleA = 0;
        int64_t gmGroupOffsetMxScaleB = 0;
        int64_t mxScaleAlignedK = static_cast<int64_t>(CeilDiv<MX_BASEK_FACTOR>(params.problemShape.k()) * MX_SCALE_COPY_GROUP_NUM);

        int64_t totalM = 0;
        uint32_t startCoreIdx = 0;

        auto tensorA = tla::MakeTensor(gmA, params.layoutA, Arch::PositionGM{});
        auto tensorC = tla::MakeTensor(gmC, params.layoutC, Arch::PositionGM{});

        for (uint32_t groupIdx = 0; groupIdx < params.problemCount; ++groupIdx) {
            uint32_t currentM = groupList.GetValue(groupIdx);
            GemmCoord inGroupProblemShape{currentM, params.problemShape.n(), params.problemShape.k()};

            auto layoutMxScaleA = tla::MakeMxScaleLayout<ElementMxScaleA, layout::RowMajor, false>(
                inGroupProblemShape.m(), CeilDiv<MX_SCALE_GROUP_NUM>(inGroupProblemShape.k()));

            BlockScheduler matmulBlockScheduler(inGroupProblemShape, MakeCoord(L1_TILE_M, L1_TILE_N));
            uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

            AscendC::GlobalTensor<ElementB> gmB;
            gmB.SetGlobalBuffer(params.ptrB + gmGroupOffsetB);
            AscendC::GlobalTensor<ElementMxScaleB> gmMxScaleB;
            gmMxScaleB.SetGlobalBuffer(params.ptrMxScaleB + gmGroupOffsetMxScaleB);

            AscendC::GlobalTensor<ElementMxScaleA> gmMxScaleA;
            gmMxScaleA.SetGlobalBuffer(params.ptrMxScaleA + gmGroupOffsetMxScaleA);

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
            auto tensorMxScaleA = tla::MakeTensor(gmMxScaleA, layoutMxScaleA, Arch::PositionGM{});
            auto tensorMxScaleB = tla::MakeTensor(gmMxScaleB, params.layoutMxScaleB, Arch::PositionGM{});

            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += coreNum) {
                GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
                GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);

                auto tensorBlockA = GetTile(tensorA,
                    tla::MakeCoord(totalM + blockCoord.m() * L1_TILE_M, blockCoord.k() * L1_TILE_K),
                    tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));

                auto tensorBlockB = GetTile(tensorB,
                    tla::MakeCoord(blockCoord.k() * L1_TILE_K, blockCoord.n() * L1_TILE_N),
                    tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));

                auto tensorBlockC = GetTile(tensorC,
                    tla::MakeCoord(totalM + blockCoord.m() * L1_TILE_M, blockCoord.n() * L1_TILE_N),
                    tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));

                auto tensorBlockMxScaleA = GetTile(
                    tensorMxScaleA,
                    tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.k() * L1_TILE_K / MX_SCALE_GROUP_NUM),
                    tla::MakeShape(actualBlockShape.m(), CeilDiv<MX_SCALE_GROUP_NUM>(actualBlockShape.k())));

                auto tensorBlockMxScaleB = GetTile(
                    tensorMxScaleB,
                    tla::MakeCoord(blockCoord.k() * L1_TILE_K / MX_SCALE_GROUP_NUM, blockCoord.n() * L1_TILE_N),
                    tla::MakeShape(CeilDiv<MX_SCALE_GROUP_NUM>(actualBlockShape.k()), actualBlockShape.n()));

                blockMmad(tensorBlockA, tensorBlockB, tensorBlockC, actualBlockShape,
                    tensorBlockMxScaleA, tensorBlockMxScaleB);
            }

            totalM += inGroupProblemShape.m();
            if constexpr (AscendC::Std::is_one_of_v<ElementB, float4_e2m1x2_t, float4_e1m2x2_t>) {
                gmGroupOffsetB += std::is_same_v<LayoutTagB, layout::ColumnMajor> ?
                    static_cast<int64_t>(CeilDiv<2>(inGroupProblemShape.k())) * inGroupProblemShape.n() :
                    static_cast<int64_t>(CeilDiv<2>(inGroupProblemShape.n())) * inGroupProblemShape.k();
            }else{
                gmGroupOffsetB += static_cast<int64_t>(inGroupProblemShape.k()) * inGroupProblemShape.n();
            }
            gmGroupOffsetMxScaleA += inGroupProblemShape.m() * mxScaleAlignedK;
            gmGroupOffsetMxScaleB += mxScaleAlignedK * inGroupProblemShape.n();

            startCoreIdx = (startCoreIdx + coreLoops) % coreNum;
        }

        if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
            blockMmad.template SynchronizeBlock<decltype(tensorC)>();
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const &params) {}

private:
    Arch::Resource<ArchTag> resource;
};

#endif // (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)

} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_GROUPED_MX_MATMUL_SLICE_M_TLA_HPP
