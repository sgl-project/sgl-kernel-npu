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

#ifndef CATLASS_GEMM_KERNEL_GROUPED_MX_MATMUL_SLICE_M_ASWT_TLA_HPP
#define CATLASS_GEMM_KERNEL_GROUPED_MX_MATMUL_SLICE_M_ASWT_TLA_HPP

#include "catlass_a5/catlass.hpp"
#include "catlass_a5/arch/resource.hpp"
#include "catlass_a5/coord.hpp"
#include "catlass_a5/detail/alignment.hpp"
#include "catlass_a5/gemm_coord.hpp"
#include "catlass_a5/matrix_coord.hpp"
#include "tla_a5/layout.hpp"
#include "tla_a5/tensor.hpp"

namespace Catlass::Gemm::Kernel {

#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)

template <class BlockMmad_, class BlockEpilogue_, class BlockScheduler_, class ElementGroupList_,
          bool EnableBaseMBalance_ = false>
class GroupedMxMatmulSliceMAswtTla
{
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
    static constexpr bool ENABLE_BASE_M_BALANCE = EnableBaseMBalance_;
    static constexpr uint32_t BASE_M_ALIGN = 16;  // cube alignment for base-M rebalance

    // Check given epilogue should be void
    static_assert(std::is_void_v<BlockEpilogue_>,
                  "Current kernel: GroupedMxMatmulSliceMAswtTla does not support epilogue.");

    /// Parameters structure
    struct Params {
        // Data members
        GemmCoord problemShape;
        uint32_t problemCount;
        __gm__ ElementGroupList *ptrGroupList;  // int64_t
        __gm__ ElementA *ptrA;                  // fp8_e4m3 or fp8_e5m2 or fp4_e2m1
        LayoutA layoutA;                        // {m, k}
        __gm__ ElementB *ptrB;                  // fp8_e4m3 or fp8_e5m2 or fp4_e2m1
        LayoutB layoutB;                        // {k, n}
        __gm__ ElementMxScaleA *ptrMxScaleA;    // fp8_e8m0
        LayoutMxScaleA layoutMxScaleA;          // {m // groups, k // 32}
        __gm__ ElementMxScaleB *ptrMxScaleB;    // fp8_e8m0
        LayoutMxScaleB layoutMxScaleB;          // {groups, k // 32, n}
        __gm__ ElementC *ptrC;
        LayoutC layoutC;

        // Methods
        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(GemmCoord const &problemShape_, uint32_t problemCount_, GM_ADDR ptrGroupList_, GM_ADDR ptrA_,
               LayoutA const &layoutA_, GM_ADDR ptrB_, LayoutB const &layoutB_, GM_ADDR ptrMxScaleA_,
               LayoutMxScaleA layoutMxScaleA_, GM_ADDR ptrMxScaleB_, LayoutMxScaleB layoutMxScaleB_, GM_ADDR ptrC_,
               LayoutC const &layoutC_)
            : problemShape(problemShape_),
              problemCount(problemCount_),
              ptrGroupList(reinterpret_cast<__gm__ ElementGroupList *>(ptrGroupList_)),
              ptrA(reinterpret_cast<__gm__ ElementA *>(ptrA_)),
              layoutA(layoutA_),
              ptrB(reinterpret_cast<__gm__ ElementB *>(ptrB_)),
              layoutB(layoutB_),
              ptrMxScaleA(reinterpret_cast<__gm__ ElementMxScaleA *>(ptrMxScaleA_)),
              layoutMxScaleA(layoutMxScaleA_),
              ptrMxScaleB(reinterpret_cast<__gm__ ElementMxScaleB *>(ptrMxScaleB_)),
              layoutMxScaleB(layoutMxScaleB_),
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
               std::is_same_v<ElementMxScaleA, float8_e8m0_t> && std::is_same_v<ElementMxScaleB, float8_e8m0_t> &&
               std::is_same_v<LayoutTagA, layout::RowMajor>;
    }
    static size_t GetWorkspaceSize(const Arguments &args)
    {
        return 0;
    }
    static Params ToUnderlyingArguments(const Arguments &args, [[maybe_unused]] uint8_t *workspace)
    {
        Params params{args.problemShape,   args.problemCount, args.ptrGroupList, args.ptrA,           args.layoutA,
                      args.ptrB,           args.layoutB,      args.ptrMxScaleA,  args.layoutMxScaleA, args.ptrMxScaleB,
                      args.layoutMxScaleB, args.ptrC,         args.layoutC};
        return params;
    }
    // Methods
    CATLASS_DEVICE
    GroupedMxMatmulSliceMAswtTla() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const &params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const &params)
    {
        AscendC::ICachePreLoad(1);
        BlockMmad blockMmad(resource);

        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);

        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrC);

        AscendC::GlobalTensor<ElementGroupList> groupList;
        groupList.SetGlobalBuffer(params.ptrGroupList);

        int64_t gmGroupOffsetB = 0;
        int64_t gmGroupOffsetMxScaleA = 0;
        int64_t gmGroupOffsetMxScaleB = 0;
        int64_t mxScaleAlignedK =
            static_cast<int64_t>(CeilDiv<MX_BASEK_FACTOR>(params.problemShape.k()) * MX_SCALE_COPY_GROUP_NUM);

        int64_t totalM = 0;

        auto tensorA = tla::MakeTensor(gmA, params.layoutA, Arch::PositionGM{});
        auto tensorC = tla::MakeTensor(gmC, params.layoutC, Arch::PositionGM{});

        // ASWT scheduler: constructed once, carries the rolling per-core assignment across groups
        // (no barrier between groups) and the last-group tail-split state.
        BlockScheduler scheduler(L1_TILE_M, L1_TILE_N);

        uint32_t kScaleActual = CeilDiv<MX_SCALE_GROUP_NUM>(params.problemShape.k());

        for (uint32_t groupIdx = 0; groupIdx < params.problemCount; ++groupIdx) {
            uint32_t currentM = static_cast<uint32_t>(groupList.GetValue(groupIdx));

            // Empty group still consumes a per-group B / scaleB slot; skip its tiles (the scheduler
            // is not defined for m==0) and keep the cross-group rolling untouched.
            if (currentM == 0) {
                if constexpr (AscendC::Std::is_one_of_v<ElementB, float4_e2m1x2_t, float4_e1m2x2_t>) {
                    gmGroupOffsetB +=
                        std::is_same_v<LayoutTagB, layout::ColumnMajor>
                            ? static_cast<int64_t>(CeilDiv<2>(params.problemShape.k())) * params.problemShape.n()
                            : static_cast<int64_t>(CeilDiv<2>(params.problemShape.n())) * params.problemShape.k();
                } else {
                    gmGroupOffsetB += static_cast<int64_t>(params.problemShape.k()) * params.problemShape.n();
                }
                gmGroupOffsetMxScaleB += mxScaleAlignedK * params.problemShape.n();
                continue;
            }

            GemmCoord inGroupProblemShape{currentM, params.problemShape.n(), params.problemShape.k()};

            // Base tile M for this group. With rebalance off, curBaseM == L1_TILE_M (plain ASWT).
            uint32_t curBaseM = L1_TILE_M;
            if constexpr (ENABLE_BASE_M_BALANCE) {
                if (currentM > 0) {
                    uint32_t mCntOrig = CeilDiv(currentM, L1_TILE_M);
                    curBaseM = RoundUp<BASE_M_ALIGN>(CeilDiv(currentM, mCntOrig));
                    if (curBaseM > L1_TILE_M) {
                        curBaseM = L1_TILE_M;
                    }
                }
            }
            scheduler.UpdateBaseM(curBaseM);
            scheduler.UpdateNextProblem(inGroupProblemShape);

            // Last group: split the tail tiles to fill the otherwise idle cores of the final wave.
            bool isLastGroup = (groupIdx + 1 == params.problemCount);
            bool doTailSplit = isLastGroup && scheduler.NeedTailSplit();
            if (doTailSplit) {
                scheduler.UpdateTailTile();
            }

            AscendC::GlobalTensor<ElementB> gmB;
            gmB.SetGlobalBuffer(params.ptrB + gmGroupOffsetB);
            AscendC::GlobalTensor<ElementMxScaleB> gmMxScaleB;
            gmMxScaleB.SetGlobalBuffer(params.ptrMxScaleB + gmGroupOffsetMxScaleB);
            AscendC::GlobalTensor<ElementMxScaleA> gmMxScaleA;
            gmMxScaleA.SetGlobalBuffer(params.ptrMxScaleA + gmGroupOffsetMxScaleA);

            // Keep the baseline L2-cache heuristic: a single M-block group => B is not worth caching.
            // Skip it when splitting the last group's tail (those tiles reuse B across cores).
            if (!doTailSplit && CeilDiv(currentM, curBaseM) == 1) {
                gmB.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            }

            auto layoutMxScaleA = tla::MakeMxScaleLayout<ElementMxScaleA, layout::RowMajor, false>(
                inGroupProblemShape.m(), CeilDiv<MX_SCALE_GROUP_NUM>(inGroupProblemShape.k()));

            auto tensorB = tla::MakeTensor(gmB, params.layoutB, Arch::PositionGM{});
            auto tensorMxScaleA = tla::MakeTensor(gmMxScaleA, layoutMxScaleA, Arch::PositionGM{});
            auto tensorMxScaleB = tla::MakeTensor(gmMxScaleB, params.layoutMxScaleB, Arch::PositionGM{});

            uint32_t kActual = inGroupProblemShape.k();

            GemmCoord blockCoord;
            while (scheduler.GetTileIdx(blockCoord)) {
                auto shape = scheduler.GetBlockShape(blockCoord);
                if (shape.m == 0 || shape.n == 0) {
                    continue;
                }
                uint32_t mInGroup = blockCoord.m() * curBaseM + shape.mOffset;
                uint32_t nOffset = blockCoord.n() * L1_TILE_N + shape.nOffset;
                int64_t mGlobal = totalM + static_cast<int64_t>(mInGroup);
                GemmCoord actualBlockShape{shape.m, shape.n, kActual};

                auto tensorBlockA = GetTile(tensorA, tla::MakeCoord(mGlobal, static_cast<uint32_t>(0)),
                                            tla::MakeShape(shape.m, kActual));

                auto tensorBlockB = GetTile(tensorB, tla::MakeCoord(static_cast<uint32_t>(0), nOffset),
                                            tla::MakeShape(kActual, shape.n));

                auto tensorBlockC =
                    GetTile(tensorC, tla::MakeCoord(mGlobal, nOffset), tla::MakeShape(shape.m, shape.n));

                auto tensorBlockMxScaleA = GetTile(tensorMxScaleA, tla::MakeCoord(mInGroup, static_cast<uint32_t>(0)),
                                                   tla::MakeShape(shape.m, kScaleActual));

                auto tensorBlockMxScaleB = GetTile(tensorMxScaleB, tla::MakeCoord(static_cast<uint32_t>(0), nOffset),
                                                   tla::MakeShape(kScaleActual, shape.n));

                blockMmad(tensorBlockA, tensorBlockB, tensorBlockC, actualBlockShape, tensorBlockMxScaleA,
                          tensorBlockMxScaleB);
            }

            totalM += inGroupProblemShape.m();
            if constexpr (AscendC::Std::is_one_of_v<ElementB, float4_e2m1x2_t, float4_e1m2x2_t>) {
                gmGroupOffsetB +=
                    std::is_same_v<LayoutTagB, layout::ColumnMajor>
                        ? static_cast<int64_t>(CeilDiv<2>(inGroupProblemShape.k())) * inGroupProblemShape.n()
                        : static_cast<int64_t>(CeilDiv<2>(inGroupProblemShape.n())) * inGroupProblemShape.k();
            } else {
                gmGroupOffsetB += static_cast<int64_t>(inGroupProblemShape.k()) * inGroupProblemShape.n();
            }
            gmGroupOffsetMxScaleA += inGroupProblemShape.m() * mxScaleAlignedK;
            gmGroupOffsetMxScaleB += mxScaleAlignedK * inGroupProblemShape.n();
        }

        if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
            blockMmad.template SynchronizeBlock<decltype(tensorC)>();
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const &params)
    {}

private:
    Arch::Resource<ArchTag> resource;
};

#endif  // (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)

}  // namespace Catlass::Gemm::Kernel

#endif  // CATLASS_GEMM_KERNEL_GROUPED_MX_MATMUL_SLICE_M_ASWT_TLA_HPP
