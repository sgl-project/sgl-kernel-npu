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

#ifndef CATLASS_GEMM_KERNEL_GROUPED_MX_MATMUL_SWIGLU_QUANT_SLICE_M_TLA_HPP
#define CATLASS_GEMM_KERNEL_GROUPED_MX_MATMUL_SWIGLU_QUANT_SLICE_M_TLA_HPP

#include "catlass_a5/catlass.hpp"
#include "catlass_a5/arch/resource.hpp"
#include "catlass_a5/coord.hpp"
#include "catlass_a5/gemm_coord.hpp"
#include "catlass_a5/matrix_coord.hpp"
#include "catlass_a5/epilogue/block/block_epilogue.hpp"
#include "tla_a5/layout.hpp"
#include "tla_a5/tensor.hpp"

namespace Catlass::Gemm::Kernel {

#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)

template <class BlockMmad_, class BlockEpilogue_, class BlockScheduler_, class ElementGroupList_, class LayoutQ_,
          class LayoutQScale_>
class GroupedMxMatmulSliceMSwigluMxQuantTla
{
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
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
    using LayoutTagC = typename BlockMmad::TileCopy::LayoutTagC;
    using ElementAccumulator = typename BlockMmad::ElementAccumulator;

    using BlockEpilogue = BlockEpilogue_;
    using ElementQ = typename BlockEpilogue::QuantOutType;
    using ElementQScale = typename BlockEpilogue::QuantScaleType;
    using LayoutQ = LayoutQ_;
    using LayoutQScale = LayoutQScale_;
    using ElementGroupList = ElementGroupList_;
    using BlockScheduler = BlockScheduler_;
    using BlockEpilogueParams = typename BlockEpilogue::Params;

    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});
    static constexpr uint32_t N_HALF = 2;

    static constexpr uint16_t AIV_SYNC_AIC_FLAG = 6;
    static constexpr uint16_t AIC_SYNC_AIV_FLAG = 8;
    static constexpr uint8_t AIC_SYNC_AIV_MODE = 2;
    static constexpr uint16_t FLAG_ID_MAX = 16;
    static constexpr uint32_t UB_TWO_BANK_ELEMS_B32 = 128U;
    static constexpr uint32_t AAA = 64U;
    static constexpr int64_t PER_BLOCK_SIZE = 128LL;
    static constexpr uint32_t UB_SUB_BANK_NUM = 2U;
    static constexpr int8_t FLOAT_OVERFLOW_MODE_CTRL = 60;

    struct Params {
        GemmCoord problemShape;
        uint32_t problemCount;
        __gm__ ElementGroupList *ptrGroupList;
        __gm__ ElementA *ptrA;
        LayoutA layoutA;
        __gm__ ElementB *ptrB;
        LayoutB layoutB;
        __gm__ ElementMxScaleA *ptrMxScaleA;
        LayoutMxScaleA layoutMxScaleA;
        __gm__ ElementMxScaleB *ptrMxScaleB;
        LayoutMxScaleB layoutMxScaleB;
        __gm__ ElementQ *ptrQ;
        LayoutQ layoutQ;
        __gm__ ElementQScale *ptrQScale;
        LayoutQScale layoutQScale;
        BlockEpilogueParams epilogueParams;

        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(GemmCoord const &problemShape_, uint32_t problemCount_, GM_ADDR ptrGroupList_, GM_ADDR ptrA_,
               LayoutA const &layoutA_, GM_ADDR ptrB_, LayoutB const &layoutB_, GM_ADDR ptrMxScaleA_,
               LayoutMxScaleA layoutMxScaleA_, GM_ADDR ptrMxScaleB_, LayoutMxScaleB layoutMxScaleB_, GM_ADDR ptrQ_,
               LayoutQ const &layoutQ_, GM_ADDR ptrQScale_, LayoutQScale const &layoutQScale_,
               BlockEpilogueParams epilogueParams_)
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
              ptrQ(reinterpret_cast<__gm__ ElementQ *>(ptrQ_)),
              layoutQ(layoutQ_),
              ptrQScale(reinterpret_cast<__gm__ ElementQScale *>(ptrQScale_)),
              layoutQScale(layoutQScale_),
              epilogueParams(epilogueParams_)
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
        uint8_t *ptrQ;
        LayoutQ layoutQ;
        uint8_t *ptrQScale;
        LayoutQScale layoutQScale;
        BlockEpilogueParams epilogueParams;
    };

    static bool CanImplement(const Arguments &args)
    {
        return AscendC::Std::is_one_of_v<ElementA, float8_e4m3_t, float8_e5m2_t> &&
               AscendC::Std::is_one_of_v<ElementB, float8_e4m3_t, float8_e5m2_t> &&
               std::is_same_v<ElementMxScaleA, float8_e8m0_t> && std::is_same_v<ElementMxScaleB, float8_e8m0_t> &&
               AscendC::Std::is_one_of_v<ElementQ, float8_e4m3_t, float8_e5m2_t> &&
               std::is_same_v<ElementQScale, float8_e8m0_t> && args.problemCount <= 1024 &&
               args.problemShape.n() % 128 == 0;
    }

    static size_t GetWorkspaceSize(const Arguments &args)
    {
        return 0;
    }

    static Params ToUnderlyingArguments(const Arguments &args, [[maybe_unused]] uint8_t *workspace)
    {
        Params params{args.problemShape,   args.problemCount, args.ptrGroupList,   args.ptrA,
                      args.layoutA,        args.ptrB,         args.layoutB,        args.ptrMxScaleA,
                      args.layoutMxScaleA, args.ptrMxScaleB,  args.layoutMxScaleB, args.ptrQ,
                      args.layoutQ,        args.ptrQScale,    args.layoutQScale,   args.epilogueParams};
        return params;
    }

    CATLASS_DEVICE
    GroupedMxMatmulSliceMSwigluMxQuantTla()
    {
        oriOverflowMode = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
        // enable overflow mode to avoid nan/inf value
        AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
    }

    CATLASS_DEVICE ~GroupedMxMatmulSliceMSwigluMxQuantTla()
    {
        AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(oriOverflowMode);
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const &params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const &params)
    {
        AscendC::ICachePreLoad(1);
        BlockMmad blockMmad(resource);

        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();

        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);
        AscendC::GlobalTensor<ElementQ> gmQ;
        gmQ.SetGlobalBuffer((__gm__ ElementQ *)params.ptrQ);
        AscendC::GlobalTensor<ElementGroupList> groupList;
        groupList.SetGlobalBuffer(params.ptrGroupList);

        constexpr uint32_t elems = UB_TWO_BANK_ELEMS_B32 * PER_BLOCK_SIZE;
        mmResUb_ping_ = AscendC::LocalTensor<ElementC>(AscendC::TPosition::VECCALC, 0, elems);
        mmResUb_pong_ = AscendC::LocalTensor<ElementC>(AscendC::TPosition::VECCALC, elems * sizeof(ElementC), elems);

        int64_t gmGroupOffsetB = 0;
        int64_t gmGroupOffsetMxScaleA = 0;
        int64_t gmGroupOffsetMxScaleB = 0;
        int64_t mxScaleAlignedK =
            static_cast<int64_t>(CeilDiv<MX_BASEK_FACTOR>(params.problemShape.k()) * MX_SCALE_COPY_GROUP_NUM);

        int64_t totalM = 0;
        uint32_t startCoreIdx = 0;

        auto tensorA = tla::MakeTensor(gmA, params.layoutA, Arch::PositionGM{});

        for (uint32_t groupIdx = 0; groupIdx < params.problemCount; ++groupIdx) {
            uint32_t currentM = groupList.GetValue(groupIdx);
            if (currentM == 0) continue;

            uint32_t N = params.problemShape.n();
            uint32_t N_half = N / N_HALF;

            GemmCoord inGroupProblemShape{currentM, N_half, params.problemShape.k()};

            auto layoutMxScaleA = tla::MakeMxScaleLayout<ElementMxScaleA, layout::RowMajor, false>(
                currentM, CeilDiv<MX_SCALE_GROUP_NUM>(inGroupProblemShape.k()));

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
                uint32_t actualN = actualBlockShape.n();
                uint32_t actualN_half = actualN;

                auto tensorBlockA =
                    GetTile(tensorA, tla::MakeCoord(totalM + blockCoord.m() * L1_TILE_M, blockCoord.k() * L1_TILE_K),
                            tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));

                auto tensorBlockB_act =
                    GetTile(tensorB, tla::MakeCoord(blockCoord.k() * L1_TILE_K, blockCoord.n() * L1_TILE_N),
                            tla::MakeShape(actualBlockShape.k(), actualN_half));

                auto tensorBlockMxScaleA_act =
                    GetTile(tensorMxScaleA,
                            tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.k() * L1_TILE_K / MX_SCALE_GROUP_NUM),
                            tla::MakeShape(actualBlockShape.m(), CeilDiv<MX_SCALE_GROUP_NUM>(actualBlockShape.k())));

                auto tensorBlockMxScaleB_act =
                    GetTile(tensorMxScaleB,
                            tla::MakeCoord(blockCoord.k() * L1_TILE_K / MX_SCALE_GROUP_NUM, blockCoord.n() * L1_TILE_N),
                            tla::MakeShape(CeilDiv<MX_SCALE_GROUP_NUM>(actualBlockShape.k()), actualN_half));

                auto ubLayoutAct = tla::MakeLayout(tla::MakeShape(actualBlockShape.m(), actualN_half),
                                                   tla::MakeStride(static_cast<int64_t>(actualN_half), tla::Int<1>{}));
                auto tensorBlockC_act = tla::MakeTensor(mmResUb_ping_, ubLayoutAct, Arch::PositionUB{});
                if (isVecSetSyncCom_) {
                    AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE, PIPE_FIX>(AIV_SYNC_AIC_FLAG);
                }
                GemmCoord actBlockShape{actualBlockShape.m(), actualN_half, actualBlockShape.k()};
                blockMmad(tensorBlockA, tensorBlockB_act, tensorBlockC_act, actBlockShape, tensorBlockMxScaleA_act,
                          tensorBlockMxScaleB_act);

                auto tensorBlockB_gate =
                    GetTile(tensorB, tla::MakeCoord(blockCoord.k() * L1_TILE_K, N_half + blockCoord.n() * L1_TILE_N),
                            tla::MakeShape(actualBlockShape.k(), actualN_half));

                auto tensorBlockMxScaleB_gate =
                    GetTile(tensorMxScaleB,
                            tla::MakeCoord(blockCoord.k() * L1_TILE_K / MX_SCALE_GROUP_NUM,
                                           N_half + blockCoord.n() * L1_TILE_N),
                            tla::MakeShape(CeilDiv<MX_SCALE_GROUP_NUM>(actualBlockShape.k()), actualN_half));

                auto ubLayoutGate = tla::MakeLayout(tla::MakeShape(actualBlockShape.m(), actualN_half),
                                                    tla::MakeStride(static_cast<int64_t>(actualN_half), tla::Int<1>{}));
                auto tensorBlockC_gate = tla::MakeTensor(mmResUb_pong_, ubLayoutGate, Arch::PositionUB{});

                GemmCoord gateBlockShape{actualBlockShape.m(), actualN_half, actualBlockShape.k()};
                blockMmad(tensorBlockA, tensorBlockB_gate, tensorBlockC_gate, gateBlockShape, tensorBlockMxScaleA_act,
                          tensorBlockMxScaleB_gate);

                AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE, PIPE_FIX>(AIC_SYNC_AIV_FLAG);
                isVecSetSyncCom_ = true;
            }

            totalM += currentM;
            gmGroupOffsetB += inGroupProblemShape.k() * params.problemShape.n();
            gmGroupOffsetMxScaleA += currentM * mxScaleAlignedK;
            gmGroupOffsetMxScaleB += mxScaleAlignedK * params.problemShape.n();
            startCoreIdx = (startCoreIdx + coreLoops) % coreNum;
        }
        if (isVecSetSyncCom_) {
            AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE, PIPE_FIX>(AIV_SYNC_AIC_FLAG);
        }
        if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
            blockMmad.template SynchronizeBlock<decltype(mmResUb_ping_)>();
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const &params)
    {
        BlockEpilogue blockEpilogue;
        blockEpilogue.Init(&params.epilogueParams);

        uint32_t aicoreIndex = AscendC::GetBlockIdx() / AscendC::GetTaskRation();
        uint32_t aicoreNum = AscendC::GetBlockNum();

        AscendC::GlobalTensor<ElementQ> gmQ;
        gmQ.SetGlobalBuffer((__gm__ ElementQ *)params.ptrQ);
        AscendC::GlobalTensor<ElementQScale> gmQScale;
        gmQScale.SetGlobalBuffer((__gm__ ElementQScale *)params.ptrQScale);
        AscendC::GlobalTensor<ElementGroupList> groupList;
        groupList.SetGlobalBuffer(params.ptrGroupList);

        blockEpilogue.UpdateGlobalAddr(gmQ, gmQScale);

        constexpr uint32_t elems = UB_TWO_BANK_ELEMS_B32 * PER_BLOCK_SIZE;
        mmResUb_ping_ = AscendC::LocalTensor<ElementC>(AscendC::TPosition::VECCALC, 0, elems);
        mmResUb_pong_ = AscendC::LocalTensor<ElementC>(AscendC::TPosition::VECCALC, elems * sizeof(ElementC), elems);

        int64_t gmGroupOffsetMxScaleA = 0;
        int64_t mxScaleAlignedK =
            static_cast<int64_t>(CeilDiv<MX_BASEK_FACTOR>(params.problemShape.k()) * MX_SCALE_COPY_GROUP_NUM);

        int64_t totalM = 0;
        uint32_t startCoreIdx = 0;

        uint32_t N = params.problemShape.n();
        uint32_t N_half = N / N_HALF;

        for (uint32_t groupIdx = 0; groupIdx < params.problemCount; ++groupIdx) {
            uint32_t currentM = groupList.GetValue(groupIdx);
            if (currentM == 0) continue;

            GemmCoord inGroupProblemShape{currentM, N_half, params.problemShape.k()};

            auto layoutMxScaleA = tla::MakeMxScaleLayout<ElementMxScaleA, layout::RowMajor, false>(
                currentM, CeilDiv<MX_SCALE_GROUP_NUM>(inGroupProblemShape.k()));

            BlockScheduler matmulBlockScheduler(inGroupProblemShape, MakeCoord(L1_TILE_M, L1_TILE_N));
            uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

            uint32_t startLoopIdx;
            if (aicoreIndex < startCoreIdx) {
                startLoopIdx = aicoreIndex + aicoreNum - startCoreIdx;
            } else {
                startLoopIdx = aicoreIndex - startCoreIdx;
            }

            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += aicoreNum) {
                GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
                GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);
                uint32_t actualN_half = actualBlockShape.n();

                AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE, PIPE_V>(AIC_SYNC_AIV_FLAG);

                GemmCoord resShape{actualBlockShape.m(), actualN_half, actualBlockShape.k()};
                blockEpilogue(resShape, totalM, blockCoord, mmResUb_ping_, mmResUb_pong_, L1_TILE_M, L1_TILE_N);

                AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE, PIPE_V>(AIV_SYNC_AIC_FLAG);
            }

            totalM += currentM;
            startCoreIdx = (startCoreIdx + coreLoops) % aicoreNum;
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    Arch::Resource<ArchTag> resource;
    AscendC::LocalTensor<ElementC> mmResUb_ping_;
    AscendC::LocalTensor<ElementC> mmResUb_pong_;
    bool isVecSetSyncCom_ = false;
    int64_t oriOverflowMode = 0;
};

#endif  // (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)

}  // namespace Catlass::Gemm::Kernel

#endif  // CATLASS_GEMM_KERNEL_GROUPED_MX_MATMUL_SWIGLU_QUANT_SLICE_M_TLA_HPP
