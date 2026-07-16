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

#ifndef CATLASS_GEMM_KERNEL_MX_GMM2_CAST_COMBINE_H
#define CATLASS_GEMM_KERNEL_MX_GMM2_CAST_COMBINE_H

#include "ascendc/basic_api/interface/kernel_operator_list_tensor_intf.h"
#include "../../raw_distributed/cam_moe_distribute_combine.h"
#include "../../fused_deep_moe_utils.h"
#include "catlass_a5/catlass.hpp"
#include "catlass_a5/arch/resource.hpp"
#include "catlass_a5/coord.hpp"
#include "catlass_a5/detail/callback.hpp"
#include "catlass_a5/gemm_coord.hpp"
#include "catlass_a5/matrix_coord.hpp"
#include "tla_a5/layout.hpp"
#include "tla_a5/tensor.hpp"

using namespace Cam;

namespace Catlass::Gemm::Kernel {
namespace GMM2 {
    constexpr uint64_t SOFT_SYNC_OFFSET = 982 * 1024;
    constexpr int64_t AIV_NUM_PER_GROUP = 2;
    constexpr int64_t CORE_NUM_PER_GROUP = 3;
    constexpr int64_t INT32_COUNT_PER_BLOCK = 32 / sizeof(int32_t);
}

#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)

// Template for GroupedMxMatmulSliceM kernel
template <
    TemplateMC2TypeClass,
    class BlockMmad_,
    class BlockEpilogue_,
    class BlockScheduler_,
    class ElementGroupList_
>
class MxGmm2CastCombine {
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
    ///////////////////////////////////////////
    using ElementD = ExpandXType;
    //////////////////////////////////////////

    using ElementAccumulator = typename BlockMmad::ElementAccumulator;

    using BlockEpilogue = BlockEpilogue_;
    using EpilogueParams = typename BlockEpilogue::Params;

    using ElementGroupList = ElementGroupList_;
    using BlockScheduler = BlockScheduler_;
    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});

    // Check given epilogue should be void
    // static_assert(std::is_void_v<BlockEpilogue_>, "Current kernel: MxGmm2CastCombine does not support epilogue.");


    /// Parameters structure
    struct Params {
        // Data members
        GemmCoord problemShape;
        uint32_t problemCount;
        __gm__ ElementGroupList *ptrGroupList;
        __gm__ ElementA *ptrA, *ptrSharedA;
        LayoutA layoutA, layoutSharedA;
        __gm__ ElementB *ptrB, *ptrSharedB;
        LayoutB layoutB, layoutSharedB;
        __gm__ ElementMxScaleA *ptrMxScaleA, *ptrSharedMxScaleA;
        LayoutMxScaleA layoutMxScaleA, layoutSharedMxScaleA;
        __gm__ ElementMxScaleB *ptrMxScaleB, *ptrSharedMxScaleB;
        LayoutMxScaleB layoutMxScaleB, layoutSharedMxScaleB;
        __gm__ ElementC *ptrC, *ptrSharedC;
        LayoutC layoutC;
        __gm__ ElementD *ptrD, *ptrSharedD;

        GemmCoord sharedProblemShape;
        void *combiner;

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
            GM_ADDR ptrC_, LayoutC const &layoutC_,
            GM_ADDR ptrD_,
            GemmCoord const &sharedProblemShape_, 
            GM_ADDR ptrSharedA_, LayoutA const &layoutSharedA_,
            GM_ADDR ptrSharedB_, LayoutB const &layoutSharedB_,
            GM_ADDR ptrSharedMxScaleA_, LayoutMxScaleA layoutSharedMxScaleA_,
            GM_ADDR ptrSharedMxScaleB_, LayoutMxScaleB layoutSharedMxScaleB_,
            GM_ADDR ptrSharedC_,
            GM_ADDR ptrSharedD_,
            void *combiner_
        ) : problemShape(problemShape_),
            problemCount(problemCount_), ptrGroupList(reinterpret_cast<__gm__ ElementGroupList *>(ptrGroupList_)),
            ptrA(reinterpret_cast<__gm__ ElementA *>(ptrA_)), layoutA(layoutA_),
            ptrB(reinterpret_cast<__gm__ ElementB *>(ptrB_)), layoutB(layoutB_),
            ptrMxScaleA(reinterpret_cast<__gm__ ElementMxScaleA *>(ptrMxScaleA_)), layoutMxScaleA(layoutMxScaleA_),
            ptrMxScaleB(reinterpret_cast<__gm__ ElementMxScaleB *>(ptrMxScaleB_)), layoutMxScaleB(layoutMxScaleB_),
            ptrC(reinterpret_cast<__gm__ ElementC *>(ptrC_)), layoutC(layoutC_),
            ptrD(reinterpret_cast<__gm__ ElementD *>(ptrD_)),
            sharedProblemShape(sharedProblemShape_),
            ptrSharedA(reinterpret_cast<__gm__ ElementA *>(ptrSharedA_)), layoutSharedA(layoutSharedA_),
            ptrSharedB(reinterpret_cast<__gm__ ElementB *>(ptrSharedB_)), layoutSharedB(layoutSharedB_),
            ptrSharedMxScaleA(reinterpret_cast<__gm__ ElementMxScaleA *>(ptrSharedMxScaleA_)), layoutSharedMxScaleA(layoutSharedMxScaleA_),
            ptrSharedMxScaleB(reinterpret_cast<__gm__ ElementMxScaleB *>(ptrSharedMxScaleB_)), layoutSharedMxScaleB(layoutSharedMxScaleB_),
            ptrSharedC(reinterpret_cast<__gm__ ElementC *>(ptrSharedC_)),
            ptrSharedD(reinterpret_cast<__gm__ ElementD *>(ptrSharedD_)),
            combiner(combiner_)
        {}
    };

    // Methods
    CATLASS_DEVICE
    MxGmm2CastCombine() {
        winContext_ = (__gm__ Mc2Kernel::HcclOpParam *)AscendC::GetHcclContext<AscendC::HCCL_GROUP_ID_0>();
        syncGmAddr = Mc2Kernel::GetStatusDataSpaceGm(winContext_);
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params const &params);

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params const &params)
    {
        AscendC::ICachePreLoad(1);

        BlockScheduler blockScheduler;
        BlockMmad blockMmad(resource);
        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();

        AscendC::GlobalTensor<ElementA> gmA;
        AscendC::GlobalTensor<ElementMxScaleA> gmMxScaleA;
        AscendC::GlobalTensor<ElementB> gmB;
        AscendC::GlobalTensor<ElementMxScaleB> gmMxScaleB;
        AscendC::GlobalTensor<ElementC> gmC;

        uint32_t currentM = 0;
        uint32_t startCoreIdx = 0;
        aicSetFunc = {syncGmAddr + GMM2::SOFT_SYNC_OFFSET, static_cast<uint8_t>(AscendC::GetBlockIdx())};
        Callback callbackAfterFixpipe = MakeCallback(&aicSetFunc);
        {
            AscendC::GlobalTensor<ElementGroupList> groupList;
            groupList.SetGlobalBuffer(params.ptrGroupList);
            gmA.SetGlobalBuffer(params.ptrA);
            gmC.SetGlobalBuffer(params.ptrC);
            AscendC::ListTensorDesc gmBlistTensorDesc(reinterpret_cast<__gm__ void *>(params.ptrB));
            AscendC::ListTensorDesc gmBScalelistTensorDesc(reinterpret_cast<__gm__ void *>(params.ptrMxScaleB));

            int64_t gmGroupOffsetB = 0;
            int64_t gmGroupOffsetMxScaleA = 0;
            int64_t gmGroupOffsetMxScaleB = 0;
            int64_t mxScaleAlignedK = static_cast<int64_t>(CeilDiv<MX_BASEK_FACTOR>(params.problemShape.k()) * MX_SCALE_COPY_GROUP_NUM);

            int64_t totalM = 0;
            auto tensorA = tla::MakeTensor(gmA, params.layoutA, Arch::PositionGM{});
            auto tensorC = tla::MakeTensor(gmC, params.layoutC, Arch::PositionGM{});

            for (uint32_t groupIdx = 0; groupIdx < params.problemCount; ++groupIdx) {
                gmMxScaleA.SetGlobalBuffer(params.ptrMxScaleA + gmGroupOffsetMxScaleA);
                if constexpr (EXEC_FLAG & EXEC_FLAG_TENSOR_LIST) {
                    gmB.SetGlobalBuffer(gmBlistTensorDesc.GetDataPtr<ElementB>(groupIdx));
                    gmMxScaleB.SetGlobalBuffer(gmBScalelistTensorDesc.GetDataPtr<ElementMxScaleB>(groupIdx));
                } else {
                    gmB.SetGlobalBuffer(gmBlistTensorDesc.GetDataPtr<ElementB>(0) + gmGroupOffsetB);
                    gmMxScaleB.SetGlobalBuffer(gmBScalelistTensorDesc.GetDataPtr<ElementMxScaleB>(0) + gmGroupOffsetMxScaleB);
                }
            currentM = (groupIdx == 0) ? groupList.GetValue(groupIdx)
                                                    : (groupList.GetValue(groupIdx) - groupList.GetValue(groupIdx - 1));
                GemmCoord inGroupProblemShape{currentM, params.problemShape.n(), params.problemShape.k()};

                BlockScheduler matmulBlockScheduler(inGroupProblemShape, MakeCoord(L1_TILE_M, L1_TILE_N));
                uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

                if (CeilDiv(currentM, L1_TILE_M) == 1) {
                    gmB.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
                } else {
                    gmB.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_NORMAL);
                }

                uint32_t startLoopIdx;
                if (coreIdx < startCoreIdx) {
                    startLoopIdx = coreIdx + coreNum - startCoreIdx;
                } else {
                    startLoopIdx = coreIdx - startCoreIdx;
                }

                auto tensorB = tla::MakeTensor(gmB, params.layoutB, Arch::PositionGM{});
                auto tensorMxScaleA = tla::MakeTensor(gmMxScaleA, params.layoutMxScaleA, Arch::PositionGM{});
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
                    callbackAfterFixpipe();
                }
                totalM += inGroupProblemShape.m();

                if constexpr (!(EXEC_FLAG & EXEC_FLAG_TENSOR_LIST)) {
                    if constexpr (AscendC::Std::is_one_of_v<ElementB, float4_e2m1x2_t, float4_e1m2x2_t>) {
                        gmGroupOffsetB += std::is_same_v<LayoutB, layout::ColumnMajor> ? 
                            CeilDiv<2>(inGroupProblemShape.k()) * inGroupProblemShape.n() : 
                            CeilDiv<2>(inGroupProblemShape.n()) * inGroupProblemShape.k();
                    }else{
                        gmGroupOffsetB += inGroupProblemShape.k() * inGroupProblemShape.n();
                    }
                    gmGroupOffsetMxScaleB += mxScaleAlignedK * inGroupProblemShape.n();
                }
                gmGroupOffsetMxScaleA += inGroupProblemShape.m() * mxScaleAlignedK;

                startCoreIdx = (startCoreIdx + coreLoops) % coreNum;
            }

            if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
                blockMmad.template SynchronizeBlock<decltype(tensorC)>();
            }
        }
        if constexpr (EXEC_FLAG & EXEC_FLAG_SHARED_EXPERT) {
            currentM = params.sharedProblemShape.m();
            gmA.SetGlobalBuffer(params.ptrSharedA);
            gmMxScaleA.SetGlobalBuffer(params.ptrSharedMxScaleA);
            gmB.SetGlobalBuffer(params.ptrSharedB);
            gmMxScaleB.SetGlobalBuffer(params.ptrSharedMxScaleB);
            gmC.SetGlobalBuffer(params.ptrSharedC);
            GemmCoord inGroupProblemShape{currentM, params.sharedProblemShape.n(), params.sharedProblemShape.k()};

            BlockScheduler matmulBlockScheduler(inGroupProblemShape, MakeCoord(L1_TILE_M, L1_TILE_N));
            uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

            if (CeilDiv(currentM, L1_TILE_M) == 1) {
                gmB.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            } else {
                gmB.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_NORMAL);
            }

            uint32_t startLoopIdx;
            if (coreIdx < startCoreIdx) {
                startLoopIdx = coreIdx + coreNum - startCoreIdx;
            } else {
                startLoopIdx = coreIdx - startCoreIdx;
            }

            auto tensorA = tla::MakeTensor(gmA, params.layoutSharedA, Arch::PositionGM{});
            auto tensorMxScaleA = tla::MakeTensor(gmMxScaleA, params.layoutSharedMxScaleA, Arch::PositionGM{});
            auto tensorB = tla::MakeTensor(gmB, params.layoutSharedB, Arch::PositionGM{});
            auto tensorMxScaleB = tla::MakeTensor(gmMxScaleB, params.layoutSharedMxScaleB, Arch::PositionGM{});
            auto tensorC = tla::MakeTensor(gmC, params.layoutC, Arch::PositionGM{});
            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += coreNum) {
                GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
                GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);

                auto tensorBlockA = GetTile(tensorA,
                    tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.k() * L1_TILE_K),
                    tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));

                auto tensorBlockB = GetTile(tensorB,
                    tla::MakeCoord(blockCoord.k() * L1_TILE_K, blockCoord.n() * L1_TILE_N),
                    tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));

                auto tensorBlockC = GetTile(tensorC,
                    tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.n() * L1_TILE_N),
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
                callbackAfterFixpipe();
            }

            startCoreIdx = (startCoreIdx + coreLoops) % coreNum;
            if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
                blockMmad.template SynchronizeBlock<decltype(tensorC)>();
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const &params) {
        auto *combiner = (MoeDistributeCombineImpl::CamMoeDistributeCombine<TemplateMC2TypeFunc> *)params.combiner;

        uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t coreNum = AscendC::GetBlockNum();

        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer(params.ptrC);
        AscendC::GlobalTensor<ElementGroupList> groupList;
        groupList.SetGlobalBuffer(params.ptrGroupList);

        AscendC::GlobalTensor<ElementD> gmD;
        gmD.SetGlobalBuffer(params.ptrD);

        do {
            if constexpr (EXEC_FLAG & EXEC_FLAG_DEEP_FUSE) {
                if (AscendC::GetSubBlockIdx() == 0) {
                    AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(MoeDistributeCombineImpl::RECV_SYNC_EVENT_ID);
                }
            }
            BlockEpilogue blockEpilogue(resource, combiner->GetCalcInfo());
            uint32_t target = 1;
            uint32_t startCoreIdx = 0;

            int64_t mxScaleAlignedK = static_cast<int64_t>(CeilDiv<MX_BASEK_FACTOR>(params.problemShape.k()) * MX_SCALE_COPY_GROUP_NUM);

            uint32_t currentM = 0;
            int64_t totalM = 0;

            auto tensorC = tla::MakeTensor(gmC, params.layoutC, Arch::PositionGM{});
            auto tensorD = tla::MakeTensor(gmD, params.layoutC, Arch::PositionGM{});

            for (uint32_t groupIdx = 0; groupIdx < params.problemCount; ++groupIdx) {
                currentM = (groupIdx == 0) ? groupList.GetValue(groupIdx)
                                                    : (groupList.GetValue(groupIdx) - groupList.GetValue(groupIdx - 1));
                GemmCoord inGroupProblemShape{currentM, params.problemShape.n(), params.problemShape.k()};
                BlockScheduler matmulBlockScheduler(inGroupProblemShape, MakeCoord(L1_TILE_M, L1_TILE_N));
                uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

                uint32_t startLoopIdx;
                if (coreIdx < startCoreIdx) {
                    startLoopIdx = coreIdx + coreNum - startCoreIdx;
                } else {
                    startLoopIdx = coreIdx - startCoreIdx;
                }

                for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += coreNum) {
                    GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
                    GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);

                    auto tensorBlockC = GetTile(tensorC,
                        tla::MakeCoord(totalM + blockCoord.m() * L1_TILE_M, blockCoord.n() * L1_TILE_N),
                        tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));

                    auto tensorBlockD = GetTile(tensorD,
                        tla::MakeCoord(totalM + blockCoord.m() * L1_TILE_M, blockCoord.n() * L1_TILE_N),
                        tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));

                    CheckSyncFlag(syncGmAddr + GMM2::SOFT_SYNC_OFFSET, static_cast<uint8_t>(coreIdx), target);
                    target += 1;
                    blockEpilogue(tensorBlockC, tensorBlockD, actualBlockShape, groupIdx, totalM + blockCoord.m() * L1_TILE_M, blockCoord.n() * L1_TILE_N);
                }

                totalM += inGroupProblemShape.m();

                startCoreIdx = (startCoreIdx + coreLoops) % coreNum;
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            if constexpr (EXEC_FLAG & EXEC_FLAG_SHARED_EXPERT) {
                if (AscendC::GetSubBlockIdx() == 0) {
                    AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(MoeDistributeCombineImpl::SEND_SYNC_EVENT_ID);
                    if constexpr ((EXEC_FLAG & EXEC_FLAG_DEEP_FUSE) == 0) {
                        AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(MoeDistributeCombineImpl::RECV_SYNC_EVENT_ID);
                    }
                    gmC.SetGlobalBuffer(params.ptrSharedC); // 必须重新赋值，不然下面还是用的路由专家数据
                    gmD.SetGlobalBuffer(params.ptrSharedD); // 必须重新赋值，不然下面还是用的路由专家数据
                    tensorC = tla::MakeTensor(gmC, params.layoutC, Arch::PositionGM{});
                    tensorD = tla::MakeTensor(gmD, params.layoutC, Arch::PositionGM{});
                    currentM = params.sharedProblemShape.m();
                    GemmCoord inGroupProblemShape{currentM, params.sharedProblemShape.n(), params.sharedProblemShape.k()};
                    BlockScheduler matmulBlockScheduler(inGroupProblemShape, MakeCoord(L1_TILE_M, L1_TILE_N));
                    uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

                    uint32_t startLoopIdx;
                    if (coreIdx < startCoreIdx) {
                        startLoopIdx = coreIdx + coreNum - startCoreIdx;
                    } else {
                        startLoopIdx = coreIdx - startCoreIdx;
                    }

                    for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += coreNum) {
                        GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
                        GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);

                        auto tensorBlockC = GetTile(tensorC,
                            tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.n() * L1_TILE_N),
                            tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));

                        auto tensorBlockD = GetTile(tensorD,
                            tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.n() * L1_TILE_N),
                            tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));

                        CheckSyncFlag(syncGmAddr + GMM2::SOFT_SYNC_OFFSET, static_cast<uint8_t>(coreIdx), target);
                        target += 1;
                        blockEpilogue(tensorBlockC, tensorBlockD, actualBlockShape, UINT32_MAX, 0, 0);
                    }
                    AscendC::CrossCoreWaitFlag(MoeDistributeCombineImpl::SEND_SYNC_EVENT_ID);
                    AscendC::CrossCoreWaitFlag(MoeDistributeCombineImpl::RECV_SYNC_EVENT_ID);
                }
            }
        } while (false);

        icache_preload(4);
        if constexpr (EXEC_FLAG & EXEC_FLAG_SHARED_EXPERT) {
            if (AscendC::GetSubBlockIdx() == 1) {
                resource.pipe.Init();
                combiner->TPipeSet(&resource.pipe);
                combiner->ProcessCombine();
                combiner->TPipeSet(nullptr);
                resource.pipe.Destroy();
            }
        } else if constexpr (EXEC_FLAG & EXEC_FLAG_DEEP_FUSE) {
            if (AscendC::GetSubBlockIdx() == 0) {
                resource.pipe.Init();
                combiner->TPipeSet(&resource.pipe);
                combiner->AllToAllSend();
                combiner->TPipeSet(nullptr);
                resource.pipe.Destroy();
            } else {
                resource.pipe.Init();
                combiner->TPipeSet(&resource.pipe);
                combiner->ReducePermute();
                combiner->TPipeSet(nullptr);
                resource.pipe.Destroy();
            }
        } else {
            resource.pipe.Init();
            combiner->TPipeSet(&resource.pipe);
            combiner->Process();
            combiner->TPipeSet(nullptr);
            resource.pipe.Destroy();
        }
        if (AscendC::GetSubBlockIdx() == 0) {
            AscendC::GlobalTensor<int32_t> softSyncTensor;
            softSyncTensor.SetGlobalBuffer((__gm__ int32_t*)(syncGmAddr + GMM2::SOFT_SYNC_OFFSET));
            AscendC::LocalTensor<int32_t> tmpZeroLocalTensor = resource.ubBuf.template GetBufferByByte<int32_t>(0);
            AscendC::Duplicate(tmpZeroLocalTensor, (int32_t)0, GMM2::INT32_COUNT_PER_BLOCK);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
            AscendC::DataCopy(softSyncTensor[coreIdx * CVSoftSync::SOFT_SYNC_SPACE_SIZE / sizeof(int32_t)],
                                                tmpZeroLocalTensor, GMM2::INT32_COUNT_PER_BLOCK);
        }
    }

private:
    friend struct AicSetFunc;
    struct AicSetFunc {
        CATLASS_DEVICE
        AicSetFunc() = default;

        CATLASS_DEVICE
        void operator()() const
        {
            EncreaseSyncFlag(flagAddr, idx);
        }

        __gm__ uint8_t *flagAddr;
        uint8_t idx;
    };

    AicSetFunc aicSetFunc;

    AscendC::GlobalTensor<GM_ADDR> epWinContext_;
    __gm__ Mc2Kernel::HcclOpParam *winContext_;
    GM_ADDR syncGmAddr;
    Arch::Resource<ArchTag> resource;
};

#endif // (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)

} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_MX_GMM2_CAST_COMBINE_H
