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

#ifndef CATLASS_GEMM_KERNEL_W8A8_MATMUL_PER_CHANNEL_HPP
#define CATLASS_GEMM_KERNEL_W8A8_MATMUL_PER_CHANNEL_HPP

#include <iostream>

#include "catlass_a5/catlass.hpp"
#include "catlass_a5/arch/cross_core_sync.hpp"
#include "catlass_a5/arch/resource.hpp"
#include "catlass_a5/coord.hpp"
#include "catlass_a5/layout/layout.hpp"
#include "catlass_a5/detail/callback.hpp"
#include "catlass_a5/gemm_coord.hpp"
#include "catlass_a5/matrix_coord.hpp"

namespace Catlass::Gemm::Kernel {

template <class BlockMmad_, class BlockEpilogue_, class BlockScheduler_, uint32_t WORKSPACE_STAGES_,
          class ElementScale_, class LayoutScale_>
class DynamicW8A8MatmulPerTokenPerChannelDequant
{
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using ElementScale = ElementScale_;
    using LayoutScale = LayoutScale_;
    using ElementA = typename BlockMmad::ElementA;
    using LayoutA = typename BlockMmad::LayoutA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutB = typename BlockMmad::LayoutB;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;
    using ElementAccumulator = typename BlockMmad::ElementAccumulator;

    using BlockEpilogue = BlockEpilogue_;
    using ElementPerTokenScale = typename BlockEpilogue::ElementPerTokenScale;
    using LayoutPerTokenScale = typename BlockEpilogue::LayoutPerTokenScale;
    using ElementD = typename BlockEpilogue::ElementD;
    using LayoutD = typename BlockEpilogue::LayoutD;
    using EpilogueParams = typename BlockEpilogue::Params;

    using BlockScheduler = BlockScheduler_;

    using EpilogueTileShape = typename BlockEpilogue::TileShape;

    static constexpr uint32_t WORKSPACE_STAGES = WORKSPACE_STAGES_;

    /// Parameters structure
    struct Params {
        // Data members
        GemmCoord problemShape;
        GemmCoord l1TileShape;
        GemmCoord l0TileShape;
        __gm__ ElementA *ptrA;
        LayoutA layoutA;
        __gm__ ElementB *ptrB;
        LayoutB layoutB;
        __gm__ ElementScale *ptrScale;
        LayoutScale layoutScale;
        __gm__ ElementPerTokenScale *ptrPerTokenScale;
        LayoutPerTokenScale layoutPerTokenScale;
        __gm__ ElementD *ptrD;
        LayoutD layoutD;
        GM_ADDR ptrWorkspace;

        // Methods
        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(GemmCoord problemShape_, GemmCoord l1TileShape_, GemmCoord l0TileShape_, GM_ADDR ptrA_, LayoutA layoutA_,
               GM_ADDR ptrB_, LayoutB layoutB_, GM_ADDR ptrScale_, LayoutScale layoutScale_, GM_ADDR ptrPerTokenScale_,
               LayoutPerTokenScale layoutPerTokenScale_, GM_ADDR ptrD_, LayoutD layoutD_, GM_ADDR ptrWorkspace_)
            : problemShape(problemShape_),
              l1TileShape(l1TileShape_),
              l0TileShape(l0TileShape_),
              ptrA(reinterpret_cast<__gm__ ElementA *>(ptrA_)),
              layoutA(layoutA_),
              ptrB(reinterpret_cast<__gm__ ElementB *>(ptrB_)),
              layoutB(layoutB_),
              ptrScale(reinterpret_cast<__gm__ ElementScale *>(ptrScale_)),
              layoutScale(layoutScale_),
              ptrPerTokenScale(reinterpret_cast<__gm__ ElementPerTokenScale *>(ptrPerTokenScale_)),
              layoutPerTokenScale(layoutPerTokenScale_),
              ptrD(reinterpret_cast<__gm__ ElementD *>(ptrD_)),
              layoutD(layoutD_),
              ptrWorkspace(ptrWorkspace_)
        {}
    };

    struct Arguments {
        GemmCoord problemShape;
        uint32_t aicCoreNum;
        uint8_t *ptrA;
        LayoutA layoutA;
        uint8_t *ptrB;
        LayoutB layoutB;
        uint8_t *ptrScale;
        LayoutScale layoutScale;
        uint8_t *ptrPerTokenScale;
        LayoutPerTokenScale layoutPerTokenScale;
        uint8_t *ptrD;
        LayoutD layoutD;
    };

    static bool CanImplement(const Arguments &args)
    {
        uint32_t n = args.problemShape.n();
        uint32_t k = args.problemShape.k();

        bool isValid = true;
        if constexpr (std::is_same_v<LayoutB, layout::zN>) {
            isValid = (k % 16 == 0 && n % 64 == 0);
            if (!isValid) {
                std::cerr << "matmulOp cannot be implemented, " << "please mind that n must be divisible by 64 "
                          << "while k must be divisible by 16" << std::endl;
            }
        }

        if constexpr (std::is_same_v<LayoutB, layout::nZ>) {
            isValid = (k % 64 == 0 && n % 16 == 0);
            if (!isValid) {
                std::cerr << "matmulOp cannot be implemented, " << "please mind that k must be divisible by 64 "
                          << "while n must be divisible by 16" << std::endl;
            }
        }

        return isValid;
    }

    static Params ToUnderlyingArguments(const Arguments &args, uint8_t *workspace)
    {
        Params params{args.problemShape,
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
                      workspace};
        return params;
    }

    // Methods
    CATLASS_DEVICE
    DynamicW8A8MatmulPerTokenPerChannelDequant()
    {
        Arch::FlagID flagId = 0;
        for (uint32_t stageId = 0; stageId < WORKSPACE_STAGES; ++stageId) {
            flagAicFinishStoreList[stageId] = Arch::CrossCoreFlag(flagId++);
            flagAivFinishComputeList[stageId] = Arch::CrossCoreFlag(flagId++);
            aicWaitFuncList[stageId] = {this, stageId};
            aicSetFuncList[stageId] = {this, stageId};
        }
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const &params, Catlass::Arch::Resource<ArchTag> &resource);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const &params, Catlass::Arch::Resource<ArchTag> &resource)
    {
        BlockScheduler blockScheduler;
        BlockMmad blockMmad(params.l1TileShape, params.l0TileShape, resource);

        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer(params.ptrA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer(params.ptrB);
        AscendC::GlobalTensor<ElementC> gmC;

        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();

        gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrWorkspace));
        auto layoutC = layout::RowMajor{params.l1TileShape.m() * coreNum * WORKSPACE_STAGES, params.l1TileShape.n()};

        uint32_t stageId = 0;
        uint32_t stageUsed = 0;
        uint32_t currentM = params.problemShape.m();
        GemmCoord inProblemShape{currentM, params.problemShape.n(), params.problemShape.k()};

        LayoutScale layoutScale = params.layoutScale;
        LayoutA layoutA = params.layoutA.GetTileLayout(inProblemShape.GetCoordMK());
        LayoutB layoutB = params.layoutB;

        blockScheduler.Update(inProblemShape, MakeCoord(params.l1TileShape.m(), params.l1TileShape.n()));
        uint32_t coreLoops = blockScheduler.GetCoreLoops();

        if (CeilDiv(currentM, params.l1TileShape.m()) == 1) {
            gmB.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        }
        // Loop through the matmul of each groupIdx
        for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += coreNum) {
            // Compute block location
            GemmCoord blockCoord = blockScheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBlockShape = blockScheduler.GetActualBlockShape(blockCoord);

            Callback callbackBeforeFixpipe{};
            if (stageUsed == WORKSPACE_STAGES) {
                callbackBeforeFixpipe = MakeCallback(&aicWaitFuncList[stageId]);
            } else {
                ++stageUsed;
            }
            Callback callbackAfterFixpipe = MakeCallback(&aicSetFuncList[stageId]);

            // Compute initial location in logical coordinates
            MatrixCoord offsetA{blockCoord.m() * params.l1TileShape.m(), blockCoord.k() * params.l1TileShape.k()};
            MatrixCoord offsetB{blockCoord.k() * params.l1TileShape.k(), blockCoord.n() * params.l1TileShape.n()};
            MatrixCoord offsetC{(stageId * coreNum + coreIdx) * params.l1TileShape.m(), 0};
            int64_t gmOffsetA = layoutA.GetOffset(offsetA);
            int64_t gmOffsetB = layoutB.GetOffset(offsetB);
            int64_t gmOffsetC = layoutC.GetOffset(offsetC);

            // Compute block-scoped matrix multiply-add
            if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
                blockMmad(gmA[gmOffsetA], layoutA, gmB[gmOffsetB], layoutB, gmC[gmOffsetC], layoutC, actualBlockShape,
                          callbackBeforeFixpipe, callbackAfterFixpipe);
            } else {
                callbackBeforeFixpipe();
                blockMmad(gmA[gmOffsetA], layoutA, gmB[gmOffsetB], layoutB, gmC[gmOffsetC], layoutC, actualBlockShape);
                callbackAfterFixpipe();
            }

            stageId = (stageId + 1 < WORKSPACE_STAGES) ? (stageId + 1) : 0;
        }

        if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
            blockMmad.SynchronizeBlock();
        }

        while (stageUsed > 0) {
            uint32_t aivComputeStageId =
                (stageId >= stageUsed) ? (stageId - stageUsed) : (stageId + WORKSPACE_STAGES - stageUsed);
            Arch::CrossCoreWaitFlag(flagAivFinishComputeList[aivComputeStageId]);
            --stageUsed;
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const &params, Catlass::Arch::Resource<ArchTag> &resource)
    {
        BlockScheduler blockScheduler;
        BlockEpilogue blockEpilogue(resource);

        uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t coreNum = AscendC::GetBlockNum();

        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrWorkspace));
        auto layoutC = layout::RowMajor{params.l1TileShape.m() * coreNum * WORKSPACE_STAGES, params.l1TileShape.n()};

        uint32_t stageId = 0;
        uint32_t startCoreIdx = 0;

        GemmCoord problemShape{params.problemShape.m(), params.problemShape.n(), params.problemShape.k()};

        LayoutScale layoutScale = params.layoutScale.GetTileLayout(problemShape.template GetCoordByAxis<0>());
        LayoutPerTokenScale layoutPerTokenScale =
            params.layoutPerTokenScale.GetTileLayout(problemShape.template GetCoordByAxis<0>());
        LayoutD layoutD = params.layoutD.GetTileLayout(problemShape.GetCoordMN());

        EpilogueParams epilogueParams{params.ptrScale,     layoutScale, params.ptrPerTokenScale,
                                      layoutPerTokenScale, params.ptrD, layoutD};

        blockScheduler.Update(problemShape, MakeCoord(params.l1TileShape.m(), params.l1TileShape.n()));
        blockEpilogue.UpdateParams(epilogueParams);
        uint32_t coreLoops = blockScheduler.GetCoreLoops();

        GemmCoord blockShapeMNK = MakeCoord(params.l1TileShape.m(), params.l1TileShape.n(), params.l1TileShape.k());
        for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += coreNum) {
            GemmCoord blockCoordMNK = blockScheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBlockShapeMNK = blockScheduler.GetActualBlockShape(blockCoordMNK);

            MatrixCoord offsetC{(stageId * coreNum + coreIdx) * params.l1TileShape.m(), 0};
            int64_t gmOffsetC = layoutC.GetOffset(offsetC);
            auto gmBlockC = gmC[gmOffsetC];
            auto layoutBlockC = layoutC.GetTileLayout(actualBlockShapeMNK.GetCoordMN());

            Arch::CrossCoreWaitFlag(flagAicFinishStoreList[stageId]);
            blockEpilogue(blockShapeMNK, blockCoordMNK, actualBlockShapeMNK, gmBlockC, layoutBlockC);
            Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishComputeList[stageId]);

            stageId = (stageId + 1 < WORKSPACE_STAGES) ? (stageId + 1) : 0;
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    friend struct AicWaitFunc;
    friend struct AicSetFunc;

    struct AicWaitFunc {
        using MatmulKernel = DynamicW8A8MatmulPerTokenPerChannelDequant<BlockMmad, BlockEpilogue, BlockScheduler,
                                                                        WORKSPACE_STAGES, ElementScale, LayoutScale>;

        CATLASS_DEVICE
        AicWaitFunc() = default;

        CATLASS_DEVICE
        void operator()() const
        {
            Arch::CrossCoreWaitFlag(ptr->flagAivFinishComputeList[stageId]);
        }

        MatmulKernel *ptr{nullptr};
        uint32_t stageId;
    };

    struct AicSetFunc {
        using MatmulKernel = DynamicW8A8MatmulPerTokenPerChannelDequant<BlockMmad, BlockEpilogue, BlockScheduler,
                                                                        WORKSPACE_STAGES, ElementScale, LayoutScale>;

        CATLASS_DEVICE
        AicSetFunc() = default;

        CATLASS_DEVICE
        void operator()() const
        {
            Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(ptr->flagAicFinishStoreList[stageId]);
        }

        MatmulKernel *ptr{nullptr};
        uint32_t stageId;
    };

    Arch::CrossCoreFlag flagAicFinishStoreList[WORKSPACE_STAGES];
    Arch::CrossCoreFlag flagAivFinishComputeList[WORKSPACE_STAGES];

    AicWaitFunc aicWaitFuncList[WORKSPACE_STAGES];
    AicSetFunc aicSetFuncList[WORKSPACE_STAGES];
};

}  // namespace Catlass::Gemm::Kernel

#endif  // CATLASS_GEMM_KERNEL_W8A8_MATMUL_PER_CHANNEL_HPP
