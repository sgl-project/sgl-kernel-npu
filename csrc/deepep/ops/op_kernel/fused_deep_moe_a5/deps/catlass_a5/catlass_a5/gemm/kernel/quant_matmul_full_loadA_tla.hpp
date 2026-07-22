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

#ifndef CATLASS_GEMM_KERNEL_QUANT_MATMUL_FULL_LOADA_TLA_HPP
#define CATLASS_GEMM_KERNEL_QUANT_MATMUL_FULL_LOADA_TLA_HPP

#include "catlass_a5/catlass.hpp"
#include "catlass_a5/coord.hpp"
#include "catlass_a5/gemm_coord.hpp"
#include "catlass_a5/matrix_coord.hpp"
#include "catlass_a5/arch/resource.hpp"
#include "catlass_a5/arch/cross_core_sync.hpp"
#include "catlass_a5/detail/callback.hpp"
#include "tla_a5/layout.hpp"
#include "tla_a5/tensor.hpp"

namespace Catlass::Gemm::Kernel {

// Template for Matmul kernel. Compute C = A * B
template <class BlockMmad_, class BlockEpilogue_, class BlockScheduler_, uint32_t WORKSPACE_STAGES_>
class QuantMatmulFullLoadATla
{
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using ElementA = typename BlockMmad::ElementA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;
    using ElementBias = typename BlockMmad::ElementBias;
    using ElementAccumulator = typename BlockMmad::ElementAccumulator;

    using BlockEpilogue = BlockEpilogue_;
    using ElementScale = typename BlockEpilogue::ElementScale;
    using LayoutScale = typename BlockEpilogue::LayoutScale;
    using ElementPerTokenScale = typename BlockEpilogue::ElementPerTokenScale;
    using LayoutPerTokenScale = typename BlockEpilogue::LayoutPerTokenScale;
    using ElementD = typename BlockEpilogue::ElementD;
    using LayoutD = typename BlockEpilogue::LayoutD;

    using BlockScheduler = BlockScheduler_;
    static constexpr uint32_t WORKSPACE_STAGES = WORKSPACE_STAGES_;

    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});

    /**
     * @struct BlockMmadArguments
     * @brief Kernel arguments for the host side
     */
    struct BlockMmadArguments {
        GM_ADDR ptrA{nullptr};  ///< The global memory address of matrix A
        LayoutA layoutA;
        GM_ADDR ptrB{nullptr};  ///< The global memory address of matrix B
        LayoutB layoutB;
        GM_ADDR ptrBias{nullptr};  ///< The global memory address of bias
    };

    using BlockMmadParams = BlockMmadArguments;
    using EpilogueParams = typename BlockEpilogue::Params;
    using EpilogueArguments = EpilogueParams;

    /// Parameters structure
    struct Params {
        // Data members
        GemmCoord problemShape;
        uint32_t aicCoreNum;
        BlockMmadParams mmadParams;
        EpilogueParams epiParams;
        GM_ADDR ptrWorkspace;
        Params() = default;
    };

    struct Arguments {
        GemmCoord problemShape;
        uint32_t aicCoreNum;
        BlockMmadArguments mmadArgs;
        EpilogueArguments epiArgs;
        Arguments() = default;
    };

    static bool CanImplement(const Arguments &args)
    {
        uint32_t L1UsedSpace =
            L1_TILE_M * args.problemShape.k() * sizeof(ElementA) + L1_TILE_K * L1_TILE_N * 2 * sizeof(ElementB);
        if (L1UsedSpace > ArchTag::L1_SIZE) {
            return false;
        }
        return true;
    }

    static size_t GetWorkspaceSize(const Arguments &args)
    {
        size_t lenWorkspace = static_cast<size_t>(L1_TILE_M) * L1_TILE_N * args.aicCoreNum * WORKSPACE_STAGES;
        size_t sizeWorkspace = lenWorkspace * sizeof(ElementC);
        return sizeWorkspace;
    }

    static Params ToUnderlyingArguments(const Arguments &args, uint8_t *workspace)
    {
        BlockMmadParams mmadParams = {args.mmadArgs.ptrA, args.mmadArgs.layoutA, args.mmadArgs.ptrB,
                                      args.mmadArgs.layoutB, args.mmadArgs.ptrBias};
        EpilogueParams epiParams = {args.epiArgs.ptrScale,
                                    args.epiArgs.layoutScale,
                                    args.epiArgs.ptrPerTokenScale,
                                    args.epiArgs.layoutPerTokenScale,
                                    args.epiArgs.ptrD,
                                    args.epiArgs.layoutD};
        // mmad params with epilogue takes workspaceGm as output
        Params params = {args.problemShape, args.aicCoreNum, mmadParams, epiParams, workspace};
        return params;
    }

    // Methods
    CATLASS_DEVICE
    QuantMatmulFullLoadATla()
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
    CATLASS_DEVICE void operator()(Params const &params);

    /// Executes one Matmul
    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const &params)
    {
        BlockScheduler matmulBlockScheduler(params.problemShape, MakeCoord(L1_TILE_M, L1_TILE_N));
        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA *)params.mmadParams.ptrA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB *)params.mmadParams.ptrB);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrWorkspace);

        // Matrix A or Matrix B does not have duplicate data reads. Setting L2 Cache to Disable,
        // data reads will bypass L2 Cache.
        if (CeilDiv(params.problemShape.m(), L1_TILE_M) == 1) {
            gmB.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        }
        if (CeilDiv(params.problemShape.n(), L1_TILE_N) == 1) {
            gmA.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        }

        using GlobalTensorBiasType = std::conditional_t<std::is_void_v<ElementBias>, uint8_t, ElementBias>;
        AscendC::GlobalTensor<GlobalTensorBiasType> gmBias;
        if constexpr (!std::is_void_v<ElementBias>) {
            gmBias.SetGlobalBuffer((__gm__ ElementBias *)params.mmadParams.ptrBias);
        }

        auto layoutBias = tla::MakeLayout(params.problemShape.n());
        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();

        // Represent the full tensors
        auto tensorA = tla::MakeTensor(gmA, params.mmadParams.layoutA, Arch::PositionGM{});
        auto tensorB = tla::MakeTensor(gmB, params.mmadParams.layoutB, Arch::PositionGM{});
        auto layoutC = tla::MakeLayout<ElementC, layout::RowMajor>(L1_TILE_M * coreNum * WORKSPACE_STAGES, L1_TILE_N);
        auto tensorC = tla::MakeTensor(gmC, layoutC, Arch::PositionGM{});
        auto tensorBias = tla::MakeTensor(gmBias, layoutBias, Arch::PositionGM{});

        uint32_t stageId = 0;
        uint32_t stageUsed = 0;

        BlockMmad blockMmad(resource);
        int64_t gmOffsetAPreload{0};

        for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += coreNum) {
            // Compute block location
            GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);

            Callback callbackBeforeFixpipe{};
            if (stageUsed == WORKSPACE_STAGES) {
                callbackBeforeFixpipe = MakeCallback(&aicWaitFuncList[stageId]);
            } else {
                ++stageUsed;
            }
            Callback callbackAfterFixpipe = MakeCallback(&aicSetFuncList[stageId]);

            // Make tiled views
            auto tensorBlockA = GetTileA(tensorA, blockCoord.m() * L1_TILE_M, blockCoord.k() * L1_TILE_K,
                                         actualBlockShape.m(), actualBlockShape.k());
            auto tensorBlockB = GetTile(tensorB, tla::MakeCoord(blockCoord.k() * L1_TILE_K, blockCoord.n() * L1_TILE_N),
                                        tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
            auto tensorBlockC = GetTile(tensorC, tla::MakeCoord((stageId * coreNum + coreIdx) * L1_TILE_M, 0),
                                        tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));

            int64_t gmOffsetA =
                params.mmadParams.layoutA(tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.k() * L1_TILE_K));

            // Judge whether the current blockA is already on L1Cache
            bool isFirstBlock = (loopIdx == AscendC::GetBlockIdx());
            bool needLoadL1 = true;
            if (isFirstBlock) {
                gmOffsetAPreload = gmOffsetA;
            } else {
                if (gmOffsetA == gmOffsetAPreload) {
                    needLoadL1 = false;
                } else {
                    gmOffsetAPreload = gmOffsetA;
                }
            }

            // Compute block-scoped matrix multiply-add
            if constexpr (std::is_void_v<ElementBias>) {
                callbackBeforeFixpipe();
                blockMmad(tensorBlockA, tensorBlockB, tensorBlockC, actualBlockShape, needLoadL1);
                callbackAfterFixpipe();
            } else {
                auto tensorBlockBias = GetTile(tensorBias, tla::MakeCoord(blockCoord.n() * L1_TILE_N),
                                               tla::MakeShape(actualBlockShape.n()));
                callbackBeforeFixpipe();
                blockMmad(tensorBlockA, tensorBlockB, tensorBlockC, actualBlockShape, needLoadL1, tensorBlockBias);
                callbackAfterFixpipe();
            }
            stageId = (stageId + 1 < WORKSPACE_STAGES) ? (stageId + 1) : 0;
        }

        if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
            blockMmad.template SynchronizeBlock<decltype(tensorC)>();
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
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const &params)
    {
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.mmadParams.ptrA));
        auto tensorA = tla::MakeTensor(gmA, params.mmadParams.layoutA, Arch::PositionGM{});

        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.mmadParams.ptrB));
        auto tensorB = tla::MakeTensor(gmB, params.mmadParams.layoutB, Arch::PositionGM{});

        BlockScheduler blockScheduler;
        BlockEpilogue blockEpilogue(resource);

        uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t coreNum = AscendC::GetBlockNum();

        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrWorkspace));
        auto layoutC = tla::MakeLayout<ElementC, layout::RowMajor>(L1_TILE_M * coreNum * WORKSPACE_STAGES, L1_TILE_N);
        auto tensorC = tla::MakeTensor(gmC, layoutC, Arch::PositionGM{});

        AscendC::GlobalTensor<ElementScale> gmScale;
        gmScale.SetGlobalBuffer(reinterpret_cast<__gm__ ElementScale *>(params.epiParams.ptrScale));
        auto tensorScale = tla::MakeTensor(gmScale, params.epiParams.layoutScale, Arch::PositionGM{});

        AscendC::GlobalTensor<ElementPerTokenScale> gmPerTokenScale;
        gmPerTokenScale.SetGlobalBuffer(
            reinterpret_cast<__gm__ ElementPerTokenScale *>(params.epiParams.ptrPerTokenScale));
        auto tensorPerTokenScale =
            tla::MakeTensor(gmPerTokenScale, params.epiParams.layoutPerTokenScale, Arch::PositionGM{});

        AscendC::GlobalTensor<ElementD> gmD;
        gmD.SetGlobalBuffer(reinterpret_cast<__gm__ ElementD *>(params.epiParams.ptrD));
        auto tensorD = tla::MakeTensor(gmD, params.epiParams.layoutD, Arch::PositionGM{});

        uint32_t stageId = 0;

        blockScheduler.Update(params.problemShape, MakeCoord(L1_TILE_M, L1_TILE_N));
        blockEpilogue.UpdateParams(params.epiParams);
        uint32_t coreLoops = blockScheduler.GetCoreLoops();

        for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += coreNum) {
            GemmCoord blockCoord = blockScheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBlockShapeMNK = blockScheduler.GetActualBlockShape(blockCoord);

            auto tensorBlockC = GetTile(tensorC, tla::MakeCoord((stageId * coreNum + coreIdx) * L1_TILE_M, 0),
                                        tla::MakeShape(actualBlockShapeMNK.m(), actualBlockShapeMNK.n()));

            auto tensorBlockScale = GetTile(tensorScale, tla::MakeCoord(0, blockCoord.n() * L1_TILE_N),
                                            tla::MakeShape(tla::Int<1>{}, actualBlockShapeMNK.n()));

            auto tensorBlockPerTokenScaleScale =
                GetTile(tensorPerTokenScale, tla::MakeCoord(0, blockCoord.m() * L1_TILE_M),
                        tla::MakeShape(tla::Int<1>{}, actualBlockShapeMNK.m()));

            auto tensorBlockD = GetTile(tensorD, tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.n() * L1_TILE_N),
                                        tla::MakeShape(actualBlockShapeMNK.m(), actualBlockShapeMNK.n()));

            Arch::CrossCoreWaitFlag(flagAicFinishStoreList[stageId]);
            blockEpilogue(tensorBlockC, tensorBlockScale, tensorBlockPerTokenScaleScale, tensorBlockD,
                          actualBlockShapeMNK);
            Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishComputeList[stageId]);

            stageId = (stageId + 1 < WORKSPACE_STAGES) ? (stageId + 1) : 0;
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    friend struct AicWaitFunc;
    friend struct AicSetFunc;

    struct AicWaitFunc {
        using MatmulKernel = QuantMatmulFullLoadATla<BlockMmad, BlockEpilogue, BlockScheduler, WORKSPACE_STAGES>;

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
        using MatmulKernel = QuantMatmulFullLoadATla<BlockMmad, BlockEpilogue, BlockScheduler, WORKSPACE_STAGES>;

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

    static constexpr Arch::FlagID FLAG_AIV_FINISH_STORE = 4;
    Arch::CrossCoreFlag flagAivFinishPadding{FLAG_AIV_FINISH_STORE};
    Arch::Resource<ArchTag> resource;

    template <class TensorA>
    CATLASS_DEVICE auto GetTileA(TensorA &tensorA, uint32_t mIndex, uint32_t kIndex, uint32_t mSize, uint32_t kSize)
    {
        if constexpr (tla::detail::isVector<LayoutA>::value) {
            return GetTile(tensorA, tla::MakeCoord(kIndex), tla::MakeShape(kSize));
        } else {
            return GetTile(tensorA, tla::MakeCoord(mIndex, kIndex), tla::MakeShape(mSize, kSize));
        }
    }
};

}  // namespace Catlass::Gemm::Kernel

#endif  // CATLASS_GEMM_KERNEL_QUANT_MATMUL_FULL_LOADA_TLA_HPP
