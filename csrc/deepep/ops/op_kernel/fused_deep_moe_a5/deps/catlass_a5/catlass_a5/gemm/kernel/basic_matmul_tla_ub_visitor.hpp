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

#ifndef CATLASS_GEMM_KERNEL_BASIC_MATMUL_TLA_UB_VISITOR_HPP
#define CATLASS_GEMM_KERNEL_BASIC_MATMUL_TLA_UB_VISITOR_HPP

#include "catlass_a5/catlass.hpp"
#include "catlass_a5/arch/resource.hpp"
#include "catlass_a5/arch/cross_core_sync.hpp"
#include "catlass_a5/layout/layout.hpp"
#include "catlass_a5/gemm_coord.hpp"
#include "catlass_a5/matrix_coord.hpp"
#include "tla_a5/tensor.hpp"
#include "tla_a5/layout.hpp"

namespace Catlass::Gemm::Kernel {

template <class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
class BasicMatmulTlaUbVisitor
{
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using LayoutA = typename BlockMmad::LayoutA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutB = typename BlockMmad::LayoutB;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;
    using ElementBias = typename BlockMmad::ElementBias;

    using BlockEpilogue = BlockEpilogue_;
    using EpilogueParams = typename BlockEpilogue::Params;
    static_assert(std::is_same_v<ArchTag, Arch::Ascend950>, "BasicMatmulTlaUbVisitor only support Ascend950");
    static_assert(BlockEpilogue::USE_UB_WORKSPACE == true,
                  "BlockEpilogue's DispatchPolicy must be EpilogueVisitor<true>");

    using BlockScheduler = BlockScheduler_;

    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});

    struct Params {
        GemmCoord problemShape;
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrB;
        LayoutB layoutB;
        GM_ADDR ptrBias;
        EpilogueParams epilogueParams;

        Params() {}

        Params(GemmCoord const &problemShape_, GM_ADDR ptrA_, LayoutA const &layoutA_, GM_ADDR ptrB_,
               LayoutB const &layoutB_, GM_ADDR ptrBias_, EpilogueParams const &epilogueParams_)
            : problemShape(problemShape_),
              ptrA(ptrA_),
              layoutA(layoutA_),
              ptrB(ptrB_),
              layoutB(layoutB_),
              ptrBias(ptrBias_),
              epilogueParams(epilogueParams_)
        {}
    };

    struct Arguments {
        GemmCoord problemShape;
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrB;
        LayoutB layoutB;
        GM_ADDR ptrC;
        LayoutC layoutC;
        GM_ADDR ptrBias{nullptr};
        typename BlockEpilogue::EVG::Arguments evg_args;
    };

    static bool CanImplement(const Arguments &args)
    {
        return BlockEpilogue::EVG::can_implement(args.problemShape, args.evg_args);
    }

    static size_t GetWorkspaceSize(const Arguments &args)
    {
        return BlockEpilogue::EVG::get_workspace_size(args.problemShape, args.evg_args);
    }

    static Params ToUnderlyingArguments(const Arguments &args, uint8_t *workspace)
    {
        GemmCoord problemShape = args.problemShape;

        uint8_t *evg_workspace = workspace;
        BlockEpilogue::EVG::initialize_workspace(problemShape, args.evg_args, evg_workspace);

        // 转换 EVG Arguments 到 Params
        typename BlockEpilogue::EVG::Params fusion_params =
            BlockEpilogue::EVG::to_underlying_arguments(problemShape, args.evg_args, evg_workspace);

        EpilogueParams epilogueParams{fusion_params};
        Params params{problemShape, args.ptrA, args.layoutA, args.ptrB, args.layoutB, args.ptrBias, epilogueParams};
        return params;
    }

    CATLASS_DEVICE
    BasicMatmulTlaUbVisitor() {}

    CATLASS_DEVICE
    void operator()(Params const &params)
    {
        BlockScheduler matmulBlockScheduler(params.problemShape, MakeCoord(L1_TILE_M, L1_TILE_N));
        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

        BlockMmad blockMmad(resource);
        BlockEpilogue blockEpilogue(resource, params.epilogueParams);

        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrB);

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
            gmBias.SetGlobalBuffer((__gm__ ElementBias *)params.ptrBias);
        }

        // Represent the full tensors
        auto tensorA = tla::MakeTensor(gmA, params.layoutA, Arch::PositionGM{});
        auto tensorB = tla::MakeTensor(gmB, params.layoutB, Arch::PositionGM{});
        auto layoutBias = tla::MakeLayout(params.problemShape.n());
        auto tensorBias = tla::MakeTensor(gmBias, layoutBias, Arch::PositionGM{});

        if ASCEND_IS_AIV {
            Catlass::Arch::CrossCoreSetFlag<0x4, PIPE_V>(flagAivFinish);
        }

        uint32_t aicoreIndex = AscendC::GetBlockIdx();
        if ASCEND_IS_AIV {
            aicoreIndex /= AscendC::GetSubBlockNum();
        }
        uint32_t loopStep = AscendC::GetBlockNum();

        GemmCoord blockShape(L1_TILE_M, L1_TILE_N, L1_TILE_K);
        // 统一循环
        for (uint32_t loopIdx = aicoreIndex; loopIdx < coreLoops; loopIdx += loopStep) {
            GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);

            // 使用 GetMmadUbTensor 获取 tensorBlockC
            auto tensorBlockC = blockEpilogue.GetMmadUbTensor(actualBlockShape);

            if ASCEND_IS_AIC {
                auto tensorBlockA =
                    GetTile(tensorA, tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.k() * L1_TILE_K),
                            tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
                auto tensorBlockB =
                    GetTile(tensorB, tla::MakeCoord(blockCoord.k() * L1_TILE_K, blockCoord.n() * L1_TILE_N),
                            tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));

                // 核间同步：等待 AIV 完成
                Catlass::Arch::CrossCoreWaitFlag<0x4, PIPE_FIX>(flagAivFinish);
                Catlass::Arch::CrossCoreWaitFlag<0x4, PIPE_FIX>(flagAiv1FinishNotifyAic);

                // Compute block-scoped matrix multiply-add
                if constexpr (std::is_void_v<ElementBias>) {
                    blockMmad(tensorBlockA, tensorBlockB, tensorBlockC, actualBlockShape);
                } else {
                    auto tensorBlockBias = GetTile(tensorBias, tla::MakeCoord(blockCoord.n() * L1_TILE_N),
                                                   tla::MakeShape(actualBlockShape.n()));
                    blockMmad(tensorBlockA, tensorBlockB, tensorBlockC, actualBlockShape, tensorBlockBias);
                }
                // 核间同步：通知 AIV 可以开始处理
                Catlass::Arch::CrossCoreSetFlag<0x4, PIPE_FIX>(flagAicFinish);
                Catlass::Arch::CrossCoreSetFlag<0x4, PIPE_FIX>(flagAicFinishNotifyAiv1);
            } else if ASCEND_IS_AIV {
                // AIV 侧：从 UB 读取数据，执行 epilogue
                // 等待 AIC 完成计算
                Catlass::Arch::CrossCoreWaitFlag<0x4, PIPE_V>(flagAicFinish);

                // 执行 epilogue
                blockEpilogue(blockShape, blockCoord, actualBlockShape, tensorBlockC);

                // 通知 AIC 可以继续
                Catlass::Arch::CrossCoreSetFlag<0x4, PIPE_V>(flagAivFinish);
            }
        }

        if ASCEND_IS_AIC {
            Catlass::Arch::CrossCoreWaitFlag<0x4, PIPE_FIX>(flagAivFinish);
            Catlass::Arch::CrossCoreWaitFlag<0x4, PIPE_FIX>(flagAiv1FinishNotifyAic);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    // mode 4 flag id: AIV0 0-10 对应 AIC 0-10, AIV1 0-10 对应 AIC 16-26
    // 因此AIV侧用flagAicFinish(0)，而AIC侧需要使用flagAicFinish(0)和flagAicFinishNotifyAiv1(16)同时跟AIV0和AIV1进行同步，flagAivFinish同理。
    static constexpr Arch::FlagID FLAG_AIC_FINISH = 0;
    static constexpr Arch::FlagID FLAG_AIV_FINISH = 1;
    Arch::CrossCoreFlag flagAicFinish{FLAG_AIC_FINISH};                 // for AIV0/1 <- AIC and AIC -> AIV0
    Arch::CrossCoreFlag flagAicFinishNotifyAiv1{FLAG_AIC_FINISH + 16};  // for AIC -> AIV1
    Arch::CrossCoreFlag flagAivFinish{FLAG_AIV_FINISH};                 // for AIV0/1 -> AIC and AIC <- AIV1
    Arch::CrossCoreFlag flagAiv1FinishNotifyAic{FLAG_AIV_FINISH + 16};  // for AIC <- AIV1
    Arch::Resource<ArchTag> resource;
};

}  // namespace Catlass::Gemm::Kernel

#endif  // CATLASS_GEMM_KERNEL_BASIC_MATMUL_TLA_UB_VISITOR_HPP
