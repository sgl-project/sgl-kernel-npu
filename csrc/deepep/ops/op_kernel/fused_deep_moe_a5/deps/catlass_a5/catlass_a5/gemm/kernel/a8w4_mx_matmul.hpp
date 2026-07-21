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

#ifndef CATLASS_GEMM_KERNEL_A8W4_MX_MATMUL_HPP
#define CATLASS_GEMM_KERNEL_A8W4_MX_MATMUL_HPP

#include "catlass_a5/arch/resource.hpp"
#include "catlass_a5/catlass.hpp"
#include "catlass_a5/coord.hpp"
#include "catlass_a5/gemm_coord.hpp"
#include "catlass_a5/matrix_coord.hpp"
#include "tla_a5/layout.hpp"
#include "tla_a5/tensor.hpp"

namespace Catlass::Gemm::Kernel {

#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)

// Template for MxMatmul kernel. Compute C = (MxScaleA x A) * (MxScaleB x B) + Bias
template <class BlockMmad_, class BlockPrologue_, class BlockEpilogue_, class BlockScheduler_>
class A8W4MxMatmul
{
public:
    using BlockMmad = BlockMmad_;
    using BlockPrologue = BlockPrologue_;
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
    using ElementBias = typename BlockMmad::ElementBias;
    using ElementPrologueB = typename BlockMmad::ElementPrologueB;
    using ElementAccumulator = typename BlockMmad::ElementAccumulator;
    using LayoutTagL1B = typename BlockMmad::LayoutTagL1B;

    using BlockScheduler = BlockScheduler_;

    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});

    // L1B tile size
    static constexpr uint32_t L1B_TILE_SIZE = L1_TILE_N * L1_TILE_K * SizeOfBits<ElementB>::value / 8;

    // L1B STAGES
    static constexpr uint32_t L1B_STAGES = 2;

    /// Parameters structure
    struct Params {
        // Data members
        GemmCoord problemShape;
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrB;
        LayoutB layoutB;
        GM_ADDR ptrMxScaleA;
        LayoutMxScaleA layoutMxScaleA;
        GM_ADDR ptrMxScaleB;
        LayoutMxScaleB layoutMxScaleB;
        GM_ADDR ptrC;
        LayoutC layoutC;
        GM_ADDR ptrBias;

        // Methods
        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(GemmCoord const &problemShape_, GM_ADDR ptrA_, LayoutA layoutA_, GM_ADDR ptrB_, LayoutB layoutB_,
               GM_ADDR ptrMxScaleA_, LayoutMxScaleA layoutMxScaleA_, GM_ADDR ptrMxScaleB_,
               LayoutMxScaleB layoutMxScaleB_, GM_ADDR ptrC_, LayoutC layoutC_, GM_ADDR ptrBias_ = nullptr)
            : problemShape(problemShape_),
              ptrA(ptrA_),
              layoutA(layoutA_),
              ptrB(ptrB_),
              layoutB(layoutB_),
              ptrMxScaleA(ptrMxScaleA_),
              layoutMxScaleA(layoutMxScaleA_),
              ptrMxScaleB(ptrMxScaleB_),
              layoutMxScaleB(layoutMxScaleB_),
              ptrC(ptrC_),
              layoutC(layoutC_),
              ptrBias(ptrBias_)
        {}
    };

    struct Arguments {
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
        uint8_t *ptrBias{nullptr};
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
        Params params{args.problemShape,   args.ptrA,        args.layoutA,        args.ptrB,
                      args.layoutB,        args.ptrMxScaleA, args.layoutMxScaleA, args.ptrMxScaleB,
                      args.layoutMxScaleB, args.ptrC,        args.layoutC,        args.ptrBias};
        return params;
    }

    // Methods
    CATLASS_DEVICE
    A8W4MxMatmul(uint32_t l1BufAddrStart = 0)
    {
        Arch::Resource<ArchTag> resource;
        if constexpr (tla::detail::isRowMajor<LayoutC>::value) {
            AscendC::SetMMLayoutTransform(true);
        }
        uint32_t l1Offset = l1BufAddrStart;
        for (uint32_t i = 0; i < L1B_STAGES; i++) {
            // Assign L1/L0A/L0B space for each stages
            l1BTensorList[i] = resource.l1Buf.template GetBufferByByte<ElementB>(l1Offset);
            l1Offset += L1B_TILE_SIZE;
            // Assign event ID for each stages
            l1BEventList[i] = i;
            // The event id that needs to be set before the loop
        }
        if ASCEND_IS_AIC {
            AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE, PIPE_MTE1>(AIV_SYNC_AIC_FLAG);
            AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE, PIPE_MTE1>(AIC_SYNC_AIV_FLAG);
        }
    }

    // Destructor
    CATLASS_DEVICE
    ~A8W4MxMatmul()
    {
        if ASCEND_IS_AIV {
            AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE, PIPE_MTE3>(AIV_SYNC_AIC_FLAG);
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

        Arch::Resource<ArchTag> resource;
        BlockMmad blockMmad(resource, L1B_TILE_SIZE * L1B_STAGES);

        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);
        AscendC::GlobalTensor<ElementMxScaleA> gmMxScaleA;
        gmMxScaleA.SetGlobalBuffer((__gm__ ElementMxScaleA *)params.ptrMxScaleA);
        AscendC::GlobalTensor<ElementMxScaleB> gmMxScaleB;
        gmMxScaleB.SetGlobalBuffer((__gm__ ElementMxScaleB *)params.ptrMxScaleB);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrC);

        // Setting L2 Cache to Disable, data reads will bypass L2 Cache.
        if (CeilDiv(params.problemShape.n(), L1_TILE_N) == 1) {
            gmA.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        }

        using GlobalTensorBiasType = std::conditional_t<std::is_void_v<ElementBias>, uint8_t, ElementBias>;
        AscendC::GlobalTensor<GlobalTensorBiasType> gmBias;
        if constexpr (!std::is_void_v<ElementBias>) {
            gmBias.SetGlobalBuffer((__gm__ ElementBias *)params.ptrBias);
        }

        auto layoutBias = tla::MakeLayout(params.problemShape.n());

        // Represent the full tensors
        auto tensorA = tla::MakeTensor(gmA, params.layoutA, Arch::PositionGM{});
        auto tensorMxScaleA = tla::MakeTensor(gmMxScaleA, params.layoutMxScaleA, Arch::PositionGM{});
        auto tensorMxScaleB = tla::MakeTensor(gmMxScaleB, params.layoutMxScaleB, Arch::PositionGM{});
        auto tensorC = tla::MakeTensor(gmC, params.layoutC, Arch::PositionGM{});
        auto tensorBias = tla::MakeTensor(gmBias, layoutBias, Arch::PositionGM{});

        // make L1 TensorB
        auto tensorL1B = tla::MakeTensor(l1BTensorList[l1BListId], L1B_LAYOUT, Arch::PositionL1{});

        for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
            GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);

            // Make tiled views
            auto tensorBlockA = GetTile(tensorA, tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.k() * L1_TILE_K),
                                        tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
            auto tensorBlockMxScaleA =
                GetTile(tensorMxScaleA,
                        tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.k() * L1_TILE_K / MX_SCALE_GROUP_NUM),
                        tla::MakeShape(actualBlockShape.m(), CeilDiv<MX_SCALE_GROUP_NUM>(actualBlockShape.k())));
            auto tensorBlockMxScaleB =
                GetTile(tensorMxScaleB,
                        tla::MakeCoord(blockCoord.k() * L1_TILE_K / MX_SCALE_GROUP_NUM, blockCoord.n() * L1_TILE_N),
                        tla::MakeShape(CeilDiv<MX_SCALE_GROUP_NUM>(actualBlockShape.k()), actualBlockShape.n()));

            auto tensorBlockC = GetTile(tensorC, tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.n() * L1_TILE_N),
                                        tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));

            // Compute block-scoped matrix multiply-add
            if constexpr (std::is_void_v<ElementBias>) {
                blockMmad(tensorBlockA, tensorBlockC, actualBlockShape, tensorL1B, tensorBlockMxScaleA,
                          tensorBlockMxScaleB);
            } else {
                auto tensorBlockBias = GetTile(tensorBias, tla::MakeCoord(blockCoord.n() * L1_TILE_N),
                                               tla::MakeShape(actualBlockShape.n()));
                blockMmad(tensorBlockA, tensorBlockC, actualBlockShape, tensorL1B, tensorBlockMxScaleA,
                          tensorBlockMxScaleB, tensorBlockBias);
            }
        }
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const &params)
    {
        using PrologueParams = typename BlockPrologue::Params;
        PrologueParams prologueParams{L1TileShape{}, params.layoutB, L1_TILE_N, L1_TILE_K, false};

        BlockPrologue blockPrologue(prologueParams);

        // Represent the full gmB
        AscendC::GlobalTensor<ElementPrologueB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementPrologueB *)params.ptrB);

        auto tensorGmB = tla::MakeTensor(gmB, params.layoutB, Arch::PositionGM{});

        BlockScheduler matmulBlockScheduler(params.problemShape, MakeCoord(L1_TILE_M, L1_TILE_N));
        // veccore
        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

        // make L1 TensorB
        auto tensorL1B = tla::MakeTensor(l1BTensorList[l1BListId], L1B_LAYOUT, Arch::PositionL1{});

        for (uint32_t loopIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum(); loopIdx < coreLoops;
             loopIdx += AscendC::GetBlockNum()) {
            GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);
            auto blockTensorB =
                GetTile(tensorGmB, tla::MakeCoord(blockCoord.k() * L1_TILE_K, blockCoord.n() * L1_TILE_N),
                        tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
            blockPrologue(blockTensorB, tensorL1B, actualBlockShape, prologueParams);
        }
    }

    // L1B TensorList
    AscendC::LocalTensor<ElementB> l1BTensorList[L1B_STAGES];
    int32_t l1BEventList[L1B_STAGES];

    // The id of current stage
    uint32_t l1BListId{0};

    constexpr static uint8_t AIC_SYNC_AIV_MODE = 4;
    static constexpr uint16_t AIV_SYNC_AIC_FLAG = 0;
    static constexpr uint16_t AIC_SYNC_AIV_FLAG = 16;

    // L1BLayout
    static constexpr auto L1B_LAYOUT =
        tla::MakeLayout<ElementB, LayoutTagL1B>(tla::Int<L1_TILE_K>{}, tla::Int<L1_TILE_N>{});
};

#endif  // (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)

}  // namespace Catlass::Gemm::Kernel

#endif  // CATLASS_GEMM_KERNEL_A8W4_MX_MATMUL_HPP
