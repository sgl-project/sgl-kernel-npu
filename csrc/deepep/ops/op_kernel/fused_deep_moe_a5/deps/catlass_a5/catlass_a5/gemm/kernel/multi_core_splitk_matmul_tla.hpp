/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_KERNEL_MULTI_CORE_SPLITK_MATMUL_TLA_HPP
#define CATLASS_GEMM_KERNEL_MULTI_CORE_SPLITK_MATMUL_TLA_HPP

#include "catlass_a5/catlass.hpp"
#include "catlass_a5/arch/resource.hpp"
#include "catlass_a5/arch/cross_core_sync.hpp"
#include "catlass_a5/layout/layout.hpp"
#include "catlass_a5/gemm_coord.hpp"
#include "catlass_a5/matrix_coord.hpp"
#include "tla_a5/tensor.hpp"
#include "tla_a5/layout.hpp"
#include "catlass_a5/gemm/kernel/splitk_matmul.hpp"

namespace Catlass::Gemm::Kernel {

// Template for Matmul kernel. Compute C = A * B
template <
    class BlockMmad_,
    class BlockEpilogue_,
    class BlockScheduler_
>
class MultiCoreSplitkMatmulTla {
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
    using ElementAccumulator = typename BlockMmad::ElementAccumulator;
    using ElementBias = typename BlockMmad::ElementBias;

    using BlockScheduler = BlockScheduler_;
    
    static_assert(BlockMmad::TileCopy::ReluEnable == false, "Splitk template can not use the Relu with FixPipe !!!");

    static constexpr uint32_t computeLength = 192 * 1024 / sizeof(ElementAccumulator);
    using ReduceAdd = Catlass::Gemm::Kernel::SplitkReduceAdd<ArchTag, ElementAccumulator, ElementC, 1, computeLength>;

    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});

    /// Parameters structure
    struct Params {
        // Data members
        GemmCoord problemShape;
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrB;
        LayoutB layoutB;
        GM_ADDR ptrC;
        LayoutC layoutC;
        GM_ADDR ptrWorkspace;
        uint32_t splitkFactor = 1;
        GM_ADDR ptrBias;

        // Methods
        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(GemmCoord const &problemShape_, GM_ADDR ptrA_, LayoutA layoutA_, GM_ADDR ptrB_,
               LayoutB layoutB_, GM_ADDR ptrC_, LayoutC layoutC_, GM_ADDR ptrWorkspace_,
               uint32_t splitkFactor_, GM_ADDR ptrBias_ = nullptr)
            : problemShape(problemShape_), ptrA(ptrA_), layoutA(layoutA_), ptrB(ptrB_), layoutB(layoutB_),
              ptrC(ptrC_), layoutC(layoutC_), ptrWorkspace(ptrWorkspace_), splitkFactor(splitkFactor_),
              ptrBias(ptrBias_) {}
    };

    struct Arguments {
        GemmCoord problemShape;
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrB;
        LayoutB layoutB;
        GM_ADDR ptrC;
        LayoutC layoutC;
        uint32_t aicCoreNum;
        GM_ADDR ptrBias{nullptr};
    };

    static uint32_t GetSplitkFactor(uint32_t m, uint32_t n, uint32_t k, uint32_t aicCoreNum)
    {
        uint32_t splitkFactor = 2;
        uint32_t blockNum = CeilDiv(m, L1_TILE_M) * CeilDiv(n, L1_TILE_N);
        uint32_t kTileNum = CeilDiv(k, L1_TILE_K);

        if (aicCoreNum / blockNum > 0) {
            splitkFactor = aicCoreNum / blockNum;
        }
        // splitkFactor = std::min(splitkFactor, kTileNum);
        splitkFactor = splitkFactor < kTileNum ? splitkFactor:kTileNum;
        return splitkFactor;
    }

    static bool CanImplement(const Arguments &args)
    {
        return true;
    }

    static size_t GetWorkspaceSize(const Arguments &args)
    {
        uint32_t m = args.problemShape.m();
        uint32_t n = args.problemShape.n();
        uint32_t k = args.problemShape.k();
        uint32_t splitkFactor = GetSplitkFactor(m, n, k, args.aicCoreNum);
        size_t minSpaceSize = 2 * 1024 * 1024; // 2M
        size_t workspaceSize = static_cast<size_t>(m) * n * sizeof(ElementAccumulator) * splitkFactor;
        return minSpaceSize > workspaceSize ? minSpaceSize : workspaceSize;
    }

    static Params ToUnderlyingArguments(const Arguments &args, uint8_t *workspace)
    {
        uint32_t splitkFactor = GetSplitkFactor(
            args.problemShape.m(), args.problemShape.n(), args.problemShape.k(), args.aicCoreNum);
        Params params{
            args.problemShape,
            args.ptrA,
            args.layoutA,
            args.ptrB,
            args.layoutB,
            args.ptrC,
            args.layoutC,
            workspace,
            splitkFactor,
            args.ptrBias
        };
        return params;
    }

    // Methods
    CATLASS_DEVICE
    MultiCoreSplitkMatmulTla() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params const &params);

    /// Executes one Matmul
    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params const &params)
    {
        BlockScheduler matmulBlockScheduler(
            params.problemShape, GemmCoord(L1_TILE_M, L1_TILE_N, L1_TILE_K), params.splitkFactor
        );
        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

        BlockMmad blockMmad(resource);

        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrB);
        AscendC::GlobalTensor<ElementAccumulator> gmW;
        gmW.SetGlobalBuffer((__gm__ ElementAccumulator *)params.ptrWorkspace);

        using GlobalTensorBiasType = std::conditional_t<std::is_void_v<ElementBias>, uint8_t, ElementBias>;
        AscendC::GlobalTensor<GlobalTensorBiasType> gmBias;
        if constexpr (!std::is_void_v<ElementBias>) {
            gmBias.SetGlobalBuffer((__gm__ ElementBias *)params.ptrBias);
        }

        // Matrix A or Matrix B does not have duplicate data reads. Setting L2 Cache to Disable,
        // data reads will bypass L2 Cache.
        if (CeilDiv(params.problemShape.m(), L1_TILE_M) == 1) {
            gmB.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        }
        if (CeilDiv(params.problemShape.n(), L1_TILE_N) == 1) {
            gmA.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        }

        auto layoutC =
            tla::MakeLayout<ElementAccumulator, layout::RowMajor>(params.problemShape.m(), params.problemShape.n());
        auto layoutBias = tla::MakeLayout(params.problemShape.n());

        // Represent the full tensors
        auto tensorA = tla::MakeTensor(gmA, params.layoutA, Arch::PositionGM{});
        auto tensorB = tla::MakeTensor(gmB, params.layoutB, Arch::PositionGM{});
        auto tensorBias = tla::MakeTensor(gmBias, layoutBias, Arch::PositionGM{});

        size_t lenC = static_cast<size_t>(params.problemShape.m()) * params.problemShape.n();

        for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
            // Compute block location
            GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBlockShape =
                matmulBlockScheduler.GetActualBlockShape(blockCoord, matmulBlockScheduler.GetSplitkSliceIdx(loopIdx));

            // Make tiled views
            auto tensorBlockA = GetTile(
                tensorA, tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.k() * L1_TILE_K),
                tla::MakeShape(actualBlockShape.m(), actualBlockShape.k())
            );
            auto tensorBlockB = GetTile(
                tensorB, tla::MakeCoord(blockCoord.k() * L1_TILE_K, blockCoord.n() * L1_TILE_N),
                tla::MakeShape(actualBlockShape.k(), actualBlockShape.n())
            );

            auto tensorC = tla::MakeTensor(
                gmW[lenC * matmulBlockScheduler.GetSplitkSliceIdx(loopIdx)], layoutC, Arch::PositionGM{}
            );
            auto tensorBlockC = GetTile(
                tensorC, tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.n() * L1_TILE_N),
                tla::MakeShape(actualBlockShape.m(), actualBlockShape.n())
            );

            // Compute block-scoped matrix multiply-add
            if constexpr (std::is_void_v<ElementBias>) {
                blockMmad(tensorBlockA, tensorBlockB, tensorBlockC, actualBlockShape);
            } else {
                if (blockCoord.k() == 0) {
                    auto tensorBlockBias = GetTile(
                        tensorBias, tla::MakeCoord(blockCoord.n() * L1_TILE_N), tla::MakeShape(actualBlockShape.n())
                    );
                    blockMmad(tensorBlockA, tensorBlockB, tensorBlockC, actualBlockShape, tensorBlockBias);
                } else {
                    blockMmad(tensorBlockA, tensorBlockB, tensorBlockC, actualBlockShape);
                }
            }
        }

        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(flagAicFinish);

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const &params)
    {
        Catlass::Arch::CrossCoreWaitFlag<0x2, PIPE_MTE2>(flagAicFinish);
        Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE2>();

        AscendC::GlobalTensor<ElementC> gmC;
        AscendC::GlobalTensor<ElementAccumulator> gmWorkspace;
        gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrC));
        gmWorkspace.SetGlobalBuffer(reinterpret_cast<__gm__ ElementAccumulator *>(params.ptrWorkspace));
        ReduceAdd reduceAdd(resource);
        reduceAdd(
            gmC, gmWorkspace,
            static_cast<uint64_t>(params.problemShape.m()) * static_cast<uint64_t>(params.problemShape.n()),
            params.splitkFactor
        );

        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    static constexpr Arch::FlagID FLAG_AIC_FINISH = 0;
    Arch::CrossCoreFlag flagAicFinish{FLAG_AIC_FINISH};
    Arch::Resource<ArchTag> resource;
};

} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_MULTI_CORE_SPLITK_MATMUL_TLA_HPP