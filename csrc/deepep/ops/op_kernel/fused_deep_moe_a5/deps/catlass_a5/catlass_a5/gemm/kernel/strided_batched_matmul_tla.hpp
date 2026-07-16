/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_KERNEL_STRIDED_BATCHED_MATMUL_TLA_HPP
#define CATLASS_GEMM_KERNEL_STRIDED_BATCHED_MATMUL_TLA_HPP

#include "catlass_a5/catlass.hpp"
#include "catlass_a5/arch/resource.hpp"
#include "catlass_a5/coord.hpp"
#include "catlass_a5/gemm_coord.hpp"
#include "catlass_a5/matrix_coord.hpp"
#include "tla_a5/layout.hpp"
#include "tla_a5/tensor.hpp"

namespace Catlass::Gemm::Kernel {

// Template for Strided Batched Matmul kernel. Compute strided batched C = A * B
template <
    class BlockMmad_,
    class BlockEpilogue_,
    class BlockScheduler_
>
class StridedBatchedMatmulTla {
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using LayoutA2D = typename BlockMmad::LayoutA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutB2D = typename BlockMmad::LayoutB;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC2D = typename BlockMmad::LayoutC;
    using ElementAccumulator = typename BlockMmad::ElementAccumulator;

    using LayoutA = tla::MakeBatchedLayout_t<LayoutA2D>;
    using LayoutB = tla::MakeBatchedLayout_t<LayoutB2D>;
    using LayoutC = tla::MakeBatchedLayout_t<LayoutC2D>;

    using BlockScheduler = BlockScheduler_;

    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});

    /// Parameters structure
    struct Params {
        uint32_t batchCount;
        GemmCoord problemShape;
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrB;
        LayoutB layoutB;
        GM_ADDR ptrC;
        LayoutC layoutC;

        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(uint32_t batchCount_, GemmCoord const &problemShape_,
               GM_ADDR ptrA_, LayoutA layoutA_,
               GM_ADDR ptrB_, LayoutB layoutB_,
               GM_ADDR ptrC_, LayoutC layoutC_)
            : batchCount(batchCount_), problemShape(problemShape_),
              ptrA(ptrA_), layoutA(layoutA_),
              ptrB(ptrB_), layoutB(layoutB_),
              ptrC(ptrC_), layoutC(layoutC_) {}
    };

    struct Arguments {
        uint32_t batchCount;
        GemmCoord problemShape;
        uint8_t *ptrA; LayoutA layoutA;
        uint8_t *ptrB; LayoutB layoutB;
        uint8_t *ptrC; LayoutC layoutC;
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
        Params params{args.batchCount, args.problemShape,
                      args.ptrA, args.layoutA,
                      args.ptrB, args.layoutB,
                      args.ptrC, args.layoutC};
        return params;
    }

    CATLASS_DEVICE
    StridedBatchedMatmulTla() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params const &params);

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params const &params) {
        BlockScheduler matmulBlockScheduler(params.problemShape, MakeCoord(L1_TILE_M, L1_TILE_N));
        uint32_t coreLoops = params.batchCount * matmulBlockScheduler.GetCoreLoops();

        Arch::Resource<ArchTag> resource;
        BlockMmad blockMmad(resource);

        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrB);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrC);

        auto tensorA3 = tla::MakeTensor(gmA, params.layoutA, Arch::PositionGM{});
        auto tensorB3 = tla::MakeTensor(gmB, params.layoutB, Arch::PositionGM{});
        auto tensorC3 = tla::MakeTensor(gmC, params.layoutC, Arch::PositionGM{});

        for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
            uint32_t batchIdx = matmulBlockScheduler.GetBatchIdx(loopIdx);
            GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);

            // Slice to rank-2 tensors: Tensor(batchIdx, _, _)
            auto tensorA = tensorA3(batchIdx, tla::_, tla::_);
            auto tensorB = tensorB3(batchIdx, tla::_, tla::_);
            auto tensorC = tensorC3(batchIdx, tla::_, tla::_);

            auto tensorBlockA = tla::TileView(
                tensorA,
                tla::MakeCoord(blockCoord.m(), 0u),
                tla::MakeShape(L1_TILE_M, params.problemShape.k())
            );
            auto tensorBlockB = tla::TileView(
                tensorB,
                tla::MakeCoord(0u, blockCoord.n()),
                tla::MakeShape(params.problemShape.k(), L1_TILE_N)
            );
            auto tensorBlockC = tla::TileView(
                tensorC,
                tla::MakeCoord(blockCoord.m(), blockCoord.n()),
                tla::MakeShape(L1_TILE_M, L1_TILE_N)
            );

            blockMmad(tensorBlockA, tensorBlockB, tensorBlockC);
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const &params) {}
};

} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_STRIDED_BATCHED_MATMUL_TLA_HPP
