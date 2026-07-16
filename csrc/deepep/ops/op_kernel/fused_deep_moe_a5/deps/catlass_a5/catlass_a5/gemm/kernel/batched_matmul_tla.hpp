/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_KERNEL_BATCHED_MATMUL_TLA_HPP
#define CATLASS_GEMM_KERNEL_BATCHED_MATMUL_TLA_HPP

#include "catlass_a5/catlass.hpp"
#include "catlass_a5/arch/resource.hpp"
#include "catlass_a5/coord.hpp"
#include "catlass_a5/gemm_coord.hpp"
#include "catlass_a5/gemm/dispatch_policy.hpp"
#include "catlass_a5/matrix_coord.hpp"
#include "tla_a5/tensor.hpp"
#include "tla_a5/layout.hpp"

namespace Catlass::Gemm::Kernel {

// Template for Batched Matmul kernel. Compute batched C = A * B
template <
    class BlockMmad_,
    class BlockEpilogue_,
    class BlockScheduler_,
    class Enable = void
>
class BatchedMatmulTla {
    static_assert(DEPENDENT_FALSE<typename BlockMmad_::DispatchPolicy>,
        "BatchedMatmulTla is not implemented for this DispatchPolicy");
};

template <
    class BlockMmad_,
    class BlockEpilogue_,
    class BlockScheduler_
>
class BatchedMatmulTla<BlockMmad_, BlockEpilogue_, BlockScheduler_,
    std::enable_if_t<!std::is_same_v<
        typename BlockMmad_::DispatchPolicy,
        MmadMultiBatch<typename BlockMmad_::ArchTag, BlockMmad_::USE_HF32_MODE, BlockMmad_::L0C_STAGES>>>> {
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

    using BlockScheduler = BlockScheduler_;

    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});

    /// Parameters structure
    struct Params {
        // Data members
        uint32_t batchCount;
        GemmCoord problemShape;
        GM_ADDR ptrA;
        LayoutA layoutA;
        int64_t strideA;
        GM_ADDR ptrB;
        LayoutB layoutB;
        int64_t strideB;
        GM_ADDR ptrC;
        LayoutC layoutC;
        int64_t strideC;

        // Methods
        CATLASS_HOST_DEVICE
        Params()
        {}

        CATLASS_HOST_DEVICE
        Params(uint32_t batchCount_, GemmCoord const &problemShape_,
               GM_ADDR ptrA_, LayoutA layoutA_, int64_t strideA_,
               GM_ADDR ptrB_, LayoutB layoutB_, int64_t strideB_,
               GM_ADDR ptrC_, LayoutC layoutC_, int64_t strideC_)
            : batchCount(batchCount_), problemShape(problemShape_),
              ptrA(ptrA_), layoutA(layoutA_), strideA(strideA_),
              ptrB(ptrB_), layoutB(layoutB_), strideB(strideB_),
              ptrC(ptrC_), layoutC(layoutC_), strideC(strideC_) {}
    };

    struct Arguments {
        uint32_t batchCount;
        GemmCoord problemShape;
        GM_ADDR ptrA; LayoutA layoutA;
        GM_ADDR ptrB; LayoutB layoutB;
        GM_ADDR ptrC; LayoutC layoutC;
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
        GemmCoord problemShape = args.problemShape;
        uint32_t m = problemShape.m();
        uint32_t n = problemShape.n();
        uint32_t k = problemShape.k();
        int64_t strideA = m * k;
        int64_t strideB = k * n;
        int64_t strideC = m * n;
        Params params{args.batchCount,
            problemShape,
            args.ptrA,
            args.layoutA,
            strideA,
            args.ptrB,
            args.layoutB,
            strideB,
            args.ptrC,
            args.layoutC,
            strideC};
        return params;
    }

    // Methods
    CATLASS_DEVICE
    BatchedMatmulTla() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params const &params);

    /// Executes one BatchedMatmul
    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params const &params)
    {
        BlockScheduler matmulBlockScheduler(params.problemShape, MakeCoord(L1_TILE_M, L1_TILE_N));
        uint32_t coreLoops = params.batchCount * matmulBlockScheduler.GetCoreLoops();

        Arch::Resource<ArchTag> resource;
        BlockMmad blockMmad(resource);

        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrB);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrC);

        for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
            // Compute block location
            uint32_t batchIdx = matmulBlockScheduler.GetBatchIdx(loopIdx);
            GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);

            // batchOffset
            int64_t batchOffsetA = batchIdx * params.strideA;
            int64_t batchOffsetB = batchIdx * params.strideB;
            int64_t batchOffsetC = batchIdx * params.strideC;

            // Represent the full tensors
            auto tensorA = tla::MakeTensor(gmA[batchOffsetA], params.layoutA, Arch::PositionGM{});
            auto tensorB = tla::MakeTensor(gmB[batchOffsetB], params.layoutB, Arch::PositionGM{});
            auto tensorC = tla::MakeTensor(gmC[batchOffsetC], params.layoutC, Arch::PositionGM{});

            // Make tiled views
            auto tensorBlockA = GetTile(
                tensorA, tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.k() * L1_TILE_K),
                tla::MakeShape(actualBlockShape.m(), actualBlockShape.k())
            );
            auto tensorBlockB = GetTile(
                tensorB, tla::MakeCoord(blockCoord.k() * L1_TILE_K, blockCoord.n() * L1_TILE_N),
                tla::MakeShape(actualBlockShape.k(), actualBlockShape.n())
            );
            auto tensorBlockC = GetTile(
                tensorC, tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.n() * L1_TILE_N),
                tla::MakeShape(actualBlockShape.m(), actualBlockShape.n())
            );

            // Compute block-scoped matrix multiply-add
            blockMmad(tensorBlockA, tensorBlockB, tensorBlockC, actualBlockShape);
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const &params) {}
};

template <
    class BlockMmad_,
    class BlockEpilogue_,
    class BlockScheduler_
>
class BatchedMatmulTla<BlockMmad_, BlockEpilogue_, BlockScheduler_,
    std::enable_if_t<std::is_same_v<
        typename BlockMmad_::DispatchPolicy,
        MmadMultiBatch<typename BlockMmad_::ArchTag, BlockMmad_::USE_HF32_MODE, BlockMmad_::L0C_STAGES>>>> {
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using L0TileShape = typename BlockMmad::L0TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using LayoutA = typename BlockMmad::LayoutA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutB = typename BlockMmad::LayoutB;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;
    using ElementAccumulator = typename BlockMmad::ElementAccumulator;

    using BlockScheduler = BlockScheduler_;

    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});
    static constexpr uint32_t L0_TILE_M = tla::get<0>(L0TileShape{});
    static constexpr uint32_t L0_TILE_N = tla::get<1>(L0TileShape{});
    static constexpr uint32_t L0_TILE_K = tla::get<2>(L0TileShape{});

    // L1 tile size
    static constexpr uint32_t L1A_TILE_SIZE = L1_TILE_M * L1_TILE_K * sizeof(ElementA);
    static constexpr uint32_t L1B_TILE_SIZE = L1_TILE_N * L1_TILE_K * sizeof(ElementB);
    // L0 tile size
    static constexpr uint32_t L0A_TILE_SIZE = L0_TILE_M * L0_TILE_K * sizeof(ElementA);
    static constexpr uint32_t L0B_TILE_SIZE = L0_TILE_K * L0_TILE_N * sizeof(ElementB);
    static constexpr uint32_t L0C_TILE_SIZE = L1_TILE_M * L1_TILE_N * sizeof(ElementAccumulator);

    /// Parameters structure
    struct Params {
        // Data members
        uint32_t batchCount;
        GemmCoord problemShape;
        GM_ADDR ptrA;
        LayoutA layoutA;
        int64_t strideA;
        GM_ADDR ptrB;
        LayoutB layoutB;
        int64_t strideB;
        GM_ADDR ptrC;
        LayoutC layoutC;
        int64_t strideC;

        // Methods
        CATLASS_HOST_DEVICE
        Params()
        {}

        CATLASS_HOST_DEVICE
        Params(uint32_t batchCount_, GemmCoord const &problemShape_,
               GM_ADDR ptrA_, LayoutA layoutA_, int64_t strideA_,
               GM_ADDR ptrB_, LayoutB layoutB_, int64_t strideB_,
               GM_ADDR ptrC_, LayoutC layoutC_, int64_t strideC_)
            : batchCount(batchCount_), problemShape(problemShape_),
              ptrA(ptrA_), layoutA(layoutA_), strideA(strideA_),
              ptrB(ptrB_), layoutB(layoutB_), strideB(strideB_),
              ptrC(ptrC_), layoutC(layoutC_), strideC(strideC_) {}
    };

    struct Arguments {
        uint32_t batchCount;
        GemmCoord problemShape;
        GM_ADDR ptrA; LayoutA layoutA;
        GM_ADDR ptrB; LayoutB layoutB;
        GM_ADDR ptrC; LayoutC layoutC;
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
        GemmCoord problemShape = args.problemShape;
        uint32_t m = problemShape.m();
        uint32_t n = problemShape.n();
        uint32_t k = problemShape.k();
        int64_t strideA = m * k;
        int64_t strideB = k * n;
        int64_t strideC = m * n;
        Params params{args.batchCount,
            problemShape,
            args.ptrA,
            args.layoutA,
            strideA,
            args.ptrB,
            args.layoutB,
            strideB,
            args.ptrC,
            args.layoutC,
            strideC};
        return params;
    }

    // Methods
    CATLASS_DEVICE
    BatchedMatmulTla() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params const &params);

    /// Executes one BatchedMatmul
    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params const &params)
    {
        uint32_t maxL1Batch = ArchTag::L1_SIZE / BlockMmad::STAGES / (L1A_TILE_SIZE + L1B_TILE_SIZE);
        uint32_t maxL0Batch = AscendC::Std::min(
            AscendC::Std::min(ArchTag::L0A_SIZE / BlockMmad::STAGES / L0A_TILE_SIZE,
                ArchTag::L0B_SIZE / BlockMmad::STAGES / L0B_TILE_SIZE),
            ArchTag::L0C_SIZE / BlockMmad::STAGES / L0C_TILE_SIZE);
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();

        maxL1Batch = AscendC::Std::min(maxL1Batch, AscendC::Std::max(params.batchCount / blockNum, (uint32_t)1));
        maxL0Batch = AscendC::Std::max(maxL0Batch, (uint32_t)1);

        uint32_t coreLoops = CeilDiv(params.batchCount, maxL1Batch);

        Arch::Resource<ArchTag> resource;
        BlockMmad blockMmad(resource, maxL1Batch);
        GemmCoord actualBlockShape = params.problemShape;

        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrB);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrC);

        bool isFirstBlock = true;
        for (uint32_t loopIdx = blockIdx; loopIdx < coreLoops; loopIdx += blockNum) {
            uint32_t actualL1Batch = (loopIdx == coreLoops - 1) ? (params.batchCount - loopIdx * maxL1Batch)
                                                                : maxL1Batch;
            uint32_t batchIdx = loopIdx * maxL1Batch;
            int64_t batchOffsetA = batchIdx * params.strideA;
            int64_t batchOffsetB = batchIdx * params.strideB;
            int64_t batchOffsetC = batchIdx * params.strideC;

            auto tensorA = tla::MakeTensor(gmA[batchOffsetA], params.layoutA, Arch::PositionGM{});
            auto tensorB = tla::MakeTensor(gmB[batchOffsetB], params.layoutB, Arch::PositionGM{});
            auto tensorC = tla::MakeTensor(gmC[batchOffsetC], params.layoutC, Arch::PositionGM{});

            uint32_t nextLoopIdx = loopIdx + blockNum;
            uint32_t nextActualL1Batch = 0;
            int64_t nextBatchOffsetA = 0;
            int64_t nextBatchOffsetB = 0;
            bool hasNextBlock = false;
            if (nextLoopIdx < coreLoops) {
                hasNextBlock = true;
                nextActualL1Batch = (nextLoopIdx == coreLoops - 1) ? (params.batchCount - nextLoopIdx * maxL1Batch)
                                                                   : maxL1Batch;
                uint32_t nextBatchIdx = nextLoopIdx * maxL1Batch;
                nextBatchOffsetA = nextBatchIdx * params.strideA;
                nextBatchOffsetB = nextBatchIdx * params.strideB;
            }
            auto nextTensorA = tla::MakeTensor(gmA[nextBatchOffsetA], params.layoutA, Arch::PositionGM{});
            auto nextTensorB = tla::MakeTensor(gmB[nextBatchOffsetB], params.layoutB, Arch::PositionGM{});

            // Compute block-scoped matrix multiply-add
            blockMmad(
                tensorA, tensorB, tensorC, nextTensorA, nextTensorB, actualBlockShape, actualL1Batch, nextActualL1Batch,
                maxL0Batch, isFirstBlock, hasNextBlock
            );
            isFirstBlock = false;
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const &params) {}
};

} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_BATCHED_MATMUL_TLA_HPP