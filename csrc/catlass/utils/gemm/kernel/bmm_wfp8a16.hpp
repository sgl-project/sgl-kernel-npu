/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_KERNEL_BATCH_MATMUL_M_FP8_HPP
#define CATLASS_GEMM_KERNEL_BATCH_MATMUL_M_FP8_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"

inline __gm__ struct OpSystemRunCfg g_opSystemRunCfg {
    Catlass::L2_OFFSET
};

namespace Catlass::Gemm::Kernel {

// Template for grouped matmul kernel. Compute grouped C = A * B
template <class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
class BatchMatmulFP8
{
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using LayoutA = typename BlockMmad::LayoutA;
    using ElementPrologueB = typename BlockMmad::PrologueB::ElementSrc;
    using LayoutPrologueB = typename BlockMmad::PrologueB::LayoutSrc;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutB = typename BlockMmad::LayoutB;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;

    using ElementScale = typename BlockMmad::PrologueB::ElementScale;
    using LayoutScale = typename BlockMmad::PrologueB::LayoutScale;

    // using L1TileShape = typename BlockMmad::L1TileShape;
    using MmadParams = typename BlockMmad::Params;
    using BlockScheduler = BlockScheduler_;

    /// Parameters structure
    struct Params {
        // Data members
        GemmCoord problemShape;
        uint32_t problemCount;
        GM_ADDR ptrA;
        LayoutA layoutA;
        int64_t strideA;
        GM_ADDR ptrPrologueB;
        LayoutPrologueB layoutPrologueB;
        int64_t strideB;
        GM_ADDR ptrC;
        LayoutC layoutC;
        int64_t strideC;
        MmadParams mmadParams;
        GM_ADDR ptrPerGroupScale;
        LayoutScale layoutScale;
        int64_t strideS;
        uint32_t groupSize;
        GM_ADDR ptrWorkspace;

        // Methods
        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(GemmCoord const &problemShape_, uint32_t problemCount_, GM_ADDR ptrA_, LayoutA const &layoutA_,
               int64_t strideA_, GM_ADDR ptrPrologueB_, LayoutPrologueB const &layoutPrologueB_, int64_t strideB_,
               GM_ADDR ptrC_, LayoutC const &layoutC_, int64_t strideC_, MmadParams const &mmadParams_,
               GM_ADDR ptrPerGroupScale_, LayoutScale layoutScale_, int64_t strideS_, uint32_t groupSize_,
               GM_ADDR ptrWorkspace_)
            : problemShape(problemShape_),
              problemCount(problemCount_),
              ptrA(ptrA_),
              layoutA(layoutA_),
              strideA(strideA_),
              ptrPrologueB(ptrPrologueB_),
              layoutPrologueB(layoutPrologueB_),
              strideB(strideB_),
              ptrC(ptrC_),
              layoutC(layoutC_),
              strideC(strideC_),
              mmadParams(mmadParams_),
              ptrPerGroupScale(ptrPerGroupScale_),
              layoutScale(layoutScale_),
              strideS(strideS_),
              groupSize(groupSize_),
              ptrWorkspace(ptrWorkspace_)
        {}
    };
    struct Arguments {
        GemmCoord problemShape;
        uint32_t problemCount;
        GM_ADDR deviceA;
        LayoutA layoutA;
        GM_ADDR devicePrologueB;
        LayoutPrologueB layoutPrologueB;
        GM_ADDR deviceC;
        LayoutC layoutC;
        half deqScalar;
        half deqZeroPoint;
        GM_ADDR deviceScale;
        LayoutScale layoutScale;
        uint32_t groupSize;
        uint32_t aicoreNum;
    };
    static bool CanImplement(const Arguments &args)
    {
        return true;
    }
    static size_t GetWorkspaceSize(const Arguments &args)
    {
        return BlockMmad::STAGES * L1TileShape::K * L1TileShape::N * sizeof(ElementB) * args.aicoreNum;
    }
    static Params ToUnderlyingArguments(const Arguments &args, uint8_t *workspace)
    {
        GemmCoord problemShape = args.problemShape;
        int64_t strideA = problemShape.m() * problemShape.k();
        int64_t strideB = problemShape.k() * problemShape.n();
        int64_t strideC = problemShape.m() * problemShape.n();
        int64_t strideS = ((problemShape.k() + 127) / args.groupSize) * ((problemShape.n() + 127) / args.groupSize);
        Params params{args.problemShape,    args.problemCount, args.deviceA,
                      args.layoutA,         strideA,           args.devicePrologueB,
                      args.layoutPrologueB, strideB,           args.deviceC,
                      args.layoutC,         strideC,           {{}, {args.deqScalar, args.deqZeroPoint}, {}},
                      args.deviceScale,     args.layoutScale,  strideS,
                      args.groupSize,       workspace};
        return params;
    }
    // Methods
    CATLASS_HOST_DEVICE
    BatchMatmulFP8() {}
    // Methods
    CATLASS_HOST_DEVICE
    ~BatchMatmulFP8() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const &params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const &params)
    {
        BlockScheduler matmulBlockScheduler(params.problemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
        uint32_t coreLoops = params.problemCount * matmulBlockScheduler.GetCoreLoops();
        BlockMmad blockMmad(resource, params.mmadParams);

        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrA));
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrC));

        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();
        int64_t gmGroupOffsetA = 0;
        int64_t gmGroupOffsetC = 0;

        // AscendC::printf("AIC coreNum is %d\n", coreNum);
        LayoutB layoutBlockB{L1TileShape::K, L1TileShape::N};
        auto gmOffsetB = coreIdx * layoutBlockB.Capacity() * BlockMmad::STAGES;
        AscendC::GlobalTensor<ElementB> gmBlockB;
        gmBlockB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.ptrWorkspace) + gmOffsetB);

        for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
            // Compute block location
            uint32_t batchIdx = matmulBlockScheduler.GetBatchIdx(loopIdx);
            GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);

            // batchOffset
            int64_t batchOffsetA = batchIdx * params.strideA;
            int64_t batchOffsetC = batchIdx * params.strideC;

            // Compute initial location in logical coordinates
            MatrixCoord offsetA{blockCoord.m() * L1TileShape::M, blockCoord.k() * L1TileShape::K};
            MatrixCoord offsetC{blockCoord.m() * L1TileShape::M, blockCoord.n() * L1TileShape::N};
            int64_t gmOffsetA = params.layoutA.GetOffset(offsetA);
            int64_t gmOffsetC = params.layoutC.GetOffset(offsetC);
            // Compute block-scoped matrix multiply-add
            blockMmad(gmA[batchOffsetA + gmOffsetA], params.layoutA, gmBlockB, layoutBlockB,
                      gmC[batchOffsetC + gmOffsetC], params.layoutC, actualBlockShape);
        }
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const &params)
    {
        BlockScheduler matmulBlockScheduler(params.problemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
        uint32_t coreLoops = params.problemCount * matmulBlockScheduler.GetCoreLoops();
        BlockMmad blockMmad(resource, params.mmadParams);

        uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t coreNum = AscendC::GetBlockNum();
        int64_t gmGroupOffsetB = 0;
        int64_t gmGroupOffsetS = 0;

        LayoutB layoutBlockB{L1TileShape::K, L1TileShape::N};
        auto gmOffsetB = coreIdx * layoutBlockB.Capacity() * BlockMmad::STAGES;
        AscendC::GlobalTensor<ElementB> gmBlockB;
        gmBlockB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.ptrWorkspace) + gmOffsetB);
        AscendC::GlobalTensor<ElementPrologueB> gmB;
        gmB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementPrologueB *>(params.ptrPrologueB));
        AscendC::GlobalTensor<ElementScale> gmScaleFull;
        gmScaleFull.SetGlobalBuffer(reinterpret_cast<__gm__ ElementScale *>(params.ptrPerGroupScale));

        uint32_t groupSize = params.groupSize;
        for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
            // Compute block location
            uint32_t batchIdx = matmulBlockScheduler.GetBatchIdx(loopIdx);
            GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);

            // batchOffset
            int64_t batchOffsetB = batchIdx * params.strideB;
            int64_t batchOffsetS = batchIdx * params.strideS;

            // Compute initial location in logical coordinates
            MatrixCoord offsetB{blockCoord.k() * L1TileShape::K, blockCoord.n() * L1TileShape::N};
            int64_t gmOffsetB = params.layoutPrologueB.GetOffset(offsetB);

            auto layoutBlockPrologueB = params.layoutPrologueB.GetTileLayout(actualBlockShape.GetCoordKN());
            // [FP8] 增加传入scale
            uint32_t kOffset = offsetB[0];
            uint32_t nOffset = offsetB[1];
            uint32_t kAct = actualBlockShape.k();
            uint32_t nAct = actualBlockShape.n();

            uint32_t scaleRowStart = kOffset / groupSize;
            uint32_t scaleRowEnd = (kOffset + kAct - 1) / groupSize;
            uint32_t scaleColStart = nOffset / groupSize;
            uint32_t scaleColEnd = (nOffset + nAct - 1) / groupSize;

            uint32_t scaleTileRows = scaleRowEnd - scaleRowStart + 1;
            uint32_t scaleTileCols = scaleColEnd - scaleColStart + 1;

            MatrixCoord offsetScaleCoord{scaleRowStart, scaleColStart};
            MatrixCoord scaleTileShape{scaleTileRows, scaleTileCols};

            auto gmBlockScale = gmScaleFull[batchOffsetS + params.layoutScale.GetOffset(offsetScaleCoord)];
            auto layoutBlockScale = params.layoutScale.GetTileLayout(scaleTileShape);
            // Compute block-scoped matrix multiply-add

            blockMmad.Prologue(gmB[batchOffsetB + gmOffsetB], layoutBlockPrologueB, gmBlockB, layoutBlockB,
                               gmBlockScale, layoutBlockScale, groupSize, actualBlockShape);
        }
    }

private:
    Arch::Resource<ArchTag> resource;
};

}  // namespace Catlass::Gemm::Kernel

#endif  // CATLASS_GEMM_KERNEL_BATCH_MATMUL_M_FP8_HPP
