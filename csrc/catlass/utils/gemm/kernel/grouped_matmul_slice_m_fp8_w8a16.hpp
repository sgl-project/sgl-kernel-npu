/**

This program is free software, you can redistribute it and/or modify.
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This file is a part of the CANN Open Software.
Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of the
software repository for the full text of the License.
*/
#ifndef CATLASS_GEMM_KERNEL_GROUPED_MATMUL_M_FP8_W8A16_HPP
#define CATLASS_GEMM_KERNEL_GROUPED_MATMUL_M_FP8_W8A16_HPP

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
template <class BlockMmad_, class BlockEpilogue_, class BlockScheduler_, class ElementGroupList_>
class GroupedMatmulSliceMFP8W8A16
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
    using ElementGroupList = ElementGroupList_;
    using BlockScheduler = BlockScheduler_;

    /// Parameters structure
    struct Params {
        // Data members
        GemmCoord problemShape;
        uint32_t problemCount;
        GM_ADDR ptrGroupList;
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrPrologueB;
        LayoutPrologueB layoutPrologueB;
        GM_ADDR ptrC;
        LayoutC layoutC;

        MmadParams mmadParams;
        GM_ADDR ptrPerGroupScale;
        LayoutScale layoutScale;
        uint32_t groupSize;
        GM_ADDR ptrWorkspace;

        // Methods
        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(GemmCoord const &problemShape_, uint32_t problemCount_, GM_ADDR ptrGroupList_, GM_ADDR ptrA_,
               LayoutA const &layoutA_, GM_ADDR ptrPrologueB_, LayoutPrologueB const &layoutPrologueB_, GM_ADDR ptrC_,
               LayoutC const &layoutC_, MmadParams const &mmadParams_, GM_ADDR ptrPerGroupScale_,
               LayoutScale layoutScale_, uint32_t groupSize_, GM_ADDR ptrWorkspace_)
            : problemShape(problemShape_),
              problemCount(problemCount_),
              ptrGroupList(ptrGroupList_),
              ptrA(ptrA_),
              layoutA(layoutA_),
              ptrPrologueB(ptrPrologueB_),
              layoutPrologueB(layoutPrologueB_),
              ptrC(ptrC_),
              layoutC(layoutC_),
              mmadParams(mmadParams_),
              ptrPerGroupScale(ptrPerGroupScale_),
              layoutScale(layoutScale_),
              groupSize(groupSize_),
              ptrWorkspace(ptrWorkspace_)
        {}
    };
    struct Arguments {
        GemmCoord problemShape;
        uint32_t problemCount;
        GM_ADDR ptrGroupList;
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
        Params params{args.problemShape,    args.problemCount,
                      args.ptrGroupList,    args.deviceA,
                      args.layoutA,         args.devicePrologueB,
                      args.layoutPrologueB, args.deviceC,
                      args.layoutC,         {{}, {args.deqScalar, args.deqZeroPoint}, {}},
                      args.deviceScale,     args.layoutScale,
                      args.groupSize,       workspace};
        return params;
    }
    // Methods
    CATLASS_HOST_DEVICE
    GroupedMatmulSliceMFP8W8A16() {}
    // Methods
    CATLASS_HOST_DEVICE
    ~GroupedMatmulSliceMFP8W8A16() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params const &params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const &params)
    {
        BlockScheduler blockScheduler;
        BlockMmad blockMmad(resource, params.mmadParams);

        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrA));
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrC));
        AscendC::GlobalTensor<ElementGroupList> groupList;
        groupList.SetGlobalBuffer(reinterpret_cast<__gm__ ElementGroupList *>(params.ptrGroupList));

        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();
        int64_t gmGroupOffsetA = 0;
        int64_t gmGroupOffsetC = 0;

        // AscendC::printf("AIC coreNum is %d\n", coreNum);
        LayoutB layoutBlockB{L1TileShape::K, L1TileShape::N};
        auto gmOffsetB = coreIdx * layoutBlockB.Capacity() * BlockMmad::STAGES;
        AscendC::GlobalTensor<ElementB> gmBlockB;
        gmBlockB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.ptrWorkspace) + gmOffsetB);

        uint32_t startCoreIdx = 0;
        for (uint32_t groupIdx = 0; groupIdx < params.problemCount; ++groupIdx) {
            uint32_t currentM = (groupIdx == 0) ? groupList.GetValue(groupIdx)
                                                : (groupList.GetValue(groupIdx) - groupList.GetValue(groupIdx - 1));
            // AscendC::printf("AIC currentM: %d\n", currentM);
            GemmCoord inGroupProblemShape{currentM, params.problemShape.n(), params.problemShape.k()};

            LayoutA layoutA = params.layoutA.GetTileLayout(inGroupProblemShape.GetCoordMK());
            LayoutB layoutB = params.layoutPrologueB;
            LayoutC layoutC = params.layoutC.GetTileLayout(inGroupProblemShape.GetCoordMN());

            blockScheduler.Update(inGroupProblemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
            uint32_t coreLoops = blockScheduler.GetCoreLoops();

            // Determine the starting loopIdx of the current core under the current groupIdx
            uint32_t startLoopIdx;
            if (coreIdx < startCoreIdx) {
                startLoopIdx = coreIdx + coreNum - startCoreIdx;
            } else {
                startLoopIdx = coreIdx - startCoreIdx;
            }
            // Loop through the matmul of each groupIdx
            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += coreNum) {
                // Compute block location
                GemmCoord blockCoord = blockScheduler.GetBlockCoord(loopIdx);
                GemmCoord actualBlockShape = blockScheduler.GetActualBlockShape(blockCoord);

                // Compute initial location in logical coordinates
                MatrixCoord offsetA{blockCoord.m() * L1TileShape::M, blockCoord.k() * L1TileShape::K};
                MatrixCoord offsetC{blockCoord.m() * L1TileShape::M, blockCoord.n() * L1TileShape::N};
                int64_t gmOffsetA = layoutA.GetOffset(offsetA);
                int64_t gmOffsetC = layoutC.GetOffset(offsetC);

                // Compute block-scoped matrix multiply-add
                blockMmad(gmA[gmGroupOffsetA + gmOffsetA], layoutA, gmBlockB, layoutBlockB,
                          gmC[gmGroupOffsetC + gmOffsetC], layoutC, actualBlockShape);
            }

            gmGroupOffsetA += inGroupProblemShape.m() * inGroupProblemShape.k();
            gmGroupOffsetC += inGroupProblemShape.m() * inGroupProblemShape.n();

            startCoreIdx = (startCoreIdx + coreLoops) % coreNum;
        }
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const &params)
    {
        BlockScheduler blockScheduler;
        BlockMmad blockMmad(resource, params.mmadParams);

        AscendC::GlobalTensor<ElementGroupList> groupList;
        groupList.SetGlobalBuffer(reinterpret_cast<__gm__ ElementGroupList *>(params.ptrGroupList));

        uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t coreNum = AscendC::GetBlockNum();
        int64_t gmGroupOffsetB = 0;
        int64_t gmGroupOffsetS = 0;

        // AscendC::printf("AIV coreIdx is %d\n", coreIdx);
        LayoutB layoutBlockB{L1TileShape::K, L1TileShape::N};
        auto gmOffsetB = coreIdx * layoutBlockB.Capacity() * BlockMmad::STAGES;
        AscendC::GlobalTensor<ElementB> gmBlockB;
        gmBlockB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.ptrWorkspace) + gmOffsetB);
        uint32_t groupSize = params.groupSize;
        uint32_t startCoreIdx = 0;
        uint32_t bufferIndex = 0;
        for (uint32_t groupIdx = 0; groupIdx < params.problemCount; ++groupIdx) {
            uint32_t currentM = (groupIdx == 0) ? groupList.GetValue(groupIdx)
                                                : (groupList.GetValue(groupIdx) - groupList.GetValue(groupIdx - 1));
            GemmCoord inGroupProblemShape{currentM, params.problemShape.n(), params.problemShape.k()};
            // AscendC::printf("currentM is %d\n", currentM);
            LayoutB layoutPrologueB = params.layoutPrologueB;
            blockScheduler.Update(inGroupProblemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
            uint32_t coreLoops = blockScheduler.GetCoreLoops();

            AscendC::GlobalTensor<ElementPrologueB> gmPrologueB;
            gmPrologueB.SetGlobalBuffer(
                reinterpret_cast<__gm__ ElementPrologueB *>(params.ptrPrologueB + gmGroupOffsetB));
            if (CeilDiv(currentM, L1TileShape::M) == 1) {
                gmPrologueB.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            }

            AscendC::GlobalTensor<ElementScale> gmScaleFull;
            gmScaleFull.SetGlobalBuffer(reinterpret_cast<__gm__ ElementScale *>(params.ptrPerGroupScale +
                                                                                gmGroupOffsetS * sizeof(ElementScale)));
        
            // Determine the starting loopIdx of the current core under the current groupIdx
            uint32_t startLoopIdx;
            if (coreIdx < startCoreIdx) {
                startLoopIdx = coreIdx + coreNum - startCoreIdx;
            } else {
                startLoopIdx = coreIdx - startCoreIdx;
            }
            // Loop through the matmul of each groupIdx
            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += coreNum) {
                // Compute block location
                GemmCoord blockCoord = blockScheduler.GetBlockCoord(loopIdx);
                GemmCoord actualBlockShape = blockScheduler.GetActualBlockShape(blockCoord);

                // Compute initial location in logical coordinates
                MatrixCoord offsetB{blockCoord.k() * L1TileShape::K, blockCoord.n() * L1TileShape::N};
                // ~ auto offsetCoordB = blockIdxCoord.GetCoordKN() * L1TileShape::ToCoordKN(); // return MakeCoord(K, N);

                int64_t gmOffsetB = layoutPrologueB.GetOffset(offsetB);
                auto layoutBlockPrologueB = layoutPrologueB.GetTileLayout(actualBlockShape.GetCoordKN());

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

                auto gmBlockScale = gmScaleFull[params.layoutScale.GetOffset(offsetScaleCoord)];
                auto layoutBlockScale = params.layoutScale.GetTileLayout(scaleTileShape);
            
                // Compute block-scoped matrix multiply-add
                blockMmad.Prologue(gmPrologueB[gmOffsetB], layoutBlockPrologueB, gmBlockB, layoutBlockB, gmBlockScale,
                                   layoutBlockScale, groupSize, actualBlockShape);
            }

            gmGroupOffsetB += inGroupProblemShape.k() * inGroupProblemShape.n();
            gmGroupOffsetS +=
                ((inGroupProblemShape.k() + 127) / groupSize) * ((inGroupProblemShape.n() + 127) / groupSize);
            startCoreIdx = (startCoreIdx + coreLoops) % coreNum;
        }
    }

private:
    Arch::Resource<ArchTag> resource;
};

}  // namespace Catlass::Gemm::Kernel

#endif  // CATLASS_GEMM_KERNEL_GROUPED_MATMUL_M_FP8_W8A16_HPP
