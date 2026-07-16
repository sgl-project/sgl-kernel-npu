/*
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef CATLASS_GEMM_KERNEL_GROUPED_MATMUL_SLICE_M_PER_TOKEN_DEQUANT_TLA_HPP
#define CATLASS_GEMM_KERNEL_GROUPED_MATMUL_SLICE_M_PER_TOKEN_DEQUANT_TLA_HPP

#include "catlass_a5/catlass.hpp"
#include "catlass_a5/arch/resource.hpp"
#include "catlass_a5/coord.hpp"
#include "catlass_a5/gemm_coord.hpp"
#include "catlass_a5/matrix_coord.hpp"
#include "tla_a5/layout.hpp"
#include "tla_a5/tensor.hpp"

namespace Catlass::Gemm::Kernel {

template <
    class BlockMmadTla_,
    class BlockEpilogue_,
    class BlockScheduler_,
    class ElementGroupList_
>
class GroupedMatmulSliceMPerTokenTla {
public:
    using BlockMmadTla = BlockMmadTla_;
    using BlockEpilogue = BlockEpilogue_;
    using ArchTag = typename BlockMmadTla::ArchTag;
    using L1TileShape = typename BlockMmadTla::L1TileShape;
    using ElementA = typename BlockMmadTla::ElementA;
    using LayoutA = typename BlockMmadTla::TileCopy::LayoutTagA;
    using ElementB = typename BlockMmadTla::ElementB;
    using LayoutB = typename BlockMmadTla::TileCopy::LayoutTagB;
    using ElementC = typename BlockMmadTla::ElementC;
    using LayoutC = typename BlockMmadTla::TileCopy::LayoutTagC;

    using ElementAccumulator = typename BlockMmadTla::ElementAccumulator;
    using ElementGroupList = ElementGroupList_;
    using BlockScheduler = BlockScheduler_;
    using ProblemShape = tla::Shape<int64_t, int64_t, int64_t, int64_t>;

    using ElementScale = typename BlockEpilogue::ElementScale;
    using LayoutScale = typename BlockEpilogue::LayoutScale;
    using ElementPerToken = typename BlockEpilogue::ElementPerToken;
    using LayoutPerToken = typename BlockEpilogue::LayoutPerToken;
    using ElementDequant = typename BlockEpilogue::ElementDst;
    using LayoutDequant = typename BlockEpilogue::LayoutDst;
    using TileShape = typename BlockEpilogue::TileShape;
    static constexpr uint32_t UB_STAGES = BlockEpilogue::UB_STAGES;
    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});

    struct Params {
        GemmCoord problemShape;
        uint32_t problemCount;
        __gm__ ElementGroupList *ptrGroupList;
        __gm__ ElementA *ptrA;
        __gm__ ElementB *ptrB;
        __gm__ ElementScale *ptrScale;
        __gm__ ElementPerToken *ptrPerToken;
        __gm__ ElementDequant *ptrDequant;

        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(
            GemmCoord problemShape_,
            uint32_t problemCount_, 
            GM_ADDR ptrGroupList_,
            GM_ADDR ptrA_, 
            GM_ADDR ptrB_,
            GM_ADDR ptrScale_, 
            GM_ADDR ptrPerToken_, 
            GM_ADDR ptrDequant_
        ) : problemShape(problemShape_),
            problemCount(problemCount_),
            ptrGroupList(reinterpret_cast<__gm__ ElementGroupList *>(ptrGroupList_)),
            ptrA(reinterpret_cast<__gm__ ElementA *>(ptrA_)),
            ptrB(reinterpret_cast<__gm__ ElementB *>(ptrB_)),
            ptrScale(reinterpret_cast<__gm__ ElementScale *>(ptrScale_)),
            ptrPerToken(reinterpret_cast<__gm__ ElementPerToken *>(ptrPerToken_)),
            ptrDequant(reinterpret_cast<__gm__ ElementDequant *>(ptrDequant_))
        {
        }
    };
    struct Arguments{
        GemmCoord problemShape;
        uint32_t problemCount;
        uint8_t *ptrGroupList;
        uint8_t *ptrA;
        uint8_t *ptrB;
        uint8_t *ptrScale;
        uint8_t *ptrPerToken;
        uint8_t *ptrDequant;
    };
    static bool CanImplement(const Arguments &args)
    {
        return true;
    }
    static size_t GetWorkspaceSize(const Arguments &args)
    {
        return 0;
    }
    static Params ToUnderlyingArguments(const Arguments &args, void* workspace)
    {
        Params params{
            args.problemShape, 
            args.problemCount, 
            args.ptrGroupList,
            args.ptrA,
            args.ptrB,
            args.ptrScale,
            args.ptrPerToken,
            args.ptrDequant
        };
        return params;
    }
    
    CATLASS_HOST_DEVICE
    GroupedMatmulSliceMPerTokenTla() {
#ifdef __DAV_VEC__
        for(int ubListId = 0; ubListId < UB_STAGES; ubListId++){
            AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(AIV_SYNC_AIC_FLAG + ubListId);
        }
#endif
    }

    CATLASS_HOST_DEVICE
    ~GroupedMatmulSliceMPerTokenTla(){
#ifdef __DAV_CUBE__
        for(int ubListId = 0; ubListId < UB_STAGES; ubListId++){
            AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIV_SYNC_AIC_FLAG + ubListId);
            AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIV_SYNC_AIC_FLAG + FLAG_ID_MAX + ubListId);
        }
#endif
    }

    CATLASS_DEVICE
    void operator()(Params const &params)
    {
        uint32_t m = params.problemShape.m();
        uint32_t n = params.problemShape.n();
        uint32_t k = params.problemShape.k();

        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer(params.ptrA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer(params.ptrB);
        AscendC::GlobalTensor<ElementGroupList> groupList;
        groupList.SetGlobalBuffer(params.ptrGroupList);
        AscendC::GlobalTensor<ElementScale> gmScale;
        gmScale.SetGlobalBuffer(params.ptrScale);
        AscendC::GlobalTensor<ElementPerToken> gmPerToken;
        gmPerToken.SetGlobalBuffer(params.ptrPerToken);
        AscendC::GlobalTensor<ElementDequant> gmDequant;
        gmDequant.SetGlobalBuffer(params.ptrDequant);

#ifdef __DAV_CUBE__
        int64_t curBlockIdx = AscendC::GetBlockIdx();

        auto layoutA = tla::MakeLayout<ElementA, LayoutA>(m, k);
        auto gmATensor = tla::MakeTensor(gmA, layoutA, Arch::PositionGM{});
        
        auto layoutB = tla::MakeLayout<ElementB, LayoutB>(k * params.problemCount, n);
        auto gmBTensor = tla::MakeTensor(gmB, layoutB, Arch::PositionGM{});
        
        Arch::Resource<ArchTag> resource;
        BlockMmadTla blockMmadTla(resource);
#endif
        
#ifdef __DAV_VEC__
        int64_t subBlockIdx = AscendC::GetSubBlockIdx();
        int64_t curBlockIdx = AscendC::GetBlockIdx() >> 1;

        auto layoutScale = tla::MakeLayout<ElementScale>(n * params.problemCount);
        auto gmScaleVector = tla::MakeTensor(gmScale, layoutScale, Arch::PositionGM{});
        
        auto layoutPerToken = tla::MakeLayout<ElementPerToken>(m);
        auto gmPerTokenVector = tla::MakeTensor(gmPerToken, layoutPerToken, Arch::PositionGM{});

        auto layoutDequant= tla::MakeLayout<ElementDequant, LayoutDequant>(m, n);
        auto gmDequantTensor = tla::MakeTensor(gmDequant, layoutDequant, Arch::PositionGM{});
        
        Arch::Resource<ArchTag> resource;
        uint32_t ubOffset = 0;
        for(int i = 0; i < UB_STAGES; i++){
            ubGmmResList[i] = resource.ubBuf.template GetBufferByByte<ElementC>(ubOffset);
            ubOffset += TileShape::COUNT / BlockEpilogue::CV_RATIO * sizeof(ElementC);
        }
        BlockEpilogue blockEpilogue(resource, ubOffset);
#endif
        int64_t totalM = 0;
        int64_t totalN = 0;
        int64_t totalK = 0;
        uint32_t ubListId = 0;
        int64_t blockNum = AscendC::GetBlockNum();
        BlockScheduler blockScheduler(curBlockIdx, blockNum);
        for (uint32_t groupIdx = 0; groupIdx < params.problemCount; ++groupIdx) {
            uint32_t currentM = (groupIdx == 0) ? groupList.GetValue(groupIdx) :
                (groupList.GetValue(groupIdx) - groupList.GetValue(groupIdx - 1));
            GemmCoord inGroupProblemShape{currentM, n, k};
            blockScheduler.UpdateGroupParams(inGroupProblemShape);
            if (groupIdx == params.problemCount - 1 && (blockScheduler.GetEndBlockIdx() + 1) <= blockNum / 2) {
                blockScheduler.UpdateTailTile();
            }
            uint32_t coreLoops = blockScheduler.round_;
            for (uint32_t loopIdx = 0; loopIdx < coreLoops; ++loopIdx) {
                bool isLastGroupRound = groupIdx == params.problemCount - 1 && loopIdx == blockScheduler.round_ - 1 &&
                    curBlockIdx <= blockScheduler.GetEndBlockIdx();
                blockScheduler.UpdateMNTileIdx(loopIdx, isLastGroupRound);
                blockScheduler.UpdateBlockShape(loopIdx, isLastGroupRound);
                auto blockShape = blockScheduler.GetBlockShape();
                auto blkElemCoord = blockScheduler.GetBlockCoordByElement();

                uint32_t mCoord = blkElemCoord.m();
                uint32_t blockM = blockShape.m();
                uint32_t nCoord = blkElemCoord.n();
                uint32_t blockN = blockShape.n();
#ifdef __DAV_CUBE__
                uint32_t kCoord = blkElemCoord.k();
                uint32_t blockK = blockShape.k();
                auto gmATile = GetTile(gmATensor,
                                            tla::MakeCoord(totalM + mCoord, kCoord),
                                            tla::MakeShape(blockM, blockK)
                                    );
                auto gmBTile = GetTile(gmBTensor,
                                            tla::MakeCoord(totalK + kCoord, nCoord),
                                            tla::MakeShape(blockK, blockN)
                                    );
                auto layoutGmm = tla::MakeLayout(
                    tla::MakeShape(blockM, blockN),
                    tla::MakeStride(TileShape::COLUMN, tla::Int<1>{})
                );
                auto ubGmmTensor = tla::MakeTensor(ubGmmResList[ubListId], layoutGmm, Arch::PositionUB{});

                AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIV_SYNC_AIC_FLAG + ubListId);
                AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIV_SYNC_AIC_FLAG + FLAG_ID_MAX + ubListId);
                blockMmadTla(
                    gmATile, gmBTile, ubGmmTensor,
                    blockShape
                );
                AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIC_SYNC_AIV_FLAG + ubListId);
                AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIC_SYNC_AIV_FLAG + FLAG_ID_MAX + ubListId);
#endif
#ifdef __DAV_VEC__
                int64_t firstHalfBlockM = (blockM + 1) >> 1;
                int64_t halfBlockM = subBlockIdx ? (blockM - firstHalfBlockM) : firstHalfBlockM;
                auto gmScaleTile = GetTile(gmScaleVector,
                                        tla::MakeCoord(totalN + nCoord),
                                        tla::MakeShape(blockN)
                                    );
                auto gmPerTokenTile = GetTile(gmPerTokenVector,
                                        tla::MakeCoord(totalM + mCoord + firstHalfBlockM * subBlockIdx),
                                        tla::MakeShape(halfBlockM)
                                    );
                auto gmDequantTile = GetTile(gmDequantTensor,
                                        tla::MakeCoord(totalM + mCoord + firstHalfBlockM * subBlockIdx, nCoord),
                                        tla::MakeShape(halfBlockM, blockN)
                                    );
                auto layoutGmm = tla::MakeLayout(
                    tla::MakeShape(halfBlockM, blockN),
                    tla::MakeStride(TileShape::COLUMN, tla::Int<1>{})
                );
                auto ubGmmTensor = tla::MakeTensor(ubGmmResList[ubListId], layoutGmm, Arch::PositionUB{});
                AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_V>(AIC_SYNC_AIV_FLAG + ubListId);
                blockEpilogue(
                    gmDequantTile,
                    ubGmmTensor,
                    gmScaleTile,
                    gmPerTokenTile,
                    ubListId
                );
                AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(AIV_SYNC_AIC_FLAG + ubListId);
#endif
                ubListId = (ubListId + 1 < UB_STAGES) ? (ubListId + 1) : 0;
            }
            totalM += inGroupProblemShape.m();
            totalN += inGroupProblemShape.n();
            totalK += inGroupProblemShape.k();
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }
private:
    AscendC::LocalTensor<ElementC> ubGmmResList[UB_STAGES];
    constexpr static uint16_t AIC_SYNC_AIV_MODE_4 = 4;
    constexpr static uint16_t AIV_SYNC_AIC_FLAG = 6;
    constexpr static uint16_t AIC_SYNC_AIV_FLAG = 8;
    constexpr static uint16_t FLAG_ID_MAX = 16;
};

} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_GROUPED_MATMUL_SLICE_M_PER_TOKEN_DEQUANT_TLA_HPP