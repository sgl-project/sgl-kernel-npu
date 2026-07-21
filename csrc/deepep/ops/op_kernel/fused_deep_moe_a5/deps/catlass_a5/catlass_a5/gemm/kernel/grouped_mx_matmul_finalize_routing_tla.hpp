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

#ifndef CATLASS_GEMM_KERNEL_GROUPED_MX_MATMUL_FINALIZE_ROUTING_TLA_HPP
#define CATLASS_GEMM_KERNEL_GROUPED_MX_MATMUL_FINALIZE_ROUTING_TLA_HPP

#include "catlass_a5/catlass.hpp"
#include "catlass_a5/arch/resource.hpp"
#include "catlass_a5/coord.hpp"
#include "catlass_a5/gemm_coord.hpp"
#include "catlass_a5/matrix_coord.hpp"
#include "catlass_a5/epilogue/block/block_epilogue_finalize_routing.hpp"
#include "tla_a5/layout.hpp"
#include "tla_a5/tensor.hpp"

namespace Catlass::Gemm::Kernel {

#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)

template <class BlockMmad_, class BlockEpilogue_, class BlockScheduler_, class ElementGroupList_,
          class ElementSharedInput_>
class GroupedMxMatmulFinalizeRoutingTla
{
public:
    using BlockMmad = BlockMmad_;
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
    using ElementAccumulator = typename BlockMmad::ElementAccumulator;
    using ElementRowIndex = int64_t;

    using BlockScheduler = BlockScheduler_;
    using BlockEpilogue = BlockEpilogue_;
    using ElementGroupList = ElementGroupList_;
    using ElementSharedInput = ElementSharedInput_;
    static constexpr uint32_t UB_STAGES = BlockEpilogue::UB_STAGES;
    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});

    struct Params {
        uint32_t aicCoreNum;
        GemmCoord problemShape;
        uint32_t problemCount;
        __gm__ ElementGroupList *ptrGroupList;
        __gm__ ElementA *ptrA;
        LayoutA layoutA;
        __gm__ ElementB *ptrB;
        LayoutB layoutB;
        __gm__ ElementMxScaleA *ptrMxScaleA;
        LayoutMxScaleA layoutMxScaleA;
        __gm__ ElementMxScaleB *ptrMxScaleB;
        LayoutMxScaleB layoutMxScaleB;
        __gm__ ElementC *ptrC;
        LayoutC layoutC;
        __gm__ bfloat16_t *ptrBias;
        __gm__ float *ptrLogit;
        __gm__ int64_t *ptrRowIndex;
        __gm__ ElementSharedInput *ptrSharedInput;
        uint32_t groupListType;
        float sharedInputWeight;
        int64_t sharedInputOffset;
        int64_t batchSize;
        int64_t bsdp;
        __gm__ float *ptrOut;

        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(uint32_t aicCoreNum_, GemmCoord const &problemShape_, uint32_t problemCount_, GM_ADDR ptrGroupList_,
               GM_ADDR ptrA_, LayoutA const &layoutA_, GM_ADDR ptrB_, LayoutB const &layoutB_, GM_ADDR ptrMxScaleA_,
               LayoutMxScaleA layoutMxScaleA_, GM_ADDR ptrMxScaleB_, LayoutMxScaleB layoutMxScaleB_, GM_ADDR ptrC_,
               LayoutC const &layoutC_, GM_ADDR ptrBias_, GM_ADDR ptrLogit_, GM_ADDR ptrRowIndex_,
               GM_ADDR ptrSharedInput_, uint32_t groupListType_, float sharedInputWeight_, int64_t sharedInputOffset_,
               int64_t batchSize_, int64_t bsdp_, GM_ADDR ptrOut_)
            : aicCoreNum(aicCoreNum_),
              problemShape(problemShape_),
              problemCount(problemCount_),
              ptrGroupList(reinterpret_cast<__gm__ ElementGroupList *>(ptrGroupList_)),
              ptrA(reinterpret_cast<__gm__ ElementA *>(ptrA_)),
              layoutA(layoutA_),
              ptrB(reinterpret_cast<__gm__ ElementB *>(ptrB_)),
              layoutB(layoutB_),
              ptrMxScaleA(reinterpret_cast<__gm__ ElementMxScaleA *>(ptrMxScaleA_)),
              layoutMxScaleA(layoutMxScaleA_),
              ptrMxScaleB(reinterpret_cast<__gm__ ElementMxScaleB *>(ptrMxScaleB_)),
              layoutMxScaleB(layoutMxScaleB_),
              ptrC(reinterpret_cast<__gm__ ElementC *>(ptrC_)),
              layoutC(layoutC_),
              ptrBias(reinterpret_cast<__gm__ bfloat16_t *>(ptrBias_)),
              ptrLogit(reinterpret_cast<__gm__ float *>(ptrLogit_)),
              ptrRowIndex(reinterpret_cast<__gm__ int64_t *>(ptrRowIndex_)),
              ptrSharedInput(reinterpret_cast<__gm__ ElementSharedInput *>(ptrSharedInput_)),
              groupListType(groupListType_),
              sharedInputWeight(sharedInputWeight_),
              sharedInputOffset(sharedInputOffset_),
              batchSize(batchSize_),
              bsdp(bsdp_),
              ptrOut(reinterpret_cast<__gm__ float *>(ptrOut_))
        {}
    };

    struct Arguments {
        uint32_t aicCoreNum;
        GemmCoord problemShape;
        uint32_t problemCount;
        uint8_t *ptrGroupList;
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
        uint8_t *ptrBias;
        uint8_t *ptrLogit;
        uint8_t *ptrRowIndex;
        uint8_t *ptrSharedInput;
        uint32_t groupListType;
        float sharedInputWeight;
        int64_t sharedInputOffset;
        int64_t batchSize;
        int64_t bsdp;
        uint8_t *ptrOut;
    };

    static bool CanImplement(const Arguments &args)
    {
        return AscendC::Std::is_one_of_v<ElementA, float8_e4m3_t, float8_e5m2_t> &&
               AscendC::Std::is_one_of_v<ElementB, float8_e4m3_t, float8_e5m2_t> &&
               std::is_same_v<ElementMxScaleA, float8_e8m0_t> && std::is_same_v<ElementMxScaleB, float8_e8m0_t>;
    }

    static size_t GetWorkspaceSize(const Arguments &args)
    {
        int64_t lenWorkspace = static_cast<int64_t>(L1_TILE_M) * L1_TILE_N * args.aicCoreNum;
        int64_t sizeWorkspace = lenWorkspace * sizeof(ElementC);
        return sizeWorkspace;
    }

    static Params ToUnderlyingArguments(const Arguments &args, uint8_t *workspace)
    {
        Params params{args.aicCoreNum,
                      args.problemShape,
                      args.problemCount,
                      args.ptrGroupList,
                      args.ptrA,
                      args.layoutA,
                      args.ptrB,
                      args.layoutB,
                      args.ptrMxScaleA,
                      args.layoutMxScaleA,
                      args.ptrMxScaleB,
                      args.layoutMxScaleB,
                      workspace,
                      args.layoutC,
                      args.ptrBias,
                      args.ptrLogit,
                      args.ptrRowIndex,
                      args.ptrSharedInput,
                      args.groupListType,
                      args.sharedInputWeight,
                      args.sharedInputOffset,
                      args.batchSize,
                      args.bsdp,
                      args.ptrOut};
        return params;
    }

    CATLASS_DEVICE
    GroupedMxMatmulFinalizeRoutingTla() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const &params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const &params)
    {
        AscendC::ICachePreLoad(1);
        BlockMmad blockMmad(resource);

        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);

        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrC);

        AscendC::GlobalTensor<ElementGroupList> groupList;
        groupList.SetGlobalBuffer(params.ptrGroupList);
        int64_t gmGroupOffsetB = 0;
        int64_t gmGroupOffsetMxScaleA = 0;
        int64_t gmGroupOffsetMxScaleB = 0;
        int64_t gmGroupOffsetBias = 0;
        int64_t mxScaleAlignedK =
            static_cast<int64_t>(CeilDiv<MX_BASEK_FACTOR>(params.problemShape.k()) * MX_SCALE_COPY_GROUP_NUM);
        uint32_t coreIdx = AscendC::GetBlockIdx();

        int64_t totalM = 0;

        auto tensorA = tla::MakeTensor(gmA, params.layoutA, Arch::PositionGM{});
        auto tensorC = tla::MakeTensor(gmC[coreIdx * L1_TILE_M * L1_TILE_N], params.layoutC, Arch::PositionGM{});

        int64_t tailOffset = 0;
        uint32_t ubListId = 0;
        ubGmmResList[ubListId] = resource.ubBuf.template GetBufferByByte<ElementC>(0);

        uint32_t lastGroupIdx = 0;
        for (uint32_t groupIdx = 0; groupIdx < params.problemCount; ++groupIdx) {
            uint32_t groupValue = groupList.GetValue(groupIdx);
            uint32_t currentM = groupValue;
            if (params.groupListType == 0) {
                currentM = groupValue - lastGroupIdx;
                lastGroupIdx = groupValue;
            }
            GemmCoord inGroupProblemShape{currentM, params.problemShape.n(), params.problemShape.k()};

            auto layoutMxScaleA = tla::MakeMxScaleLayout<ElementMxScaleA, layout::RowMajor, false>(
                inGroupProblemShape.m(), CeilDiv<MX_SCALE_GROUP_NUM>(inGroupProblemShape.k()));

            BlockScheduler matmulBlockScheduler(inGroupProblemShape, MakeCoord(L1_TILE_M, L1_TILE_N));
            uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

            if (groupIdx == 0) {
                tailOffset = matmulBlockScheduler.GetTailOffset();
            }

            AscendC::GlobalTensor<ElementB> gmB;
            gmB.SetGlobalBuffer(params.ptrB + gmGroupOffsetB);
            AscendC::GlobalTensor<ElementMxScaleB> gmMxScaleB;
            gmMxScaleB.SetGlobalBuffer(params.ptrMxScaleB + gmGroupOffsetMxScaleB);

            AscendC::GlobalTensor<ElementMxScaleA> gmMxScaleA;
            gmMxScaleA.SetGlobalBuffer(params.ptrMxScaleA + gmGroupOffsetMxScaleA);

            using GlobalTensorBiasType = std::conditional_t<std::is_void_v<ElementBias>, uint8_t, ElementBias>;
            AscendC::GlobalTensor<GlobalTensorBiasType> gmBias;
            if constexpr (!std::is_void_v<ElementBias>) {
                gmBias.SetGlobalBuffer((__gm__ ElementBias *)params.ptrBias + gmGroupOffsetBias);
            }
            if (CeilDiv(currentM, L1_TILE_M) == 1) {
                gmB.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            }

            auto tensorB = tla::MakeTensor(gmB, params.layoutB, Arch::PositionGM{});
            auto tensorMxScaleA = tla::MakeTensor(gmMxScaleA, layoutMxScaleA, Arch::PositionGM{});
            auto tensorMxScaleB = tla::MakeTensor(gmMxScaleB, params.layoutMxScaleB, Arch::PositionGM{});
            auto layoutBias = tla::MakeLayout(params.problemShape.n());
            auto tensorBias = tla::MakeTensor(gmBias, layoutBias, Arch::PositionGM{});

            for (uint32_t loopIdx = 0; loopIdx < coreLoops; ++loopIdx) {
                GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
                GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);
                if (actualBlockShape.n() == 0) continue;

                int64_t realNStart = blockCoord.n() * L1_TILE_N;
                if (matmulBlockScheduler.GetIsTail(loopIdx)) {
                    realNStart += tailOffset;
                }

                auto tensorBlockA =
                    GetTile(tensorA, tla::MakeCoord(totalM + blockCoord.m() * L1_TILE_M, blockCoord.k() * L1_TILE_K),
                            tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));

                auto tensorBlockB = GetTile(tensorB, tla::MakeCoord(blockCoord.k() * L1_TILE_K, realNStart),
                                            tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));

                auto layoutWorkSpace = tla::MakeLayout(tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()),
                                                       tla::MakeStride(actualBlockShape.n(), tla::Int<1>{}));
                auto tensorBlockC =
                    tla::MakeTensor(gmC[coreIdx * L1_TILE_M * L1_TILE_N], layoutWorkSpace, Arch::PositionGM{});
                auto tensorBlockMxScaleA =
                    GetTile(tensorMxScaleA,
                            tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.k() * L1_TILE_K / MX_SCALE_GROUP_NUM),
                            tla::MakeShape(actualBlockShape.m(), CeilDiv<MX_SCALE_GROUP_NUM>(actualBlockShape.k())));

                auto tensorBlockMxScaleB =
                    GetTile(tensorMxScaleB, tla::MakeCoord(blockCoord.k() * L1_TILE_K / MX_SCALE_GROUP_NUM, realNStart),
                            tla::MakeShape(CeilDiv<MX_SCALE_GROUP_NUM>(actualBlockShape.k()), actualBlockShape.n()));

                if (groupIdx != 0 || loopIdx != 0) {
                    AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_2>(AIV_SYNC_AIC_FLAG + ubListId);
                }

                if constexpr (!std::is_void_v<ElementBias>) {
                    auto tensorBlockBias =
                        GetTile(tensorBias, tla::MakeCoord(realNStart), tla::MakeShape(actualBlockShape.n()));
                    blockMmad(tensorBlockA, tensorBlockB, tensorBlockC, actualBlockShape, tensorBlockMxScaleA,
                              tensorBlockMxScaleB, tensorBlockBias);
                } else {
                    blockMmad(tensorBlockA, tensorBlockB, tensorBlockC, actualBlockShape, tensorBlockMxScaleA,
                              tensorBlockMxScaleB);
                }
                AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_2, PIPE_FIX>(AIC_SYNC_AIV_FLAG + ubListId);
            }

            totalM += inGroupProblemShape.m();
            gmGroupOffsetB += inGroupProblemShape.k() * inGroupProblemShape.n();
            if constexpr (!std::is_void_v<ElementBias>) {
                gmGroupOffsetBias += inGroupProblemShape.n();
            }
            gmGroupOffsetMxScaleA += inGroupProblemShape.m() * mxScaleAlignedK;
            gmGroupOffsetMxScaleB += mxScaleAlignedK * inGroupProblemShape.n();
        }

        AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_2>(AIV_SYNC_AIC_FLAG + ubListId);
        if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
            blockMmad.template SynchronizeBlock<decltype(tensorC)>();
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const &params)
    {
        AscendC::ICachePreLoad(1);
        BlockScheduler blockScheduler(params.problemShape, MakeCoord(L1_TILE_M, L1_TILE_N));
        OutSplitScheduler outSplitScheduler;
        BlockEpilogue blockEpilogue(resource);

        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();
        int64_t gmGroupOffset = 0;

        AscendC::GlobalTensor<ElementGroupList> groupList;
        groupList.SetGlobalBuffer(params.ptrGroupList);

        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrC));

        AscendC::GlobalTensor<ElementC> gmLogit;
        gmLogit.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrLogit));
        AscendC::GlobalTensor<ElementRowIndex> gmRowIndex;
        gmRowIndex.SetGlobalBuffer(reinterpret_cast<__gm__ ElementRowIndex *>(params.ptrRowIndex));
        AscendC::GlobalTensor<ElementC> gmOut;
        gmOut.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrOut));
        int64_t totalM = 0;
        uint32_t ubListId = 0;
        ubGmmResList[ubListId] = resource.ubBuf.template GetBufferByByte<ElementC>(0);

        int64_t tailOffset = 0;
        blockEpilogue.Update(params.problemShape);

        MatrixCoord outSplitCoord = outSplitScheduler.GetTask(params.batchSize, coreIdx, coreNum);
        blockEpilogue.ClearOutTile(gmOut[static_cast<int64_t>(outSplitCoord.row()) * params.problemShape.n()],
                                   outSplitCoord);

        if constexpr (!std::is_void_v<ElementSharedInput>) {
            AscendC::GlobalTensor<ElementSharedInput> gmSharedInput;
            gmSharedInput.SetGlobalBuffer(reinterpret_cast<__gm__ ElementSharedInput *>(params.ptrSharedInput));
            auto outSharedSplitCoord = outSplitScheduler.GetTask(params.bsdp, coreIdx, coreNum);
            AscendC::SyncAll();
            blockEpilogue.AssignSharedInputTile(
                gmSharedInput[static_cast<int64_t>(outSharedSplitCoord.row()) * params.problemShape.n()],
                gmOut[static_cast<int64_t>(params.sharedInputOffset + outSharedSplitCoord.row()) *
                      params.problemShape.n()],
                outSharedSplitCoord, params.sharedInputWeight);
        }

        AscendC::SyncAll();
        uint32_t lastGroupIdx = 0;
        for (uint32_t groupIdx = 0; groupIdx < params.problemCount; ++groupIdx) {
            uint32_t groupValue = groupList.GetValue(groupIdx);
            uint32_t currentM = groupValue;
            if (params.groupListType == 0) {
                currentM = groupValue - lastGroupIdx;
                lastGroupIdx = groupValue;
            }
            GemmCoord inGroupProblemShape{currentM, params.problemShape.n(), params.problemShape.k()};

            blockScheduler.Update(inGroupProblemShape, MakeCoord(L1_TILE_M, L1_TILE_N));
            blockEpilogue.Update(inGroupProblemShape);
            uint32_t coreLoops = blockScheduler.GetCoreLoops();

            if (groupIdx == 0) {
                tailOffset = blockScheduler.GetTailOffset();
            }
            for (uint32_t loopIdx = 0; loopIdx < coreLoops; ++loopIdx) {
                GemmCoord blockCoord = blockScheduler.GetBlockCoord(loopIdx);
                GemmCoord actualBlockShape = blockScheduler.GetActualBlockShape(blockCoord);
                if (actualBlockShape.n() == 0) {
                    continue;
                }
                int64_t realNStart = blockCoord.n() * L1_TILE_N;
                if (blockScheduler.GetIsTail(loopIdx)) {
                    realNStart += tailOffset;
                }
                uint32_t coreZeroN = CeilDiv(actualBlockShape.n(), 2);
                uint32_t coreOneN = actualBlockShape.n() - coreZeroN;
                uint32_t curN = AscendC::GetSubBlockIdx() == 0 ? coreZeroN : coreOneN;
                uint32_t curNOffset = AscendC::GetSubBlockIdx() == 0 ? 0 : coreZeroN;

                realNStart += curNOffset;
                int64_t gmOffsetC = coreIdx / 2 * L1_TILE_M * L1_TILE_N + curNOffset;
                int64_t gmOffsetLogit = gmGroupOffset + blockCoord.m() * L1_TILE_M;
                GemmCoord workBlockShape = GemmCoord{actualBlockShape.m(), curN, 0};
                GemmCoord gmmBlockShape = GemmCoord{actualBlockShape.m(), actualBlockShape.n(), 0};
                AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_2>(AIC_SYNC_AIV_FLAG + ubListId);

                blockEpilogue.LogitScatterAddTile(ubGmmResList[ubListId], gmC[gmOffsetC], gmLogit[gmOffsetLogit],
                                                  gmRowIndex[gmOffsetLogit], gmOut[realNStart], workBlockShape,
                                                  gmmBlockShape);
                AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_2, PIPE_MTE3>(AIV_SYNC_AIC_FLAG + ubListId);
            }

            totalM += inGroupProblemShape.m();
            gmGroupOffset += inGroupProblemShape.m();
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    Arch::Resource<ArchTag> resource;
    AscendC::LocalTensor<ElementC> ubGmmResList[UB_STAGES];
    constexpr static uint16_t AIC_SYNC_AIV_MODE_2 = 2;
    constexpr static uint16_t AIV_SYNC_AIC_FLAG = 4;
    constexpr static uint16_t AIC_SYNC_AIV_FLAG = 6;
    constexpr static uint16_t FLAG_ID_MAX = 16;
    struct OutSplitScheduler {
        CATLASS_DEVICE
        OutSplitScheduler() {}

        CATLASS_DEVICE
        MatrixCoord GetTask(uint32_t batch, uint32_t coreIdx, uint32_t aicoreNum)
        {
            uint32_t coreNum = aicoreNum * 2;
            uint32_t perCoreRow = batch / coreNum;
            uint32_t remainRow = batch % coreNum;
            uint32_t rowStart = coreIdx * perCoreRow;
            uint32_t curCoreRow = perCoreRow;
            if (coreIdx < remainRow) {
                rowStart += coreIdx;
                curCoreRow += 1;
            } else {
                rowStart += remainRow;
            }
            return MatrixCoord(rowStart, curCoreRow);
        }
    };
};

#endif  // (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)

}  // namespace Catlass::Gemm::Kernel

#endif  // CATLASS_GEMM_KERNEL_GROUPED_MX_MATMUL_FINALIZE_ROUTING_TLA_HPP
