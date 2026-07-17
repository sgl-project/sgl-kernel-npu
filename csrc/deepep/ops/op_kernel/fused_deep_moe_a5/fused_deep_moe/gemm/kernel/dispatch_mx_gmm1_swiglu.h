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

#ifndef CATLASS_GEMM_KERNEL_DISPATCH_MX_GMM1_SWIGLU_H
#define CATLASS_GEMM_KERNEL_DISPATCH_MX_GMM1_SWIGLU_H

#include "ascendc/basic_api/interface/kernel_operator_list_tensor_intf.h"
#include "catlass_a5/catlass.hpp"
#include "catlass_a5/arch/cross_core_sync.hpp"
#include "catlass_a5/arch/resource.hpp"
#include "catlass_a5/coord.hpp"
#include "catlass_a5/detail/callback.hpp"
#include "catlass_a5/gemm_coord.hpp"
#include "catlass_a5/matrix_coord.hpp"
#include "tla_a5/layout.hpp"
#include "tla_a5/tensor.hpp"
#include "catlass_a5/epilogue/tile/tile_swizzle.hpp"
#include "catlass_a5/epilogue/tile/tile_copy.hpp"

#include "dynamic_mx_quant.h"
#include "../../../fused_deep_moe_a5_base.h"
#include "../../fused_deep_moe_utils.h"
#include "../../../fused_deep_moe_a5_tiling.h"

constexpr uint32_t STATE_OFFSET = 512;
constexpr uint64_t WIN_STATE_OFFSET = 512 * 1024;
constexpr uint64_t STATE_WIN_OFFSET = 900 * 1024;
constexpr uint64_t GROUP_TOKEN_NUM_OFFSET = 932 * 1024;
constexpr uint64_t SOFT_SYNC_OFFSET = 964 * 1024;
constexpr uint64_t SHARE_QUANT_SOFT_SYNC_OFFSET = 1000 * 1024;
constexpr uint32_t SELF_STATE_OFFSET = 256 * 1024;
constexpr uint32_t SUM_TMP_TENSOR_SIZE = 1024;
constexpr uint32_t UB_ALIGN = 32;
constexpr uint32_t TOKEN_EXTRA_SPACE = 512;
constexpr uint32_t INT32_COUNT_PER_BLOCK = 8;
constexpr int64_t REDUCE_SUM_WORK_SIZE = 4096; // 最大支持64k-fp32累加
constexpr int32_t SUB_AIV_NUM = 2;
constexpr int32_t ODD_EVEN_BASE = 2;
constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t GATHER_SECOND_NUM = 2;
#define OPT_RANK_OFFSET 512

#define CEIL_UP(x) ((x + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN)
#define CEIL(x, y) (((x) + (y - 1)) / (y))
#define UB_BLOCK_SIZE (32)
#define TOKEN_FLAG_1 (0x55555555)
#define TOKEN_FLAG_2 (0x33333333)
#define V_TO_C_FLAG_1 (0x03030303)
#define V_TO_C_FLAG_2 (0x05050505)
#define CV_FLAG_INDEX 0
#define GROUP_ID_INDEX 1
#define PRE_COUNT_INDEX 2
#define SELF_COUNT_INDEX 3
#define TOTAL_COUNT_INDEX 4
#define GROUP_TOKEN_COUNT 3  // equal to SELF_COUNT_INDEX
#define GROUP_INFO_SIZE 32

using namespace Cam;
namespace Catlass::Gemm::Kernel {

#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)

template <typename ElementMx>
CATLASS_DEVICE constexpr uint32_t MxCount2Byte(uint32_t count)
{
    if constexpr (AscendC::Std::is_one_of_v<ElementMx, float4_e2m1x2_t, float4_e1m2x2_t>) {
        return (count + 1U) / 2U;
    }
    return count * sizeof(ElementMx);
}

template <typename ElementMx>
CATLASS_DEVICE constexpr uint32_t MxByte2Count(uint32_t byte)
{
    if constexpr (AscendC::Std::is_one_of_v<ElementMx, float4_e2m1x2_t, float4_e1m2x2_t>) {
        return byte * 2U;
    }
    return byte / sizeof(ElementMx);
}

// Template for GroupedMxMatmulSliceM kernel
template <
    TemplateMC2TypeClass,
    class BlockMmad_,
    class BlockEpilogue_,
    class BlockScheduler_,
    class ElementGroupList_
>
class DispatchMxGmm1Swiglu {
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
    using ElementAccumulator = typename BlockMmad::ElementAccumulator;

    using BlockEpilogue = BlockEpilogue_;
    using EpilogueParams = typename BlockEpilogue::Params;

    using ElementGroupList = ElementGroupList_;
    using BlockScheduler = BlockScheduler_;
    using XType = ExpandXType;

    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});

    /// Parameters structure
    struct Params {
        // Data members
        GemmCoord problemShape, shareProblemShape;
        uint32_t problemCount;
        __gm__ ElementGroupList *ptrGroupList;
        __gm__ ElementA *ptrA, *ptrShareA;
        LayoutA layoutA;
        __gm__ ElementB *ptrB, *ptrShareB;
        LayoutB layoutB, layoutShareB;
        __gm__ ElementMxScaleA *ptrMxScaleA, *ptrShareMxScaleA;
        LayoutMxScaleA layoutMxScaleA;
        __gm__ ElementMxScaleB *ptrMxScaleB, *ptrShareMxScaleB;
        LayoutMxScaleB layoutMxScaleB, layoutShareMxScaleB;
        __gm__ ElementC *ptrC, *ptrShareC;
        LayoutC layoutC, layoutShareC;

        __gm__ ElementC *gmSwigluOut;
        __gm__ ElementC *gmShareSwigluOut;
        __gm__ ElementA *ptrX2, *ptrShareX2;
        __gm__ ElementMxScaleA *gmX2Scale, *gmShareX2Scale;

        GM_ADDR gmX;
        GM_ADDR gmExpertIds;
        GM_ADDR gmXActiveMask;
        GM_ADDR gmMoeSmoothScales;
        GM_ADDR gmShareSmoothScales;
        GM_ADDR gmExpandIdx;
        GM_ADDR gmEpSendCount;
        GM_ADDR gmExpertTokenNums;

        uint32_t epRankSize;
        uint32_t epRankId;
        uint32_t moeExpertNum;
        uint32_t moeExpertNumPerRank;
        uint32_t quantMode;
        uint32_t globalBs;
        uint32_t bs;
        uint32_t topK;
        uint32_t tokenLen;
        uint32_t shareN;
        // Methods
        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(
            GemmCoord const &problemShape_, uint32_t problemCount_, GM_ADDR ptrGroupList_,
            GM_ADDR ptrA_, LayoutA const &layoutA_,
            GM_ADDR ptrB_, LayoutB const &layoutB_,
            GM_ADDR ptrMxScaleA_, LayoutMxScaleA layoutMxScaleA_,
            GM_ADDR ptrMxScaleB_, LayoutMxScaleB layoutMxScaleB_,
            GM_ADDR ptrC_, LayoutC const &layoutC_,
            GM_ADDR gmSwigluOut_,GM_ADDR ptrX2_, GM_ADDR gmX2Scale_,
            GemmCoord const &shareProblemShape_,
            GM_ADDR ptrShareA_,
            GM_ADDR ptrShareB_, LayoutB const &layoutShareB_,
            GM_ADDR ptrShareMxScaleA_,
            GM_ADDR ptrShareMxScaleB_, LayoutMxScaleB layoutShareMxScaleB_,
            GM_ADDR ptrShareC_, LayoutC const &layoutShareC_,
            GM_ADDR gmShareSwigluOut_,GM_ADDR ptrShareX2_, GM_ADDR gmShareX2Scale_,
            GM_ADDR gmX_, GM_ADDR gmExpertIds_, GM_ADDR gmXActiveMask_, GM_ADDR gmMoeSmoothScales_, GM_ADDR gmShareSmoothScales_, 
            GM_ADDR gmExpandIdx_, GM_ADDR gmEpSendCount_, GM_ADDR gmExpertTokenNums_, const FusedDeepMoeInfo &fusedDeepMoeInfo
        ) : problemShape(problemShape_),
            problemCount(problemCount_), ptrGroupList(reinterpret_cast<__gm__ ElementGroupList *>(ptrGroupList_)),
            ptrA(reinterpret_cast<__gm__ ElementA *>(ptrA_)), layoutA(layoutA_),
            ptrB(reinterpret_cast<__gm__ ElementB *>(ptrB_)), layoutB(layoutB_),
            ptrMxScaleA(reinterpret_cast<__gm__ ElementMxScaleA *>(ptrMxScaleA_)), layoutMxScaleA(layoutMxScaleA_),
            ptrMxScaleB(reinterpret_cast<__gm__ ElementMxScaleB *>(ptrMxScaleB_)), layoutMxScaleB(layoutMxScaleB_),
            ptrC(reinterpret_cast<__gm__ ElementC *>(ptrC_)), layoutC(layoutC_),
            gmSwigluOut(reinterpret_cast<__gm__ ElementC *>(gmSwigluOut_)), ptrX2(reinterpret_cast<__gm__ ElementA *>(ptrX2_)), gmX2Scale(reinterpret_cast<__gm__ ElementMxScaleA *>(gmX2Scale_)),
            shareProblemShape(shareProblemShape_),
            ptrShareA(reinterpret_cast<__gm__ ElementA *>(ptrShareA_)),
            ptrShareB(reinterpret_cast<__gm__ ElementB *>(ptrShareB_)), layoutShareB(layoutShareB_),
            ptrShareMxScaleA(reinterpret_cast<__gm__ ElementMxScaleA *>(ptrShareMxScaleA_)),
            ptrShareMxScaleB(reinterpret_cast<__gm__ ElementMxScaleB *>(ptrShareMxScaleB_)), layoutShareMxScaleB(layoutShareMxScaleB_),
            ptrShareC(reinterpret_cast<__gm__ ElementC *>(ptrShareC_)), layoutShareC(layoutShareC_),
            gmShareSwigluOut(reinterpret_cast<__gm__ ElementC *>(gmShareSwigluOut_)), ptrShareX2(reinterpret_cast<__gm__ ElementA *>(ptrShareX2_)), gmShareX2Scale(reinterpret_cast<__gm__ ElementMxScaleA *>(gmShareX2Scale_)),
            gmX(gmX_), gmExpertIds(gmExpertIds_), gmXActiveMask(gmXActiveMask_),
            gmMoeSmoothScales(gmMoeSmoothScales_), gmShareSmoothScales(gmShareSmoothScales_),
            gmExpandIdx(gmExpandIdx_), gmEpSendCount(gmEpSendCount_), gmExpertTokenNums(gmExpertTokenNums_),
            epRankSize(fusedDeepMoeInfo.epRankSize), epRankId(fusedDeepMoeInfo.epRankId),
            moeExpertNum(fusedDeepMoeInfo.moeExpertNum), moeExpertNumPerRank(fusedDeepMoeInfo.moeExpertNumPerRank),
            quantMode(fusedDeepMoeInfo.quantMode), globalBs(fusedDeepMoeInfo.globalBs), bs(fusedDeepMoeInfo.bs),
            topK(fusedDeepMoeInfo.k), tokenLen(fusedDeepMoeInfo.h), shareN(fusedDeepMoeInfo.shareGmm1HLen)
        {}
    };

    // Methods
    CATLASS_DEVICE
    DispatchMxGmm1Swiglu() {
        aiCoreGroupNum = AscendC::GetBlockNum();
        subBlockNum = AscendC::GetSubBlockNum();
        aiCoreGroupIdx = AscendC::GetBlockIdx() / subBlockNum;
        aicNum = aiCoreGroupNum;
        aivNum = aiCoreGroupNum * SUB_AIV_NUM; // 1C2V
        if ASCEND_IS_AIC {
            aicIdx = AscendC::GetBlockIdx();
        }
        if ASCEND_IS_AIV {
            aivIdx = AscendC::GetBlockIdx();
        }

        winContext_ = (__gm__ Mc2Kernel::HcclOpParam *)AscendC::GetHcclContext<AscendC::HCCL_GROUP_ID_0>();
        statusDataSpaceGm = Mc2Kernel::GetStatusDataSpaceGm(winContext_);

        if ASCEND_IS_AIV {
            compCoreNum = aiCoreGroupNum;
            isCompCore = true;
            compCoreIdx = aiCoreGroupIdx;
        }
        if constexpr ((EXEC_FLAG & EXEC_FLAG_DEEP_FUSE) == 0) {
            return ;
        }

        recvCoreNum = aiCoreGroupNum;
        sendCoreNum = aiCoreGroupNum;
        if constexpr (EXEC_FLAG & EXEC_FLAG_SHARED_EXPERT) {
            shareQuantCoreNum = recvCoreNum;
        }
        AscendC::GlobalTensor<int32_t> selfDataStatusTensor;
        selfDataStatusTensor.SetGlobalBuffer((__gm__ int32_t *)(statusDataSpaceGm + STATE_WIN_OFFSET));
        if ASCEND_IS_AIC {
            aicStateGlobalCoreIdx = aivNum + aicIdx;
            cvDataState = FlushAndSpinValue<int32_t>(selfDataStatusTensor, aicStateGlobalCoreIdx * UB_ALIGN);
            vToCFlag = (cvDataState == 0) ? V_TO_C_FLAG_1 : V_TO_C_FLAG_2;
        }
        if ASCEND_IS_AIV {
            isRecvCore = ((aivIdx % ODD_EVEN_BASE) == 0);
            recvCoreIdx = aiCoreGroupIdx;
            isSendCore = ((aivIdx % ODD_EVEN_BASE) == 1);
            sendCoreIdx = aiCoreGroupIdx;
            if constexpr (EXEC_FLAG & EXEC_FLAG_SHARED_EXPERT) {
                isShareQuantCore = isRecvCore;
                shareQuantCoreIdx = recvCoreIdx;
            }
            aivStateGlobalCoreIdx = aivNum + aicNum + aivIdx;

            dataState = FlushAndSpinValue<int32_t>(selfDataStatusTensor, aivIdx * UB_ALIGN);
            cvDataState = FlushAndSpinValue<int32_t>(selfDataStatusTensor, aivStateGlobalCoreIdx * UB_ALIGN);
            vToCFlag = (cvDataState == 0) ? V_TO_C_FLAG_1 : V_TO_C_FLAG_2;
        }
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params const &params);
    __aicore__ inline void WaitGroupTokenNumReady(AscendC::GlobalTensor<int32_t>& groupTokenNumStateTensor,
                                                      uint32_t expected)
    {
        while (true) {
            if (FlushAndGetValue<int32_t>(groupTokenNumStateTensor, 0) == static_cast<int32_t>(expected)) {
                break;
            }
            SPIN_WAIT_CYCLES();
        }
    }

    __aicore__ inline GM_ADDR GetWindStateAddrByRankId(int64_t rankId)
    {
        return Mc2Kernel::GetBaseWindStateAddrByRankId(winContext_, rankId, epRankId) + dataState * WIN_STATE_OFFSET;
    }

    __aicore__ inline GM_ADDR GetWindAddrByRankId(int64_t rankId)
    {
        return Mc2Kernel::GetBaseWindAddrByRankId(winContext_, rankId, epRankId) + winDataSizeOffset + rankId * OPT_RANK_OFFSET;
    }

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params const &params)
    {
        AscendC::ICachePreLoad(1);
        // uint32_t actualRecvCoreNumPerGroup = recvCoreNum < params.epRankSize ? recvCoreNum : params.epRankSize;
        uint32_t actualRecvCoreNumPerGroup = recvCoreNum;

        BlockScheduler blockScheduler;
        BlockMmad blockMmad(resource);

        AscendC::GlobalTensor<ElementA> gmA;
        AscendC::GlobalTensor<ElementMxScaleA> gmMxScaleA;
        AscendC::GlobalTensor<ElementB> gmB;
        AscendC::GlobalTensor<ElementMxScaleB> gmMxScaleB;
        AscendC::GlobalTensor<ElementC> gmC;

        uint32_t currentM = 0;
        uint32_t startCoreIdx = 0;
        aicSetFunc = {reinterpret_cast<__gm__ int32_t *>(statusDataSpaceGm + SOFT_SYNC_OFFSET), static_cast<int32_t>(AscendC::GetBlockIdx())};
        Callback callbackAfterFixpipe = MakeCallback(&aicSetFunc);
        if constexpr (EXEC_FLAG & EXEC_FLAG_SHARED_EXPERT) {
            currentM = params.bs;
            gmA.SetGlobalBuffer(params.ptrShareA);
            gmMxScaleA.SetGlobalBuffer(params.ptrShareMxScaleA);
            gmB.SetGlobalBuffer(params.ptrShareB);
            gmMxScaleB.SetGlobalBuffer(params.ptrShareMxScaleB);
            gmC.SetGlobalBuffer(params.ptrShareC);
            GemmCoord inGroupProblemShape{currentM, params.shareProblemShape.n(), params.shareProblemShape.k()};

            BlockScheduler matmulBlockScheduler(inGroupProblemShape, MakeCoord(L1_TILE_M, L1_TILE_N));
            uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

            if (CeilDiv(currentM, L1_TILE_M) == 1) {
                gmB.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            } else {
                gmB.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_NORMAL);
            }

            uint32_t startLoopIdx;
            if (aicIdx < startCoreIdx) {
                startLoopIdx = aicIdx + aicNum - startCoreIdx;
            } else {
                startLoopIdx = aicIdx - startCoreIdx;
            }

            auto tensorA = tla::MakeTensor(gmA, params.layoutA, Arch::PositionGM{});
            auto tensorMxScaleA = tla::MakeTensor(gmMxScaleA, params.layoutMxScaleA, Arch::PositionGM{});
            auto tensorB = tla::MakeTensor(gmB, params.layoutShareB, Arch::PositionGM{});
            auto tensorMxScaleB = tla::MakeTensor(gmMxScaleB, params.layoutShareMxScaleB, Arch::PositionGM{});
            auto tensorC = tla::MakeTensor(gmC, params.layoutShareC, Arch::PositionGM{});
            if constexpr (EXEC_FLAG & EXEC_FLAG_DEEP_FUSE) {
                // wait AIV quantize needed tokens
                AscendC::GlobalTensor<int32_t> shareQuantTokenStateTensor;
                uint32_t waitFlagCount = params.bs < shareQuantCoreNum ? params.bs : shareQuantCoreNum;
                shareQuantTokenStateTensor.SetGlobalBuffer((__gm__ int32_t*)(
                    statusDataSpaceGm + SHARE_QUANT_SOFT_SYNC_OFFSET));
                uint32_t expected = waitFlagCount * vToCFlag;
                WaitGroupTokenNumReady(shareQuantTokenStateTensor, expected);
            }
            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += aicNum) {
                GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
                GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);

                auto tensorBlockA = GetTile(tensorA,
                    tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.k() * L1_TILE_K),
                    tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));

                auto tensorBlockB = GetTile(tensorB,
                    tla::MakeCoord(blockCoord.k() * L1_TILE_K, blockCoord.n() * L1_TILE_N),
                    tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));

                auto tensorBlockC = GetTile(tensorC,
                    tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.n() * L1_TILE_N),
                    tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));

                auto tensorBlockMxScaleA = GetTile(
                    tensorMxScaleA,
                    tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.k() * L1_TILE_K / MX_SCALE_GROUP_NUM),
                    tla::MakeShape(actualBlockShape.m(), CeilDiv<MX_SCALE_GROUP_NUM>(actualBlockShape.k())));

                auto tensorBlockMxScaleB = GetTile(
                    tensorMxScaleB,
                    tla::MakeCoord(blockCoord.k() * L1_TILE_K / MX_SCALE_GROUP_NUM, blockCoord.n() * L1_TILE_N),
                    tla::MakeShape(CeilDiv<MX_SCALE_GROUP_NUM>(actualBlockShape.k()), actualBlockShape.n()));

                blockMmad(tensorBlockA, tensorBlockB, tensorBlockC, actualBlockShape,
                    tensorBlockMxScaleA, tensorBlockMxScaleB);
                callbackAfterFixpipe();
            }

            startCoreIdx = (startCoreIdx + coreLoops) % aicNum;
        }
        {
            AscendC::GlobalTensor<ElementGroupList> groupList;
            groupList.SetGlobalBuffer(params.ptrGroupList);
            gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);
            gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrC);
            AscendC::ListTensorDesc gmBlistTensorDesc(reinterpret_cast<__gm__ void *>(params.ptrB));
            AscendC::ListTensorDesc gmBScalelistTensorDesc(reinterpret_cast<__gm__ void *>(params.ptrMxScaleB));

            int64_t gmGroupOffsetB = 0;
            int64_t gmGroupOffsetMxScaleA = 0;
            int64_t gmGroupOffsetMxScaleB = 0;
            int64_t mxScaleAlignedK = static_cast<int64_t>(CeilDiv<MX_BASEK_FACTOR>(params.problemShape.k()) * MX_SCALE_COPY_GROUP_NUM);

            int64_t totalM = 0;
            auto tensorA = tla::MakeTensor(gmA, params.layoutA, Arch::PositionGM{});
            auto tensorC = tla::MakeTensor(gmC, params.layoutC, Arch::PositionGM{});

            AscendC::GlobalTensor<int32_t> groupTokenNumStateTensor;
            for (uint32_t groupIdx = 0; groupIdx < params.problemCount; ++groupIdx) {
                gmMxScaleA.SetGlobalBuffer(params.ptrMxScaleA + gmGroupOffsetMxScaleA);
                if constexpr (EXEC_FLAG & EXEC_FLAG_TENSOR_LIST) {
                    gmB.SetGlobalBuffer(gmBlistTensorDesc.GetDataPtr<ElementB>(groupIdx));
                    gmMxScaleB.SetGlobalBuffer(gmBScalelistTensorDesc.GetDataPtr<ElementMxScaleB>(groupIdx));
                } else {
                    gmB.SetGlobalBuffer(gmBlistTensorDesc.GetDataPtr<ElementB>(0) + gmGroupOffsetB);
                    gmMxScaleB.SetGlobalBuffer(gmBScalelistTensorDesc.GetDataPtr<ElementMxScaleB>(0) + gmGroupOffsetMxScaleB);
                }
                if constexpr (EXEC_FLAG & EXEC_FLAG_DEEP_FUSE) {
                    groupTokenNumStateTensor.SetGlobalBuffer((__gm__ int32_t *)(
                        statusDataSpaceGm + GROUP_TOKEN_NUM_OFFSET) + groupIdx * GROUP_INFO_SIZE);
                    // wait AIV recv needed tokens
                    uint32_t expected = actualRecvCoreNumPerGroup * vToCFlag;
                    WaitGroupTokenNumReady(groupTokenNumStateTensor, expected);
                    callbackAfterFixpipe();
                    currentM = groupTokenNumStateTensor.GetValue(GROUP_TOKEN_COUNT);
                } else {
                    currentM = (groupIdx == 0) ? groupList.GetValue(groupIdx)
                                                    : (groupList.GetValue(groupIdx) - groupList.GetValue(groupIdx - 1));
                }
                GemmCoord inGroupProblemShape{currentM, params.problemShape.n(), params.problemShape.k()};

                BlockScheduler matmulBlockScheduler(inGroupProblemShape, MakeCoord(L1_TILE_M, L1_TILE_N));
                uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();


                if (CeilDiv(currentM, L1_TILE_M) == 1) {
                    gmB.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
                } else {
                    gmB.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_NORMAL);
                }

                uint32_t startLoopIdx;
                if (aicIdx < startCoreIdx) {
                    startLoopIdx = aicIdx + aicNum - startCoreIdx;
                } else {
                    startLoopIdx = aicIdx - startCoreIdx;
                }

                auto tensorB = tla::MakeTensor(gmB, params.layoutB, Arch::PositionGM{});
                auto tensorMxScaleA = tla::MakeTensor(gmMxScaleA, params.layoutMxScaleA, Arch::PositionGM{});
                auto tensorMxScaleB = tla::MakeTensor(gmMxScaleB, params.layoutMxScaleB, Arch::PositionGM{});

                for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += aicNum) {
                    GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
                    GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);

                    auto tensorBlockA = GetTile(tensorA,
                        tla::MakeCoord(totalM + blockCoord.m() * L1_TILE_M, blockCoord.k() * L1_TILE_K),
                        tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));

                    auto tensorBlockB = GetTile(tensorB,
                        tla::MakeCoord(blockCoord.k() * L1_TILE_K, blockCoord.n() * L1_TILE_N),
                        tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));

                    auto tensorBlockC = GetTile(tensorC,
                        tla::MakeCoord(totalM + blockCoord.m() * L1_TILE_M, blockCoord.n() * L1_TILE_N),
                        tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));

                    auto tensorBlockMxScaleA = GetTile(
                        tensorMxScaleA,
                        tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.k() * L1_TILE_K / MX_SCALE_GROUP_NUM),
                        tla::MakeShape(actualBlockShape.m(), CeilDiv<MX_SCALE_GROUP_NUM>(actualBlockShape.k())));

                    auto tensorBlockMxScaleB = GetTile(
                        tensorMxScaleB,
                        tla::MakeCoord(blockCoord.k() * L1_TILE_K / MX_SCALE_GROUP_NUM, blockCoord.n() * L1_TILE_N),
                        tla::MakeShape(CeilDiv<MX_SCALE_GROUP_NUM>(actualBlockShape.k()), actualBlockShape.n()));

                    blockMmad(tensorBlockA, tensorBlockB, tensorBlockC, actualBlockShape,
                        tensorBlockMxScaleA, tensorBlockMxScaleB);
                    callbackAfterFixpipe();
                }
                totalM += inGroupProblemShape.m();

                if constexpr (!(EXEC_FLAG & EXEC_FLAG_TENSOR_LIST)) {
                    if constexpr (AscendC::Std::is_one_of_v<ElementB, float4_e2m1x2_t, float4_e1m2x2_t>) {
                        gmGroupOffsetB += std::is_same_v<LayoutB, layout::ColumnMajor> ? 
                            CeilDiv<2>(inGroupProblemShape.k()) * inGroupProblemShape.n() : 
                            CeilDiv<2>(inGroupProblemShape.n()) * inGroupProblemShape.k();
                    }else{
                        gmGroupOffsetB += inGroupProblemShape.k() * inGroupProblemShape.n();
                    }
                    gmGroupOffsetMxScaleB += mxScaleAlignedK * inGroupProblemShape.n();
                }
                gmGroupOffsetMxScaleA += inGroupProblemShape.m() * mxScaleAlignedK;

                startCoreIdx = (startCoreIdx + coreLoops) % aicNum;
            }

            if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
                blockMmad.template SynchronizeBlock<decltype(tensorC)>();
            }
        }

        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::SyncAll<false>();
    }

    // template <>
    CATLASS_DEVICE
    void QuantDynamicMx(
        AscendC::LocalTensor<ElementA>& outLocal, AscendC::LocalTensor<XType>& inLocal, AscendC::LocalTensor<float>& tokenF32LT, uint32_t quantLength, uint32_t mxScaleNumPerToken)
    {
        __ubuf__ XType* srcAddr = (__ubuf__ XType*)inLocal.GetPhyAddr();
        __ubuf__ uint16_t* maxExpAddr = (__ubuf__ uint16_t*)tokenF32LT.GetPhyAddr();
        __ubuf__ uint16_t* halfScaleLocalAddr = (__ubuf__ uint16_t*)tokenF32LT[mxScaleNumPerToken].GetPhyAddr();
        __ubuf__ int8_t* outLocalAddr = (__ubuf__ int8_t*)outLocal.GetPhyAddr();
        __ubuf__ uint16_t* mxScaleLocalAddr = (__ubuf__ uint16_t*)outLocal[quantLength].GetPhyAddr();

        quant::ComputeMaxExp(srcAddr, maxExpAddr, quantLength);
        quant::ComputeScale<ElementA>(maxExpAddr, mxScaleLocalAddr, halfScaleLocalAddr, mxScaleNumPerToken);
        if constexpr (AscendC::Std::is_one_of_v<ElementA, float4_e2m1x2_t, float4_e1m2x2_t>) {
            quant::ComputeFp4Data<XType, ElementA, AscendC::RoundMode::CAST_TRUNC, AscendC::RoundMode::CAST_RINT>(
                srcAddr, halfScaleLocalAddr, outLocalAddr, quantLength);
        } else {
            quant::ComputeFp8Data<XType, ElementA, AscendC::RoundMode::CAST_TRUNC, AscendC::RoundMode::CAST_RINT>(
                srcAddr, halfScaleLocalAddr, outLocalAddr, quantLength);
        }
    }

    CATLASS_DEVICE
    void TokenActiveMaskCal(GM_ADDR gmXActiveMask, int64_t ubOffset)
    {
        int64_t subUbOffset = ubOffset;

        AscendC::GlobalTensor<bool> xActiveMaskGMTensor;
        xActiveMaskGMTensor.SetGlobalBuffer((__gm__ bool *)gmXActiveMask);
        uint32_t axisBsAlignSize = CEIL_UP(axisBS * sizeof(bool));

        AscendC::DataCopyExtParams maskParams = {1U, static_cast<uint32_t>(axisBS * sizeof(bool)), 0U, 0U, 0U};
        AscendC::DataCopyPadExtParams<bool> maskCopyPadParams{false, 0U, 0U, 0U};
        AscendC::DataCopyPad(maskInputTensor, xActiveMaskGMTensor, maskParams, maskCopyPadParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::Cast(maskTmpTensor, maskInputInt8Tensor, AscendC::RoundMode::CAST_NONE, axisBS);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SumParams params{1, axisBsAlignSize, axisBS};
        AscendC::Sum(sumOutTensor, maskTmpTensor, sharedTmpBuffer, params);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
        activeMaskBsCnt = static_cast<int32_t>(sumOutTensor.GetValue(0));
    }

    CATLASS_DEVICE
    void CalExpandxIdx(int32_t dstExpertId, uint32_t tokenIndex, int32_t &curExpertCnt, int64_t ubOffset)
    {
        // calculate index in remote
        int64_t subUbOffset = ubOffset;
        AscendC::Duplicate<int32_t>(dstExpIdTensor_, dstExpertId, tokenIndex);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(dstExpIdTensor_, expertIdsTensor_, dstExpIdTensor_, tokenIndex);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Abs(dstExpIdFp32Tensor_, dstExpIdFp32Tensor_, tokenIndex);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mins(dstExpIdTensor_, dstExpIdTensor_, 1, tokenIndex);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ReduceSum<float>(dstExpIdFp32Tensor_, dstExpIdFp32Tensor_, reduceSumWorkLocalTensor, tokenIndex);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
        int32_t curOtherExpertCnt = dstExpIdTensor_(0);
        if (tokenIndex > curOtherExpertCnt) {
            curExpertCnt = tokenIndex - curOtherExpertCnt;
        }
    }

    CATLASS_DEVICE
    void CalAndSendTokenCount()
    {
        uint32_t totalExpertNum = moeExpertNum;
        uint32_t sendCountExpertNum = totalExpertNum / sendCoreNum;
        uint32_t remainderRankNum = totalExpertNum % sendCoreNum;
        uint32_t startExpertId = sendCountExpertNum * sendCoreIdx;
        if (sendCoreIdx < remainderRankNum) {
            sendCountExpertNum += 1;
            startExpertId += sendCoreIdx;
        } else {
            startExpertId += remainderRankNum;
        }
        uint32_t endExpertId = startExpertId + sendCountExpertNum;
        if (startExpertId >= totalExpertNum) {
            return;
        }

        AscendC::Duplicate(statusTensor_, (int32_t)0,
                           expertCntUp * INT32_COUNT_PER_BLOCK);
        if (state == 0) {
            // set the first number of every 8 numbers as 0x3F800000(float 1.0)
            uint64_t mask[2] = {0x101010101010101, 0};
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Duplicate<int32_t>(statusTensor_, 0x3F800000, mask, CEIL(expertCntUp, INT32_COUNT_PER_BLOCK), 1,
                                        INT32_COUNT_PER_BLOCK);
        }

        AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);

        for (uint32_t curExpertId = startExpertId; curExpertId < endExpertId; ++curExpertId) {
            int32_t curExpertCnt = 0;
            int32_t dstExpertId = curExpertId;
            CalExpandxIdx(dstExpertId, expertIdsCnt, curExpertCnt, ubOffset);
            int32_t cntPosIndex = curExpertId * INT32_COUNT_PER_BLOCK + 1;
            statusTensor_(cntPosIndex) = curExpertCnt;
        }

        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);

        AscendC::GlobalTensor<int32_t> rankGMTensor;
        uint32_t offset = stateOffset * epRankId;
        for (uint32_t rankIndex = startExpertId; rankIndex < endExpertId; ++rankIndex) {
            uint32_t dstRankId = rankIndex;
            if (moeExpertNumPerRank > 1) {
                dstRankId = ((rankIndex) / moeExpertNumPerRank);
                offset =
                    (epRankId + (rankIndex) % moeExpertNumPerRank * epRankSize) * stateOffset;
            }
            GM_ADDR rankGM = (__gm__ uint8_t *)(GetWindStateAddrByRankId(dstRankId) + offset);
            rankGMTensor.SetGlobalBuffer((__gm__ int32_t *)rankGM);
            AscendC::DataCopy<int32_t>(rankGMTensor, statusTensor_[rankIndex * INT32_COUNT_PER_BLOCK], 8UL);
        }
    }

    CATLASS_DEVICE
    void QuantToken(AscendC::LocalTensor<XType> &xInTensor, AscendC::LocalTensor<float> &smoothScaleTensor,
        AscendC::LocalTensor<ElementA> &yInt8Tensor, int64_t ubOffset)
    {
        int64_t subUbOffset = ubOffset;
        AscendC::LocalTensor<int32_t> yInt32Tensor = (yInt8Tensor[tokenLength].template ReinterpretCast<ElementMxScaleA>())[x1MxScaleNum].template ReinterpretCast<int32_t>();
        if constexpr(EXEC_FLAG & EXEC_FLAG_SMOOTH_QUANT) {
            AscendC::Cast(xFp32TmpTensor, xInTensor, AscendC::RoundMode::CAST_NONE, tokenLength);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(xFp32TmpTensor, xFp32TmpTensor, smoothScaleTensor, tokenLength);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(xInTensor, xFp32TmpTensor, AscendC::RoundMode::CAST_RINT, tokenLength);
            AscendC::PipeBarrier<PIPE_V>();
        }
        QuantDynamicMx(yInt8Tensor, xInTensor, tokenF32LT, tokenLength, x1MxScaleNum);
        yInt32Tensor.SetValue(0, tokenFlag);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
    }

    CATLASS_DEVICE
    void SendToMoeExprt(GM_ADDR gmX, GM_ADDR gmExpandIdx, GM_ADDR gmMoeSmoothScales)
    {
        uint32_t sendTokenNum = expertIdsCnt / sendToMoeAivNum;
        uint32_t remainderTokenNum = expertIdsCnt % sendToMoeAivNum;
        uint32_t startTokenId = sendTokenNum * sendCoreIdx;
        if (sendCoreIdx < remainderTokenNum) {
            sendTokenNum += 1;
            startTokenId += sendCoreIdx;
        } else {
            startTokenId += remainderTokenNum;
        }
        uint32_t endTokenId = startTokenId + sendTokenNum;
        if (startTokenId >= expertIdsCnt) {
            return;
        }
        AscendC::Duplicate(expertCountTensor, (int32_t)0, expertIdsCnt);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(1);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(1);

        AscendC::GlobalTensor<XType> srcWinGMTensor;
        srcWinGMTensor.SetGlobalBuffer((__gm__ XType *)gmX);
        AscendC::GlobalTensor<float> moeSmoothScaleGMTensor;

        if constexpr(EXEC_FLAG & EXEC_FLAG_SMOOTH_QUANT) {
            moeSmoothScaleGMTensor.SetGlobalBuffer((__gm__ float*) gmMoeSmoothScales);
        }
        AscendC::GlobalTensor<ElementA> dstWinGMTensor;
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(1);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(1);
        uint32_t sendValidTokenIndex = 0;
        for (uint32_t sendGroupIndex = 0; sendGroupIndex < moeExpertNumPerRank; ++sendGroupIndex) {
            for (uint32_t tokenIndex = startTokenId; tokenIndex < endTokenId; ++tokenIndex) {
                int32_t dstExpertId = expertIdsTensor_(tokenIndex);
                if (dstExpertId < 0) {
                    continue;
                }
                // Send to preferentically to the specicied expert
                if ((dstExpertId % moeExpertNumPerRank) != sendGroupIndex) {
                    continue;
                }
                uint32_t index = (sendValidTokenIndex & 1) ? 0 : 1;
                int32_t eventId = (sendValidTokenIndex & 1) ? 0 : 1;
                sendValidTokenIndex += 1;
                int32_t curExpertCnt = 0;
                CalExpandxIdx(dstExpertId, tokenIndex, curExpertCnt, ubOffset);
                expertCountTensor(tokenIndex - startTokenId) = curExpertCnt;
                uint32_t tempRankId = dstExpertId / moeExpertNumPerRank;
                GM_ADDR rankGM = (__gm__ uint8_t *)(
                    GetWindAddrByRankId(tempRankId) +
                    (expertPerSizeOnWin * (epRankId * moeExpertNumPerRank + dstExpertId % moeExpertNumPerRank)) +
                    hCommuSize * curExpertCnt);
                dstWinGMTensor.SetGlobalBuffer((__gm__ ElementA *)rankGM);

                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventId);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
                AscendC::DataCopy(xInTensor[index], srcWinGMTensor[tokenIndex / axisK * tokenLength], tokenLength);
                if constexpr(EXEC_FLAG & EXEC_FLAG_SMOOTH_QUANT) {
                    AscendC::PipeBarrier<PIPE_MTE2>();
                    AscendC::DataCopy(
                        moeSmoothScaleTensor[index], moeSmoothScaleGMTensor[dstExpertId * tokenLength], tokenLength);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventId);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventId);
                QuantToken(xInTensor[index], moeSmoothScaleTensor[index], yInt8Tensor[index], ubOffset);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventId);

                AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventId);

                AscendC::DataCopy(dstWinGMTensor, yInt8Tensor[index], tokenLength);
                AscendC::PipeBarrier<PIPE_MTE3>();
                AscendC::DataCopy(dstWinGMTensor[tokenLength], yInt8Tensor[index][tokenLength], MxByte2Count<ElementA>(scaleFlagSize));
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventId);
            }
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(1);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(1);

        AscendC::GlobalTensor<int32_t> expandIdxGMTensor;
        expandIdxGMTensor.SetGlobalBuffer((__gm__ int32_t *)gmExpandIdx + startTokenId);
        AscendC::DataCopyExtParams expertIdsCntParams = {1U, static_cast<uint32_t>(sendTokenNum * sizeof(uint32_t)),
                                                         0U, 0U, 0U};
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
        AscendC::DataCopyPad(expandIdxGMTensor, expertCountTensor, expertIdsCntParams);
    }

    CATLASS_DEVICE void
    SendCoreFunc(GM_ADDR gmX, GM_ADDR gmExpertIds, GM_ADDR gmMoeSmoothScales,
                 GM_ADDR gmExpandIdx, GM_ADDR gmXActiveMask)
    {
        if constexpr (EXEC_FLAG & EXEC_FLAG_X_ACTIVE_MASK) {
            ubOffset = 0;
            maskInputTensor = resource.ubBuf.template GetBufferByByte<bool>(ubOffset);
            ubOffset += CEIL_UP(axisBS * sizeof(bool));
            maskInputInt8Tensor = maskInputTensor.template ReinterpretCast<int8_t>();
            maskTmpTensor = resource.ubBuf.template GetBufferByByte<half>(ubOffset);
            ubOffset += CEIL_UP(axisBS * sizeof(half));
            sumOutTensor = resource.ubBuf.template GetBufferByByte<half>(ubOffset);
            ubOffset += CEIL_UP(SUM_TMP_TENSOR_SIZE);
            sharedTmpBuffer = resource.ubBuf.template GetBufferByByte<uint8_t>(ubOffset);
            TokenActiveMaskCal(gmXActiveMask, ubOffset);
        }

        ubOffset = 0;
        expertIdsCnt = activeMaskBsCnt * axisK;
        expertIdsTensor_ = (resource.ubBuf.template GetBufferByByte<int32_t>(ubOffset));
        ubOffset += CEIL_UP(expertIdsCnt * sizeof(int32_t));
        statusTensor_ = resource.ubBuf.template GetBufferByByte<int32_t>(ubOffset);
        ubOffset += CEIL_UP(CEIL(expertCntUp, INT32_COUNT_PER_BLOCK) * INT32_COUNT_PER_BLOCK * UB_BLOCK_SIZE);
        expertCountTensor = (resource.ubBuf.template GetBufferByByte<int32_t>(ubOffset));
        ubOffset += CEIL_UP(expertIdsCnt * sizeof(int32_t));

        for (uint32_t i = 0; i < BUFFER_NUM; ++i) {
            xInTensor[i] = resource.ubBuf.template GetBufferByByte<XType>(ubOffset);
            ubOffset += CEIL_UP(tokenLength * sizeof(XType));
            yInt8Tensor[i] = resource.ubBuf.template GetBufferByByte<ElementA>(ubOffset);
            yScaleTensor[i] = yInt8Tensor[i][tokenLength].template ReinterpretCast<ElementMxScaleA>();
            ubOffset += CEIL_UP(hCommuSize);
            if constexpr(EXEC_FLAG & EXEC_FLAG_SMOOTH_QUANT) { 
                moeSmoothScaleTensor[i] = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
                ubOffset += CEIL_UP(tokenLength * sizeof(float));
            }
        }
        xFp32TmpTensor = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
        ubOffset += CEIL_UP(tokenLength * sizeof(float));
        tokenF32LT = resource.ubBuf.template GetBufferByByte<ElementC>(ubOffset);
        ubOffset += x1MxScaleNum * 2 * sizeof(float);

        dstExpIdTensor_ = resource.ubBuf.template GetBufferByByte<int32_t>(ubOffset);
        dstExpIdFp32Tensor_ = dstExpIdTensor_.ReinterpretCast<float>();
        ubOffset += CEIL_UP(expertIdsCnt * sizeof(float));
        reduceSumWorkLocalTensor = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
        ubOffset += REDUCE_SUM_WORK_SIZE;

        AscendC::GlobalTensor<int32_t> expertIdsGMTensor_;
        expertIdsGMTensor_.SetGlobalBuffer((__gm__ int32_t *)gmExpertIds);
        AscendC::DataCopyExtParams expertIdsCntParams = {1U, static_cast<uint32_t>(expertIdsCnt * sizeof(uint32_t)),
                                                         0U, 0U, 0U};
        AscendC::DataCopyPadExtParams<int32_t> copyPadParams{false, 0U, 0U, 0U};
        AscendC::DataCopyPad(expertIdsTensor_, expertIdsGMTensor_, expertIdsCntParams, copyPadParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);

        CalAndSendTokenCount();
        AscendC::PipeBarrier<PIPE_ALL>();
        sendToMoeAivNum = sendCoreNum;
        SendToMoeExprt(gmX, gmExpandIdx, gmMoeSmoothScales);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    CATLASS_DEVICE
    void shareQuantCoreFunc(GM_ADDR gmX, GM_ADDR gmShareSmoothScales, GM_ADDR gmShareX1Token, GM_ADDR gmShareX1Scale)
    {
        ubOffset = 0;
        uint32_t quantTokenPerCore = axisBS / shareQuantCoreNum;
        uint32_t remainTokenNum = axisBS % shareQuantCoreNum;
        uint32_t startTokenId = quantTokenPerCore * shareQuantCoreIdx;
        if (shareQuantCoreIdx < remainTokenNum) {
            quantTokenPerCore += 1;
            startTokenId += shareQuantCoreIdx;
        } else {
            startTokenId += remainTokenNum;
        }
        uint32_t endTokenId = startTokenId + quantTokenPerCore;
        if (startTokenId >= axisBS) {
            return;
        }
        AscendC::GlobalTensor<XType> srcXGMTensor;
        srcXGMTensor.SetGlobalBuffer((__gm__ XType*)gmX);
        AscendC::GlobalTensor<ElementA> dstXInt8GMTensor;
        dstXInt8GMTensor.SetGlobalBuffer((__gm__ ElementA*)gmShareX1Token);
        AscendC::GlobalTensor<ElementMxScaleA> dstXScaleGMTensor;
        dstXScaleGMTensor.SetGlobalBuffer((__gm__ ElementMxScaleA*)gmShareX1Scale);
        AscendC::GlobalTensor<float> shareSmoothScaleGMTensor;
        shareSmoothScaleGMTensor.SetGlobalBuffer((__gm__ float*)gmShareSmoothScales);

        for (uint32_t i = 0; i < BUFFER_NUM; ++i) {
            xInTensor[i] = resource.ubBuf.template GetBufferByByte<XType>(ubOffset);
            ubOffset += CEIL_UP(tokenLength * sizeof(XType));
            yInt8Tensor[i] = resource.ubBuf.template GetBufferByByte<ElementA>(ubOffset);
            yScaleTensor[i] = yInt8Tensor[i][tokenLength].template ReinterpretCast<ElementMxScaleA>();
            ubOffset += CEIL_UP(hCommuSize);
        }
        xFp32TmpTensor = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
        ubOffset += CEIL_UP(tokenLength * sizeof(float));
        tokenF32LT = resource.ubBuf.template GetBufferByByte<ElementC>(ubOffset);
        ubOffset += x1MxScaleNum * 2 * sizeof(float);
        tmpLocalTensor = resource.ubBuf.template GetBufferByByte<int32_t>(ubOffset);
        ubOffset += CEIL_UP(UB_BLOCK_SIZE);
        if constexpr(EXEC_FLAG & EXEC_FLAG_SMOOTH_QUANT) {
            shareSmoothScaleTensor = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
            ubOffset += CEIL_UP(tokenLength * sizeof(float));
            AscendC::DataCopy(shareSmoothScaleTensor, shareSmoothScaleGMTensor, tokenLength);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
        }
        // double buffer
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(1);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(1);
        AscendC::DataCopyExtParams dataCopyParamsFloat = {1U, (uint32_t)(x1MxScaleNum * sizeof(ElementMxScaleA)), 0U, 0U, 0U};
        for (uint32_t tokenIndex = startTokenId; tokenIndex < endTokenId; ++tokenIndex) {
            uint32_t index = (tokenIndex & 1) ? 0 : 1;
            int32_t eventId = (tokenIndex & 1) ? 0 : 1;
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
            AscendC::DataCopy(xInTensor[index], srcXGMTensor[tokenIndex * tokenLength], tokenLength);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventId);
            QuantToken(xInTensor[index], shareSmoothScaleTensor, yInt8Tensor[index], ubOffset);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventId);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventId);
            AscendC::DataCopy(dstXInt8GMTensor[tokenIndex * tokenLength], yInt8Tensor[index], tokenLength);
            AscendC::DataCopyPad(
                dstXScaleGMTensor[tokenIndex * x1MxScaleNum], yScaleTensor[index], dataCopyParamsFloat);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventId);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(1);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(1);

        // Set GM to info AIC
        AscendC::PipeBarrier<PIPE_ALL>();
        tmpLocalTensor.SetValue(CV_FLAG_INDEX, vToCFlag);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);

        AscendC::GlobalTensor<int32_t> shareQuantTokenStateTensor;
        shareQuantTokenStateTensor.SetGlobalBuffer(
            (__gm__ int32_t*)(statusDataSpaceGm + SHARE_QUANT_SOFT_SYNC_OFFSET));
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
        AscendC::SetAtomicAdd<int32_t>();
        // Atomic add
        AscendC::DataCopy(shareQuantTokenStateTensor, tmpLocalTensor, INT32_COUNT_PER_BLOCK);
        AscendC::SetAtomicNone();
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    CATLASS_DEVICE
    void RecvCount(int64_t ubOffset)
    {
        uint32_t recStatusNumPerCore = expertCntUp;
        uint32_t startStatusIndex = 0;  // every wait for all token counts

        gatherTmpTensor.SetValue(0, 1);

        uint32_t mask = 1;
        uint64_t rsvdCnt = 0;
        AscendC::SumParams sumParams{1, recStatusNumPerCore, recStatusNumPerCore};
        float sumOfFlag = static_cast<float>(-1.0);
        float minTarget = (sumTarget * recStatusNumPerCore) - (float)0.5;
        float maxTarget = (sumTarget * recStatusNumPerCore) + (float)0.5;
        AscendC::DataCopyParams intriParams{static_cast<uint16_t>(recStatusNumPerCore), 1, static_cast<uint16_t>(15),
                                            0};
        AscendC::GlobalTensor<float> windowInstatusFp32Tensor_;
        windowInstatusFp32Tensor_.SetGlobalBuffer((__gm__ float *)GetWindStateAddrByRankId(epRankId));
        AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);

        uint32_t preRecvTokenCount = 0;
        while ((sumOfFlag < minTarget) || (sumOfFlag > maxTarget)) {
            AscendC::DataCopy(statusFp32Tensor_, windowInstatusFp32Tensor_[startStatusIndex *
                                                                           stateOffset / sizeof(float)], intriParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
            AscendC::GatherMask(gatherMaskOutTensor, statusFp32Tensor_, gatherTmpTensor, true, mask,
                                {1, (uint16_t)recStatusNumPerCore, 1, 0}, rsvdCnt);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sum(statusSumOutTensor, gatherMaskOutTensor, sumTmpTensor, sumParams);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
            sumOfFlag = statusSumOutTensor.GetValue(0);
            if ((sumOfFlag < minTarget) || (sumOfFlag > maxTarget)) {
                SPIN_WAIT_CYCLES();
            }
        }
    }

    CATLASS_DEVICE
    void GetCumSum(int32_t startRankId, int32_t recvExpertNum, int64_t ubOffset)
    {
        // calculate token index in output tensor
        int64_t subUbOffset = ubOffset;
        uint32_t recStatusNumPerCore = expertCntUp;

        uint64_t rsvdCnt = 0;
        gatherTmpTensor.SetValue(0, GATHER_SECOND_NUM);
        AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
        AscendC::GatherMask(gatherMaskOutTensor, statusFp32Tensor_, gatherTmpTensor, true, GATHER_SECOND_NUM,
                            {1, (uint16_t)recStatusNumPerCore, 1, 0}, rsvdCnt);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ReduceSum<float>(gatherMaskOutTensor, gatherMaskOutTensor, reduceSumWorkLocalTensor,
                                (startRankId + 1) <= recvExpertNum ? (startRankId + 1) : recvExpertNum);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
    }

    CATLASS_DEVICE
    void RecvToken(GM_ADDR gmX1, GM_ADDR gmX1Scale, 
                   uint32_t startRankId, uint32_t startTokenIdx, uint32_t startTokenIdxInRank, uint32_t recvTokenNum)
    {
        AscendC::DataCopyExtParams dataCopyParamsFloat = {1U, (uint32_t)(x1MxScaleNum * sizeof(ElementMxScaleA)), 0U, 0U, 0U};
        AscendC::GlobalTensor<ElementA> tokGlobal;
        AscendC::GlobalTensor<int32_t> tokGlobalInt32;
        AscendC::GlobalTensor<ElementA> expandXOutGlobal;
        AscendC::GlobalTensor<ElementMxScaleA> dynamicScalesOutGMTensor_;
        dynamicScalesOutGMTensor_.SetGlobalBuffer((__gm__ ElementMxScaleA *)(gmX1Scale));
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
        AscendC::DataCopyExtParams dataCopyOutParams = {1U, static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U};
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);

        uint32_t currentRank = startRankId;
        uint32_t currentTokenIdx = startTokenIdx;
        uint32_t currentTokenIdxInRank = startTokenIdxInRank;
        uint32_t curRecvTokenCount = 0;
        uint32_t currentRankCount = statusTensor_.GetValue(currentRank * INT32_COUNT_PER_BLOCK + 1);
        while (curRecvTokenCount < recvTokenNum) {
            while (currentTokenIdxInRank >= currentRankCount) {
                currentTokenIdxInRank = 0;
                currentRank += 1;
                currentRankCount = statusTensor_.GetValue(currentRank * INT32_COUNT_PER_BLOCK + 1);
            }

            uint32_t winOffset = currentRank;
            winOffset = (currentRank % epRankSize) * moeExpertNumPerRank + currentRank / epRankSize;
            GM_ADDR wAddr = (__gm__ uint8_t *)(GetWindAddrByRankId(epRankId)) + winOffset * expertPerSizeOnWin;

            tokGlobal.SetGlobalBuffer((__gm__ ElementA *)(wAddr + currentTokenIdxInRank * hCommuSize));
            tokGlobalInt32.SetGlobalBuffer((__gm__ int32_t *)(wAddr + currentTokenIdxInRank * hCommuSize + hOutSize + scaleSize));
            expandXOutGlobal.SetGlobalBuffer((__gm__ ElementA *)(gmX1 + currentTokenIdx * hOutSize), tokenLength);
            while (true) {
                AscendC::DataCopy(tmpLocalTensor, tokGlobalInt32, INT32_COUNT_PER_BLOCK);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
                if (tmpLocalTensor.GetValue(0) == tokenFlag) {
                    SetValueAndFlush<int32_t>(tokGlobalInt32, 1, 0);
                    break;
                }
                SPIN_WAIT_CYCLES();
            }
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
            AscendC::DataCopy(xTmpTensor_, tokGlobal, MxByte2Count<ElementA>(hCommuSize));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(0);
            AscendC::DataCopyPad(dynamicScalesOutGMTensor_[currentTokenIdx * x1MxScaleNum],
                xOutFp32Tensor_, dataCopyParamsFloat);
            AscendC::DataCopy(expandXOutGlobal, xTmpTensor_, tokenLength);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);

            curRecvTokenCount += 1;
            currentTokenIdxInRank += 1;
            currentTokenIdx += 1;
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    }

    CATLASS_DEVICE
    void RecvCoreFunc(GM_ADDR gmX1, GM_ADDR gmX1Scale, GM_ADDR gmEpSendCount)
    {
        ubOffset = 0;

        statusTensor_ = resource.ubBuf.template GetBufferByByte<int32_t>(ubOffset);
        statusFp32Tensor_ = statusTensor_.ReinterpretCast<float>();
        ubOffset += CEIL_UP(expertCntUp * UB_BLOCK_SIZE);
        gatherTmpTensor = (resource.ubBuf.template GetBufferByByte<uint32_t>(ubOffset));
        ubOffset += CEIL_UP(UB_BLOCK_SIZE);
        gatherMaskOutTensor = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
        gatherMaskOutCountTensor = gatherMaskOutTensor.template ReinterpretCast<int32_t>();
        ubOffset += CEIL_UP(expertCntUp * sizeof(float));

        statusSumOutTensor = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
        ubOffset += CEIL_UP(UB_BLOCK_SIZE);
        sumTmpTensor = resource.ubBuf.template GetBufferByByte<uint8_t>(ubOffset);
        ubOffset += CEIL_UP(SUM_TMP_TENSOR_SIZE);

        xTmpTensor_ = resource.ubBuf.template GetBufferByByte<ElementA>(ubOffset);
        xOutFp32Tensor_ = xTmpTensor_[tokenLength].template ReinterpretCast<ElementMxScaleA>();
        ubOffset += CEIL_UP(hCommuSize);

        tmpLocalTensor = resource.ubBuf.template GetBufferByByte<int32_t>(ubOffset);
        ubOffset += CEIL_UP(UB_BLOCK_SIZE);

        sendCountsLocalTensor = resource.ubBuf.template GetBufferByByte<int32_t>(ubOffset);
        ubOffset += CEIL_UP(UB_BLOCK_SIZE);

        AscendC::LocalTensor<int32_t> notifyCubeTensor = resource.ubBuf.template GetBufferByByte<int32_t>(
                                                                                                    ubOffset);

        ubOffset += CEIL_UP((expertCntUp / recvCoreNum + 1) * sizeof(int32_t));
        reduceSumWorkLocalTensor = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
        ubOffset += REDUCE_SUM_WORK_SIZE; 

        RecvCount(ubOffset);

        uint32_t recvExpertNum = expertCntUp;
        uint32_t recvCoreNumPerGroup = recvCoreNum;
        uint32_t recvRankNumPerCore = epRankSize / recvCoreNumPerGroup;
        uint32_t remainderRankNum = epRankSize % recvCoreNumPerGroup;
        uint32_t startCoreIdx = 0;

        uint32_t subUbOffset = CEIL_UP(expertCntUp * UB_BLOCK_SIZE) + CEIL_UP(UB_BLOCK_SIZE) +
                                CEIL_UP(expertCntUp * sizeof(float));
        uint32_t preExpertToken = 0;
        for (uint32_t groupId = 0; groupId < localExpertNum; ++groupId) {
            GetCumSum((groupId + 1) * epRankSize - 1, recvExpertNum, ubOffset);
            uint32_t currentM = gatherMaskOutCountTensor.GetValue(0) - preExpertToken;

            uint32_t recvTokenPerCore = currentM / recvCoreNum;
            uint32_t remainToken = currentM % recvCoreNum;

            uint32_t newRecvCoreIdx = (recvCoreIdx + recvCoreNum - startCoreIdx) % recvCoreNum;
            uint32_t startTokenIdx = newRecvCoreIdx * recvTokenPerCore;
            if (newRecvCoreIdx < remainToken) {
                recvTokenPerCore += 1;
                startTokenIdx += newRecvCoreIdx;
            } else {
                startTokenIdx += remainToken;
            }
            uint32_t endTokenIdx = startTokenIdx + recvTokenPerCore;
            uint32_t coreTokenCount = recvTokenPerCore;
            uint32_t useCoreNum = currentM < recvCoreNum ? currentM : recvCoreNum;

            if (startTokenIdx < currentM && recvTokenPerCore > 0) {
                uint32_t startRankId = groupId * epRankSize;
                uint32_t preTokenNum = 0;
                uint32_t startTokenIdxInRank = 0;
                uint32_t startRankTokenCount = statusTensor_.GetValue(startRankId * INT32_COUNT_PER_BLOCK + 1);
                while (preTokenNum + startRankTokenCount < startTokenIdx) {
                    preTokenNum += startRankTokenCount;
                    startRankId += 1;
                    startRankTokenCount = statusTensor_.GetValue(startRankId * INT32_COUNT_PER_BLOCK + 1);
                }
                startTokenIdxInRank = startTokenIdx - preTokenNum;
                RecvToken(gmX1, gmX1Scale, startRankId, startTokenIdx + preExpertToken, startTokenIdxInRank, recvTokenPerCore);
            }
            // recv finish, inform AIC
            AscendC::PipeBarrier<PIPE_ALL>();
            notifyCubeTensor.SetValue(CV_FLAG_INDEX, vToCFlag);
            notifyCubeTensor.SetValue(GROUP_ID_INDEX, groupId);
            notifyCubeTensor.SetValue(SELF_COUNT_INDEX, coreTokenCount);
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);

            AscendC::GlobalTensor<int32_t> groupTokenNumStateTensor;
            groupTokenNumStateTensor.SetGlobalBuffer((__gm__ int32_t *)(statusDataSpaceGm + GROUP_TOKEN_NUM_OFFSET));
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
            AscendC::SetAtomicAdd<int32_t>();
            AscendC::DataCopy(
                groupTokenNumStateTensor[groupId * GROUP_INFO_SIZE], notifyCubeTensor, INT32_COUNT_PER_BLOCK);
            AscendC::SetAtomicNone();
            AscendC::PipeBarrier<PIPE_ALL>();

            startCoreIdx = (startCoreIdx + currentM) % recvCoreNum;
            preExpertToken += currentM;
        }

        uint32_t sendCountNum = expertCntUp;
        uint32_t sendCountPerCore = expertCntUp / recvCoreNum;
        uint32_t remainSendCount = expertCntUp % recvCoreNum;
        uint32_t sendCountStart = sendCountPerCore * recvCoreIdx;
        if (recvCoreIdx < remainSendCount) {
            sendCountStart += recvCoreIdx;
            sendCountPerCore += 1;
        } else {
            sendCountStart += remainSendCount;
        }
        if (sendCountStart >= sendCountNum) {
            return;
        }
        AscendC::GlobalTensor<int32_t> sendCountsGlobal;
        sendCountsGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(gmEpSendCount));
        uint32_t sendCountEnd = sendCountStart + sendCountPerCore;
        GetCumSum(sendCountStart, sendCountNum, ubOffset);
        sendCountsLocalTensor(0) = gatherMaskOutCountTensor.GetValue(0);
        for (uint32_t index = 1; index < sendCountPerCore; ++ index) {
            sendCountsLocalTensor(index) = sendCountsLocalTensor(index-1) + statusTensor_.GetValue((sendCountStart + index) * INT32_COUNT_PER_BLOCK + 1); 
        }
        AscendC::DataCopyExtParams sendCountDataCopyOutParams = {1U, static_cast<uint32_t>(sendCountPerCore * sizeof(int32_t)), 0U, 0U, 0U};
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
        AscendC::DataCopyPad(sendCountsGlobal[sendCountStart], sendCountsLocalTensor, sendCountDataCopyOutParams);
    }

    CATLASS_DEVICE
    void AivInitParams(Params const &params)
    {
        moeExpertNumPerRank = params.moeExpertNumPerRank;

        epRankSize = params.epRankSize;
        epRankId = params.epRankId;
        expertCntUp = epRankSize * moeExpertNumPerRank;
        localExpertNum = moeExpertNumPerRank;
        moeExpertNum = params.moeExpertNum;
        tokenLength = params.tokenLen;

        x1MxScaleNum = CEIL(tokenLength, 32);
        hOutSize = MxCount2Byte<ElementA>(tokenLength);
        scaleSize = MxCount2Byte<ElementMxScaleA>(x1MxScaleNum);
        scaleFlagSize =  CEIL(scaleSize + sizeof(int32_t), TOKEN_EXTRA_SPACE) * TOKEN_EXTRA_SPACE; // scale and flag
        hCommuSize = hOutSize + scaleFlagSize;
        axisHCommu = MxByte2Count<ElementA>(hCommuSize);
        axisBS = params.bs;
        activeMaskBsCnt = axisBS;
        axisK = params.topK;
        uint32_t maxAxisBs = params.globalBs / epRankSize;

        stateOffset = STATE_OFFSET;
        expertPerSizeOnWin = maxAxisBs * tokenLength * sizeof(XType);
    }

    CATLASS_DEVICE
    void AivInitState()
    {
        // state of data sapce
        winDataSizeOffset = dataState * epRankSize * expertPerSizeOnWin * moeExpertNumPerRank;
        GM_ADDR statusSpaceGm_ = GetWindStateAddrByRankId(epRankId);
        AscendC::GlobalTensor<int32_t> selfStatusTensor;
        selfStatusTensor.SetGlobalBuffer((__gm__ int32_t *)(statusSpaceGm_ + SELF_STATE_OFFSET));
        state = FlushAndGetValue<int32_t>(selfStatusTensor, aivIdx * UB_ALIGN);
        sumTarget = state == 0 ? 1.0f : 0.0f;
        tokenFlag = state == 0 ? TOKEN_FLAG_1 : TOKEN_FLAG_2;
        if (state == 0) {
            SetValueAndFlush<int32_t>(selfStatusTensor, aivIdx * UB_ALIGN, 0x3F800000);
        } else {
            SetValueAndFlush<int32_t>(selfStatusTensor, aivIdx * UB_ALIGN, 0);
        }
    }

    CATLASS_DEVICE
    void UpdateAndCleanInfo(__gm__ ElementGroupList_ *ptrGroupList, GM_ADDR gmEpSendCount, GM_ADDR gmExpertTokenNums)
    {
        if (isCompCore && AscendC::GetSubBlockIdx() == 0) {
            AscendC::GlobalTensor<int32_t> softSyncTensor;
            softSyncTensor.SetGlobalBuffer((__gm__ int32_t*)(statusDataSpaceGm + SOFT_SYNC_OFFSET));
            AscendC::LocalTensor<int32_t> tmpZeroLocalTensor = resource.ubBuf.template GetBufferByByte<int32_t>(0);
            AscendC::Duplicate(tmpZeroLocalTensor, (int32_t)0, INT32_COUNT_PER_BLOCK);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
            AscendC::DataCopy(softSyncTensor[compCoreIdx * CVSoftSync::SOFT_SYNC_SPACE_SIZE / sizeof(int32_t)],
                                                tmpZeroLocalTensor, INT32_COUNT_PER_BLOCK);
        }
        if constexpr (!(EXEC_FLAG & EXEC_FLAG_DEEP_FUSE)) {
            return ;
        }
        if (aivIdx == aiCoreGroupNum * subBlockNum - 1) {
            // clean
            AscendC::GlobalTensor<int32_t> groupTokenNumStateTensor;
            groupTokenNumStateTensor.SetGlobalBuffer((__gm__ int32_t *)(statusDataSpaceGm + GROUP_TOKEN_NUM_OFFSET));
            AscendC::LocalTensor<int32_t> tmpZeroLocalTensor = resource.ubBuf.template GetBufferByByte<int32_t>(512);
            AscendC::Duplicate(tmpZeroLocalTensor, (int32_t)0, GROUP_INFO_SIZE * localExpertNum);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
            AscendC::DataCopy(groupTokenNumStateTensor, tmpZeroLocalTensor, GROUP_INFO_SIZE * localExpertNum);
            if constexpr (EXEC_FLAG & EXEC_FLAG_SHARED_EXPERT) {
                AscendC::GlobalTensor<int32_t> shareQuantTokenStateTensor;
                shareQuantTokenStateTensor.SetGlobalBuffer(
                    (__gm__ int32_t*)(statusDataSpaceGm + SHARE_QUANT_SOFT_SYNC_OFFSET));
                AscendC::DataCopy(shareQuantTokenStateTensor, tmpZeroLocalTensor, 8);
            }
        }

        if (isRecvCore && recvCoreIdx == (recvCoreNum - 1)) {
            // record token count for each local expert
            AscendC::GlobalTensor<int64_t> expertTokenNumsOutGMTensor_;
            expertTokenNumsOutGMTensor_.SetGlobalBuffer((__gm__ int64_t *)(ptrGroupList));
            AscendC::GlobalTensor<int32_t> sendCountsGlobal;
            sendCountsGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(gmEpSendCount));
            AscendC::GlobalTensor<int64_t> nonCumSumExpertTokenNumsTensor;
            nonCumSumExpertTokenNumsTensor.SetGlobalBuffer((__gm__ int64_t *)gmExpertTokenNums);
            uint32_t tmpTokenNum = 0;
            for (uint32_t localMoeIndex = 0; localMoeIndex < localExpertNum; ++localMoeIndex) {
                uint32_t tokenNum = FlushAndGetValue<int32_t>(sendCountsGlobal,
                    localMoeIndex * epRankSize + epRankSize - 1);
                SetValueAndFlush<int64_t>(expertTokenNumsOutGMTensor_, localMoeIndex, tokenNum);
                uint32_t nonCumSumTokenNum = tokenNum - tmpTokenNum;
                SetValueAndFlush<int64_t>(nonCumSumExpertTokenNumsTensor, localMoeIndex, nonCumSumTokenNum);
                tmpTokenNum = tokenNum;
            }
        }
    }

    CATLASS_DEVICE
    void PostSwigluDynamicQuant(__gm__ ElementC *swigluOutAddr, __gm__ ElementA *x2Addr, __gm__ ElementMxScaleA *x2ScaleAddr,
                                uint32_t tokenNum, uint32_t mmOutDim, uint32_t &startCoreIdx) {
        uint32_t quantLength = mmOutDim / 2;
        uint32_t quantTokenSize = MxCount2Byte<ElementA>(quantLength);
        uint32_t mxScaleNumPerToken = CeilDiv(CeilDiv(quantLength, 32), 2) * 2;
        AscendC::GlobalTensor<ElementC> gmSwigluOutTensor;
        gmSwigluOutTensor.SetGlobalBuffer(swigluOutAddr);
        AscendC::GlobalTensor<ElementA> gmX2;
        gmX2.SetGlobalBuffer(x2Addr);
        AscendC::GlobalTensor<uint8_t> gmX2MxScale;
        gmX2MxScale.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(x2ScaleAddr));

        uint32_t startTokenIdx;
        if (aivIdx < startCoreIdx) {
            startTokenIdx = aivIdx + aivNum - startCoreIdx;
        } else {
            startTokenIdx = aivIdx - startCoreIdx;
        }

        uint32_t ubOffset = 0;
        AscendC::LocalTensor<ElementC> fp32TokenLocalTensor = resource.ubBuf.template GetBufferByByte<ElementC>(ubOffset);
        ubOffset += mmOutDim * sizeof(ElementC);
        AscendC::LocalTensor<XType> bf16TokenLocalTensor = resource.ubBuf.template GetBufferByByte<XType>(ubOffset);
        ubOffset += mmOutDim * sizeof(XType);
        AscendC::LocalTensor<ElementA> fp8TokenLocalTensor = resource.ubBuf.template GetBufferByByte<ElementA>(ubOffset);
        ubOffset += quantTokenSize + CEIL_UP(mxScaleNumPerToken * sizeof(ElementMxScaleB));
        AscendC::LocalTensor<uint8_t> mxScaleLocalTensor = fp8TokenLocalTensor[quantLength].template ReinterpretCast<uint8_t>();
        AscendC::LocalTensor<ElementC> tokenF32LT = resource.ubBuf.template GetBufferByByte<ElementC>(ubOffset);
        ubOffset += CEIL_UP(mxScaleNumPerToken * 2 * sizeof(float));
        AscendC::DataCopyExtParams mnxScaleParams = {1U, static_cast<uint8_t>(mxScaleNumPerToken * sizeof(uint8_t)), 0U, 0U, 0U};
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        for (uint32_t tokenIdx = startTokenIdx; tokenIdx < tokenNum; tokenIdx += aivNum) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
            AscendC::DataCopy(fp32TokenLocalTensor, gmSwigluOutTensor[tokenIdx * mmOutDim], mmOutDim);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
            AscendC::Mul(fp32TokenLocalTensor, fp32TokenLocalTensor, fp32TokenLocalTensor[quantLength], quantLength);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(bf16TokenLocalTensor, fp32TokenLocalTensor, AscendC::RoundMode::CAST_RINT, quantLength);
            AscendC::PipeBarrier<PIPE_V>();
            QuantDynamicMx(fp8TokenLocalTensor, bf16TokenLocalTensor, tokenF32LT, quantLength, mxScaleNumPerToken);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
            AscendC::DataCopy(gmX2[tokenIdx * quantLength], fp8TokenLocalTensor, quantLength);

            AscendC::DataCopyPad(gmX2MxScale[tokenIdx * mxScaleNumPerToken], mxScaleLocalTensor, mnxScaleParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        }
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        startCoreIdx = (startCoreIdx + tokenNum) % aivNum;
    }

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const &params) {
        AscendC::SetCtrlSpr<60, 60>(0);
        AivInitParams(params);
        if constexpr (EXEC_FLAG & EXEC_FLAG_DEEP_FUSE) {
            AivInitState();
            if constexpr (EXEC_FLAG & EXEC_FLAG_SHARED_EXPERT) {
                if (isShareQuantCore) {
                    shareQuantCoreFunc((GM_ADDR)params.gmX, (GM_ADDR)params.gmShareSmoothScales,
                                        (GM_ADDR)params.ptrShareA, (GM_ADDR)params.ptrShareMxScaleA);
                }
            }
            if (isSendCore) {
                SendCoreFunc((GM_ADDR)params.gmX, (GM_ADDR)params.gmExpertIds, (GM_ADDR)params.gmMoeSmoothScales,
                            (GM_ADDR)params.gmExpandIdx,
                            (GM_ADDR)params.gmXActiveMask);
            }
            if (isRecvCore) {
                RecvCoreFunc((GM_ADDR)params.ptrA, (GM_ADDR)params.ptrMxScaleA, (GM_ADDR)params.gmEpSendCount);
            }
        }

        uint32_t totalTokenNum = 0;

        uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t coreNum = AscendC::GetBlockNum();

        AscendC::GlobalTensor<ElementC> gmC;
        AscendC::GlobalTensor<ElementC> gmSwigluOutTensor;
        AscendC::GlobalTensor<ElementC> gmShareSwigluOutTensor;
        
        BlockEpilogue blockEpilogue(resource);
        uint32_t startCoreIdx = 0;
        uint32_t currentM = 0;
        uint32_t target = 1;

        if constexpr (EXEC_FLAG & EXEC_FLAG_SHARED_EXPERT) {
            currentM = axisBS;
            gmC.SetGlobalBuffer(params.ptrShareC);
            gmShareSwigluOutTensor.SetGlobalBuffer(params.gmShareSwigluOut);

            auto tensorC = tla::MakeTensor(gmC, params.layoutShareC, Arch::PositionGM{});
            auto tensorD = tla::MakeTensor(gmShareSwigluOutTensor, params.layoutShareC, Arch::PositionGM{});

            GemmCoord inGroupProblemShape{currentM, params.shareProblemShape.n(), params.shareProblemShape.k()};
            BlockScheduler matmulBlockScheduler(inGroupProblemShape, MakeCoord(L1_TILE_M, L1_TILE_N));
            uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

            uint32_t startLoopIdx;
            if (coreIdx < startCoreIdx) {
                startLoopIdx = coreIdx + coreNum - startCoreIdx;
            } else {
                startLoopIdx = coreIdx - startCoreIdx;
            }

            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += coreNum) {
                GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
                GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);

                auto tensorBlockC = GetTile(tensorC,
                    tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.n() * L1_TILE_N),
                    tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));

                auto tensorBlockD = GetTile(tensorD,
                    tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.n() * L1_TILE_N),
                    tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));

                bool isLeft = (blockCoord.n() * L1_TILE_N < params.shareProblemShape.n() / 2);
                CheckSyncFlag(reinterpret_cast<__gm__ int32_t*>(statusDataSpaceGm + SOFT_SYNC_OFFSET), static_cast<int32_t>(compCoreIdx), target);
                target += 1;
                blockEpilogue(tensorBlockC, tensorBlockD, actualBlockShape, isLeft);
            }
            startCoreIdx = (startCoreIdx + coreLoops) % coreNum;
        }
        {
            int64_t totalM = 0;
            uint32_t coreNumPerGroup = recvCoreNum;
            gmC.SetGlobalBuffer(params.ptrC);
            gmSwigluOutTensor.SetGlobalBuffer(params.gmSwigluOut);
            AscendC::GlobalTensor<ElementGroupList> groupList;
            groupList.SetGlobalBuffer(params.ptrGroupList);

            auto tensorC = tla::MakeTensor(gmC, params.layoutC, Arch::PositionGM{});
            auto tensorD = tla::MakeTensor(gmSwigluOutTensor, params.layoutC, Arch::PositionGM{});
            AscendC::GlobalTensor<int32_t> groupTokenNumStateTensor;

            for (uint32_t groupIdx = 0; groupIdx < params.problemCount; ++groupIdx) {
                if constexpr (EXEC_FLAG & EXEC_FLAG_DEEP_FUSE) {
                    groupTokenNumStateTensor.SetGlobalBuffer((__gm__ int32_t *)
                                                            (statusDataSpaceGm + GROUP_TOKEN_NUM_OFFSET) +
                                                            groupIdx * GROUP_INFO_SIZE);
                    CheckSyncFlag(reinterpret_cast<__gm__ int32_t*>(statusDataSpaceGm + SOFT_SYNC_OFFSET), static_cast<int32_t>(compCoreIdx), target);
                    target += 1;
                    currentM = FlushAndGetValue<int32_t>(groupTokenNumStateTensor, GROUP_TOKEN_COUNT);
                } else {
                    currentM = (groupIdx == 0) ? groupList.GetValue(groupIdx)
                                                : (groupList.GetValue(groupIdx) - groupList.GetValue(groupIdx - 1));
                }
                totalTokenNum += currentM;
                GemmCoord inGroupProblemShape{currentM, params.problemShape.n(), params.problemShape.k()};
                BlockScheduler matmulBlockScheduler(inGroupProblemShape, MakeCoord(L1_TILE_M, L1_TILE_N));
                uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

                uint32_t startLoopIdx;
                if (coreIdx < startCoreIdx) {
                    startLoopIdx = coreIdx + coreNum - startCoreIdx;
                } else {
                    startLoopIdx = coreIdx - startCoreIdx;
                }

                for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += coreNum) {
                    GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
                    GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);

                    auto tensorBlockC = GetTile(tensorC,
                        tla::MakeCoord(totalM + blockCoord.m() * L1_TILE_M, blockCoord.n() * L1_TILE_N),
                        tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));

                    auto tensorBlockD = GetTile(tensorD,
                        tla::MakeCoord(totalM + blockCoord.m() * L1_TILE_M, blockCoord.n() * L1_TILE_N),
                        tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));

                    bool isLeft = (blockCoord.n() * L1_TILE_N < params.problemShape.n() / 2);
                    CheckSyncFlag(reinterpret_cast<__gm__ int32_t*>(statusDataSpaceGm + SOFT_SYNC_OFFSET), static_cast<int32_t>(compCoreIdx), target);
                    target += 1;
                    blockEpilogue(tensorBlockC, tensorBlockD, actualBlockShape, isLeft);
                }

                totalM += inGroupProblemShape.m();

                startCoreIdx = (startCoreIdx + coreLoops) % coreNum;
            }
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        icache_preload(8);
        AscendC::SyncAll<false>();
        AscendC::PipeBarrier<PIPE_ALL>();

        UpdateAndCleanInfo(params.ptrGroupList, params.gmEpSendCount, params.gmExpertTokenNums);
        AscendC::PipeBarrier<PIPE_ALL>();
        startCoreIdx = 0;
        if constexpr (EXEC_FLAG & EXEC_FLAG_SHARED_EXPERT) {
            PostSwigluDynamicQuant(params.gmShareSwigluOut, params.ptrShareX2, params.gmShareX2Scale,
                                   axisBS, params.shareProblemShape.n(), startCoreIdx);
        }
        {
            PostSwigluDynamicQuant(params.gmSwigluOut, params.ptrX2, params.gmX2Scale,
                                   totalTokenNum, params.problemShape.n(), startCoreIdx);
        }
    }

private:
    friend struct AicSetFunc;
    struct AicSetFunc {
        CATLASS_DEVICE
        AicSetFunc() = default;

        CATLASS_DEVICE
        void operator()() const
        {
            EncreaseSyncFlag(flagAddr, idx);
        }

        __gm__ int32_t *flagAddr;
        int32_t idx;
    };

    AicSetFunc aicSetFunc;
    Arch::Resource<ArchTag> resource;

    AscendC::LocalTensor<int32_t> expertIdsTensor_;
    // count info
    int32_t countPerRank[16]{0};
    int32_t curTokenIdx[16]{0};
    int32_t rankBeginIdx[16]{0};

    // rank and expert info
    uint32_t epRankSize{0};
    uint32_t epRankId{0};
    uint32_t expertCntUp{0};
    uint32_t localExpertNum{0};
    uint32_t moeExpertNumPerRank{0};
    uint32_t moeExpertNum{0};

    // token info
    uint32_t hOutSize{0};
    uint32_t scaleFlagSize{0};
    uint32_t scaleSize{0};
    uint32_t hCommuSize{0};
    uint32_t axisHCommu{0};
    uint32_t axisBS{0};
    uint32_t activeMaskBsCnt{0};
    uint32_t axisK{0};
    uint32_t totalTokenCount{0};
    uint32_t expertIdsCnt{0};
    uint32_t tokenLength{0};
    uint32_t x1MxScaleNum{0};
    uint32_t x2MxScaleNum{0};

    // state info
    int32_t tokenFlag{0};    // token flag
    int32_t vToCFlag{0};     // cv flag, decided by cvDataState
    int32_t dataState{0};    // data space state
    int32_t cvDataState{0};  // cv flag state
    int32_t state{0};        // count flag state
    float sumTarget{0.0};

    // memory info
    __gm__ Mc2Kernel::HcclOpParam *winContext_;
    GM_ADDR statusDataSpaceGm;
    uint32_t stateOffset{0};
    uint64_t expertPerSizeOnWin{0};
    uint64_t winDataSizeOffset{0};

    int64_t ubOffset;

    // core info
    bool isSendCore{false};
    bool isRecvCore{false};
    bool isCompCore{false};  // calculate deq_swiglu
    bool isShareQuantCore{false}; // calculate share quant
    uint32_t aiCoreGroupNum{0};
    uint32_t aiCoreGroupIdx{0};
    uint32_t subBlockNum{0};
    uint32_t aicNum{0};
    uint32_t aivNum{0};
    uint32_t sendCoreNum{0};
    uint32_t recvCoreNum{0};
    uint32_t compCoreNum{0};
    uint32_t shareQuantCoreNum{0};
    uint32_t aivIdx{0};
    uint32_t aicIdx{0};
    uint32_t sendCoreIdx{0};
    uint32_t recvCoreIdx{0};
    uint32_t compCoreIdx{0};
    uint32_t shareQuantCoreIdx{0};
    uint32_t aivStateGlobalCoreIdx{0};
    uint32_t aicStateGlobalCoreIdx{0};
    uint32_t sendToMoeAivNum{0};
    uint32_t sendToShareAivNum{0};

    AscendC::LocalTensor<bool> maskInputTensor;
    AscendC::LocalTensor<int8_t> maskInputInt8Tensor;
    AscendC::LocalTensor<half> maskTmpTensor;
    AscendC::LocalTensor<half> sumOutTensor;
    AscendC::LocalTensor<uint8_t> sharedTmpBuffer;

    AscendC::LocalTensor<int32_t> dstExpIdTensor_;
    AscendC::LocalTensor<float> dstExpIdFp32Tensor_;

    AscendC::LocalTensor<float> xFp32TmpTensor;
    AscendC::LocalTensor<ElementC> tokenF32LT;
    AscendC::LocalTensor<int32_t> yInt32Tensor;

    AscendC::LocalTensor<int32_t> expertCountTensor;

    AscendC::LocalTensor<XType> xInTensor[BUFFER_NUM];
    AscendC::LocalTensor<ElementA> yInt8Tensor[BUFFER_NUM];
    AscendC::LocalTensor<ElementMxScaleA> yScaleTensor[BUFFER_NUM];
    AscendC::LocalTensor<float> moeSmoothScaleTensor[BUFFER_NUM];
    AscendC::LocalTensor<float> shareSmoothScaleTensor;

    AscendC::LocalTensor<int32_t> statusTensor_;
    AscendC::LocalTensor<float> statusFp32Tensor_;
    AscendC::LocalTensor<float> gatherMaskOutTensor;
    AscendC::LocalTensor<int32_t> gatherMaskOutCountTensor;
    AscendC::LocalTensor<float> statusSumOutTensor;
    AscendC::LocalTensor<uint8_t> sumTmpTensor;
    AscendC::LocalTensor<ElementA> xTmpTensor_;
    AscendC::LocalTensor<ElementMxScaleA> xOutFp32Tensor_;
    AscendC::LocalTensor<uint32_t> gatherTmpTensor;
    AscendC::LocalTensor<int32_t> tmpLocalTensor;
    AscendC::LocalTensor<float> reduceSumWorkLocalTensor;
    AscendC::LocalTensor<int32_t> sendCountsLocalTensor;
};

#endif // (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)

} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_DISPATCH_MX_GMM1_SWIGLU_H
