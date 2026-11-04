/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mega_moe.h
 * \brief
 */

#ifndef MEGA_MOE_H
#define MEGA_MOE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "simt_api/device_sync_functions.h"
#include "../../common/op_kernel/mc2_kernel_utils.h"
#include "kernel_operator_list_tensor_intf.h"
#include "mega_moe_base.h"
#include "mega_moe_workspace_info.h"
#include "block_epilogue_swiglu_mx_quant.h"
#if __has_include("moe_distribute_dispatch_v2/op_kernel/quantize_functions.h")
#include "moe_distribute_dispatch_v2/op_kernel/quantize_functions.h"
#else
#include "moe_distribute_dispatch_v2/quantize_functions.h"
#endif
#include "tla/layout.hpp"
#include "mega_moe_catlass_impl.h"
#include "AscendTimerV2.hpp"
#include "AscendTimerV2_device.hpp"

using namespace AscendC;

namespace MegaMoeImpl {
using TupleShape = Shape<int64_t, int64_t, int64_t, int64_t>;
using BlockOffset =
    Shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;

constexpr uint32_t ALLGATHER_MAX_SHARDS_PER_RANK = 32U;
constexpr uint32_t ALLGATHER_MIN_BYTES_PER_SHARD = 2U * 1024U;
constexpr uint32_t ALLGATHER_GM_BLOCK_BYTES = 32U;
// This VF has fourteen parameters plus a binary-search/source-resolution
// loop.  1024 threads give Bisheng twice the per-thread register budget of
// the 2048-thread setting, while current token-local row slices are already
// smaller than 1024.  Larger future slices remain fully covered by the
// rowOffset += threadNum loop below.
constexpr uint32_t OWNER_DISPATCH_SIMT_THREAD_NUM = 1024U;

// The token-owner pipeline materializes one immutable endpoint per physical
// expert.  Let SIMT threads turn one contiguous canonical-row slice
// into the inverse token route plan: random GM reads and small descriptor
// stores are a much better fit for SIMT than for the AIV scalar controller.
// Every canonical row and every original top-k slot is unique, so this kernel
// needs neither atomics nor an inter-thread barrier; only the publication
// fence at the end orders its independent GM stores before the scalar vote.
__simt_vf__ __aicore__ LAUNCH_BOUND(OWNER_DISPATCH_SIMT_THREAD_NUM) inline void BuildOwnerDispatchFanoutPlanSimt(
    int32_t rowBegin, int32_t rowCount, int32_t expertPerRank, int32_t worldSize, int32_t rankId, int32_t expertAlign,
    int32_t mTopK, uint32_t scheduleBits, uint32_t payloadBits, __gm__ int32_t *expertEndpoints,
    __gm__ int32_t *physicalToSchedule, __gm__ int32_t *tokenPerExpert, __gm__ int32_t *expandedRowIdx,
    __gm__ int32_t *descriptors)
{
    int32_t threadIdx = static_cast<int32_t>(Simt::GetThreadIdx());
    int32_t threadNum = static_cast<int32_t>(Simt::GetThreadNum());
    for (int32_t rowOffset = threadIdx; rowOffset < rowCount; rowOffset += threadNum) {
        int32_t canonicalRow = rowBegin + rowOffset;

        // endpoint[e] is the exclusive local canonical-row end of physical
        // expert e.  It is monotone even when empty experts are present.
        int32_t expertLo = 0;
        int32_t expertHi = expertPerRank;
        while (expertLo < expertHi) {
            int32_t expertMid = expertLo + (expertHi - expertLo) / 2;
            if (canonicalRow < expertEndpoints[expertMid]) {
                expertHi = expertMid;
            } else {
                expertLo = expertMid + 1;
            }
        }
        if (expertLo >= expertPerRank) {
            continue;
        }
        int32_t expertIdx = expertLo;
        int32_t expertStart = expertIdx == 0 ? 0 : expertEndpoints[expertIdx - 1];
        int32_t rowInExpert = canonicalRow - expertStart;
        int32_t scheduleIdx = physicalToSchedule[expertIdx];
        if (scheduleIdx < 0 || scheduleIdx >= expertPerRank) {
            continue;
        }

        int32_t globalExpertIdx = rankId * expertPerRank + expertIdx;
        int32_t sourceBaseRow = 0;
        int32_t sourceRank = 0;
        int32_t sourcePrevious = 0;
        int32_t rowInSource = 0;
        for (; sourceRank < worldSize; ++sourceRank) {
            int32_t countOffset = sourceRank * expertAlign + globalExpertIdx;
            int32_t sourceCurrent = tokenPerExpert[countOffset];
            sourcePrevious = globalExpertIdx == 0 ? 0 : tokenPerExpert[countOffset - 1];
            int32_t sourceRows = sourceCurrent - sourcePrevious;
            if (rowInExpert < sourceBaseRow + sourceRows) {
                rowInSource = rowInExpert - sourceBaseRow;
                break;
            }
            sourceBaseRow += sourceRows;
        }
        if (sourceRank >= worldSize) {
            continue;
        }

        int32_t expandedBase = sourceRank * mTopK + sourcePrevious;
        int32_t expandedIdx = expandedRowIdx[expandedBase + rowInSource];
        // The original local [token,slot] index is exactly expandedIdx for
        // every supported token-local top-k width.  The expert-local M tile
        // remains a fixed 256-row shift, so this inverse-plan construction is
        // shared by K8 and K16 without a shape-specific branch.
        uint32_t tileIdx = static_cast<uint32_t>(rowInExpert) / EXPERT_SCHEDULE_M_TILE_ROWS;
        uint32_t descriptor = (static_cast<uint32_t>(canonicalRow) << payloadBits) | (tileIdx << scheduleBits) |
                              static_cast<uint32_t>(scheduleIdx);
        int32_t descriptorOffset = sourceRank * mTopK + expandedIdx;
        descriptors[descriptorOffset] = static_cast<int32_t>(descriptor);
    }
    // The scalar controller publishes one cross-AIV completion vote as soon
    // as VF_CALL returns.  Make every SIMT writer's descriptor globally
    // visible before that vote; a local scalar pipe barrier alone does not
    // define ordering for stores issued by SIMT threads.
    asc_threadfence();
}

// UseBf16NoQuant: dispatch_quant_mode=0. Activations and weights stay BF16
// end to end (no MXFP8 dispatch quant, no MX scales, SwiGLU without quant).
#define TemplateMegaMoeTypeClass                                                                              \
    typename XType, typename OutputType, typename TopkWeightsType, int32_t QuantMode, bool UseCoreWave,       \
        bool UseOwnerChainCombine, bool UseK16OwnerChain, bool UseSparseDecode, bool UseRaggedGmm2StripeMask, \
        bool UseBf16NoQuant
#define TemplateMegaMoeTypeFunc                                                                         \
    XType, OutputType, TopkWeightsType, QuantMode, UseCoreWave, UseOwnerChainCombine, UseK16OwnerChain, \
        UseSparseDecode, UseRaggedGmm2StripeMask, UseBf16NoQuant

// Scalar-only launch plan. No TPipe, local-memory view, hardware event or
// readiness publication may be acquired while the GMM1 context is live.
struct CoreWaveGmm2Plan {
    uint32_t nTiles;
    uint32_t groupUnits;
    uint32_t totalMGroups;
    uint32_t totalFullMGroups;
    bool useDenseTileGrid;
    bool useFullAReadyProtocol;
    bool publishStripeReady;
};

template <TemplateMegaMoeTypeClass>
class MegaMoe
{
public:
    static_assert(!UseOwnerChainCombine || UseCoreWave, "owner-chain Combine requires the core-wave producer protocol");
    static_assert(!UseK16OwnerChain || UseOwnerChainCombine,
                  "K16 owner-chain specialization requires owner-chain Combine");
    static_assert(!UseRaggedGmm2StripeMask || UseK16OwnerChain,
                  "ragged-N readiness requires the matching owner-chain consumer");
    static_assert(!UseSparseDecode || !UseCoreWave,
                  "sparse-decode is a reduced generic protocol; it must never inherit "
                  "the core-wave epilogue/readiness contract");
    static_assert(!UseBf16NoQuant || (!UseCoreWave && !UseSparseDecode),
                  "BF16 no-quant is served by the generic pipeline only");
    // Dispatch payload element. BF16 no-quant routes GMM1 (BlockMmadMxMoe)
    // and its epilogue (BlockEpilogueSwigluMxQuant) to their BF16
    // specializations; the MX scale type below is then never read.
    using QuantOutType = typename std::conditional<
        UseBf16NoQuant, bfloat16_t,
        typename std::conditional<((QuantMode >= 3) && (QuantMode != 4)), fp8_e5m2_t, fp8_e4m3fn_t>::type>::type;
    using QuantScaleOutType = typename std::conditional<(QuantMode >= 3), fp8_e8m0_t, float>::type;
    __aicore__ inline MegaMoe(){};
    __aicore__ inline void Init(GM_ADDR context, GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights, GM_ADDR weight1,
                                GM_ADDR weight2, GM_ADDR xActiveMask, GM_ADDR weightScales1, GM_ADDR weightScales2,
                                GM_ADDR timerAddr, GM_ADDR yOut, GM_ADDR expertTokenNumsOut, GM_ADDR workspaceGM,
                                MegaMoeTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void FirstBuffInit(TPipe *tPipe);
    __aicore__ inline void SecondBuffInit(TPipe *tPipe);
    __aicore__ inline void ResetData();
    __aicore__ inline void ResetDispatchTileReadyFull();
    __aicore__ inline void ResetActiveDispatchTileReady();
    __aicore__ inline void Mxfp8DynamicQuant(TPipe *tPipe);
    __aicore__ inline void CopyBf16TokensToA0();
    __aicore__ inline void AllGatherData();
    __aicore__ inline void WriteExpertTokenNumsOut();
    template <typename T>
    __aicore__ inline void CopyGMToGM(GlobalTensor<T> dst, GlobalTensor<T> src, int32_t dataLen);
    template <typename T>
    __aicore__ inline void CopyAllGatherByRank(GlobalTensor<T> src, uint64_t peerMemSt, int64_t rankOffset,
                                               int32_t dataLen, bool skipLocalRank);
    __aicore__ inline void DispatchTokens(GMMAddrInfo &gmmAddrInfo, uint32_t expertIdx,
                                          bool nextDispatchCoversJoin = false);
    __aicore__ inline void PrecomputeChainLen();
    __aicore__ inline void DispatchExperts(GMMAddrInfo &gmmAddrInfo, uint32_t scheduleBegin, uint32_t scheduleEnd,
                                           int32_t rankBaseRow);
    __aicore__ inline bool DispatchFanoutStorageCapable() const;
    __aicore__ inline bool DispatchFanoutWorkProfitable(int32_t rankBaseRow);
    __aicore__ inline void ResetDispatchFanoutDescriptors();
    __aicore__ inline void BuildDispatchFanoutPlan(int32_t rankBaseRow);
    __aicore__ inline void DispatchTokensByOwner(int32_t rankBaseRow);
    __aicore__ inline void FlushDispatchFanoutReady(LocalTensor<int32_t> countTable,
                                                    LocalTensor<int32_t> touchedBuckets, uint32_t countStride,
                                                    int32_t &touchedCount, int32_t &pendingRoutes, int32_t &copySlotIdx,
                                                    int32_t &usedCopySlots);
    __aicore__ inline uint32_t GetMGroupsPerWave();
    __aicore__ inline int32_t PrepareExpertPipeline();
    template <bool EnableBf16L1Reuse = false>
    __aicore__ inline void ProcessGenericGmm2(int32_t rankBaseRow);
    // 共享专家：输入准备（AIV，a0 去交错）、GMM1+GMM2（AIC，复用路由调用）、
    // 累加（AIV，在 UnpermuteCombineDedup 内）。sharedExpertNum_ == 0 时全部空转。
    // 共享专家三段的插入位置（generic 流水）：
    //   ProcessSharedGmm1 —— 在 PrepareExpertPipeline 的两道混核屏障之后、路由
    //                        gmm1Context 构造之前调用（AIC 矩阵乘 + sub0 的
    //                        SwiGLU epilogue；与 sub1 的 Dispatch 并发）。共享
    //                        Gmm1Context（Blaze）与路由 Gmm1Context 不可共存，
    //                        共享链必须整体先于路由上下文构造结束。末尾 credit
    //                        并入 gmm1VecSetSyncCom_：路由 GMM1 首个 tile 的
    //                        MMAD 与共享 SwiGLU 尾部并行，首个 CopyOut 才等待；
    //                        轮转游标同样续接进 startBlockIdx_，路由首个专家的
    //                        tile 优先落到没跑共享专家的核。
    //   ProcessSharedGmm2 —— 仅 AIC，紧跟本核路由 GMM2 之后执行（等 epilogue 标志）；
    //                        轮转游标以路由 GMM2 留下的相位为初值跨共享专家轮转。
    __aicore__ inline void PrepareSharedExpertInput(TPipe *tPipe);
    __aicore__ inline void ProcessSharedGmm1();
    __aicore__ inline void ProcessSharedGmm2();
    __aicore__ inline void ProcessInterleavedGmmPipeline(TPipe *firstPipe, int32_t rankBaseRow);
    __aicore__ inline void ProcessCoreWavePipeline(TPipe *firstPipe, int32_t rankBaseRow);
    __aicore__ inline CoreWaveGmm2Plan PrepareCoreWaveGmm2Plan(int32_t rankBaseRow);
    template <bool UseFullAReadyProtocol>
    __aicore__ inline void ProcessGmm2MnTiles(int32_t rankBaseRow, uint32_t totalFullMGroups, uint32_t nTiles,
                                              uint32_t gmm2StartBlockIdx, bool publishStripeReady);
    __aicore__ inline void WaitForGmm1ExpertReady(const GMMAddrInfo &gmmAddrInfo, uint32_t rows, bool &waited);
    // Experimental range helper retained for future non-static schedulers.
    // The production hot path below uses one specialized contiguous range.
    __aicore__ inline int32_t ProcessDispatchRange(GMMAddrInfo &gmmAddrInfo, uint32_t expertIdx, int32_t startRow,
                                                   int32_t endRow, int64_t widthA, int64_t widthAScale,
                                                   int32_t &copySlotIdx, int32_t &usedCopySlots,
                                                   int32_t &sourceRankHint, int32_t &sourceRankBaseRow);
    __aicore__ inline void BuildExpertOrder();
    __aicore__ inline void LoadExpertScheduleMode();
    __aicore__ inline uint32_t GetScheduledExpert(uint32_t scheduleIdx);
    __aicore__ inline bool GetExpertRange(uint32_t expertIdx, int32_t rankBaseRow, int32_t &localStart, int32_t &rows);
    __aicore__ inline bool UpdateIdentityGroupParams(uint32_t expertIdx);
    __aicore__ inline bool SetGroupParams(uint32_t expertIdx, int32_t rankBaseRow);
    template <bool PublishGmm1Ready = false>
    __aicore__ inline void UpdateGlobalBuffer(GMMAddrInfo &gmmAddrInfo, uint32_t gmmIndex);
    __aicore__ inline void ResetTokenPerExpert();
    __aicore__ inline void UnpermuteCombineDedup();
    __aicore__ inline void EndSync();
    __aicore__ inline void CrossRankSyncInWorldSize();
    __aicore__ inline void CopyGMToGMPerTokenV2(GlobalTensor<QuantOutType> dst,
                                                GlobalTensor<QuantScaleOutType> dstScale,
                                                GlobalTensor<QuantOutType> src, int32_t hiddenSize, int32_t scaleLen,
                                                int32_t copyInNum, int32_t scaleOffset, int32_t tokenInAllRank,
                                                int32_t canonicalRow, int32_t stepK, int32_t stepMetadata,
                                                int32_t copyBufferCount, uint32_t copySlotElements,
                                                int32_t &copySlotIdx, int32_t &usedCopySlots);
    // Scale is selected as a byte view so aligned local workspace rows and
    // padded remote A0 rows share the same copy implementation.
    __aicore__ inline void CopyLocalQuantToken(GlobalTensor<QuantOutType> dst, GlobalTensor<QuantScaleOutType> dstScale,
                                               GlobalTensor<QuantOutType> src, GlobalTensor<uint8_t> srcScale,
                                               int32_t hiddenSize, int32_t scaleLen, int32_t scaleOffset,
                                               int32_t scaleLenAlign, int32_t tokenInAllRank, int32_t canonicalRow,
                                               int32_t stepK, int32_t stepMetadata, int32_t copyBufferCount,
                                               uint32_t copySlotElements, int32_t &copySlotIdx, int32_t &usedCopySlots);
    __aicore__ inline int32_t GetLocalExpertStartRow();

    __aicore__ inline void TilingByCore(int32_t totalLen, int32_t &coreLen, int32_t &coreOffset,
                                        int32_t align = ALIGN_32)
    {
        int32_t coreIdx = GetBlockIdx();
        int32_t coreNum = GetBlockNum() * 2;  // 取到vec核总数
        int32_t lenPerCore = Ops::Base::CeilDiv(static_cast<uint32_t>(totalLen), static_cast<uint32_t>(coreNum));
        int32_t lenPerCoreAlign = Ops::Base::CeilAlign(static_cast<uint32_t>(lenPerCore), static_cast<uint32_t>(align));
        coreLen = lenPerCoreAlign;
        coreOffset = coreIdx * lenPerCoreAlign;
        if (coreOffset + coreLen >= totalLen) {
            coreLen = totalLen - coreOffset;
        }
        if (coreOffset >= totalLen) {
            coreLen = 0;
        }
    }

    // Common prefill shapes use topK=8 or 16.  Avoid a runtime integer divide
    // in every Dispatch/Combine row while preserving the generic fallback.
    __aicore__ inline int32_t DecodeExpandedTokenRow(int32_t expandedIdx) const
    {
        if (topK_ == 8U) {
            return expandedIdx >> 3;
        }
        if (topK_ == 16U) {
            return expandedIdx >> 4;
        }
        return expandedIdx / static_cast<int32_t>(topK_);
    }

    __aicore__ inline uint32_t DecodeExpandedTopKSlot(int32_t expandedIdx) const
    {
        if (topK_ == 8U) {
            return static_cast<uint32_t>(expandedIdx) & 7U;
        }
        if (topK_ == 16U) {
            return static_cast<uint32_t>(expandedIdx) & 15U;
        }
        return static_cast<uint32_t>(expandedIdx) % topK_;
    }

    // Power-of-two E64 avoids the per-row runtime divide used to recover the
    // owner rank; the established small-EP comparisons and generic fallback
    // remain below.
    __aicore__ inline int32_t DecodeExpertOwnerRank(int32_t expertIdx) const
    {
        int32_t experts = static_cast<int32_t>(expertPerRank_);
        if (expertPerRank_ == 64U) {
            return expertIdx >> 6;
        }
        if (worldSize_ == 4U) {
            if (expertIdx >= 3 * experts) {
                return 3;
            }
            if (expertIdx >= 2 * experts) {
                return 2;
            }
            return expertIdx >= experts ? 1 : 0;
        }
        return expertIdx / experts;
    }

    // Cross-rank barrier slot wait (CrossRankSyncInWorldSize).  Accept a peer
    // that has already published the next round into the same slot.  Back off
    // instead of a fixed 100-cycle cadence: on an imbalanced tail hundreds
    // of vector cores across all ranks probe a few GM words at high
    // frequency, competing with the slow rank's payload traffic on
    // GM/L2/NoC and delaying the very signal they wait for (FP80918 L1).
    __aicore__ inline void GmSignalWaitBarrier(__gm__ int32_t *sig_addr, uint32_t cmp_val)
    {
        uint32_t pollMisses = 0U;
        int64_t pollDelayCycles = 100;
        while (!IsCrossRankBarrierReached(static_cast<uint32_t>(ReadGmByPassDCache(sig_addr)), cmp_val)) {
            GmSpinBackoffStep(pollMisses, pollDelayCycles);
        }
    }

    TBuf<> CopyTempBuf_;
    TBuf<> DataResBuf_;
    TBuf<> RankFlagBuf_;  // UnpermuteCombineDedup中记录token需要读取的rank标记
    TQue<QuePosition::VECIN, 1> QuantXInQueue_;
    TQue<QuePosition::VECOUT, 1> QuantOutQueue_;
    TBuf<QuePosition::VECCALC> QuantMaxExpBuf_;
    TBuf<QuePosition::VECCALC> QuantInvScaleBuf_;

    __gm__ Mc2MoeContext *mc2Context_{nullptr};
    Params params_{};
    TupleShape problemShape_{};
    BlockOffset baseOffset_{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    GlobalTensor<int32_t> tokenPerExpertWin_;
    GlobalTensor<int32_t> flagSwigluToGmm2_;
    GlobalTensor<int32_t> dispatchDoneCores_;
    GlobalTensor<int32_t> dispatchTileReady_;
    GlobalTensor<int32_t> expertTokenCumsum_;
    GlobalTensor<int32_t> expertOrder_;
    GlobalTensor<int32_t> expertOrderScratch_;
    GlobalTensor<bfloat16_t> combineAccumGm_;
    GlobalTensor<bfloat16_t> combineAccumGmGlobal_;

    // 公共GlobalTensor，避免Dispatch/Combine内部重复声明
    GlobalTensor<int32_t> expandedRowIdx_;
    GlobalTensor<bfloat16_t> distProbs_;
    __gm__ int32_t *firstTokenPosPtr_ = nullptr;
    // [maxOutputSize] int32: raw stepK for small/fallback launches; otherwise
    // packed (sourceRank, tokenRow, topKSlot, stepK) at the canonical row.
    __gm__ int32_t *stepIdxPtr_ = nullptr;
    __gm__ int32_t *tokenChainDescriptorsPtr_ = nullptr;
    __gm__ int32_t *posLenPtr_ = nullptr;             // [w][m] int32: token链路长度
    __gm__ int32_t *posDonePtr_ = nullptr;            // [w][m] completed expert-major steps
    __gm__ int32_t *chainLenPtr_ = nullptr;           // [w][m] int32: token在本rank的expert数(AIV0预计算)
    __gm__ int32_t *chainLenReadyFlagPtr_ = nullptr;  // generic chainLen / core-wave final-posLen publication
    GlobalTensor<int32_t> firstTokenPos_;
    GlobalTensor<int32_t> stepIdx_;
    GlobalTensor<int32_t> posLen_;
    GlobalTensor<int32_t> posDone_;
    GlobalTensor<int32_t> chainLen_;
    GlobalTensor<int32_t> chainLenReady_;
    GlobalTensor<int32_t> gmm2ReadyFlag_;

    uint32_t m_ = 0;
    uint32_t k_ = 0;
    uint32_t topK_ = 0;
    uint32_t rankId_ = 0;
    uint32_t worldSize_ = 0;
    uint32_t expertPerRank_ = 0;
    uint64_t maxOutputSize_ = 0;
    uint32_t maxDispatchTilesPerExpert_ = 0;
    uint32_t dispatchPriorityStride_ = 1U;
    int32_t gmm1VecSetSyncCom_ = 0;
    uint32_t startBlockIdx_ = 0;
    uint32_t blockNum_ = GetBlockNum();
    uint32_t blockAivNum_ = GetBlockNum() * 2;
    uint32_t blockIdx_ = GetBlockIdx() / GetTaskRation();
    uint32_t aivCoreIdx_ = GetBlockIdx();
    uint32_t subBlockIdx_ = GetSubBlockIdx();
    bool expertScheduleEnabled_ = false;
    uint64_t preOffset_ = 0;
    // 共享专家权重基地址。列表可能为空（未启用），因此这里只保存基地址，
    // 由消费侧按 sharedExpertNum 决定是否解引用。
    GM_ADDR sharedWeight1GmAddr_ = nullptr;
    GM_ADDR sharedWeight2GmAddr_ = nullptr;
    GM_ADDR sharedWeightScales1GmAddr_ = nullptr;
    GM_ADDR sharedWeightScales2GmAddr_ = nullptr;
    // 共享专家：不参与 Dispatch/Combine，每卡对本卡全部 token 本地计算。
    uint32_t sharedExpertNum_ = 0;
    uint32_t sharedN_ = 0;  // GMM1 输出宽度 (2 * shared intermediate)
    uint32_t sharedK_ = 0;  // GMM2 输入宽度 (sharedN / SWIGLU_N_HALF)

    int32_t localExpandedRowIdxSt_;
    int32_t localTokenPerExpertSt_;

    using L1TileShape = tla::Shape<tla::Int<256>, tla::Int<256>, tla::Int<256>>;
    using BlockEpilogue = BlockEpilogueSwigluMxQuant<L1TileShape, QuantOutType, bfloat16_t, QuantScaleOutType,
                                                     QuantScaleOutType, true, false, !UseCoreWave>;
    using StreamingBlockEpilogue = BlockEpilogueSwigluMxQuant<L1TileShape, QuantOutType, bfloat16_t, QuantScaleOutType,
                                                              QuantScaleOutType, true, false, true>;
    BlockEpilogue epilogueOp_;  // GMMswiglu调用
    StreamingBlockEpilogue streamingEpilogueOp_;
    // 共享专家复用自己的 epilogue 实例：地址绑定到共享专家的 activation 缓冲，
    // 与路由实例互不干扰（UpdateGlobalAddr 写的是各自的成员）。
    using SharedBlockEpilogue = BlockEpilogueSwigluMxQuant<L1TileShape, QuantOutType, bfloat16_t, QuantScaleOutType,
                                                           QuantScaleOutType, true, false, true>;
    SharedBlockEpilogue sharedEpilogueOp_;

    // 无条件声明: release 下为空 stub (零字节), 调用都在 TIMER_BLOCK 内不展开,
    // 门控反而会造成 release 下调用点编译断裂。
    AscendTimerDevice timer;
    AscendTimerDevice timerAll;
};

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::Init(GM_ADDR context, GM_ADDR x, GM_ADDR topkIds,
                                                              GM_ADDR topkWeights, GM_ADDR weight1, GM_ADDR weight2,
                                                              GM_ADDR xActiveMask, GM_ADDR weightScales1,
                                                              GM_ADDR weightScales2, GM_ADDR timerAddr, GM_ADDR yOut,
                                                              GM_ADDR expertTokenNumsOut, GM_ADDR workspaceGM,
                                                              MegaMoeTilingData *tilingData)
{
    m_ = tilingData->m;
    k_ = tilingData->k;
    topK_ = tilingData->topK;
    worldSize_ = tilingData->epWorldSize;
    expertPerRank_ = tilingData->expertPerRank;
    maxOutputSize_ = tilingData->maxOutputSize;
    maxDispatchTilesPerExpert_ = Ops::Base::CeilDiv(tilingData->maxOutputSize, EXPERT_SCHEDULE_M_TILE_ROWS);
    Get<N_VALUE>(problemShape_) = tilingData->n;
    Get<K_VALUE>(problemShape_) = k_;
    mc2Context_ = reinterpret_cast<__gm__ Mc2MoeContext *>(context);
    rankId_ = mc2Context_->epRankId;
    for (int i = 0; i < worldSize_; i++) {
        winRankAddr_[i] = (GM_ADDR)mc2Context_->epHcclBuffer[i];
    }
    params_.aGmAddr = x;
    params_.expertIdxGmAddr = topkIds;
    params_.bGmAddr = GetTensorAddr(0, weight1);
    params_.b2GmAddr = GetTensorAddr(0, weight2);
    // 共享专家走合列表协议：共享权重/scala 是 weight1/weight2/scale 列表的
    // 追加元素（下标 1），原型不再有独立 shared 输入。未启用（sharedExpertNum
    // == 0，host tiling 由 shared_expert_num 属性判定）时不访问追加元素。BF16
    // 权重不带 MX scale，与路由 bScaleGmAddr 同款保持 nullptr。
    sharedExpertNum_ = tilingData->sharedExpertNum;
    sharedN_ = tilingData->sharedN;
    sharedK_ = sharedN_ / SWIGLU_N_HALF;
    if (sharedExpertNum_ > 0U) {
        sharedWeight1GmAddr_ = GetTensorAddr(1, weight1);
        sharedWeight2GmAddr_ = GetTensorAddr(1, weight2);
        if constexpr (UseBf16NoQuant) {
            sharedWeightScales1GmAddr_ = nullptr;
            sharedWeightScales2GmAddr_ = nullptr;
        } else {
            sharedWeightScales1GmAddr_ = GetTensorAddr(1, weightScales1);
            sharedWeightScales2GmAddr_ = GetTensorAddr(1, weightScales2);
        }
    }
    // 注意：共享 epilogue 的 Init 不能放在这里——它按值捕获 workspace 里的三个
    // 指针，而 params_.workspaceInfo 到下面才赋值，放这里会把 nullptr 绑进去，
    // 首次 DataCopy/AtomicAdd 就越界。实际初始化在 workspaceInfo 赋值之后。
    if constexpr (UseBf16NoQuant) {
        // BF16 weights carry no scales; the scale lists may be empty.
        params_.bScaleGmAddr = nullptr;
        params_.b2ScaleGmAddr = nullptr;
    } else {
        params_.bScaleGmAddr = GetTensorAddr(0, weightScales1);
        params_.b2ScaleGmAddr = GetTensorAddr(0, weightScales2);
    }

    params_.y2GmAddr = yOut;
    params_.expertTokenNumsOutGmAddr = expertTokenNumsOut;
    params_.probsGmAddr = topkWeights;
    params_.workspaceInfo = WorkspaceInfo(workspaceGM, tilingData);
    params_.peermemInfo = PeermemInfo(winRankAddr_[rankId_], tilingData);
    params_.tilingData = tilingData;
    // 共享 epilogue 绑定共享专家的 activation 缓冲，与路由实例互不干扰。
    // 必须晚于 workspaceInfo 赋值：Init 按值捕获这些指针。
    // QuantOutType = bfloat16_t 时 BlockEpilogueSwigluMxQuant 特化为 SwiGLU-only
    // 的 BlockEpilogueSwigluBf16（scale 槽位不读）， Arguments 布局保持一致。
    if (sharedExpertNum_ > 0U) {
        if constexpr (UseBf16NoQuant) {
            sharedEpilogueOp_.Init({params_.workspaceInfo.ptrSharedActivation, nullptr,
                                    params_.workspaceInfo.ptrSharedReadyFlag, nullptr, nullptr, nullptr, ALIGN_256,
                                    ALIGN_256});
        } else {
            sharedEpilogueOp_.Init(
                {params_.workspaceInfo.ptrSharedActivation, params_.workspaceInfo.ptrSharedActivationScale,
                 params_.workspaceInfo.ptrSharedReadyFlag, nullptr, nullptr, nullptr, ALIGN_256, ALIGN_256});
        }
    }
    tokenPerExpertWin_.SetGlobalBuffer((__gm__ int32_t *)params_.peermemInfo.ptrTokenPerExpert);
    expertTokenCumsum_.SetGlobalBuffer((__gm__ int32_t *)params_.peermemInfo.ptrExpertTokenCumsum);
    expertOrder_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.ptrExpertOrder));
    expertOrderScratch_.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.ptrExpertOrderScratch));
    combineAccumGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(params_.workspaceInfo.ptrCombineAccum));
    combineAccumGmGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(params_.workspaceInfo.ptrCombineAccum));
    epilogueOp_.Init({params_.workspaceInfo.ptrA2, params_.workspaceInfo.ptrA2Scale,
                      params_.workspaceInfo.ptrFlagSwiGluToGmm2, nullptr, nullptr, nullptr, ALIGN_256, ALIGN_256});
    if constexpr (UseCoreWave) {
        streamingEpilogueOp_.Init({params_.workspaceInfo.ptrA2, params_.workspaceInfo.ptrA2Scale,
                                   params_.workspaceInfo.ptrFlagSwiGluToGmm2, nullptr, nullptr, nullptr, ALIGN_256,
                                   ALIGN_256});
    }
    localExpandedRowIdxSt_ = rankId_ * m_ * topK_ * sizeof(int32_t);
    localTokenPerExpertSt_ =
        rankId_ * Ops::Base::CeilAlign(static_cast<int64_t>(worldSize_ * expertPerRank_), (int64_t)ALIGN_128) *
        sizeof(int32_t);

    expandedRowIdx_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(params_.peermemInfo.offsetExpandedRowIdx));
    distProbs_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(params_.peermemInfo.offsetExpertProb));
    firstTokenPosPtr_ = reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.ptrFirstTokenPos);
    stepIdxPtr_ = reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.ptrStepIdx);
    tokenChainDescriptorsPtr_ = reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.ptrTokenChainDescriptors);
    posLenPtr_ = reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.ptrPosLen);
    posDonePtr_ = reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.ptrPosDone);
    chainLenPtr_ = reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.ptrChainLen);
    chainLenReadyFlagPtr_ = reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.ptrChainLenReadyFlag);
    firstTokenPos_.SetGlobalBuffer(firstTokenPosPtr_);
    stepIdx_.SetGlobalBuffer(stepIdxPtr_);
    posLen_.SetGlobalBuffer(posLenPtr_);
    posDone_.SetGlobalBuffer(posDonePtr_);
    chainLen_.SetGlobalBuffer(chainLenPtr_);
    chainLenReady_.SetGlobalBuffer(chainLenReadyFlagPtr_);
    gmm2ReadyFlag_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.ptrGmm2ReadyFlag));
    flagSwigluToGmm2_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.ptrFlagSwiGluToGmm2));
    dispatchDoneCores_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.ptrDispatchDoneCores));
    dispatchTileReady_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.ptrDispatchTileReady));

    TIMER_BLOCK(timer.Init(timerAddr));
    TIMER_BLOCK(timerAll.Init(timerAddr));
    TIMER_BLOCK(if (timerAddr != nullptr) { AscendC::SyncAll<false>(); });
    TIMER_BLOCK(timerAll.Tik());
    TIMER_BLOCK(timer.Tik());

    TIMER_BLOCK(timer.setDynamicActualIter(AscendTimer::AIV_DISPATCH, expertPerRank_));
    TIMER_BLOCK(timer.setDynamicActualIter(AscendTimer::AIC_GMM1, expertPerRank_));
    TIMER_BLOCK(timer.setDynamicActualIter(AscendTimer::AIC_GMM2, expertPerRank_));
    TIMER_BLOCK(timer.setDynamicActualIter(AscendTimer::AIV_COMBINE, expertPerRank_));
    TIMER_BLOCK(timer.setDynamicActualIter(AscendTimer::AIV_GMM1_EPILOGUE, expertPerRank_));
    // 共享专家槽位的 actual iter = sharedExpertNum，0 表示未启用（解析端跳过列）。
    TIMER_BLOCK(timer.setDynamicActualIter(AscendTimer::AIC_SHARE_GMM1, sharedExpertNum_));
    TIMER_BLOCK(timer.setDynamicActualIter(AscendTimer::AIC_SHARE_GMM2, sharedExpertNum_));
    TIMER_BLOCK(timer.setDynamicActualIter(AscendTimer::AIV_SHARE_GMM1_EPILOGUE, sharedExpertNum_));
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::WriteExpertTokenNumsOut()
{
    if constexpr (g_coreType == AscendC::AIC) {
        return;
    }

    __gm__ int32_t *outputAddr = reinterpret_cast<__gm__ int32_t *>(params_.expertTokenNumsOutGmAddr);
    for (uint32_t expertIdx = aivCoreIdx_; expertIdx < expertPerRank_; expertIdx += blockAivNum_) {
        int32_t current = 0;
        int32_t previous = 0;
        if constexpr (UseCoreWave) {
            if (expertPerRank_ >= EXPERT_SCHEDULE_MIN_ACTIVE_EXPERTS) {
                __gm__ int32_t *planAddr = const_cast<__gm__ int32_t *>(expertOrderScratch_.GetPhyAddr());
                current = ReadGmByPassDCache(planAddr + 1U + expertIdx);
                previous = expertIdx == 0U ? 0 : ReadGmByPassDCache(planAddr + expertIdx);
            } else {
                __gm__ int32_t *expertCumsumAddr = const_cast<__gm__ int32_t *>(expertTokenCumsum_.GetPhyAddr());
                int32_t globalExpertIdx = static_cast<int32_t>(rankId_ * expertPerRank_ + expertIdx);
                current = ReadGmByPassDCache(expertCumsumAddr + globalExpertIdx);
                previous = globalExpertIdx == 0 ? 0 : ReadGmByPassDCache(expertCumsumAddr + globalExpertIdx - 1);
            }
        } else {
            __gm__ int32_t *expertCumsumAddr = const_cast<__gm__ int32_t *>(expertTokenCumsum_.GetPhyAddr());
            int32_t globalExpertIdx = static_cast<int32_t>(rankId_ * expertPerRank_ + expertIdx);
            current = ReadGmByPassDCache(expertCumsumAddr + globalExpertIdx);
            previous = globalExpertIdx == 0 ? 0 : ReadGmByPassDCache(expertCumsumAddr + globalExpertIdx - 1);
        }
        WriteGmByPassDCache(outputAddr + expertIdx, current - previous);
    }
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::FirstBuffInit(TPipe *tPipe)
{
    // Keep the proven 192-KiB stage-local arena.  Token-local Dispatch uses a
    // two-row ring plus compact metadata; reserving all physical A5 UB did not
    // improve capacity for the supported shapes and increased resource cost.
    if constexpr (UseBf16NoQuant) {
        // The arena is sized in bytes; BF16 copy slots index it by element.
        tPipe->InitBuffer(CopyTempBuf_, DISPATCH_FANOUT_UB_BYTES);
    } else {
        tPipe->InitBuffer(CopyTempBuf_, DISPATCH_FANOUT_UB_BYTES * sizeof(QuantOutType));
    }
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::SecondBuffInit(TPipe *tPipe)
{
    uint32_t dataResBufAlign = Ops::Base::CeilAlign(static_cast<uint32_t>(UNPERMUTE_LIST_NUM * k_ * sizeof(bfloat16_t)),
                                                    static_cast<uint32_t>(ALIGN_32));
    tPipe->InitBuffer(DataResBuf_, dataResBufAlign);
    uint32_t rankFlagBufAlign =
        Ops::Base::CeilAlign(static_cast<uint32_t>(worldSize_ * sizeof(int32_t)), static_cast<uint32_t>(ALIGN_32));
    tPipe->InitBuffer(RankFlagBuf_, rankFlagBufAlign);
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ResetDispatchTileReadyFull()
{
    // Tile-ready grows with maxOutputSize and no longer fits one AIV/UB for
    // small/generic shapes.  Split the full cache-line array across all AIVs.
    constexpr uint32_t FLAGS_PER_CHUNK = BUFFER_ALIGN / sizeof(int32_t);
    uint64_t tileFlagNum =
        static_cast<uint64_t>(expertPerRank_) * maxDispatchTilesPerExpert_ * DISPATCH_TILE_READY_STRIDE;
    uint64_t tileFlagChunks = Ops::Base::CeilDiv(tileFlagNum, static_cast<uint64_t>(FLAGS_PER_CHUNK));
    for (uint64_t chunkIdx = aivCoreIdx_; chunkIdx < tileFlagChunks; chunkIdx += blockAivNum_) {
        uint64_t offset = chunkIdx * FLAGS_PER_CHUNK;
        uint32_t curFlags =
            static_cast<uint32_t>(tileFlagNum - offset < FLAGS_PER_CHUNK ? tileFlagNum - offset : FLAGS_PER_CHUNK);
        LocalTensor<int32_t> initTensor{TPosition::VECIN, 0, 0};
        SyncFunc<AscendC::HardEvent::MTE3_V>();
        Duplicate(initTensor, 0, curFlags);
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopyPad(dispatchTileReady_[offset], initTensor,
                    DataCopyExtParams{1U, static_cast<uint32_t>(curFlags * sizeof(int32_t)), 0U, 0U, 0U});
    }
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ResetActiveDispatchTileReady()
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (aivCoreIdx_ >= expertPerRank_) {
        return;
    }

    if constexpr (UseBf16NoQuant) {
        // Called after AllGather's cross-rank join, before cumsum reduction.
        // Raw source histograms are already immutable; the two existing
        // PrepareExpertPipeline barriers publish these resets to GMM1.
        const uint32_t expertAlign =
            Ops::Base::CeilAlign(worldSize_ * expertPerRank_, static_cast<uint32_t>(ALIGN_128));
        const uint32_t rankExpertBase = rankId_ * expertPerRank_;
        __gm__ int32_t *histogram = const_cast<__gm__ int32_t *>(tokenPerExpertWin_.GetPhyAddr());
        constexpr uint32_t FLAGS_PER_CHUNK = 4096U / sizeof(int32_t);
        static_assert(FLAGS_PER_CHUNK % DISPATCH_TILE_READY_STRIDE == 0U,
                      "ready reset chunks must contain complete tile lane sets");
        LocalTensor<int32_t> initTensor{TPosition::VECIN, 0, 0};
        SyncFunc<AscendC::HardEvent::MTE3_V>();
        Duplicate(initTensor, 0, FLAGS_PER_CHUNK);
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        for (uint32_t expertIdx = aivCoreIdx_; expertIdx < expertPerRank_; expertIdx += blockAivNum_) {
            int64_t localStart = 0;
            int64_t rows = 0;
            for (uint32_t sourceRank = 0; sourceRank < worldSize_; ++sourceRank) {
                const uint64_t base = static_cast<uint64_t>(sourceRank) * expertAlign;
                const uint32_t globalExpert = rankExpertBase + expertIdx;
                const int32_t beforeRank =
                    rankExpertBase == 0U ? 0 : ReadGmByPassDCache(histogram + base + rankExpertBase - 1U);
                const int32_t beforeExpert =
                    globalExpert == 0U ? 0 : ReadGmByPassDCache(histogram + base + globalExpert - 1U);
                const int32_t throughExpert = ReadGmByPassDCache(histogram + base + globalExpert);
                localStart += static_cast<int64_t>(beforeExpert) - beforeRank;
                rows += static_cast<int64_t>(throughExpert) - beforeExpert;
            }
            if (localStart < 0 || rows <= 0 || static_cast<uint64_t>(localStart) >= maxOutputSize_) {
                continue;
            }
            const uint64_t remainingRows = maxOutputSize_ - static_cast<uint64_t>(localStart);
            const uint64_t activeRows =
                static_cast<uint64_t>(rows) < remainingRows ? static_cast<uint64_t>(rows) : remainingRows;
            const uint32_t activeTiles = static_cast<uint32_t>(
                Ops::Base::CeilDiv(activeRows, static_cast<uint64_t>(EXPERT_SCHEDULE_M_TILE_ROWS)));
            const uint64_t activeFlags = static_cast<uint64_t>(activeTiles) * DISPATCH_TILE_READY_STRIDE;
            for (uint64_t offset = 0; offset < activeFlags; offset += FLAGS_PER_CHUNK) {
                const uint32_t count = static_cast<uint32_t>(
                    activeFlags - offset < FLAGS_PER_CHUNK ? activeFlags - offset : FLAGS_PER_CHUNK);
                const uint64_t readyOffset =
                    static_cast<uint64_t>(expertIdx) * maxDispatchTilesPerExpert_ * DISPATCH_TILE_READY_STRIDE + offset;
                DataCopy(dispatchTileReady_[readyOffset], initTensor, count);
            }
        }
        // AllGather's reduction reuses this UB immediately afterwards.
        PipeBarrier<PIPE_ALL>();
        return;
    }

    // AllGather has published the real per-expert histogram before this
    // helper runs.  Clear only the cache-line counters that this launch can
    // address.  Stale tail slots are harmless: a later larger launch clears
    // every newly used slot before Dispatch/GMM1 can observe it.
    __gm__ int32_t *cumsumAddr = const_cast<__gm__ int32_t *>(expertTokenCumsum_.GetPhyAddr());
    int32_t rankExpertBase = static_cast<int32_t>(rankId_ * expertPerRank_);
    int32_t rankBaseRow = rankExpertBase == 0 ? 0 : ReadGmByPassDCache(cumsumAddr + rankExpertBase - 1);
    LocalTensor<int32_t> initTensor{TPosition::VECIN, 0, 0};
    SyncFunc<AscendC::HardEvent::MTE3_V>();
    Duplicate(initTensor, 0, DISPATCH_TILE_READY_STRIDE);
    SyncFunc<AscendC::HardEvent::V_MTE3>();

    for (uint32_t expertIdx = aivCoreIdx_; expertIdx < expertPerRank_; expertIdx += blockAivNum_) {
        int32_t globalExpertIdx = rankExpertBase + static_cast<int32_t>(expertIdx);
        int32_t globalStart = globalExpertIdx == 0 ? 0 : ReadGmByPassDCache(cumsumAddr + globalExpertIdx - 1);
        int32_t globalEnd = ReadGmByPassDCache(cumsumAddr + globalExpertIdx);
        int32_t localStart = globalStart - rankBaseRow;
        int32_t rows = globalEnd - globalStart;
        if (localStart < 0 || rows <= 0 || static_cast<uint64_t>(localStart) >= maxOutputSize_) {
            continue;
        }
        uint64_t remainingRows = maxOutputSize_ - static_cast<uint64_t>(localStart);
        if (static_cast<uint64_t>(rows) > remainingRows) {
            rows = static_cast<int32_t>(remainingRows);
        }
        uint32_t activeTiles = Ops::Base::CeilDiv(static_cast<uint32_t>(rows), EXPERT_SCHEDULE_M_TILE_ROWS);
        for (uint32_t tileIdx = 0; tileIdx < activeTiles; ++tileIdx) {
            // Reset the entire lane set.  lane0 is reused by Dispatch/GMM1;
            // Dispatch contributors occupy the first cacheline and GMM2
            // completion occupies the second.  The existing third mixed-core
            // barrier in PrepareExpertPipeline publishes every reset before
            // either producer can run.
            uint64_t readyOffset =
                (static_cast<uint64_t>(expertIdx) * maxDispatchTilesPerExpert_ + tileIdx) * DISPATCH_TILE_READY_STRIDE;
            DataCopy(dispatchTileReady_[readyOffset], initTensor, DISPATCH_TILE_READY_STRIDE);
        }
    }
    PipeBarrier<PIPE_ALL>();
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline bool MegaMoe<TemplateMegaMoeTypeFunc>::DispatchFanoutStorageCapable() const
{
    if (worldSize_ == 0U || blockNum_ == 0U || m_ > static_cast<uint32_t>(INT32_MAX) / worldSize_) {
        return false;
    }
    uint32_t sourceTokens = worldSize_ * m_;
    if (topK_ == 0U || sourceTokens > static_cast<uint32_t>(INT32_MAX) / topK_) {
        return false;
    }
    uint32_t descriptorCount = sourceTokens * topK_;
    if (blockAivNum_ == 0U || Ops::Base::CeilDiv(descriptorCount, blockAivNum_) > BUFFER_ALIGN / sizeof(int32_t)) {
        return false;
    }
    static_assert(sizeof(QuantOutType) == 1U || UseBf16NoQuant,
                  "Dispatch fanout UB accounting assumes one-byte quant rows");
    int64_t sourceRowBytes64 =
        UseBf16NoQuant ? GetA0RowStrideBf16(static_cast<int64_t>(k_)) : GetA0RowStride(static_cast<int64_t>(k_));
    if (sourceRowBytes64 <= 0 || sourceRowBytes64 > static_cast<int64_t>(UINT32_MAX)) {
        return false;
    }
    uint32_t sourceRowBytes = static_cast<uint32_t>(sourceRowBytes64);
    uint32_t privateTileCapacity = dispatchPriorityStride_ > 0U ? dispatchPriorityStride_ : 1U;
    // Reject decode/small launches before descriptor-width and UB-capacity
    // arithmetic. This keeps their fallback setup identical to C5.
    return HasProfitableDispatchFanoutWork(descriptorCount, sourceTokens, blockNum_) &&
           CanPackDispatchFanoutDescriptor(expertPerRank_, maxDispatchTilesPerExpert_, maxOutputSize_) &&
           CanFitDispatchFanoutInUb(sourceRowBytes, expertPerRank_, privateTileCapacity, topK_, sourceTokens,
                                    worldSize_, blockNum_);
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline bool MegaMoe<TemplateMegaMoeTypeFunc>::DispatchFanoutWorkProfitable(int32_t rankBaseRow)
{
    if (!DispatchFanoutStorageCapable()) {
        return false;
    }
    uint32_t sourceTokens = worldSize_ * m_;
    __gm__ int32_t *cumsumAddr = const_cast<__gm__ int32_t *>(expertTokenCumsum_.GetPhyAddr());
    int32_t rankExpertEnd = static_cast<int32_t>((rankId_ + 1U) * expertPerRank_ - 1U);
    int32_t rankEndRow = ReadGmByPassDCache(cumsumAddr + rankExpertEnd);
    int64_t localRoutes = static_cast<int64_t>(rankEndRow) - rankBaseRow;
    if (localRoutes <= 0) {
        return false;
    }
    if (static_cast<uint64_t>(localRoutes) > maxOutputSize_) {
        localRoutes = static_cast<int64_t>(maxOutputSize_);
    }
    if (!HasProfitableDispatchFanoutWork(static_cast<uint64_t>(localRoutes), sourceTokens, blockNum_)) {
        return false;
    }
    uint32_t priorityStride = 1U;
    bool priorityFromPlan = false;
    if constexpr (UseCoreWave) {
        if (expertPerRank_ >= EXPERT_SCHEDULE_MIN_ACTIVE_EXPERTS) {
            // The plan builder already published the largest clamped
            // per-expert tile count in the scratch header during AllGather;
            // one read replaces this second complete endpoint scan before
            // the first token row moves.
            __gm__ int32_t *planAddr = const_cast<__gm__ int32_t *>(expertOrderScratch_.GetPhyAddr());
            int32_t planStride =
                ReadGmByPassDCache(planAddr + 1U + 2U * expertPerRank_ + EXPERT_PLAN_HEADER_MAX_ACTIVE_TILES);
            priorityStride = planStride > 0 ? static_cast<uint32_t>(planStride) : 1U;
            priorityFromPlan = true;
        }
    }
    if (!priorityFromPlan) {
        // Small expert sets have no published plan.  Compute the compact
        // schedule/tile stride from the reduced histogram as before.
        int32_t globalExpertBase = static_cast<int32_t>(rankId_ * expertPerRank_);
        int32_t previous = rankBaseRow;
        for (uint32_t expertIdx = 0U; expertIdx < expertPerRank_; ++expertIdx) {
            int32_t current = ReadGmByPassDCache(cumsumAddr + globalExpertBase + static_cast<int32_t>(expertIdx));
            int32_t localStart = previous - rankBaseRow;
            int32_t rows = current - previous;
            previous = current;
            if (localStart < 0 || rows <= 0 || static_cast<uint64_t>(localStart) >= maxOutputSize_) {
                continue;
            }
            uint64_t remainingRows = maxOutputSize_ - static_cast<uint64_t>(localStart);
            if (static_cast<uint64_t>(rows) > remainingRows) {
                rows = static_cast<int32_t>(remainingRows);
            }
            uint32_t activeTiles = Ops::Base::CeilDiv(static_cast<uint32_t>(rows), EXPERT_SCHEDULE_M_TILE_ROWS);
            priorityStride = activeTiles > priorityStride ? activeTiles : priorityStride;
        }
    }
    int64_t sourceRowBytes64 =
        UseBf16NoQuant ? GetA0RowStrideBf16(static_cast<int64_t>(k_)) : GetA0RowStride(static_cast<int64_t>(k_));
    if (sourceRowBytes64 <= 0 || sourceRowBytes64 > static_cast<int64_t>(UINT32_MAX) ||
        !CanFitDispatchFanoutInUb(static_cast<uint32_t>(sourceRowBytes64), expertPerRank_, priorityStride, topK_,
                                  sourceTokens, worldSize_, blockNum_)) {
        return false;
    }
    dispatchPriorityStride_ = priorityStride;
    return true;
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ResetDispatchFanoutDescriptors()
{
    if constexpr (!UseBf16NoQuant) {
        if (!DispatchFanoutStorageCapable()) {
            return;
        }
    } else if (worldSize_ == 0U || blockNum_ == 0U || m_ > static_cast<uint32_t>(INT32_MAX) / worldSize_ ||
               topK_ == 0U || worldSize_ * m_ > static_cast<uint32_t>(INT32_MAX) / topK_) {
        return;
    }
    int32_t descriptorCount = static_cast<int32_t>(worldSize_ * m_ * topK_);
    int32_t coreLen = 0;
    int32_t coreOffset = 0;
    TilingByCore(descriptorCount, coreLen, coreOffset, static_cast<int32_t>(ALIGN_32 / sizeof(int32_t)));
    if (coreLen <= 0) {
        return;
    }
    GlobalTensor<int32_t> descriptors;
    descriptors.SetGlobalBuffer(tokenChainDescriptorsPtr_);
    // ResetData runs before FirstBuffInit; use the same raw stage-local UB
    // view as the other reset helpers rather than dereferencing CopyTempBuf_.
    LocalTensor<int32_t> initTensor{TPosition::VECIN, 0, 0};
    SyncFunc<AscendC::HardEvent::MTE3_V>();
    Duplicate(initTensor, -1, coreLen);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopyPad(descriptors[coreOffset], initTensor,
                DataCopyExtParams{1U, static_cast<uint32_t>(coreLen * sizeof(int32_t)), 0U, 0U, 0U});
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ResetData()
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    // workSpace groupFlagList清零
    if (aivCoreIdx_ == 0) {
        int32_t flagNum = expertPerRank_ * INT_CACHELINE;
        int32_t dispatchDoneFlagNum = expertPerRank_ * DISPATCH_DONE_STRIDE;
        LocalTensor<int32_t> initTensor{TPosition::VECIN, 0, 0};
        Duplicate(initTensor, 0, dispatchDoneFlagNum);
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopy(flagSwigluToGmm2_, initTensor, flagNum);
        // Core-wave reuses lane0 as the AIV1 fanout-plan publication counter;
        // generic mode keeps the established chainLen-ready meaning.
        DataCopy(chainLenReady_, initTensor, INT_CACHELINE);
        DataCopy(dispatchDoneCores_, initTensor, dispatchDoneFlagNum);
        DataCopy(gmm2ReadyFlag_, initTensor, flagNum);
        // 共享专家标志：[0,S) 为 GMM1 tile ready（AIC->AIV），[S,2S) 为 GMM2 完成
        // （AIC->Unpermute），[2S,3S) 为预留行（历史 input-ready/门控槽已退役，
        // 无消费者；保持清零，布局不变）。
        if (sharedExpertNum_ > 0U) {
            const int32_t sharedFlagNum = static_cast<int32_t>(sharedExpertNum_) * INT_CACHELINE * 3;
            if (sharedFlagNum > dispatchDoneFlagNum) {
                Duplicate(initTensor, 0, sharedFlagNum);
                SyncFunc<AscendC::HardEvent::V_MTE3>();
            }
            GlobalTensor<int32_t> sharedReadyFlagGm;
            sharedReadyFlagGm.SetGlobalBuffer(
                reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.ptrSharedReadyFlag));
            DataCopy(sharedReadyFlagGm, initTensor, sharedFlagNum);
            // DataCopy 是异步 MTE3 写：必须显式排空。否则核 0 的清零可能晚于
            // 其它核对本区域首个写入者（阶段 3.8 共享 epilogue 对行 [0,S) 的
            // AtomicAdd）落地，把累加打回 0，AIC 的共享 GMM2 就永远等不到 tile
            // 计数（纯互等死锁）。后续的 SyncAll 只负责跨核可见性，不保证本核
            // MTE3 已经完成。
            SyncFunc<AscendC::HardEvent::MTE3_S>();
        }
    }
    // MXFP generic and very small expert sets do not have the third mixed-core
    // publication barrier used by core-wave scheduling, so retain the full
    // early reset for those paths.  Core-wave E>=3 defers to the actual
    // post-AllGather histogram in PrepareExpertPipeline().
    // Sparse-decode neither publishes nor consumes dispatchTileReady (GMM1
    // is gated by the single dispatch-complete flag), so the reset is dead
    // work there and compiles out.
    if constexpr (UseBf16NoQuant) {
        // BF16 uses the same unconditional full reset as the MXFP8 generic
        // path below.  The earlier BF16-only deferred/partial reset
        // (ResetActiveDispatchTileReady, run after CrossRankSyncInWorldSize)
        // derived its coverage from remote-written histogram slices read
        // right after that scalar barrier.  A peer's MTE3 histogram payload
        // that only becomes visible after the barrier could make the partial
        // reset skip a tile the GMM1 consumer still polls; because that
        // consumer spins on exact equality and these counters are never
        // cleared at launch end, the value persisted from an earlier launch
        // plus this launch's AtomicAdds then overshoots the target and the
        // AIC spins forever, stranding its AIV pair and hanging every rank at
        // the terminal barrier.  A full zeroing has no cross-rank data
        // dependence and is joined by the existing SyncAll before AllGather.
        ResetDispatchTileReadyFull();
    } else if constexpr (UseSparseDecode) {
        // No per-tile readiness accounting in the one-wave decode protocol.
    } else if constexpr (!UseCoreWave) {
        ResetDispatchTileReadyFull();
    } else if (expertPerRank_ < EXPERT_SCHEDULE_MIN_ACTIVE_EXPERTS) {
        ResetDispatchTileReadyFull();
    }
    uint16_t cumsumNum = rankId_ == 0 ? expertPerRank_ : expertPerRank_ + 1;
    int32_t cumsumSt = rankId_ == 0 ? 0 : expertPerRank_ * rankId_ - 1;
    if (aivCoreIdx_ == 1) {
        LocalTensor<int32_t> initTensor{TPosition::VECIN, 0, 0};
        // AIV1 may have issued the final distributed tile-ready reset from
        // the same UB.  Join MTE3 before V overwrites it for cumsum reset.
        SyncFunc<AscendC::HardEvent::MTE3_V>();
        Duplicate(initTensor, 0, cumsumNum);
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopyPad(expertTokenCumsum_[cumsumSt], initTensor,
                    DataCopyExtParams{1U, static_cast<uint32_t>(cumsumNum * sizeof(int32_t)), 0U, 0U, 0U});
    }
    // reset firstTokenPosPtr_ (=-1), posLen_ (int32, =0), posDone_ (int32, =0)
    // stepIdx_ 无需预初始化，Dispatch 会覆盖每个有效 canonical row。
    uint32_t totalTokens = static_cast<uint32_t>(worldSize_ * m_);
    startBlockIdx_ = 2;
    uint32_t startLoopIdx = (aivCoreIdx_ < startBlockIdx_ ? aivCoreIdx_ + blockAivNum_ : aivCoreIdx_) - startBlockIdx_;
    // 1.优化点：可以细分搬运
    uint16_t numPerCore = static_cast<uint16_t>(BUFFER_ALIGN / sizeof(int32_t));
    uint32_t coreLoops = Ops::Base::CeilDiv(totalTokens, static_cast<uint32_t>(numPerCore));
    for (uint32_t idx = startLoopIdx; idx < coreLoops; idx += blockAivNum_) {
        uint32_t curTokens = totalTokens - idx * numPerCore < numPerCore ? totalTokens - idx * numPerCore : numPerCore;
        LocalTensor<int32_t> initTensor{TPosition::VECIN, 0, 0};
        SyncFunc<AscendC::HardEvent::MTE3_V>();
        Duplicate(initTensor, -1, curTokens);
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopyPad(firstTokenPos_[idx * numPerCore], initTensor,
                    DataCopyExtParams{1U, static_cast<uint32_t>(curTokens * sizeof(int32_t)), 0U, 0U, 0U});
    }
    // posDone_ (int32) 初始化为 0
    startBlockIdx_ = (startBlockIdx_ + coreLoops) % blockAivNum_;
    startLoopIdx = (aivCoreIdx_ < startBlockIdx_ ? aivCoreIdx_ + blockAivNum_ : aivCoreIdx_) - startBlockIdx_;
    for (uint32_t idx = startLoopIdx; idx < coreLoops; idx += blockAivNum_) {
        uint32_t curTokens = totalTokens - idx * numPerCore < numPerCore ? totalTokens - idx * numPerCore : numPerCore;
        LocalTensor<int32_t> initTensor{TPosition::VECIN, 0, 0};
        SyncFunc<AscendC::HardEvent::MTE3_V>();
        Duplicate(initTensor, 0, curTokens);
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopyPad(posDone_[idx * numPerCore], initTensor,
                    DataCopyExtParams{1U, static_cast<uint32_t>(curTokens * sizeof(int32_t)), 0U, 0U, 0U});
    }
    // posLen_ (int32) 初始化为 0 (与posDone_相同大小, 复用循环结构)。
    // generic 同时清 chainLen_；core-wave 在全量 Dispatch 后直接复用最终
    // posLen_，跳过这份不再读取的 GM 写。两种路径都不额外推进
    // startBlockIdx_ 轮转，避免改变后续 AllGather 的 C3 物理核映射。
    startBlockIdx_ = (startBlockIdx_ + coreLoops) % blockAivNum_;
    uint32_t posLenStartBlockIdx = startBlockIdx_;
    uint32_t allGatherCoreSpan = 3U * worldSize_;
    if (worldSize_ == 4U && startBlockIdx_ + allGatherCoreSpan + coreLoops <= blockAivNum_) {
        posLenStartBlockIdx += allGatherCoreSpan;
    }
    startLoopIdx = (aivCoreIdx_ < posLenStartBlockIdx ? aivCoreIdx_ + blockAivNum_ : aivCoreIdx_) - posLenStartBlockIdx;
    for (uint32_t idx = startLoopIdx; idx < coreLoops; idx += blockAivNum_) {
        uint32_t curTokens = totalTokens - idx * numPerCore < numPerCore ? totalTokens - idx * numPerCore : numPerCore;
        LocalTensor<int32_t> initTensor{TPosition::VECIN, 0, 0};
        SyncFunc<AscendC::HardEvent::MTE3_V>();
        Duplicate(initTensor, 0, curTokens);
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopyPad(posLen_[idx * numPerCore], initTensor,
                    DataCopyExtParams{1U, static_cast<uint32_t>(curTokens * sizeof(int32_t)), 0U, 0U, 0U});
        if constexpr (!UseCoreWave) {
            DataCopyPad(chainLen_[idx * numPerCore], initTensor,
                        DataCopyExtParams{1U, static_cast<uint32_t>(curTokens * sizeof(int32_t)), 0U, 0U, 0U});
        }
    }
    if constexpr (UseCoreWave) {
        ResetDispatchFanoutDescriptors();
    }
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

template <TemplateMegaMoeTypeClass>
template <typename T>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::CopyGMToGM(GlobalTensor<T> dst, GlobalTensor<T> src,
                                                                    int32_t dataLen)
{
    constexpr int32_t BUFFER_NUM = 2;
    int32_t moveNum = BUFFER_ALIGN / sizeof(T) / BUFFER_NUM;
    LocalTensor<T> tmpBuffer1 = CopyTempBuf_.Get<T>();
    LocalTensor<T> tmpBuffer2 = tmpBuffer1[moveNum];
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    int32_t pingpongId = 0;
    for (int32_t offset = 0; offset < dataLen; offset += moveNum) {
        int32_t curLen = dataLen - offset;
        if (curLen > moveNum) {
            curLen = moveNum;
        }
        TEventID eventId = pingpongId == 0 ? EVENT_ID0 : EVENT_ID1;
        LocalTensor<T> tmpBuffer = pingpongId == 0 ? tmpBuffer1 : tmpBuffer2;
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
        DataCopyExtParams copyParams{1U, static_cast<uint32_t>(curLen * sizeof(T)), 0U, 0U, 0U};
        DataCopyPadExtParams<T> padParams{false, 0U, 0U, 0U};
        DataCopyPad(tmpBuffer, src[offset], copyParams, padParams);
        SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
        WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
        DataCopyPad(dst[offset], tmpBuffer, copyParams);
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
        pingpongId = 1 - pingpongId;
    }
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
}

template <TemplateMegaMoeTypeClass>
template <typename T>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::CopyAllGatherByRank(GlobalTensor<T> src, uint64_t peerMemSt,
                                                                             int64_t rankOffset, int32_t dataLen,
                                                                             bool skipLocalRank)
{
    uint32_t dstCount = skipLocalRank ? (worldSize_ > 0U ? worldSize_ - 1U : 0U) : worldSize_;
    if (dataLen <= 0 || dstCount == 0U) {
        return;
    }

    // The task space is chunk-major: the first wave spans destinations before
    // assigning a second chunk to any destination.  This remains bijective
    // when worldSize exceeds the available AIVs and avoids the rank-major
    // layout's long idle groups for EP2.
    uint32_t shardSlotsPerRank = dstCount > blockAivNum_ ? 1U : blockAivNum_ / dstCount;
    shardSlotsPerRank = shardSlotsPerRank > 0U ? shardSlotsPerRank : 1U;
    shardSlotsPerRank =
        shardSlotsPerRank < ALLGATHER_MAX_SHARDS_PER_RANK ? shardSlotsPerRank : ALLGATHER_MAX_SHARDS_PER_RANK;
    uint64_t dataBytes = static_cast<uint64_t>(dataLen) * sizeof(T);
    uint32_t activeShards =
        static_cast<uint32_t>(Ops::Base::CeilDiv(dataBytes, static_cast<uint64_t>(ALLGATHER_MIN_BYTES_PER_SHARD)));
    activeShards = activeShards > 0U ? activeShards : 1U;
    activeShards = activeShards < shardSlotsPerRank ? activeShards : shardSlotsPerRank;

    // DataCopyPad may update one complete 32-byte GM block.  Split the
    // absolute destination interval into aligned units so two AIVs never
    // write different bytes of the same remote block.  Only the global first
    // and last chunks can be unaligned, and each is owned by exactly one AIV.
    constexpr uint32_t elemsPerGmBlock = ALLGATHER_GM_BLOCK_BYTES / sizeof(T);
    uint64_t absoluteBegin = static_cast<uint64_t>(rankOffset);
    uint64_t absoluteEnd = absoluteBegin + static_cast<uint64_t>(dataLen);
    uint64_t firstUnit = absoluteBegin / elemsPerGmBlock;
    uint64_t unitEnd = Ops::Base::CeilDiv(absoluteEnd, static_cast<uint64_t>(elemsPerGmBlock));
    uint32_t alignedUnits = static_cast<uint32_t>(unitEnd - firstUnit);
    activeShards = activeShards < alignedUnits ? activeShards : alignedUnits;
    // CopyGMToGM uses DataCopyPad for GM->UB.  On A5 that input path is only
    // reliable when each shard source begins on a 32-byte boundary (the
    // per-token dispatch path documents the same hardware restriction).
    // An unaligned destination interval cannot have both its internal source
    // and destination boundaries aligned, because source offset zero maps to
    // rankOffset.  Keep that interval on one AIV; aligned intervals retain
    // the multi-AIV split.
    if (absoluteBegin % elemsPerGmBlock != 0U) {
        activeShards = 1U;
    }

    uint32_t taskSlots = dstCount * shardSlotsPerRank;
    uint32_t startTaskIdx = (aivCoreIdx_ < startBlockIdx_ ? aivCoreIdx_ + blockAivNum_ : aivCoreIdx_) - startBlockIdx_;
    GlobalTensor<T> remote;
    for (uint32_t taskIdx = startTaskIdx; taskIdx < taskSlots; taskIdx += blockAivNum_) {
        uint32_t shardIdx = taskIdx / dstCount;
        uint32_t dstOrdinal = taskIdx % dstCount;
        if (shardIdx >= activeShards) {
            continue;
        }
        uint32_t dstRankIdx = skipLocalRank && dstOrdinal >= rankId_ ? dstOrdinal + 1U : dstOrdinal;
        uint32_t unitsPerShard = alignedUnits / activeShards;
        uint32_t extraUnits = alignedUnits % activeShards;
        uint64_t shardFirstUnit = firstUnit + static_cast<uint64_t>(shardIdx) * unitsPerShard +
                                  (shardIdx < extraUnits ? shardIdx : extraUnits);
        uint32_t shardUnits = unitsPerShard + (shardIdx < extraUnits ? 1U : 0U);
        uint64_t shardAbsoluteBegin = shardFirstUnit * elemsPerGmBlock;
        shardAbsoluteBegin = shardAbsoluteBegin > absoluteBegin ? shardAbsoluteBegin : absoluteBegin;
        uint64_t shardAbsoluteEnd = (shardFirstUnit + shardUnits) * elemsPerGmBlock;
        shardAbsoluteEnd = shardAbsoluteEnd < absoluteEnd ? shardAbsoluteEnd : absoluteEnd;
        if (shardAbsoluteBegin >= shardAbsoluteEnd) {
            continue;
        }
        uint32_t elemBegin = static_cast<uint32_t>(shardAbsoluteBegin - absoluteBegin);
        int32_t shardLen = static_cast<int32_t>(shardAbsoluteEnd - shardAbsoluteBegin);
        remote.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(GetRankWinAddrWithOffset(dstRankIdx, peerMemSt)) +
                               rankOffset + elemBegin);
        CopyGMToGM(remote, src[elemBegin], shardLen);
    }
    // Preserve the established downstream physical-core rotation.  The task
    // count is deliberately decoupled from this cursor (EP2 has 64 slots).
    startBlockIdx_ = (startBlockIdx_ + worldSize_) % blockAivNum_;
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::Mxfp8DynamicQuant(TPipe *tPipe)
{
    // Match the standalone MXFP8 kernels: assert per-instruction saturation after TPipe construction and
    // immediately before the vector conversion path.
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(0);
    int64_t cols = k_;
    int64_t validScaleCols = Ops::Base::CeilDiv<int64_t>(cols, MoeInitRoutingV3::MX_BLOCK_SIZE);
    int64_t scaleCols = Ops::Base::CeilAlign<int64_t>(validScaleCols, 2);
    // scale在UB/GM行内的偏移: 数据段须补齐到256B。ComputeFp8Data每循环固定写
    // 4*64B=256B(尾部溢出段按掩码补0), cols非256倍数时溢出段会覆盖紧贴cols的scale,
    // 故scale必须放在溢出段之后(见GetA0RowStride注释/dispatch_v2::QuantDynamicMxFp8)。
    int64_t a0ScaleOffset = GetA0ScaleOffset(cols);
    // ComputeMaxExp每循环处理2*VL个元素产出vRegSize/ubBlockSize个maxExp, 总数按该粒度
    // 上取整, 可能超过scaleCols(H=3200: 104>100), 缓冲须覆盖全部产出避免越界写相邻TBuf。
    uint32_t ubBlockSize = Ops::Base::GetUbBlockSize();
    uint32_t vRegSize = Ops::Base::GetVRegSize();
    int64_t quantLoopElems = 2 * static_cast<int64_t>(vRegSize) / sizeof(XType);
    int64_t maxExpNum = Ops::Base::CeilDiv<int64_t>(cols, quantLoopElems) * (vRegSize / ubBlockSize);

    tPipe->InitBuffer(QuantXInQueue_, 2, AlignBytes(cols, sizeof(XType)));
    // QuantOut缓冲与a0行布局一致: data[0,cols) + pad到256B + scale[a0ScaleOffset, +scaleCols)
    // + pad到32B, 总大小即GetA0RowStride, 可整行单次对齐写出。
    tPipe->InitBuffer(QuantOutQueue_, 2, static_cast<uint32_t>(GetA0RowStride(cols)));
    tPipe->InitBuffer(QuantMaxExpBuf_, AlignBytes(maxExpNum, sizeof(XType)));
    // invScale同样按maxExpNum分配: ComputeFp8Data每循环读vRegSize/ubBlockSize个,
    // 循环数按256粒度上取整(H=3200读104个>scaleCols 100), 尾部4个仅作用于溢出pad段。
    tPipe->InitBuffer(QuantInvScaleBuf_, AlignBytes(maxExpNum, sizeof(XType)));

    GlobalTensor<XType> xInGm;
    GlobalTensor<uint8_t> peerA0Gm;
    xInGm.SetGlobalBuffer(reinterpret_cast<__gm__ XType *>(params_.aGmAddr));
    peerA0Gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(params_.peermemInfo.ptrA0));

    int32_t coreLen = 0;
    int32_t coreOffset = 0;
    TilingByCore(m_, coreLen, coreOffset, 1);
    int32_t coreEnd = coreOffset + coreLen;
    DataCopyExtParams copyInParams{1U, static_cast<uint32_t>(cols * sizeof(XType)), 0U, 0U, 0U};
    DataCopyPadExtParams<XType> padParams{false, 0U, 0U, 0U};
    int64_t loopColsTail = cols % MoeInitRoutingV3::MX_BLOCK_SIZE;
    if (loopColsTail != 0) {
        padParams.isPad = true;
        if (loopColsTail > ubBlockSize / sizeof(XType)) {
            padParams.rightPadding = ubBlockSize / sizeof(XType) - (loopColsTail - ubBlockSize / sizeof(XType));
        } else {
            padParams.rightPadding = ubBlockSize / sizeof(XType);
        }
    }
    // a0 行布局见GetA0RowStride: data[0,cols) + pad到256B + scale + pad到32B。
    // 整行按步长(32B对齐)单次DataCopyPad写出, data段溢出pad与尾部pad为未初始化/补0字节,
    // 下游只读data/scale段。
    int64_t a0RowStride = GetA0RowStride(cols);
    DataCopyExtParams copyOutParams{1U, static_cast<uint32_t>(a0RowStride), 0U, 0U, 0U};

    // 先搬入首个token。循环内在计算当前token前预取下一个token，实现MTE2/V流水重叠。
    if (coreLen > 0) {
        LocalTensor<XType> firstXLocal = QuantXInQueue_.AllocTensor<XType>();
        DataCopyPad(firstXLocal, xInGm[static_cast<int64_t>(coreOffset) * cols], copyInParams, padParams);
        QuantXInQueue_.EnQue(firstXLocal);
    }
    for (int32_t tokenIdx = coreOffset; tokenIdx < coreOffset + coreLen; ++tokenIdx) {
        LocalTensor<XType> xLocal = QuantXInQueue_.DeQue<XType>();
        int32_t nextTokenIdx = tokenIdx + 1;
        if (nextTokenIdx < coreEnd) {
            LocalTensor<XType> nextXLocal = QuantXInQueue_.AllocTensor<XType>();
            DataCopyPad(nextXLocal, xInGm[static_cast<int64_t>(nextTokenIdx) * cols], copyInParams, padParams);
            QuantXInQueue_.EnQue(nextXLocal);
        }

        LocalTensor<int8_t> xQuantLocal = QuantOutQueue_.AllocTensor<int8_t>();
        auto xLocalAddr = reinterpret_cast<__ubuf__ XType *>(xLocal.GetPhyAddr());
        auto xQuantLocalAddr = reinterpret_cast<__ubuf__ int8_t *>(xQuantLocal.GetPhyAddr());
        auto mxScaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t *>(xQuantLocalAddr + a0ScaleOffset);
        LocalTensor<uint16_t> maxExpLocal = QuantMaxExpBuf_.Get<uint16_t>();
        LocalTensor<uint16_t> halfScaleLocal = QuantInvScaleBuf_.Get<uint16_t>();
        auto maxExpLocalAddr = reinterpret_cast<__ubuf__ uint16_t *>(maxExpLocal.GetPhyAddr());
        auto halfScaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t *>(halfScaleLocal.GetPhyAddr());
        // Use the CANN 9.1 canonical inline path. ComputeFp8Data also clips finite values to the target FP8 range,
        // so correctness does not depend solely on the CANN saturation-control state.
        quant::ComputeMaxExp(xLocalAddr, maxExpLocalAddr, static_cast<uint32_t>(cols));
        quant::ComputeScale<QuantOutType>(maxExpLocalAddr, mxScaleLocalAddr, halfScaleLocalAddr,
                                          static_cast<uint32_t>(scaleCols));
        quant::ComputeFp8Data<XType, QuantOutType, AscendC::RoundMode::CAST_TRUNC, AscendC::RoundMode::CAST_RINT>(
            xLocalAddr, halfScaleLocalAddr, xQuantLocalAddr, static_cast<uint32_t>(cols));
        QuantXInQueue_.FreeTensor(xLocal);
        QuantOutQueue_.EnQue(xQuantLocal);

        xQuantLocal = QuantOutQueue_.DeQue<int8_t>();
        LocalTensor<uint8_t> xQuantOutLocal = xQuantLocal.template ReinterpretCast<uint8_t>();
        DataCopyPad(peerA0Gm[static_cast<int64_t>(tokenIdx) * a0RowStride], xQuantOutLocal, copyOutParams);
        QuantOutQueue_.FreeTensor(xQuantLocal);
    }
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::CopyBf16TokensToA0()
{
    // BF16 no-quant publishes x unchanged into this rank's a0 window. Host
    // requires H % 16 == 0, so an a0 row is exactly one x row (2H bytes, 32B
    // aligned) and every AIV's token slice is one contiguous aligned stream.
    // Remote Dispatch reads start only after AllGatherData's local SyncAll
    // and cross-rank handshake, which also publish these stores.
    if constexpr (g_coreType == AscendC::AIC) {
        return;
    }
    int32_t coreLen = 0;
    int32_t coreOffset = 0;
    TilingByCore(m_, coreLen, coreOffset, 1);
    if (coreLen <= 0) {
        return;
    }
    GlobalTensor<bfloat16_t> xInGm;
    GlobalTensor<bfloat16_t> peerA0Gm;
    xInGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(params_.aGmAddr));
    peerA0Gm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(params_.peermemInfo.ptrA0));
    int64_t elemOffset = static_cast<int64_t>(coreOffset) * k_;
    CopyGMToGM(peerA0Gm[elemOffset], xInGm[elemOffset], static_cast<int32_t>(static_cast<int64_t>(coreLen) * k_));
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::AllGatherData()
{
    if constexpr (g_coreType == AIC) {
        return;
    }

    // 1.probs: split each destination payload across aligned AIV chunks.
    int64_t rankOffset = static_cast<int64_t>(rankId_) * m_ * topK_;
    int32_t copyElemNum = static_cast<int32_t>(m_ * topK_);
    uint64_t peerMemSt = params_.peermemInfo.offsetExpertProb - params_.peermemInfo.ptrBase;

    GlobalTensor<bfloat16_t> probsLocal;
    probsLocal.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(params_.probsGmAddr));
    CopyAllGatherByRank(probsLocal, peerMemSt, rankOffset, copyElemNum, false);

    // 2.expandedRowIdxs: routing output is sorted by global expert, so each
    // destination only needs its owner slice at the original sparse offset.
    // Use the same GM->UB->remote-GM DataCopyPad path as the established bulk
    // gather for the entire slice, including short/non-32B topK=6 ranges.
    // Do not use remote bypass scalar stores here: the original head/tail
    // prototype triggered cross-rank AIC/AICPU failures on fast.
    rankOffset = static_cast<int64_t>(rankId_) * m_ * topK_;
    peerMemSt = params_.peermemInfo.offsetExpandedRowIdx - params_.peermemInfo.ptrBase;
    GlobalTensor<int32_t> expandedRowIdxLocal;
    expandedRowIdxLocal = expandedRowIdx_[rankOffset];
    uint32_t expertNumAlign =
        Ops::Base::CeilAlign(static_cast<int64_t>(worldSize_ * expertPerRank_), static_cast<int64_t>(ALIGN_128));
    GlobalTensor<int32_t> localExpertCumsum;
    localExpertCumsum = tokenPerExpertWin_[static_cast<int64_t>(rankId_) * expertNumAlign];
    uint32_t expandedDstCount = worldSize_ > 0U ? worldSize_ - 1U : 0U;
    uint32_t expandedShardSlots =
        expandedDstCount > blockAivNum_ ? 1U : (expandedDstCount > 0U ? blockAivNum_ / expandedDstCount : 0U);
    expandedShardSlots = expandedShardSlots > 0U ? expandedShardSlots : 1U;
    expandedShardSlots =
        expandedShardSlots < ALLGATHER_MAX_SHARDS_PER_RANK ? expandedShardSlots : ALLGATHER_MAX_SHARDS_PER_RANK;
    uint32_t expandedTaskSlots = expandedDstCount * expandedShardSlots;
    uint32_t expandedStartTask =
        (aivCoreIdx_ < startBlockIdx_ ? aivCoreIdx_ + blockAivNum_ : aivCoreIdx_) - startBlockIdx_;
    GlobalTensor<int32_t> expandedRemote;
    for (uint32_t taskIdx = expandedStartTask; taskIdx < expandedTaskSlots; taskIdx += blockAivNum_) {
        uint32_t shardIdx = taskIdx / expandedDstCount;
        uint32_t dstOrdinal = taskIdx % expandedDstCount;
        uint32_t dstRankIdx = dstOrdinal >= rankId_ ? dstOrdinal + 1U : dstOrdinal;
        int32_t ownerBegin = dstRankIdx == 0U ? 0 : localExpertCumsum.GetValue(dstRankIdx * expertPerRank_ - 1U);
        int32_t ownerEnd = localExpertCumsum.GetValue((dstRankIdx + 1U) * expertPerRank_ - 1U);
        int32_t ownerRows = ownerEnd - ownerBegin;
        if (ownerRows <= 0) {
            continue;
        }
        uint64_t ownerBytes = static_cast<uint64_t>(ownerRows) * sizeof(int32_t);
        uint32_t activeShards =
            static_cast<uint32_t>(Ops::Base::CeilDiv(ownerBytes, static_cast<uint64_t>(ALLGATHER_MIN_BYTES_PER_SHARD)));
        activeShards = activeShards > 0U ? activeShards : 1U;
        activeShards = activeShards < expandedShardSlots ? activeShards : expandedShardSlots;
        constexpr uint32_t elemsPerGmBlock = ALLGATHER_GM_BLOCK_BYTES / sizeof(int32_t);
        uint64_t absoluteBegin = static_cast<uint64_t>(rankOffset + ownerBegin);
        uint64_t absoluteEnd = absoluteBegin + static_cast<uint64_t>(ownerRows);
        uint64_t firstUnit = absoluteBegin / elemsPerGmBlock;
        uint64_t unitEnd = Ops::Base::CeilDiv(absoluteEnd, static_cast<uint64_t>(elemsPerGmBlock));
        uint32_t alignedUnits = static_cast<uint32_t>(unitEnd - firstUnit);
        activeShards = activeShards < alignedUnits ? activeShards : alignedUnits;
        if (shardIdx >= activeShards) {
            continue;
        }
        uint32_t unitsPerShard = alignedUnits / activeShards;
        uint32_t extraUnits = alignedUnits % activeShards;
        uint64_t shardFirstUnit = firstUnit + static_cast<uint64_t>(shardIdx) * unitsPerShard +
                                  (shardIdx < extraUnits ? shardIdx : extraUnits);
        uint32_t shardUnits = unitsPerShard + (shardIdx < extraUnits ? 1U : 0U);
        uint64_t shardAbsoluteBegin = shardFirstUnit * elemsPerGmBlock;
        shardAbsoluteBegin = shardAbsoluteBegin > absoluteBegin ? shardAbsoluteBegin : absoluteBegin;
        uint64_t shardAbsoluteEnd = (shardFirstUnit + shardUnits) * elemsPerGmBlock;
        shardAbsoluteEnd = shardAbsoluteEnd < absoluteEnd ? shardAbsoluteEnd : absoluteEnd;
        if (shardAbsoluteBegin >= shardAbsoluteEnd) {
            continue;
        }
        uint32_t shardBegin = static_cast<uint32_t>(shardAbsoluteBegin - absoluteBegin);
        int32_t shardLen = static_cast<int32_t>(shardAbsoluteEnd - shardAbsoluteBegin);
        expandedRemote.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t *>(GetRankWinAddrWithOffset(dstRankIdx, peerMemSt)) + rankOffset +
            ownerBegin + shardBegin);
        CopyGMToGM(expandedRemote, expandedRowIdxLocal[ownerBegin + shardBegin], shardLen);
    }
    // Keep the pre-sharding cursor contract for downstream TokenPerExpert.
    startBlockIdx_ = (startBlockIdx_ + worldSize_) % blockAivNum_;

    // 3.TokenPerExpert  可只搬运前dstRank的数据，并将数据累加到expertTokenCumsum_
    uint32_t startLoopIdx = (aivCoreIdx_ < startBlockIdx_ ? aivCoreIdx_ + blockAivNum_ : aivCoreIdx_) - startBlockIdx_;
    rankOffset = static_cast<int64_t>(rankId_) * expertNumAlign;
    copyElemNum = expertNumAlign;
    peerMemSt = params_.peermemInfo.ptrTokenPerExpert - params_.peermemInfo.ptrBase;
    uint64_t peerMemExpertTokenCumsumSt = params_.peermemInfo.ptrExpertTokenCumsum - params_.peermemInfo.ptrBase;
    GlobalTensor<int32_t> tokenPerExpertLocal;
    GlobalTensor<int32_t> tokenPerExpertRemote;
    GlobalTensor<int32_t> expertTokenCumsumRemote;
    LocalTensor<int32_t> tmpBuffer = CopyTempBuf_.Get<int32_t>();
    tokenPerExpertLocal = tokenPerExpertWin_[rankOffset];
    for (uint32_t dstRankIdx = startLoopIdx; dstRankIdx < worldSize_; dstRankIdx += blockAivNum_) {
        if (dstRankIdx == rankId_) {
            continue;
        }
        // 可只搬运前dstRank的数据
        copyElemNum = dstRankIdx == 0 ? expertPerRank_ : expertPerRank_ + 1;
        int32_t expertOffset = dstRankIdx == 0 ? 0 : dstRankIdx * expertPerRank_ - 1;
        tokenPerExpertRemote.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t *>(GetRankWinAddrWithOffset(dstRankIdx, peerMemSt)) + rankOffset);
        DataCopyExtParams copyParams{1U, static_cast<uint32_t>(copyElemNum * sizeof(int32_t)), 0U, 0U, 0U};
        DataCopyPadExtParams<int32_t> padParams{false, 0U, 0U, 0U};

        SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
        DataCopyPad(tmpBuffer, tokenPerExpertLocal[expertOffset], copyParams, padParams);
        SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
        DataCopyPad(tokenPerExpertRemote[expertOffset], tmpBuffer, copyParams);
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
    }
    SyncAll<true>();
    CrossRankSyncInWorldSize();

    copyElemNum = rankId_ == 0 ? expertPerRank_ : expertPerRank_ + 1;
    int32_t expertOffset = rankId_ == 0 ? 0 : rankId_ * expertPerRank_ - 1;
    DataCopyExtParams copyParams{1U, static_cast<uint32_t>(copyElemNum * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> padParams{false, 0U, 0U, 0U};
    for (uint32_t dstRankIdx = startLoopIdx; dstRankIdx < worldSize_; dstRankIdx += blockAivNum_) {
        DataCopyPad(tmpBuffer, tokenPerExpertWin_[dstRankIdx * expertNumAlign + expertOffset], copyParams, padParams);
        SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
        SetAtomicAdd<int32_t>();
        DataCopyPad(expertTokenCumsum_[expertOffset], tmpBuffer, copyParams);
        SetAtomicNone();
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
    }
    if constexpr (UseCoreWave) {
        // The raw per-rank slices this loop reduces are already globally
        // stable, so the expert plan (endpoint table, schedule order and the
        // aggregate header) builds here from the same window data - one core
        // works overlapped with the others' reduction shares instead of
        // every core serializing behind it between the histogram barriers.
        // The two PrepareExpertPipeline barriers publish it to all readers.
        if (expertPerRank_ >= EXPERT_SCHEDULE_MIN_ACTIVE_EXPERTS) {
            BuildExpertOrder();
        }
    }
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::EndSync()
{
    if (gmm1VecSetSyncCom_ == 0) {
        return;
    }
    if constexpr (g_coreType == AIC) {
        WaitForVector();
        // Consume the final outstanding cube/vector credit exactly once.
        // Interleaved staging closes it at every expert boundary before GMM2;
        // complete staging closes it once at the end of the GMM1 stage.
        gmm1VecSetSyncCom_ = 0;
    }
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline int32_t MegaMoe<TemplateMegaMoeTypeFunc>::GetLocalExpertStartRow()
{
    if (rankId_ == 0) {
        return 0;
    }
    __gm__ int32_t *cumsumAddr = const_cast<__gm__ int32_t *>(expertTokenCumsum_.GetPhyAddr());
    return ReadGmByPassDCache(cumsumAddr + static_cast<int32_t>(rankId_ * expertPerRank_ - 1));
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::CopyGMToGMPerTokenV2(
    GlobalTensor<QuantOutType> dst, GlobalTensor<QuantScaleOutType> dstScale, GlobalTensor<QuantOutType> src,
    int32_t hiddenSize, int32_t scaleLen, int32_t copyInNum, int32_t scaleOffset, int32_t tokenInAllRank,
    int32_t canonicalRow, int32_t stepK, int32_t stepMetadata, int32_t copyBufferCount, uint32_t copySlotElements,
    int32_t &copySlotIdx, int32_t &usedCopySlots)
{
    // a0行步长已按32B补齐(GetA0RowStride), 整行拷入的源地址/长度均对齐,
    // 退回plain DataCopy(最古老最被验证的路径)。非对齐GM->UB搬运(DataCopyPad,
    // 含isPad=true/false两种配置)在本芯片实测均为全0, 不可用。
    // a0行内布局: data[0,hiddenSize)+pad到256B+scale[scaleOffset,+scaleLen), padding段不外拷。
    TEventID eventId = static_cast<TEventID>(copySlotIdx);
    if (usedCopySlots == copyBufferCount) {
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
    } else {
        ++usedCopySlots;
    }
    LocalTensor<QuantOutType> dataBuf = CopyTempBuf_.Get<QuantOutType>()[copySlotIdx * copySlotElements];
    LocalTensor<QuantScaleOutType> scaleBuf = dataBuf[scaleOffset].template ReinterpretCast<QuantScaleOutType>();

    DataCopy(dataBuf, src, copyInNum);
    // MTE2 is now in flight.  Publish independent scalar metadata before the
    // MTE2->MTE3 join instead of leaving this interval empty.
    WriteGmByPassDCache(stepIdxPtr_ + canonicalRow, stepMetadata);
    WriteGmByPassDCache(posLenPtr_ + tokenInAllRank, stepK + 1);
    WriteGmByPassDCache(firstTokenPosPtr_ + tokenInAllRank, canonicalRow);
    SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
    WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
    DataCopyPad(dst, dataBuf,
                {static_cast<uint16_t>(1), static_cast<uint32_t>(hiddenSize * sizeof(QuantOutType)), 0, 0, 0});
    if constexpr (!UseBf16NoQuant) {
        // BF16 no-quant a0 rows carry only BF16 data.
        DataCopyPad(dstScale, scaleBuf, {static_cast<uint16_t>(1), static_cast<uint16_t>(scaleLen), 0, 0, 0});
    }
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
    ++copySlotIdx;
    if (copySlotIdx == copyBufferCount) {
        copySlotIdx = 0;
    }
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::CopyLocalQuantToken(
    GlobalTensor<QuantOutType> dst, GlobalTensor<QuantScaleOutType> dstScale, GlobalTensor<QuantOutType> src,
    GlobalTensor<uint8_t> srcScale, int32_t hiddenSize, int32_t scaleLen, int32_t scaleOffset, int32_t scaleLenAlign,
    int32_t tokenInAllRank, int32_t canonicalRow, int32_t stepK, int32_t stepMetadata, int32_t copyBufferCount,
    uint32_t copySlotElements, int32_t &copySlotIdx, int32_t &usedCopySlots)
{
    // a0行内scale段偏移(=补齐到256B的数据段长度, 32B对齐)。
    TEventID eventId = static_cast<TEventID>(copySlotIdx);
    if (usedCopySlots == copyBufferCount) {
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
    } else {
        ++usedCopySlots;
    }
    LocalTensor<QuantOutType> dataBuf = CopyTempBuf_.Get<QuantOutType>()[copySlotIdx * copySlotElements];
    LocalTensor<QuantScaleOutType> scaleBuf = dataBuf[scaleOffset].template ReinterpretCast<QuantScaleOutType>();
    // data段: 源是workspace ptrA(行步长hiddenSize, H均32B倍数), 地址/长度对齐, plain DataCopy。
    // 对齐 scale 直接读本地 ptrAScale，避免 stepK>0 再走跨卡 A0；
    // 非对齐 scale 仍使用 A0 中满足 GM->UB 对齐约束的 padded 行。
    DataCopy(dataBuf, src, hiddenSize);
    if constexpr (!UseBf16NoQuant) {
        DataCopy(scaleBuf.template ReinterpretCast<uint8_t>(), srcScale, scaleLenAlign);
    }
    // The two MTE2 reads are independent of these scalar publications.
    WriteGmByPassDCache(stepIdxPtr_ + canonicalRow, stepMetadata);
    WriteGmByPassDCache(posLenPtr_ + tokenInAllRank, stepK + 1);
    SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
    WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
    // 从UB拷贝到dst: UB→GM
    DataCopyPad(dst, dataBuf,
                {static_cast<uint16_t>(1), static_cast<uint32_t>(hiddenSize * sizeof(QuantOutType)), 0, 0, 0});
    if constexpr (!UseBf16NoQuant) {
        DataCopyPad(dstScale, scaleBuf, {static_cast<uint16_t>(1), static_cast<uint16_t>(scaleLen), 0, 0, 0});
    }
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
    ++copySlotIdx;
    if (copySlotIdx == copyBufferCount) {
        copySlotIdx = 0;
    }
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline int32_t MegaMoe<TemplateMegaMoeTypeFunc>::ProcessDispatchRange(
    GMMAddrInfo &gmmAddrInfo, uint32_t expertIdx, int32_t startRow, int32_t endRow, int64_t widthA, int64_t widthAScale,
    int32_t &copySlotIdx, int32_t &usedCopySlots, int32_t &sourceRankHint, int32_t &sourceRankBaseRow)
{
    if (startRow >= endRow || sourceRankHint >= static_cast<int32_t>(worldSize_)) {
        return 0;
    }

    GlobalTensor<QuantOutType> aGlobalTensor;
    GlobalTensor<QuantScaleOutType> aScaleGlobalTensor;
    GlobalTensor<QuantOutType> allLocalA;
    GlobalTensor<QuantScaleOutType> allLocalAScale;
    GlobalTensor<QuantOutType> gmRemoteA;
    aGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ QuantOutType *>(gmmAddrInfo.aGlobal));
    aScaleGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ QuantScaleOutType *>(gmmAddrInfo.aScaleGlobal));
    allLocalA.SetGlobalBuffer(reinterpret_cast<__gm__ QuantOutType *>(params_.workspaceInfo.ptrA));
    allLocalAScale.SetGlobalBuffer(reinterpret_cast<__gm__ QuantScaleOutType *>(params_.workspaceInfo.ptrAScale));

    uint64_t remoteA0Offset = params_.peermemInfo.ptrA0 - params_.peermemInfo.ptrBase;
    int64_t a0RowStride = UseBf16NoQuant ? GetA0RowStrideBf16(widthA) / static_cast<int64_t>(sizeof(QuantOutType))
                                         : GetA0RowStride(widthA);
    int64_t a0ScaleOffset = GetA0ScaleOffset(widthA);
    int32_t hiddenSize = static_cast<int32_t>(widthA);
    int32_t scaleLen = static_cast<int32_t>(widthAScale);
    int32_t copyInNum = static_cast<int32_t>(a0RowStride);
    int32_t scaleOffset = static_cast<int32_t>(a0ScaleOffset);
    int32_t scaleLenAlign = static_cast<int32_t>(Ops::Base::CeilAlign(scaleLen, static_cast<int32_t>(ALIGN_32)));
    static_assert(sizeof(QuantOutType) == 1U || UseBf16NoQuant,
                  "Dispatch copy-ring byte and element capacities assume FP8 output");
    constexpr uint32_t MIN_SLOT_ELEMENTS = DISPATCH_COPY_MIN_SLOT_BYTES / sizeof(QuantOutType);
    int32_t copyBufferCount = copyInNum <= static_cast<int32_t>(MIN_SLOT_ELEMENTS)
                                  ? static_cast<int32_t>(DISPATCH_COPY_MAX_BUFFER_NUM)
                                  : static_cast<int32_t>(DISPATCH_COPY_FALLBACK_BUFFER_NUM);
    uint32_t copySlotElements = BUFFER_ALIGN / static_cast<uint32_t>(copyBufferCount) / sizeof(QuantOutType);
    int32_t expertBaseRow = static_cast<int32_t>(Get<IDX_A_OFFSET>(baseOffset_) / static_cast<uint64_t>(widthA));
    int32_t expertAlign =
        Ops::Base::CeilAlign(static_cast<int64_t>(worldSize_ * expertPerRank_), static_cast<int64_t>(ALIGN_128));
    bool packCombineRoute = ShouldPackCombineRouteMetadata(worldSize_, m_, topK_, maxOutputSize_, blockNum_);
    uint32_t combineRouteTokenRowBits = GetTokenOwnerFieldBits(m_);
    uint32_t combineRouteTopKBits = GetTokenOwnerFieldBits(topK_);
    int32_t tokenCount = sourceRankBaseRow;
    int32_t dispatchedRows = 0;

    for (int32_t sourceRankIdx = sourceRankHint; sourceRankIdx < static_cast<int32_t>(worldSize_); ++sourceRankIdx) {
        int32_t countOffset = sourceRankIdx * expertAlign + static_cast<int32_t>(rankId_ * expertPerRank_ + expertIdx);
        int32_t curRankTokenCumsum = tokenPerExpertWin_.GetValue(countOffset);
        int32_t curRankTokenCumsumPre =
            (rankId_ == 0U && expertIdx == 0U) ? 0 : tokenPerExpertWin_.GetValue(countOffset - 1);
        int32_t curRankTokenCount = curRankTokenCumsum - curRankTokenCumsumPre;
        int32_t rankBeginRow = tokenCount;
        int32_t rankEndRow = tokenCount + curRankTokenCount;

        if (rankEndRow <= startRow) {
            tokenCount = rankEndRow;
            sourceRankHint = sourceRankIdx + 1;
            sourceRankBaseRow = rankEndRow;
            continue;
        }
        if (rankBeginRow >= endRow) {
            return dispatchedRows;
        }

        int32_t rankStart = startRow > rankBeginRow ? startRow - rankBeginRow : 0;
        int32_t rankEnd = endRow < rankEndRow ? endRow - rankBeginRow : curRankTokenCount;
        int32_t expandedRowIdxStart = sourceRankIdx * static_cast<int32_t>(m_ * topK_) + curRankTokenCumsumPre;
        for (int32_t idx = rankStart; idx < rankEnd; ++idx) {
            int32_t expandedIdx = expandedRowIdx_.GetValue(expandedRowIdxStart + idx);
            int32_t sourceRankTokenRow = DecodeExpandedTokenRow(expandedIdx);
            int32_t tokenInAllRank = sourceRankIdx * static_cast<int32_t>(m_) + sourceRankTokenRow;
            int32_t curRowInLocalExpert = rankBeginRow + idx;
            int32_t curRowInLocalRank = expertBaseRow + curRowInLocalExpert;
            int32_t stepK = ReadGmByPassDCache(posLenPtr_ + tokenInAllRank);
            uint32_t topKSlot = DecodeExpandedTopKSlot(expandedIdx);
            int32_t stepMetadata = packCombineRoute
                                       ? PackCombineRouteMetadata(static_cast<uint32_t>(sourceRankIdx),
                                                                  static_cast<uint32_t>(sourceRankTokenRow), topKSlot,
                                                                  static_cast<uint32_t>(stepK),
                                                                  combineRouteTokenRowBits, combineRouteTopKBits)
                                       : stepK;
            if (stepK == 0) {
                gmRemoteA.SetGlobalBuffer(
                    reinterpret_cast<__gm__ QuantOutType *>(GetRankWinAddrWithOffset(sourceRankIdx, remoteA0Offset)));
                CopyGMToGMPerTokenV2(aGlobalTensor[curRowInLocalExpert * widthA],
                                     aScaleGlobalTensor[curRowInLocalExpert * widthAScale],
                                     gmRemoteA[sourceRankTokenRow * a0RowStride], hiddenSize, scaleLen, copyInNum,
                                     scaleOffset, tokenInAllRank, curRowInLocalRank, stepK, stepMetadata,
                                     copyBufferCount, copySlotElements, copySlotIdx, usedCopySlots);
            } else {
                int32_t firstPos = ReadGmByPassDCache(firstTokenPosPtr_ + tokenInAllRank);
                gmRemoteA.SetGlobalBuffer(
                    reinterpret_cast<__gm__ QuantOutType *>(GetRankWinAddrWithOffset(sourceRankIdx, remoteA0Offset)));
                CopyLocalQuantToken(
                    aGlobalTensor[curRowInLocalExpert * widthA], aScaleGlobalTensor[curRowInLocalExpert * widthAScale],
                    allLocalA[static_cast<int64_t>(firstPos) * widthA],
                    gmRemoteA[sourceRankTokenRow * a0RowStride + a0ScaleOffset].template ReinterpretCast<uint8_t>(),
                    hiddenSize, scaleLen, scaleOffset, scaleLenAlign, tokenInAllRank, curRowInLocalRank, stepK,
                    stepMetadata, copyBufferCount, copySlotElements, copySlotIdx, usedCopySlots);
            }
        }
        dispatchedRows += rankEnd - rankStart;

        tokenCount = rankEndRow;
        if (endRow < rankEndRow) {
            sourceRankHint = sourceRankIdx;
            sourceRankBaseRow = rankBeginRow;
            return dispatchedRows;
        }
        sourceRankHint = sourceRankIdx + 1;
        sourceRankBaseRow = rankEndRow;
        if (endRow <= rankEndRow) {
            return dispatchedRows;
        }
    }
    return dispatchedRows;
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::DispatchTokens(GMMAddrInfo &gmmAddrInfo, uint32_t expertIdx,
                                                                        bool nextDispatchCoversJoin)
{
    int64_t widthA = Get<K_VALUE>(problemShape_);
    int64_t widthAScale = Ops::Base::CeilDiv(widthA, static_cast<int64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
    int32_t expertBaseRow = static_cast<int32_t>(Get<IDX_A_OFFSET>(baseOffset_) / static_cast<uint64_t>(widthA));
    int32_t rowsCopyCnt = static_cast<int32_t>(Get<M_VALUE>(problemShape_));
    if (static_cast<uint64_t>(expertBaseRow) + static_cast<uint64_t>(rowsCopyCnt) > maxOutputSize_) {
        rowsCopyCnt = static_cast<int32_t>(maxOutputSize_) - expertBaseRow;
        rowsCopyCnt = rowsCopyCnt > 0 ? rowsCopyCnt : 0;
    }

    // Copy slots are initialized lazily: the first use cannot race a prior
    // MTE3, while reuse waits for that slot's completion event. Normal rows
    // use four slots; rows wider than one 48-KiB slot retain two 96-KiB slots.
    int32_t copySlotIdx = 0;
    int32_t usedCopySlots = 0;

    // Keep the full runtime AIV1 width for every expert.  Dispatch is dominated
    // by remote MTE issue throughput on large EP shapes; capping a scheduled
    // expert to half of the available workers serializes those requests.  Only
    // non-empty row owners publish completion, so naturally idle cores do not
    // add counter traffic.
    int32_t dispatchWorkerCount = static_cast<int32_t>(blockNum_);
    uint32_t dispatchMGroups =
        rowsCopyCnt > 0 ? Ops::Base::CeilDiv(static_cast<uint32_t>(rowsCopyCnt), EXPERT_SCHEDULE_M_TILE_ROWS) : 0U;
    uint32_t gmm1NTiles = Ops::Base::CeilDiv(static_cast<uint32_t>(Get<N_VALUE>(problemShape_) / SWIGLU_N_HALF),
                                             EXPERT_SCHEDULE_M_TILE_ROWS);
    uint64_t dispatchGmm1Tiles = static_cast<uint64_t>(dispatchMGroups) * gmm1NTiles;
    // Production uses one contiguous static range per active AIV1.  Keeping
    // the source scan directly in this function lets CCE fold the single-range
    // bounds and avoids the generic tail-scheduler state on every token.
    int32_t baseRowsPerCore = rowsCopyCnt / dispatchWorkerCount;
    int32_t extraRowCores = rowsCopyCnt % dispatchWorkerCount;
    int32_t activeDispatchWorkers = rowsCopyCnt < dispatchWorkerCount ? rowsCopyCnt : dispatchWorkerCount;
    // For experts whose GMM1 fits in one physical AIC wave, Dispatch is the
    // measured per-expert throughput limiter.  Defer tile
    // publication only when the next expert has at least comparable Dispatch
    // row work, so that work can cover the current expert's lost intra-expert
    // early-start window.  The final active expert, a single hot expert, and
    // a hot-to-cold transition retain fine-grained publication.
    // Per-core tile atomics (roughly one per active AIV) then cost more than
    // the intra-expert early-start window.  Preserve the exact ready-counter
    // values, but publish them once after the existing all-core completion
    // join.  Larger compute shapes keep fine-grained tile publication.
    bool publishTileReadyAfterJoin = nextDispatchCoversJoin && dispatchGmm1Tiles > 0U &&
                                     dispatchGmm1Tiles <= blockNum_ &&
                                     GetMaterialMGroupCount(static_cast<uint32_t>(rowsCopyCnt)) <= 2U;
    gmmAddrInfo.dispatchReadyTarget = activeDispatchWorkers;
    // Stable quotient/remainder ownership keeps the same logical row range on
    // the same physical AIV across expert boundaries.
    int32_t logicalWorker = static_cast<int32_t>(blockIdx_);
    int32_t rowDstOffsetPerCore =
        logicalWorker * baseRowsPerCore + (logicalWorker < extraRowCores ? logicalWorker : extraRowCores);
    int32_t curRows = baseRowsPerCore + (logicalWorker < extraRowCores ? 1 : 0);

    if (curRows > 0) {
        GlobalTensor<QuantOutType> aGlobalTensor;
        GlobalTensor<QuantScaleOutType> aScaleGlobalTensor;
        GlobalTensor<QuantOutType> allLocalA;
        GlobalTensor<QuantScaleOutType> allLocalAScale;
        GlobalTensor<QuantOutType> gmRemoteA;
        aGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ QuantOutType *>(gmmAddrInfo.aGlobal));
        aScaleGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ QuantScaleOutType *>(gmmAddrInfo.aScaleGlobal));
        allLocalA.SetGlobalBuffer(reinterpret_cast<__gm__ QuantOutType *>(params_.workspaceInfo.ptrA));
        allLocalAScale.SetGlobalBuffer(reinterpret_cast<__gm__ QuantScaleOutType *>(params_.workspaceInfo.ptrAScale));

        uint64_t remoteA0Offset = params_.peermemInfo.ptrA0 - params_.peermemInfo.ptrBase;
        // Element units of QuantOutType. BF16 no-quant a0 rows hold only the
        // H BF16 elements (32B aligned, no scale segment).
        int64_t a0RowStride = UseBf16NoQuant ? GetA0RowStrideBf16(widthA) / static_cast<int64_t>(sizeof(QuantOutType))
                                             : GetA0RowStride(widthA);
        int64_t a0ScaleOffset = GetA0ScaleOffset(widthA);
        int32_t hiddenSize = static_cast<int32_t>(widthA);
        int32_t scaleLen = static_cast<int32_t>(widthAScale);
        int32_t copyInNum = static_cast<int32_t>(a0RowStride);
        int32_t scaleOffset = static_cast<int32_t>(a0ScaleOffset);
        int32_t scaleLenAlign = static_cast<int32_t>(Ops::Base::CeilAlign(scaleLen, static_cast<int32_t>(ALIGN_32)));
        static_assert(sizeof(QuantOutType) == 1U || UseBf16NoQuant,
                      "Dispatch copy-ring byte and element capacities assume FP8 output");
        constexpr uint32_t MIN_SLOT_ELEMENTS = DISPATCH_COPY_MIN_SLOT_BYTES / sizeof(QuantOutType);
        int32_t copyBufferCount = copyInNum <= static_cast<int32_t>(MIN_SLOT_ELEMENTS)
                                      ? static_cast<int32_t>(DISPATCH_COPY_MAX_BUFFER_NUM)
                                      : static_cast<int32_t>(DISPATCH_COPY_FALLBACK_BUFFER_NUM);
        uint32_t copySlotElements = BUFFER_ALIGN / static_cast<uint32_t>(copyBufferCount) / sizeof(QuantOutType);
        bool localScaleAligned = scaleLen == scaleLenAlign;
        int32_t expertAlign =
            Ops::Base::CeilAlign(static_cast<int64_t>(worldSize_ * expertPerRank_), static_cast<int64_t>(ALIGN_128));
        bool packCombineRoute = ShouldPackCombineRouteMetadata(worldSize_, m_, topK_, maxOutputSize_, blockNum_);
        uint32_t combineRouteTokenRowBits = GetTokenOwnerFieldBits(m_);
        uint32_t combineRouteTopKBits = GetTokenOwnerFieldBits(topK_);
        int32_t taskEnd = rowDstOffsetPerCore + curRows;
        int32_t tokenCount = 0;
        for (int32_t sourceRankIdx = 0; sourceRankIdx < static_cast<int32_t>(worldSize_); ++sourceRankIdx) {
            int32_t countOffset =
                sourceRankIdx * expertAlign + static_cast<int32_t>(rankId_ * expertPerRank_ + expertIdx);
            int32_t curRankTokenCumsum = tokenPerExpertWin_.GetValue(countOffset);
            int32_t curRankTokenCumsumPre =
                (rankId_ == 0U && expertIdx == 0U) ? 0 : tokenPerExpertWin_.GetValue(countOffset - 1);
            int32_t curRankTokenCount = curRankTokenCumsum - curRankTokenCumsumPre;
            int32_t rankEndRow = tokenCount + curRankTokenCount;
            if (rankEndRow <= rowDstOffsetPerCore) {
                tokenCount = rankEndRow;
                continue;
            }
            if (tokenCount >= taskEnd) {
                break;
            }

            int32_t rankStart = rowDstOffsetPerCore > tokenCount ? rowDstOffsetPerCore - tokenCount : 0;
            int32_t rankEnd = taskEnd < rankEndRow ? taskEnd - tokenCount : curRankTokenCount;
            int32_t expandedRowIdxStart = sourceRankIdx * static_cast<int32_t>(m_ * topK_) + curRankTokenCumsumPre;
            // 源rank的a0窗口: stepK==0读data+scale整行, stepK>0仅读scale段, 两分支共用。
            gmRemoteA.SetGlobalBuffer(
                reinterpret_cast<__gm__ QuantOutType *>(GetRankWinAddrWithOffset(sourceRankIdx, remoteA0Offset)));
            // Keep the C3 one-pass issue order.  The former four-row metadata
            // batch read firstTokenPos even for first contributions and held
            // six scalar arrays live across payload copies, increasing both
            // GM traffic and vector-register pressure on the measured EP2
            // Dispatch bottleneck.
            for (int32_t idx = rankStart; idx < rankEnd; ++idx) {
                int32_t expandedIdx = expandedRowIdx_.GetValue(expandedRowIdxStart + idx);
                int32_t sourceRankTokenRow = DecodeExpandedTokenRow(expandedIdx);
                int32_t tokenInAllRank = sourceRankIdx * static_cast<int32_t>(m_) + sourceRankTokenRow;
                int32_t curRowInLocalExpert = tokenCount + idx;
                int32_t curRowInLocalRank = expertBaseRow + curRowInLocalExpert;
                int32_t stepK = ReadGmByPassDCache(posLenPtr_ + tokenInAllRank);
                uint32_t topKSlot = DecodeExpandedTopKSlot(expandedIdx);
                int32_t stepMetadata = packCombineRoute
                                           ? PackCombineRouteMetadata(static_cast<uint32_t>(sourceRankIdx),
                                                                      static_cast<uint32_t>(sourceRankTokenRow),
                                                                      topKSlot, static_cast<uint32_t>(stepK),
                                                                      combineRouteTokenRowBits, combineRouteTopKBits)
                                           : stepK;
                if (stepK == 0) {
                    CopyGMToGMPerTokenV2(aGlobalTensor[curRowInLocalExpert * widthA],
                                         aScaleGlobalTensor[curRowInLocalExpert * widthAScale],
                                         gmRemoteA[sourceRankTokenRow * a0RowStride], hiddenSize, scaleLen, copyInNum,
                                         scaleOffset, tokenInAllRank, curRowInLocalRank, stepK, stepMetadata,
                                         copyBufferCount, copySlotElements, copySlotIdx, usedCopySlots);
                } else {
                    // firstTokenPos is needed only after the first local
                    // contribution; avoid the unconditional C5 scalar read.
                    int32_t firstPos = ReadGmByPassDCache(firstTokenPosPtr_ + tokenInAllRank);
                    GlobalTensor<uint8_t> scaleSource;
                    if (localScaleAligned) {
                        scaleSource = allLocalAScale[static_cast<int64_t>(firstPos) * widthAScale]
                                          .template ReinterpretCast<uint8_t>();
                    } else {
                        scaleSource = gmRemoteA[sourceRankTokenRow * a0RowStride + a0ScaleOffset]
                                          .template ReinterpretCast<uint8_t>();
                    }
                    CopyLocalQuantToken(aGlobalTensor[curRowInLocalExpert * widthA],
                                        aScaleGlobalTensor[curRowInLocalExpert * widthAScale],
                                        allLocalA[static_cast<int64_t>(firstPos) * widthA], scaleSource, hiddenSize,
                                        scaleLen, scaleOffset, scaleLenAlign, tokenInAllRank, curRowInLocalRank, stepK,
                                        stepMetadata, copyBufferCount, copySlotElements, copySlotIdx, usedCopySlots);
                }
            }
            tokenCount = rankEndRow;
            if (tokenCount >= taskEnd) {
                break;
            }
        }
    }

    SyncFunc<AscendC::HardEvent::MTE3_S>();
    if (curRows > 0) {
        // Chain metadata uses scalar bypass-cache GM stores. Complete those
        // stores before any ready/done vote can release the next expert or
        // final chain-length reader; MTE3_S above only drains payload DMA.
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
    }
    // Publish row counts only after this core's payload MTE3 writes are
    // visible.  AIC GMM1 can consume a complete 256-row tile immediately;
    // the full-expert vote below is retained for cross-expert metadata order.
    // Sparse-decode compiles the per-tile publication out: its AIC waits for
    // the single dispatch-complete flag instead of per-tile row counters.
    if constexpr (!UseSparseDecode) {
        if (curRows > 0) {
            constexpr int32_t DISPATCH_TILE_ROWS = static_cast<int32_t>(EXPERT_SCHEDULE_M_TILE_ROWS);
            int32_t segmentBegin = rowDstOffsetPerCore;
            int32_t segmentEnd = rowDstOffsetPerCore + curRows;
            int32_t firstTile = segmentBegin / DISPATCH_TILE_ROWS;
            int32_t lastTile = (segmentEnd - 1) / DISPATCH_TILE_ROWS;
            for (int32_t tileIdx = firstTile; tileIdx <= lastTile; ++tileIdx) {
                int32_t tileBegin = tileIdx * DISPATCH_TILE_ROWS;
                int32_t tileEnd = tileBegin + DISPATCH_TILE_ROWS;
                int32_t overlapBegin = segmentBegin > tileBegin ? segmentBegin : tileBegin;
                int32_t overlapEnd = segmentEnd < tileEnd ? segmentEnd : tileEnd;
                __gm__ int32_t *readyAddr = gmmAddrInfo.dispatchTileReady +
                                            static_cast<int64_t>(tileIdx) * DISPATCH_TILE_READY_STRIDE +
                                            DISPATCH_READY_OFFSET;
                if (!publishTileReadyAfterJoin) {
                    AtomicAdd(readyAddr, overlapEnd - overlapBegin);
                }
            }
        }
    }
    if (curRows > 0) {
        AtomicAdd(gmmAddrInfo.dispatchDoneCores, int32_t(1));
    }
    for (int32_t slot = 0; slot < usedCopySlots; ++slot) {
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(static_cast<TEventID>(slot));
    }
    // Tok 在 dispatchDoneCores 屏障自旋之前 (尾部等待不计入)。
    TIMER_BLOCK(timer.template Tok<Overwrite>(AscendTimer::AIV_DISPATCH, expertIdx));
    // The per-expert join orders the non-atomic posLen/firstTokenPos writes
    // across experts, so every core that can ever own a routed row must keep
    // waiting here.  Under the sparse-decode invariant rowsCopyCnt never
    // exceeds maxOutputSize_, and the quotient/remainder split hands rows only
    // to logical workers below min(maxOutputSize_, blockNum_); a core at or
    // beyond that bound writes no payload and no chain metadata in any
    // expert, so exempting it from the join cannot reorder any writer.
    bool waitsDispatchJoin = true;
    if constexpr (UseSparseDecode) {
        uint64_t sparseWorkerBound =
            maxOutputSize_ < static_cast<uint64_t>(blockNum_) ? maxOutputSize_ : static_cast<uint64_t>(blockNum_);
        waitsDispatchJoin = static_cast<uint64_t>(logicalWorker) < sparseWorkerBound;
    }
    if (waitsDispatchJoin) {
        uint32_t pollMisses = 0U;
        int64_t pollDelayCycles = 100;
        while (AscendC::ReadGmByPassDCache(gmmAddrInfo.dispatchDoneCores) < gmmAddrInfo.dispatchReadyTarget) {
            GmSpinBackoffStep(pollMisses, pollDelayCycles);
        }
    }
    if constexpr (!UseSparseDecode) {
        if (publishTileReadyAfterJoin && blockIdx_ == 0U) {
            for (uint32_t tileIdx = 0U; tileIdx < dispatchMGroups; ++tileIdx) {
                uint32_t tileBegin = tileIdx * EXPERT_SCHEDULE_M_TILE_ROWS;
                uint32_t tileRows = static_cast<uint32_t>(rowsCopyCnt) - tileBegin;
                tileRows = tileRows < EXPERT_SCHEDULE_M_TILE_ROWS ? tileRows : EXPERT_SCHEDULE_M_TILE_ROWS;
                WriteGmByPassDCache(gmmAddrInfo.dispatchTileReady +
                                        static_cast<uint64_t>(tileIdx) * DISPATCH_TILE_READY_STRIDE +
                                        DISPATCH_READY_OFFSET,
                                    static_cast<int32_t>(tileRows));
            }
            PipeBarrier<PIPE_ALL>();
        }
    }
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::PrecomputeChainLen()
{
    // Count, for every token in all-rank token space, how many of its routed
    // experts are local to this rank.  The result equals the final posLen
    // value but is schedule independent: it only needs the AllGathered
    // expandedRowIdx/tokenPerExpertWin/expertTokenCumsum, all visible after
    // the PrepareExpertPipeline barriers.  Combine reads this array instead
    // of posLen, which removes the need for a full all-expert metadata
    // prepass before the wave pipeline may start GMM1.
    //
    // Runs on the blockNum_ AIV sub-block-0 cores while they would otherwise wait
    // for the first GMM1 tile.  Each core walks one contiguous slice of the
    // rank's received-row space and publishes one completion vote; Combine
    // waits for blockNum votes before its first chainLen read.
    int32_t expertAlign =
        Ops::Base::CeilAlign(static_cast<int64_t>(worldSize_ * expertPerRank_), static_cast<int64_t>(ALIGN_128));
    __gm__ int32_t *cumsumAddr = const_cast<__gm__ int32_t *>(expertTokenCumsum_.GetPhyAddr());
    int32_t rankExpertBase = static_cast<int32_t>(rankId_ * expertPerRank_);
    int32_t rankBaseRow = rankId_ == 0 ? 0 : ReadGmByPassDCache(cumsumAddr + rankExpertBase - 1);
    int32_t rankEndRow = ReadGmByPassDCache(cumsumAddr + rankExpertBase + static_cast<int32_t>(expertPerRank_) - 1);
    int32_t totalRows = rankEndRow - rankBaseRow;
    if (totalRows > 0 && static_cast<uint64_t>(totalRows) > maxOutputSize_) {
        totalRows = static_cast<int32_t>(maxOutputSize_);
    }
    int32_t rowsPerCore = Ops::Base::CeilDiv(totalRows, static_cast<int32_t>(blockNum_));
    int32_t rowStart = static_cast<int32_t>(blockIdx_) * rowsPerCore;
    int32_t rowEnd = rowStart + rowsPerCore < totalRows ? rowStart + rowsPerCore : totalRows;

    if (rowStart < rowEnd) {
        int32_t mTopK = static_cast<int32_t>(m_ * topK_);
        // Received rows are ordered (expert, sourceRank).  Locate the segment
        // holding rowStart: first the expert via the global cumsum, then the
        // source-rank segment via that expert's per-source counts.
        int32_t targetRowGlobal = rankBaseRow + rowStart;
        int32_t g = rankExpertBase;
        int32_t prevCumsum = rankBaseRow;
        int32_t curCumsum = ReadGmByPassDCache(cumsumAddr + g);
        while (curCumsum <= targetRowGlobal) {
            prevCumsum = curCumsum;
            ++g;
            curCumsum = ReadGmByPassDCache(cumsumAddr + g);
        }
        int32_t offInExpert = targetRowGlobal - prevCumsum;
        int32_t s = 0;
        int32_t curPre = 0;
        int32_t segRemaining = 0;
        int32_t expandedIdxPos = 0;
        {
            int32_t off = offInExpert;
            while (true) {
                curPre = g == 0 ? 0 : tokenPerExpertWin_.GetValue(s * expertAlign + g - 1);
                int32_t cur = tokenPerExpertWin_.GetValue(s * expertAlign + g);
                segRemaining = cur - curPre;
                if (off < segRemaining) {
                    expandedIdxPos = s * mTopK + curPre + off;
                    // 起点在段内第 off 项, 剩余量需扣掉已跨过的段内前缀
                    // (combine 侧游标同款: curSegCount - skipped), 否则跨段
                    // 读取会漏计头部 off 行、多计尾部 off 行。
                    segRemaining -= off;
                    break;
                }
                off -= segRemaining;
                ++s;
            }
        }
        for (int32_t r = rowStart; r < rowEnd; ++r) {
            int32_t expandedIdx = expandedRowIdx_.GetValue(expandedIdxPos);
            ++expandedIdxPos;
            int32_t token = DecodeExpandedTokenRow(expandedIdx);
            AscendC::AtomicAdd(chainLenPtr_ + s * static_cast<int32_t>(m_) + token, int32_t(1));
            if (r + 1 == rowEnd) {
                break;
            }
            if (--segRemaining == 0) {
                // Advance to the next non-empty (expert, sourceRank) segment.
                do {
                    ++s;
                    if (s == static_cast<int32_t>(worldSize_)) {
                        s = 0;
                        ++g;
                    }
                    curPre = g == 0 ? 0 : tokenPerExpertWin_.GetValue(s * expertAlign + g - 1);
                    int32_t cur = tokenPerExpertWin_.GetValue(s * expertAlign + g);
                    segRemaining = cur - curPre;
                    expandedIdxPos = s * mTopK + curPre;
                } while (segRemaining == 0);
            }
        }
    }

    // Publish completion after all GM atomics of this core are visible.
    PipeBarrier<PIPE_ALL>();
    AscendC::AtomicAdd(chainLenReadyFlagPtr_, int32_t(1));
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::DispatchExperts(GMMAddrInfo &gmmAddrInfo,
                                                                         uint32_t scheduleBegin, uint32_t scheduleEnd,
                                                                         int32_t rankBaseRow)
{
    static_assert(UseCoreWave, "Staged dispatch is only valid for the core-wave specialization");
    scheduleEnd = scheduleEnd < expertPerRank_ ? scheduleEnd : expertPerRank_;
    // Keep a logical row range on the same physical worker across experts.
    // Rotating scheduled experts made almost every token-chain dependency a
    // cross-core hand-off and turned posDone into a cache-line convoy.
    for (uint32_t scheduleIdx = scheduleBegin; scheduleIdx < scheduleEnd; ++scheduleIdx) {
        uint32_t expertIdx = GetScheduledExpert(scheduleIdx);
        bool hasGroup =
            expertScheduleEnabled_ ? SetGroupParams(expertIdx, rankBaseRow) : UpdateIdentityGroupParams(expertIdx);
        if (!hasGroup) {
            continue;
        }
        bool nextDispatchCoversJoin = false;
        if (scheduleIdx + 1U < scheduleEnd) {
            uint32_t nextExpertIdx = GetScheduledExpert(scheduleIdx + 1U);
            int32_t nextLocalStart = 0;
            int32_t nextRows = 0;
            // A comparable or larger following expert supplies enough
            // Dispatch work to hide this expert's post-join tile publication;
            // a shrinking hot-to-cold tail conservatively keeps early-ready.
            nextDispatchCoversJoin =
                GetExpertRange(nextExpertIdx, rankBaseRow, nextLocalStart, nextRows) &&
                static_cast<int64_t>(nextRows) + 1 >= static_cast<int64_t>(Get<M_VALUE>(problemShape_));
        }
        UpdateGlobalBuffer(gmmAddrInfo, 0);
        // Tik 紧贴 DispatchTokens; Tok 在其内部 (自旋之前), 见 :961。
        TIMER_BLOCK(timer.Tik(AscendTimer::AIV_DISPATCH, expertIdx));
        DispatchTokens(gmmAddrInfo, expertIdx, nextDispatchCoversJoin);
    }
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::BuildDispatchFanoutPlan(int32_t rankBaseRow)
{
    uint32_t scheduleBits = GetTokenOwnerFieldBits(expertPerRank_);
    uint32_t tileBits = GetTokenOwnerFieldBits(maxDispatchTilesPerExpert_);
    uint32_t payloadBits = scheduleBits + tileBits;
    int32_t expertAlign =
        Ops::Base::CeilAlign(static_cast<int64_t>(worldSize_ * expertPerRank_), static_cast<int64_t>(ALIGN_128));
    int32_t mTopK = static_cast<int32_t>(m_ * topK_);
    int32_t rankExpertBase = static_cast<int32_t>(rankId_ * expertPerRank_);

    bool builtWithSimt = false;
    // Every core-wave token-local path consumes the same inverse route plan;
    // owner-chain changes only the later Combine consumer.  Use the existing
    // SIMT producer for both short and long local chains so AIC startup is not
    // held behind a scalar traversal of every routed row.  The guards keep all
    // SIMT indices in int32, and unsupported future shapes retain the scalar
    // planner below.
    bool simtIndexCapable = expertPerRank_ >= EXPERT_SCHEDULE_MIN_ACTIVE_EXPERTS && m_ != 0U && expertPerRank_ != 0U &&
                            worldSize_ != 0U && blockNum_ != 0U && rankId_ < worldSize_ && blockIdx_ < blockNum_ &&
                            expertPerRank_ <= static_cast<uint32_t>(INT32_MAX) / worldSize_ && expertAlign > 0 &&
                            worldSize_ <= static_cast<uint32_t>(INT32_MAX) / static_cast<uint32_t>(expertAlign) &&
                            topK_ <= static_cast<uint32_t>(INT32_MAX) / m_ && mTopK > 0 &&
                            worldSize_ <= static_cast<uint32_t>(INT32_MAX) / static_cast<uint32_t>(mTopK) &&
                            expertPerRank_ <= (static_cast<uint32_t>(INT32_MAX) - 1U) / 2U;
    if (simtIndexCapable) {
        __gm__ int32_t *planAddr = const_cast<__gm__ int32_t *>(expertOrderScratch_.GetPhyAddr());
        __gm__ int32_t *endpointAddr = planAddr + 1U;
        __gm__ int32_t *inverseAddr = endpointAddr + expertPerRank_;
        int32_t totalRows = ReadGmByPassDCache(endpointAddr + expertPerRank_ - 1U);
        if (totalRows >= 0 && static_cast<uint32_t>(totalRows) <= maxOutputSize_) {
            uint32_t totalRowsU = static_cast<uint32_t>(totalRows);
            uint32_t baseRows = totalRowsU / blockNum_;
            uint32_t extraWorkers = totalRowsU % blockNum_;
            uint32_t rowBegin = blockIdx_ * baseRows + (blockIdx_ < extraWorkers ? blockIdx_ : extraWorkers);
            uint32_t rowCount = baseRows + (blockIdx_ < extraWorkers ? 1U : 0U);
            __gm__ int32_t *tokenPerExpertAddr = const_cast<__gm__ int32_t *>(tokenPerExpertWin_.GetPhyAddr());
            __gm__ int32_t *expandedRowIdxAddr = const_cast<__gm__ int32_t *>(expandedRowIdx_.GetPhyAddr());
            Simt::VF_CALL<BuildOwnerDispatchFanoutPlanSimt>(
                Simt::Dim3{OWNER_DISPATCH_SIMT_THREAD_NUM, 1, 1}, static_cast<int32_t>(rowBegin),
                static_cast<int32_t>(rowCount), static_cast<int32_t>(expertPerRank_), static_cast<int32_t>(worldSize_),
                static_cast<int32_t>(rankId_), expertAlign, mTopK, scheduleBits, payloadBits, endpointAddr, inverseAddr,
                tokenPerExpertAddr, expandedRowIdxAddr, tokenChainDescriptorsPtr_);
            builtWithSimt = true;
        }
    }

    if (!builtWithSimt) {
        // Preserve the established scalar capability fallback.
        for (uint32_t scheduleIdx = 0U; scheduleIdx < expertPerRank_; ++scheduleIdx) {
            uint32_t expertIdx = GetScheduledExpert(scheduleIdx);
            int32_t localStart = 0;
            int32_t rows = 0;
            if (!GetExpertRange(expertIdx, rankBaseRow, localStart, rows)) {
                continue;
            }
            int32_t baseRowsPerWorker = rows / static_cast<int32_t>(blockNum_);
            int32_t extraWorkers = rows % static_cast<int32_t>(blockNum_);
            int32_t worker = static_cast<int32_t>(blockIdx_);
            int32_t rowBegin = worker * baseRowsPerWorker + (worker < extraWorkers ? worker : extraWorkers);
            int32_t workerRows = baseRowsPerWorker + (worker < extraWorkers ? 1 : 0);
            int32_t rowEnd = rowBegin + workerRows;
            if (rowBegin >= rowEnd) {
                continue;
            }
            int32_t globalExpertIdx = rankExpertBase + static_cast<int32_t>(expertIdx);
            int32_t sourceBaseRow = 0;
            for (int32_t sourceRankIdx = 0; sourceRankIdx < static_cast<int32_t>(worldSize_) && sourceBaseRow < rowEnd;
                 ++sourceRankIdx) {
                int32_t countOffset = sourceRankIdx * expertAlign + globalExpertIdx;
                int32_t current = tokenPerExpertWin_.GetValue(countOffset);
                int32_t previous = globalExpertIdx == 0 ? 0 : tokenPerExpertWin_.GetValue(countOffset - 1);
                int32_t sourceRows = current - previous;
                int32_t sourceEndRow = sourceBaseRow + sourceRows;
                if (sourceEndRow <= rowBegin) {
                    sourceBaseRow = sourceEndRow;
                    continue;
                }
                if (sourceBaseRow >= rowEnd) {
                    break;
                }
                int32_t sourceBegin = rowBegin > sourceBaseRow ? rowBegin - sourceBaseRow : 0;
                int32_t sourceEnd = rowEnd < sourceEndRow ? rowEnd - sourceBaseRow : sourceRows;
                int32_t expandedBase = sourceRankIdx * mTopK + previous;
                for (int32_t idx = sourceBegin; idx < sourceEnd; ++idx) {
                    int32_t expandedIdx = expandedRowIdx_.GetValue(expandedBase + idx);
                    int32_t tokenRow = DecodeExpandedTokenRow(expandedIdx);
                    uint32_t topKSlot = DecodeExpandedTopKSlot(expandedIdx);
                    int32_t tokenInAllRank = sourceRankIdx * static_cast<int32_t>(m_) + tokenRow;
                    int32_t rowInExpert = sourceBaseRow + idx;
                    uint32_t tileIdx = static_cast<uint32_t>(rowInExpert) / EXPERT_SCHEDULE_M_TILE_ROWS;
                    uint32_t canonicalRow = static_cast<uint32_t>(localStart + rowInExpert);
                    uint32_t descriptor = (canonicalRow << payloadBits) | (tileIdx << scheduleBits) | scheduleIdx;
                    uint32_t descriptorOffset = static_cast<uint32_t>(tokenInAllRank) * topK_ + topKSlot;
                    WriteGmByPassDCache(tokenChainDescriptorsPtr_ + descriptorOffset, static_cast<int32_t>(descriptor));
                }
                sourceBaseRow = sourceEndRow;
            }
        }
    }

    // Planning is partitioned by expert row, while the following fanout is
    // partitioned by source token.  Every AIV1 may therefore consume
    // descriptors produced by a different AIV1.  Publish and join the whole
    // plan before any worker changes ownership, regardless of which Combine
    // specialization will consume the resulting payload.  The owner-chain
    // mode additionally reuses the completed counter as its descriptor-ready
    // indication; the established core-wave path needs the join itself.
    PipeBarrier<PIPE_ALL>();
    AtomicAdd(chainLenReadyFlagPtr_, int32_t(1));
    uint32_t pollMisses = 0U;
    int64_t pollDelayCycles = 100;
    while (ReadGmByPassDCache(chainLenReadyFlagPtr_) < static_cast<int32_t>(blockNum_)) {
        GmSpinBackoffStep(pollMisses, pollDelayCycles);
    }
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::FlushDispatchFanoutReady(
    LocalTensor<int32_t> countTable, LocalTensor<int32_t> touchedBuckets, uint32_t countStride, int32_t &touchedCount,
    int32_t &pendingRoutes, int32_t &copySlotIdx, int32_t &usedCopySlots)
{
    if (touchedCount == 0) {
        return;
    }
    // Every bucket count below covers only payload stores that precede the
    // slot's MTE3->MTE2 completion event.
    for (int32_t slot = 0; slot < usedCopySlots; ++slot) {
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(static_cast<TEventID>(slot));
    }
    PipeBarrier<PIPE_ALL>();
    __gm__ int32_t *readyBase = reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.ptrDispatchTileReady);
    for (int32_t touchedIdx = 0; touchedIdx < touchedCount; ++touchedIdx) {
        int32_t bucket = touchedBuckets.GetValue(touchedIdx);
        int32_t count = countTable.GetValue(bucket);
        uint32_t scheduleIdx = static_cast<uint32_t>(bucket) / countStride;
        uint32_t tileIdx = static_cast<uint32_t>(bucket) - scheduleIdx * countStride;
        // Keep all launch-private counters in compact schedule space.  Only
        // the physical GM ready address needs the data-dependent hot/cold
        // mapping, reducing one dependent bypass-GM lookup per route to one
        // lookup per published nonzero bucket.
        uint32_t expertIdx = GetScheduledExpert(scheduleIdx);
        AtomicAdd(readyBase + (expertIdx * maxDispatchTilesPerExpert_ + tileIdx) * DISPATCH_TILE_READY_STRIDE +
                      DISPATCH_READY_OFFSET,
                  count);
        countTable.SetValue(bucket, 0);
    }
    PipeBarrier<PIPE_ALL>();
    touchedCount = 0;
    pendingRoutes = 0;
    copySlotIdx = 0;
    usedCopySlots = 0;
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::DispatchTokensByOwner(int32_t rankBaseRow)
{
    BuildDispatchFanoutPlan(rankBaseRow);

    static_assert(sizeof(QuantOutType) == 1U || UseBf16NoQuant,
                  "Token-local Dispatch uses byte offsets inside CopyTempBuf");
    int64_t widthA = Get<K_VALUE>(problemShape_);
    int64_t widthAScale =
        UseBf16NoQuant ? 0 : Ops::Base::CeilDiv(widthA, static_cast<int64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
    int32_t a0RowStride =
        static_cast<int32_t>(UseBf16NoQuant ? GetA0RowStrideBf16(widthA) / static_cast<int64_t>(sizeof(QuantOutType))
                                            : GetA0RowStride(widthA));
    int32_t a0ScaleOffset = UseBf16NoQuant ? 0 : static_cast<int32_t>(GetA0ScaleOffset(widthA));
    int32_t scaleLen = static_cast<int32_t>(widthAScale);
    uint32_t slotElements =
        static_cast<uint32_t>(Ops::Base::CeilAlign(static_cast<int64_t>(a0RowStride), static_cast<int64_t>(ALIGN_32)));
    // Give every AIV the same Q/R row slice from every source rank.  The old
    // flat [world*m] split made one worker serve almost exclusively one peer;
    // rank-skewed first-pull traffic then translated directly into a slow
    // worker.  Source striping preserves exact token ownership while each
    // worker now carries the aggregate peer mix.  Rotating the first source
    // avoids synchronizing all workers onto the same NoC peer at launch.
    uint32_t rowsPerSourceCore = m_ / blockNum_;
    uint32_t extraSourceWorkers = m_ % blockNum_;
    uint32_t sourceRowBegin =
        blockIdx_ * rowsPerSourceCore + (blockIdx_ < extraSourceWorkers ? blockIdx_ : extraSourceWorkers);
    uint32_t sourceRowCount = rowsPerSourceCore + (blockIdx_ < extraSourceWorkers ? 1U : 0U);
    uint32_t tokenCount = sourceRowCount * worldSize_;
    uint32_t firstSource = blockIdx_ % worldSize_;
    // Descriptor packing and the physical ready workspace retain the
    // conservative max-output stride.  The private ordering table only needs
    // the largest tile count that any runtime expert actually owns.
    uint32_t priorityStride = dispatchPriorityStride_;
    // The private counter table needs only runtime material tiles.  The GM
    // ready workspace keeps its conservative max-output stride; Flush maps
    // from this compact stride when publishing.  This avoids clearing and
    // randomly touching a mostly empty expert*capacity table every launch.
    uint32_t readyCounterEntries = expertPerRank_ * priorityStride;
    uint32_t priorityEntries = expertPerRank_ * priorityStride;
    uint32_t counterByteOffset =
        DISPATCH_FANOUT_BUFFER_NUM * slotElements * static_cast<uint32_t>(sizeof(QuantOutType));
    uint32_t counterBytes = Ops::Base::CeilAlign(readyCounterEntries * static_cast<uint32_t>(sizeof(int32_t)),
                                                 static_cast<uint32_t>(ALIGN_32));
    uint32_t priorityByteOffset = counterByteOffset + counterBytes;
    uint32_t priorityBytes =
        Ops::Base::CeilAlign(priorityEntries * static_cast<uint32_t>(sizeof(int32_t)), static_cast<uint32_t>(ALIGN_32));
    uint32_t scratchByteOffset = priorityByteOffset + priorityBytes;
    uint32_t touchedCapacity = GetDispatchFanoutTouchedCapacity(topK_);

    LocalTensor<QuantOutType> byteBuffer = CopyTempBuf_.Get<QuantOutType>();
    LocalTensor<int32_t> countTable =
        byteBuffer[counterByteOffset / static_cast<uint32_t>(sizeof(QuantOutType))].template ReinterpretCast<int32_t>();
    LocalTensor<int32_t> priorityTable = byteBuffer[priorityByteOffset / static_cast<uint32_t>(sizeof(QuantOutType))]
                                             .template ReinterpretCast<int32_t>();
    LocalTensor<int32_t> routeScratch =
        byteBuffer[scratchByteOffset / static_cast<uint32_t>(sizeof(QuantOutType))].template ReinterpretCast<int32_t>();
    LocalTensor<int32_t> routeSlotScratch = routeScratch[topK_];
    LocalTensor<int32_t> touchedBuckets = routeScratch[2U * topK_];
    uint32_t tokenArrayBase =
        Ops::Base::CeilAlign(2U * topK_ + touchedCapacity, static_cast<uint32_t>(ALIGN_32 / sizeof(int32_t)));
    uint32_t alignedTokenCount = Ops::Base::CeilAlign(tokenCount, static_cast<uint32_t>(ALIGN_32 / sizeof(int32_t)));
    // Keep the existing two-token-array reservation so the proven UB gate is
    // unchanged.  Only the first array is live now: it links tokens assigned
    // to every priority after the first material one.  The second array is
    // retained as padding before the descriptor cache.
    LocalTensor<int32_t> nextToken = routeScratch[tokenArrayBase];
    LocalTensor<int32_t> descriptorStorage = nextToken[2U * alignedTokenCount];

    // Each source-rank stripe is contiguous in GM.  Start all independent
    // segment transfers before clearing the count tables, then join them only
    // when scalar decode consumes the cache.  K8/K16 issue one MTE2 per peer
    // instead of one bypass-GM scalar load per route.
    GlobalTensor<int32_t> descriptorGlobal;
    descriptorGlobal.SetGlobalBuffer(tokenChainDescriptorsPtr_);
    constexpr uint32_t descriptorAlignEntries = ALIGN_32 / sizeof(int32_t);
    LocalTensor<int32_t> descriptorCache = descriptorStorage;
    uint32_t descriptorsPerSource = sourceRowCount * topK_;
    bool useSegmentDescriptorDma = descriptorsPerSource > 0U &&
                                   (descriptorsPerSource & (descriptorAlignEntries - 1U)) == 0U &&
                                   ((sourceRowBegin * topK_) & (descriptorAlignEntries - 1U)) == 0U &&
                                   ((m_ * topK_) & (descriptorAlignEntries - 1U)) == 0U;
    for (uint32_t sourceIteration = 0U; sourceIteration < worldSize_; ++sourceIteration) {
        uint32_t sourceRank = firstSource + sourceIteration;
        if (sourceRank >= worldSize_) {
            sourceRank -= worldSize_;
        }
        uint32_t sourceTokenBegin = sourceRank * m_ + sourceRowBegin;
        uint32_t descriptorGlobalBase = sourceTokenBegin * topK_;
        uint32_t descriptorLocalBase = sourceIteration * descriptorsPerSource;
        if (useSegmentDescriptorDma) {
            DataCopyPad(
                descriptorCache[descriptorLocalBase], descriptorGlobal[descriptorGlobalBase],
                DataCopyExtParams{1U, descriptorsPerSource * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U},
                DataCopyPadExtParams<int32_t>{false, 0U, 0U, 0U});
        } else {
            // Non-cacheline top-k widths retain a scalar correctness path.
            // K8/K16 have aligned source slices and issue one DMA per peer.
            for (uint32_t descriptorIdx = 0U; descriptorIdx < descriptorsPerSource; ++descriptorIdx) {
                descriptorCache.SetValue(
                    descriptorLocalBase + descriptorIdx,
                    ReadGmByPassDCache(tokenChainDescriptorsPtr_ + descriptorGlobalBase + descriptorIdx));
            }
        }
    }
    Duplicate(countTable, 0, readyCounterEntries);
    // priorityTable is a private linked-bucket head table.  The first material
    // priority is consumed while the buckets are built, so GMM1 no longer
    // waits behind a prefix sum and a second complete token scan.
    Duplicate(priorityTable, -1, priorityEntries);
    SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    if (useSegmentDescriptorDma) {
        SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    }

    GlobalTensor<QuantOutType> allLocalA;
    GlobalTensor<QuantScaleOutType> allLocalAScale;
    GlobalTensor<QuantOutType> gmRemoteA;
    allLocalA.SetGlobalBuffer(reinterpret_cast<__gm__ QuantOutType *>(params_.workspaceInfo.ptrA));
    allLocalAScale.SetGlobalBuffer(reinterpret_cast<__gm__ QuantScaleOutType *>(params_.workspaceInfo.ptrAScale));
    uint64_t remoteA0Offset = params_.peermemInfo.ptrA0 - params_.peermemInfo.ptrBase;

    uint32_t scheduleBits = GetTokenOwnerFieldBits(expertPerRank_);
    uint32_t tileBits = GetTokenOwnerFieldBits(maxDispatchTilesPerExpert_);
    uint32_t payloadBits = scheduleBits + tileBits;
    uint32_t scheduleMask = scheduleBits == 0U ? 0U : (UINT32_MAX >> (32U - scheduleBits));
    uint32_t tileMask = tileBits == 0U ? 0U : (UINT32_MAX >> (32U - tileBits));
    bool packCombineRoute = ShouldPackCombineRouteMetadata(worldSize_, m_, topK_, maxOutputSize_, blockNum_);
    uint32_t combineRouteTokenRowBits = GetTokenOwnerFieldBits(m_);
    uint32_t combineRouteTopKBits = GetTokenOwnerFieldBits(topK_);
    int32_t copySlotIdx = 0;
    int32_t usedCopySlots = 0;
    int32_t touchedCount = 0;
    int32_t pendingRoutes = 0;
    int32_t cachedSourceRank = -1;

    // Prefix-first traversal: while classifying the owned tokens, consume the
    // first material priority immediately and link only the remaining tokens
    // into private UB buckets.  This preserves one source payload read per
    // token and the exact ready boundary, while removing the old count/prefix
    // and token-order passes from the AIC startup dependency.
    bool classifyTokens = true;
    uint32_t classifyTokenOffset = 0U;
    for (uint32_t prioritySchedule = 0U; prioritySchedule < expertPerRank_; ++prioritySchedule) {
        uint32_t priorityExpert = GetScheduledExpert(prioritySchedule);
        int32_t priorityLocalStart = 0;
        int32_t priorityRows = 0;
        if (!GetExpertRange(priorityExpert, rankBaseRow, priorityLocalStart, priorityRows)) {
            continue;
        }
        (void)priorityLocalStart;
        uint32_t activePriorityTiles =
            Ops::Base::CeilDiv(static_cast<uint32_t>(priorityRows), EXPERT_SCHEDULE_M_TILE_ROWS);
        for (uint32_t priorityTile = 0U; priorityTile < activePriorityTiles; ++priorityTile) {
            uint32_t priority = prioritySchedule * priorityStride + priorityTile;
            bool classifyCurrentPriority = classifyTokens;
            int32_t linkedTokenOffset = classifyCurrentPriority ? -1 : priorityTable.GetValue(priority);
            while (true) {
                int32_t localTokenOffset = -1;
                if (classifyCurrentPriority) {
                    // A worker may own no token for the globally first
                    // material group.  It still classifies its full stripe so
                    // later priorities remain complete and all workers cross
                    // the same ready boundary.
                    while (classifyTokenOffset < tokenCount) {
                        uint32_t candidateToken = classifyTokenOffset++;
                        uint32_t candidateDescriptorBase = candidateToken * topK_;
                        uint32_t minimumPriority = priorityEntries;
                        for (uint32_t slot = 0U; slot < topK_; ++slot) {
                            int32_t signedDescriptor = descriptorCache.GetValue(candidateDescriptorBase + slot);
                            if (signedDescriptor < 0) {
                                continue;
                            }
                            uint32_t descriptor = static_cast<uint32_t>(signedDescriptor);
                            uint32_t scheduleIdx = descriptor & scheduleMask;
                            uint32_t tileIdx = (descriptor >> scheduleBits) & tileMask;
                            uint32_t candidatePriority = scheduleIdx * priorityStride + tileIdx;
                            minimumPriority = candidatePriority < minimumPriority ? candidatePriority : minimumPriority;
                        }
                        if (minimumPriority >= priorityEntries) {
                            continue;
                        }
                        if (minimumPriority <= priority) {
                            localTokenOffset = static_cast<int32_t>(candidateToken);
                            break;
                        }
                        int32_t bucketHead = priorityTable.GetValue(minimumPriority);
                        nextToken.SetValue(candidateToken, bucketHead);
                        priorityTable.SetValue(minimumPriority, static_cast<int32_t>(candidateToken));
                    }
                } else {
                    if (linkedTokenOffset < 0) {
                        break;
                    }
                    localTokenOffset = linkedTokenOffset;
                    linkedTokenOffset = nextToken.GetValue(static_cast<uint32_t>(linkedTokenOffset));
                }
                if (localTokenOffset < 0) {
                    break;
                }
                uint32_t localToken = static_cast<uint32_t>(localTokenOffset);
                uint32_t sourceIteration = localToken / sourceRowCount;
                uint32_t sourceOffset = localToken - sourceIteration * sourceRowCount;
                uint32_t sourceRankU = firstSource + sourceIteration;
                if (sourceRankU >= worldSize_) {
                    sourceRankU -= worldSize_;
                }
                int32_t sourceRank = static_cast<int32_t>(sourceRankU);
                int32_t sourceTokenRow = static_cast<int32_t>(sourceRowBegin + sourceOffset);
                uint32_t tokenInAllRank = sourceRankU * m_ + static_cast<uint32_t>(sourceTokenRow);
                uint32_t descriptorCacheBase = localToken * topK_;
                // Insertion-sort the cached local routes by schedule
                // position/canonical row. No descriptor is reread from GM.
                int32_t localCount = 0;
                for (uint32_t slot = 0; slot < topK_; ++slot) {
                    int32_t descriptor = descriptorCache.GetValue(descriptorCacheBase + slot);
                    if (descriptor < 0) {
                        continue;
                    }
                    uint32_t scheduleIdx = static_cast<uint32_t>(descriptor) & scheduleMask;
                    uint32_t canonicalRow = static_cast<uint32_t>(descriptor) >> payloadBits;
                    int32_t insert = localCount;
                    while (insert > 0) {
                        int32_t previous = routeScratch.GetValue(insert - 1);
                        uint32_t previousSchedule = static_cast<uint32_t>(previous) & scheduleMask;
                        uint32_t previousCanonical = static_cast<uint32_t>(previous) >> payloadBits;
                        if (previousSchedule < scheduleIdx ||
                            (previousSchedule == scheduleIdx && previousCanonical <= canonicalRow)) {
                            break;
                        }
                        routeScratch.SetValue(insert, previous);
                        routeSlotScratch.SetValue(insert, routeSlotScratch.GetValue(insert - 1));
                        --insert;
                    }
                    routeScratch.SetValue(insert, descriptor);
                    routeSlotScratch.SetValue(insert, static_cast<int32_t>(slot));
                    ++localCount;
                }
                if (localCount == 0) {
                    continue;
                }
                if constexpr (UseOwnerChainCombine) {
                    // Reuse the existing [token, top-k] workspace, but turn
                    // this token's original-slot fanout descriptors into one
                    // compact chain in the exact stepK order already used by
                    // expert-major Combine.  The original slot is retained in
                    // the payload so the owner can load the same probability.
                    uint32_t ownerSlotBits = GetTokenOwnerFieldBits(topK_);
                    uint32_t ownerPayloadBits = scheduleBits + ownerSlotBits;
                    for (int32_t stepK = 0; stepK < localCount; ++stepK) {
                        uint32_t descriptor = static_cast<uint32_t>(routeScratch.GetValue(stepK));
                        uint32_t scheduleIdx = descriptor & scheduleMask;
                        uint32_t canonicalRow = descriptor >> payloadBits;
                        uint32_t topKSlot = static_cast<uint32_t>(routeSlotScratch.GetValue(stepK));
                        uint32_t ownerDescriptor =
                            (canonicalRow << ownerPayloadBits) | (scheduleIdx << ownerSlotBits) | topKSlot;
                        descriptorCache.SetValue(descriptorCacheBase + static_cast<uint32_t>(stepK),
                                                 static_cast<int32_t>(ownerDescriptor));
                    }
                    for (uint32_t stepK = static_cast<uint32_t>(localCount); stepK < topK_; ++stepK) {
                        descriptorCache.SetValue(descriptorCacheBase + stepK, -1);
                    }
                }
                if (pendingRoutes > 0 &&
                    pendingRoutes + localCount > static_cast<int32_t>(DISPATCH_FANOUT_READY_BATCH_ROUTES)) {
                    // Close the previous micro-batch before adding this
                    // complete token. Keeping a token indivisible preserves
                    // one-source-row fanout; the UB gate reserves
                    // max(topK, batch-routes) touched slots, so an unusually
                    // wide single token remains in bounds.
                    FlushDispatchFanoutReady(countTable, touchedBuckets, priorityStride, touchedCount, pendingRoutes,
                                             copySlotIdx, usedCopySlots);
                }

                if (sourceRank != cachedSourceRank) {
                    gmRemoteA.SetGlobalBuffer(
                        reinterpret_cast<__gm__ QuantOutType *>(GetRankWinAddrWithOffset(sourceRank, remoteA0Offset)));
                    cachedSourceRank = sourceRank;
                }

                TEventID eventId = static_cast<TEventID>(copySlotIdx);
                if (usedCopySlots == static_cast<int32_t>(DISPATCH_FANOUT_BUFFER_NUM)) {
                    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
                } else {
                    ++usedCopySlots;
                }
                LocalTensor<QuantOutType> dataBuf = byteBuffer[static_cast<uint32_t>(copySlotIdx) * slotElements];
                LocalTensor<QuantScaleOutType> scaleBuf =
                    dataBuf[a0ScaleOffset].template ReinterpretCast<QuantScaleOutType>();
                DataCopy(dataBuf, gmRemoteA[static_cast<int64_t>(sourceTokenRow) * a0RowStride], a0RowStride);

                // Metadata and ready-bucket accounting are independent of
                // MTE2. The descriptor sort makes stepK identical to the
                // canonical hot/cold expert traversal even though payload is
                // token-major.
                for (int32_t stepK = 0; stepK < localCount; ++stepK) {
                    uint32_t descriptor = static_cast<uint32_t>(routeScratch.GetValue(stepK));
                    uint32_t scheduleIdx = descriptor & scheduleMask;
                    uint32_t tileIdx = (descriptor >> scheduleBits) & tileMask;
                    uint32_t canonicalRow = descriptor >> payloadBits;
                    uint32_t bucket = scheduleIdx * priorityStride + tileIdx;
                    int32_t bucketCount = countTable.GetValue(bucket);
                    if (bucketCount == 0) {
                        touchedBuckets.SetValue(touchedCount++, static_cast<int32_t>(bucket));
                    }
                    countTable.SetValue(bucket, bucketCount + 1);
                    ++pendingRoutes;
                    uint32_t topKSlot = static_cast<uint32_t>(routeSlotScratch.GetValue(stepK));
                    int32_t stepMetadata =
                        packCombineRoute
                            ? PackCombineRouteMetadata(
                                  static_cast<uint32_t>(sourceRank), static_cast<uint32_t>(sourceTokenRow), topKSlot,
                                  static_cast<uint32_t>(stepK), combineRouteTokenRowBits, combineRouteTopKBits)
                            : stepK;
                    WriteGmByPassDCache(stepIdxPtr_ + canonicalRow, stepMetadata);
                    if (stepK == 0) {
                        WriteGmByPassDCache(firstTokenPosPtr_ + tokenInAllRank, static_cast<int32_t>(canonicalRow));
                    }
                }
                WriteGmByPassDCache(posLenPtr_ + tokenInAllRank, localCount);

                SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
                WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
                for (int32_t stepK = 0; stepK < localCount; ++stepK) {
                    uint32_t descriptor = static_cast<uint32_t>(routeScratch.GetValue(stepK));
                    uint32_t canonicalRow = descriptor >> payloadBits;
                    DataCopyPad(
                        allLocalA[static_cast<uint64_t>(canonicalRow) * widthA], dataBuf,
                        {static_cast<uint16_t>(1), static_cast<uint16_t>(widthA * sizeof(QuantOutType)), 0, 0, 0});
                    if constexpr (!UseBf16NoQuant) {
                        DataCopyPad(allLocalAScale[static_cast<uint64_t>(canonicalRow) * widthAScale], scaleBuf,
                                    {static_cast<uint16_t>(1), static_cast<uint16_t>(scaleLen), 0, 0, 0});
                    }
                }
                SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
                ++copySlotIdx;
                if (copySlotIdx == static_cast<int32_t>(DISPATCH_FANOUT_BUFFER_NUM)) {
                    copySlotIdx = 0;
                }

                if (pendingRoutes >= static_cast<int32_t>(DISPATCH_FANOUT_READY_BATCH_ROUTES)) {
                    FlushDispatchFanoutReady(countTable, touchedBuckets, priorityStride, touchedCount, pendingRoutes,
                                             copySlotIdx, usedCopySlots);
                }
            }
            if (classifyCurrentPriority) {
                classifyTokens = false;
            }
            // A short tail batch cannot cross a priority boundary: after
            // this flush every contribution needed by GMM1(schedule,tile)
            // has reached its exact row-count gate.
            FlushDispatchFanoutReady(countTable, touchedBuckets, priorityStride, touchedCount, pendingRoutes,
                                     copySlotIdx, usedCopySlots);
        }
    }
    FlushDispatchFanoutReady(countTable, touchedBuckets, priorityStride, touchedCount, pendingRoutes, copySlotIdx,
                             usedCopySlots);
    if constexpr (UseOwnerChainCombine) {
        // Publish one compact descriptor segment per peer.  K8/K16 use one
        // aligned MTE3 transaction per source; unusual top-k widths retain a
        // scalar correctness path.  The segments are disjoint across AIVs,
        // and the source rotation changes only their local issue order.
        PipeBarrier<PIPE_ALL>();
        if (useSegmentDescriptorDma) {
            // descriptorCache was filled through scalar SetValue.  Match the
            // established routing-kernel S->MTE3 dependency before the bulk
            // UB-to-GM publication.
            SetWaitFlag<HardEvent::S_MTE3>();
        }
        for (uint32_t sourceIteration = 0U; sourceIteration < worldSize_; ++sourceIteration) {
            uint32_t sourceRank = firstSource + sourceIteration;
            if (sourceRank >= worldSize_) {
                sourceRank -= worldSize_;
            }
            uint32_t sourceTokenBegin = sourceRank * m_ + sourceRowBegin;
            uint32_t descriptorGlobalBase = sourceTokenBegin * topK_;
            uint32_t descriptorLocalBase = sourceIteration * descriptorsPerSource;
            if (useSegmentDescriptorDma) {
                DataCopyPad(
                    descriptorGlobal[descriptorGlobalBase], descriptorCache[descriptorLocalBase],
                    DataCopyExtParams{1U, descriptorsPerSource * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U});
            } else {
                for (uint32_t descriptorIdx = 0U; descriptorIdx < descriptorsPerSource; ++descriptorIdx) {
                    WriteGmByPassDCache(tokenChainDescriptorsPtr_ + descriptorGlobalBase + descriptorIdx,
                                        descriptorCache.GetValue(descriptorLocalBase + descriptorIdx));
                }
            }
        }
        if (useSegmentDescriptorDma) {
            SyncFunc<AscendC::HardEvent::MTE3_S>();
        }
    }
    TIMER_BLOCK(timer.template Tok<Overwrite>(AscendTimer::AIV_DISPATCH, 0U));
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline uint32_t MegaMoe<TemplateMegaMoeTypeFunc>::GetMGroupsPerWave()
{
    uint32_t gmm1TilesPerMGroup =
        Ops::Base::CeilDiv(static_cast<uint32_t>(Get<N_VALUE>(problemShape_)), EXPERT_SCHEDULE_M_TILE_ROWS);
    uint32_t gmm2TilesPerMGroup = Ops::Base::CeilDiv(k_, EXPERT_SCHEDULE_M_TILE_ROWS);
    uint32_t gmm1Groups = Ops::Base::CeilDiv(blockNum_ * 4U, gmm1TilesPerMGroup);
    uint32_t gmm2Groups = Ops::Base::CeilDiv(blockNum_, gmm2TilesPerMGroup);
    return gmm1Groups > gmm2Groups ? gmm1Groups : gmm2Groups;
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::BuildExpertOrder()
{
    if constexpr (g_coreType == AscendC::AIC) {
        return;
    }
    // AIV0 destroys firstPipe immediately after AllGather; AIV1 keeps
    // CopyTempBuf_ alive for Dispatch.  Build the UB plan on the first AIV1
    // logical core before that buffer is reused by payload copies.
    if (subBlockIdx_ != 1U || blockIdx_ != 0U) {
        return;
    }

    __gm__ int32_t *orderAddr = const_cast<__gm__ int32_t *>(expertOrder_.GetPhyAddr());
    __gm__ int32_t *selectedAddr = const_cast<__gm__ int32_t *>(expertOrderScratch_.GetPhyAddr());
    int32_t rankExpertBase = static_cast<int32_t>(rankId_ * expertPerRank_);
    uint32_t activeCount = 0U;
    uint32_t minTiles = UINT32_MAX;
    uint32_t maxTiles = 0U;
    // Keep the histogram and the temporary heap in UB.  The previous skew
    // path repeatedly rescanned the GM scratch table for every selected
    // expert (O(E^2) bypass-GM reads).  Five int32 arrays fit comfortably in
    // CopyTempBuf_ even at the supported E=768 ceiling.
    LocalTensor<int32_t> planUb = CopyTempBuf_.Get<int32_t>();
    uint32_t heapBase = expertPerRank_;
    uint32_t outputBase = 2U * expertPerRank_;
    uint32_t cumsumBase = 3U * expertPerRank_;
    // DataCopyPad's local (UB) endpoint must start on a 32-byte data block.
    // cumsum occupies E+1 int32 values, so placing staging immediately after
    // it leaves a 4-byte-aligned address for the common E=32/64 shapes and can
    // raise an AICore MTE exception during the first warmup launch.  Round the
    // scratch cursor up without changing either source layout or plan math.
    constexpr uint32_t int32PerDataBlock = 32U / sizeof(int32_t);
    uint32_t stagingBaseUnaligned = 4U * expertPerRank_ + 1U;
    uint32_t stagingBase = (stagingBaseUnaligned + int32PerDataBlock - 1U) / int32PerDataBlock * int32PerDataBlock;
    // Source the plan from the raw per-rank cumsum slices already sitting in
    // the local AllGather window: each peer pushed exactly this rank's
    // expert interval, and CrossRankSyncInWorldSize made every slice
    // stable.  One batched UB copy per rank replaces the former per-expert
    // serial bypass-GM reads of the reduced table, and drops the dependency
    // on the mixed-core barriers that publish that reduction - which is what
    // lets the plan build inside the AllGather phase at all.
    SyncFunc<AscendC::HardEvent::MTE3_S>();
    uint32_t expertNumAlign = static_cast<uint32_t>(
        Ops::Base::CeilAlign(static_cast<int64_t>(worldSize_ * expertPerRank_), static_cast<int64_t>(ALIGN_128)));
    uint32_t sliceElems = rankId_ == 0U ? expertPerRank_ : expertPerRank_ + 1U;
    int32_t sliceOffset = rankId_ == 0U ? 0 : rankExpertBase - 1;
    for (uint32_t idx = 0U; idx < sliceElems; ++idx) {
        planUb.SetValue(cumsumBase + idx, 0);
    }
    DataCopyExtParams sliceCopyParams{1U, static_cast<uint32_t>(sliceElems * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> slicePadParams{false, 0U, 0U, 0U};
    for (uint32_t srcRank = 0U; srcRank < worldSize_; ++srcRank) {
        SyncFunc<AscendC::HardEvent::S_MTE2>();
        DataCopyPad(planUb[stagingBase], tokenPerExpertWin_[srcRank * expertNumAlign + sliceOffset], sliceCopyParams,
                    slicePadParams);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        for (uint32_t idx = 0U; idx < sliceElems; ++idx) {
            planUb.SetValue(cumsumBase + idx, planUb.GetValue(cumsumBase + idx) + planUb.GetValue(stagingBase + idx));
        }
    }
    int32_t rankBase = rankId_ == 0U ? 0 : planUb.GetValue(cumsumBase);
    uint32_t entryBase = rankId_ == 0U ? cumsumBase : cumsumBase + 1U;
    int32_t previous = rankBase;
    uint32_t totalMGroups = 0U;
    uint32_t totalFullMGroups = 0U;
    uint32_t totalTailRows = 0U;
    uint32_t maxActiveTiles = 1U;

    for (uint32_t expertIdx = 0; expertIdx < expertPerRank_; ++expertIdx) {
        int32_t current = planUb.GetValue(entryBase + expertIdx);
        int32_t rows = current - previous;
        previous = current;
        // Publish one physical-expert endpoint table together with the order.
        // Downstream AIC/AIV roles can reconstruct [start,end) without each
        // rescanning the all-rank cumsum table.
        WriteGmByPassDCache(selectedAddr + 1U + expertIdx, current - rankBase);
        // Identity is both the default schedule and the complete inverse table
        // for the no-skew path.  Every token-local Dispatch specialization can
        // use the common SIMT plan producer; hot-first planning below
        // overwrites these slots with physical-expert -> schedule-position.
        WriteGmByPassDCache(selectedAddr + 1U + expertPerRank_ + expertIdx, static_cast<int32_t>(expertIdx));
        planUb.SetValue(expertIdx, rows);
        if (rows <= 0) {
            continue;
        }
        uint32_t tiles = Ops::Base::CeilDiv(static_cast<uint32_t>(rows), EXPERT_SCHEDULE_M_TILE_ROWS);
        activeCount++;
        minTiles = tiles < minTiles ? tiles : minTiles;
        maxTiles = tiles > maxTiles ? tiles : maxTiles;
        // Header aggregates replicate GetExpertRange's clamp exactly, so a
        // consumer reading them is arithmetically identical to its former
        // per-expert rescan of the endpoint table.
        int32_t localStart = current - rankBase - rows;
        if (localStart >= 0 && static_cast<uint64_t>(localStart) < maxOutputSize_) {
            uint64_t remainingRows = maxOutputSize_ - static_cast<uint64_t>(localStart);
            uint32_t clampedRows = static_cast<uint64_t>(rows) > remainingRows ? static_cast<uint32_t>(remainingRows)
                                                                               : static_cast<uint32_t>(rows);
            if (clampedRows > 0U) {
                uint32_t activeTiles = Ops::Base::CeilDiv(clampedRows, EXPERT_SCHEDULE_M_TILE_ROWS);
                totalMGroups += activeTiles;
                totalFullMGroups += clampedRows / EXPERT_SCHEDULE_M_TILE_ROWS;
                totalTailRows += clampedRows % EXPERT_SCHEDULE_M_TILE_ROWS;
                maxActiveTiles = activeTiles > maxActiveTiles ? activeTiles : maxActiveTiles;
            }
        }
    }
    // Publish the aggregate header on both schedule paths; the endpoint
    // table above and these four slots travel to every consumer through the
    // same PrepareExpertPipeline barriers.
    uint32_t headerBase = 1U + 2U * expertPerRank_;
    WriteGmByPassDCache(selectedAddr + headerBase + EXPERT_PLAN_HEADER_TOTAL_MGROUPS,
                        static_cast<int32_t>(totalMGroups));
    WriteGmByPassDCache(selectedAddr + headerBase + EXPERT_PLAN_HEADER_TOTAL_FULL_MGROUPS,
                        static_cast<int32_t>(totalFullMGroups));
    WriteGmByPassDCache(selectedAddr + headerBase + EXPERT_PLAN_HEADER_TOTAL_TAIL_ROWS,
                        static_cast<int32_t>(totalTailRows));
    WriteGmByPassDCache(selectedAddr + headerBase + EXPERT_PLAN_HEADER_MAX_ACTIVE_TILES,
                        static_cast<int32_t>(maxActiveTiles));

    bool useHotFirstSchedule =
        activeCount >= EXPERT_SCHEDULE_MIN_ACTIVE_EXPERTS && maxTiles - minTiles >= EXPERT_SCHEDULE_MIN_TILE_SPREAD;
    if (!useHotFirstSchedule) {
        // Identity traversal uses the same published physical-expert endpoint
        // table as every other distribution.  There is no balanced-only mode
        // or host-selected physical-core phase.
        WriteGmByPassDCache(selectedAddr, 0);
        PipeBarrier<PIPE_ALL>();
        return;
    }

    // Materialize active physical ids in UB, then heap-sort by the exact old
    // comparator.  Since ceil(rows/256) is monotonic in rows, sorting by
    // (rows, physical id) is identical to sorting by (M tiles, rows, id).
    uint32_t heapSize = 0U;
    for (uint32_t expertIdx = 0; expertIdx < expertPerRank_; ++expertIdx) {
        if (planUb.GetValue(expertIdx) > 0) {
            planUb.SetValue(heapBase + heapSize, static_cast<int32_t>(expertIdx));
            heapSize++;
        }
    }

    // Build a max heap.
    for (int32_t start = static_cast<int32_t>(heapSize / 2U) - 1; start >= 0; --start) {
        uint32_t root = static_cast<uint32_t>(start);
        while (2U * root + 1U < heapSize) {
            uint32_t child = 2U * root + 1U;
            int32_t childExpert = planUb.GetValue(heapBase + child);
            if (child + 1U < heapSize) {
                int32_t rightExpert = planUb.GetValue(heapBase + child + 1U);
                int32_t childRows = planUb.GetValue(static_cast<uint32_t>(childExpert));
                int32_t rightRows = planUb.GetValue(static_cast<uint32_t>(rightExpert));
                if (rightRows > childRows || (rightRows == childRows && rightExpert > childExpert)) {
                    child++;
                    childExpert = rightExpert;
                }
            }
            int32_t rootExpert = planUb.GetValue(heapBase + root);
            int32_t rootRows = planUb.GetValue(static_cast<uint32_t>(rootExpert));
            int32_t childRows = planUb.GetValue(static_cast<uint32_t>(childExpert));
            if (rootRows > childRows || (rootRows == childRows && rootExpert >= childExpert)) {
                break;
            }
            planUb.SetValue(heapBase + root, childExpert);
            planUb.SetValue(heapBase + child, rootExpert);
            root = child;
        }
    }

    // Pop the max heap in place; the final array is cold-to-hot.
    for (uint32_t end = heapSize; end > 1U; --end) {
        int32_t rootExpert = planUb.GetValue(heapBase);
        int32_t tailExpert = planUb.GetValue(heapBase + end - 1U);
        planUb.SetValue(heapBase, tailExpert);
        planUb.SetValue(heapBase + end - 1U, rootExpert);
        uint32_t reducedSize = end - 1U;
        uint32_t root = 0U;
        while (2U * root + 1U < reducedSize) {
            uint32_t child = 2U * root + 1U;
            int32_t childExpert = planUb.GetValue(heapBase + child);
            if (child + 1U < reducedSize) {
                int32_t rightExpert = planUb.GetValue(heapBase + child + 1U);
                int32_t childRows = planUb.GetValue(static_cast<uint32_t>(childExpert));
                int32_t rightRows = planUb.GetValue(static_cast<uint32_t>(rightExpert));
                if (rightRows > childRows || (rightRows == childRows && rightExpert > childExpert)) {
                    child++;
                    childExpert = rightExpert;
                }
            }
            rootExpert = planUb.GetValue(heapBase + root);
            int32_t rootRows = planUb.GetValue(static_cast<uint32_t>(rootExpert));
            int32_t childRows = planUb.GetValue(static_cast<uint32_t>(childExpert));
            if (rootRows > childRows || (rootRows == childRows && rootExpert >= childExpert)) {
                break;
            }
            planUb.SetValue(heapBase + root, childExpert);
            planUb.SetValue(heapBase + child, rootExpert);
            root = child;
        }
    }

    // Complete-stage GMM1 has no expert-wave boundary to balance around.
    // Start the long experts first so their GMM2/Combine tail is exposed as
    // early as possible, then drain progressively shorter work.  The heap
    // result is cold-to-hot, hence the reverse traversal.
    for (uint32_t hotRank = 0; hotRank < activeCount; ++hotRank) {
        planUb.SetValue(outputBase + hotRank, planUb.GetValue(heapBase + activeCount - 1U - hotRank));
    }

    uint32_t emptyPos = activeCount;
    for (uint32_t expertIdx = 0; expertIdx < expertPerRank_; ++expertIdx) {
        if (planUb.GetValue(expertIdx) <= 0) {
            planUb.SetValue(outputBase + emptyPos, static_cast<int32_t>(expertIdx));
            emptyPos++;
        }
    }
    for (uint32_t scheduleIdx = 0; scheduleIdx < expertPerRank_; ++scheduleIdx) {
        int32_t physicalExpert = planUb.GetValue(outputBase + scheduleIdx);
        WriteGmByPassDCache(orderAddr + scheduleIdx, physicalExpert);
        WriteGmByPassDCache(selectedAddr + 1U + expertPerRank_ + static_cast<uint32_t>(physicalExpert),
                            static_cast<int32_t>(scheduleIdx));
    }
    WriteGmByPassDCache(selectedAddr, 1);
    PipeBarrier<PIPE_ALL>();
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::LoadExpertScheduleMode()
{
    __gm__ int32_t *selectedAddr = const_cast<__gm__ int32_t *>(expertOrderScratch_.GetPhyAddr());
    int32_t scheduleMode = ReadGmByPassDCache(selectedAddr);
    expertScheduleEnabled_ = scheduleMode != 0;
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline uint32_t MegaMoe<TemplateMegaMoeTypeFunc>::GetScheduledExpert(uint32_t scheduleIdx)
{
    if (!expertScheduleEnabled_) {
        return scheduleIdx;
    }
    __gm__ int32_t *orderAddr = const_cast<__gm__ int32_t *>(expertOrder_.GetPhyAddr());
    return static_cast<uint32_t>(ReadGmByPassDCache(orderAddr + scheduleIdx));
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline bool MegaMoe<TemplateMegaMoeTypeFunc>::GetExpertRange(uint32_t expertIdx, int32_t rankBaseRow,
                                                                        int32_t &localStart, int32_t &rows)
{
    int32_t localEnd = 0;
    if constexpr (UseCoreWave) {
        if (expertPerRank_ >= EXPERT_SCHEDULE_MIN_ACTIVE_EXPERTS) {
            __gm__ int32_t *planAddr = const_cast<__gm__ int32_t *>(expertOrderScratch_.GetPhyAddr());
            localStart = expertIdx == 0U ? 0 : ReadGmByPassDCache(planAddr + expertIdx);
            localEnd = ReadGmByPassDCache(planAddr + 1U + expertIdx);
        } else {
            __gm__ int32_t *cumsumAddr = const_cast<__gm__ int32_t *>(expertTokenCumsum_.GetPhyAddr());
            int32_t globalExpertIdx = static_cast<int32_t>(rankId_ * expertPerRank_ + expertIdx);
            int32_t globalEnd = ReadGmByPassDCache(cumsumAddr + globalExpertIdx);
            int32_t globalStart = globalExpertIdx == 0 ? 0 : ReadGmByPassDCache(cumsumAddr + globalExpertIdx - 1);
            localStart = globalStart - rankBaseRow;
            localEnd = globalEnd - rankBaseRow;
        }
    } else {
        __gm__ int32_t *cumsumAddr = const_cast<__gm__ int32_t *>(expertTokenCumsum_.GetPhyAddr());
        int32_t globalExpertIdx = static_cast<int32_t>(rankId_ * expertPerRank_ + expertIdx);
        int32_t globalEnd = ReadGmByPassDCache(cumsumAddr + globalExpertIdx);
        int32_t globalStart = globalExpertIdx == 0 ? 0 : ReadGmByPassDCache(cumsumAddr + globalExpertIdx - 1);
        localStart = globalStart - rankBaseRow;
        localEnd = globalEnd - rankBaseRow;
    }

    rows = localEnd - localStart;
    if (localStart < 0 || rows <= 0 || static_cast<uint64_t>(localStart) >= maxOutputSize_) {
        rows = 0;
        return false;
    }
    uint64_t remainingRows = maxOutputSize_ - static_cast<uint64_t>(localStart);
    if (static_cast<uint64_t>(rows) > remainingRows) {
        rows = static_cast<int32_t>(remainingRows);
    }
    return rows > 0;
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline bool MegaMoe<TemplateMegaMoeTypeFunc>::UpdateIdentityGroupParams(uint32_t expertIdx)
{
    // Identity order retains the physical-order incremental fast path: no
    // per-expert 64-bit absolute offset products.  Core-wave E>=3 reads the
    // published endpoint plan; generic/small-E keeps the original rolling
    // cumsum state.
    if (expertIdx != 0U) {
        uint64_t m = Get<M_VALUE>(problemShape_);
        uint64_t n = Get<N_VALUE>(problemShape_);
        uint64_t k = Get<K_VALUE>(problemShape_);
        uint64_t scaleK = Ops::Base::CeilDiv(k, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        uint64_t nHalf = n / SWIGLU_N_HALF;
        uint64_t scaleNHalf =
            Ops::Base::CeilDiv(nHalf, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        Get<IDX_A_OFFSET>(baseOffset_) += m * k;
        Get<IDX_B_OFFSET>(baseOffset_) += n * k;
        Get<IDX_A_SCALE_OFFSET>(baseOffset_) += m * scaleK;
        Get<IDX_B_SCALE_OFFSET>(baseOffset_) += n * scaleK;
        Get<IDX_C_OFFSET>(baseOffset_) += m * nHalf;
        Get<IDX_C_SCALE_OFFSET>(baseOffset_) += m * scaleNHalf;
        Get<IDX_FLAG_OFFSET>(baseOffset_) += 1;
        Get<IDX_B2_OFFSET>(baseOffset_) += k * nHalf;
        Get<IDX_B2_SCALE_OFFSET>(baseOffset_) += k * scaleNHalf;
        Get<IDX_Y2_OFFSET>(baseOffset_) += m * k;
        Get<IDX_M_OFFSET>(baseOffset_) += m;
    }

    if constexpr (UseCoreWave) {
        if (expertPerRank_ >= EXPERT_SCHEDULE_MIN_ACTIVE_EXPERTS) {
            // Identity traversal already carries the exact previous endpoint
            // in IDX_M_OFFSET and reads only this expert's published end.
            uint64_t localStart = Get<IDX_M_OFFSET>(baseOffset_);
            if (localStart >= maxOutputSize_) {
                Get<M_VALUE>(problemShape_) = 0;
                return false;
            }
            __gm__ int32_t *planAddr = const_cast<__gm__ int32_t *>(expertOrderScratch_.GetPhyAddr());
            int32_t localEnd = ReadGmByPassDCache(planAddr + 1U + expertIdx);
            int64_t rows = static_cast<int64_t>(localEnd) - static_cast<int64_t>(localStart);
            if (rows <= 0) {
                Get<M_VALUE>(problemShape_) = 0;
                return false;
            }
            uint64_t remainingRows = maxOutputSize_ - localStart;
            if (static_cast<uint64_t>(rows) > remainingRows) {
                rows = static_cast<int64_t>(remainingRows);
            }
            Get<M_VALUE>(problemShape_) = static_cast<int32_t>(rows);
            return true;
        }
    }

    __gm__ int32_t *cumsumAddr = const_cast<__gm__ int32_t *>(expertTokenCumsum_.GetPhyAddr());
    int32_t globalExpertIdx = static_cast<int32_t>(rankId_ * expertPerRank_ + expertIdx);
    int32_t currentOffset = ReadGmByPassDCache(cumsumAddr + globalExpertIdx);
    int32_t rows = currentOffset - preOffset_;
    preOffset_ = currentOffset;
    Get<M_VALUE>(problemShape_) = rows;
    uint64_t localStart = Get<IDX_M_OFFSET>(baseOffset_);
    if (localStart >= maxOutputSize_) {
        Get<M_VALUE>(problemShape_) = 0;
        return false;
    }
    uint64_t remainingRows = maxOutputSize_ - localStart;
    if (rows > 0 && static_cast<uint64_t>(rows) > remainingRows) {
        Get<M_VALUE>(problemShape_) = static_cast<int32_t>(remainingRows);
    }
    return Get<M_VALUE>(problemShape_) > 0;
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline bool MegaMoe<TemplateMegaMoeTypeFunc>::SetGroupParams(uint32_t expertIdx, int32_t rankBaseRow)
{
    int32_t localStart = 0;
    int32_t rows = 0;
    if (!GetExpertRange(expertIdx, rankBaseRow, localStart, rows)) {
        Get<M_VALUE>(problemShape_) = 0;
        return false;
    }

    uint64_t n = static_cast<uint64_t>(Get<N_VALUE>(problemShape_));
    uint64_t k = static_cast<uint64_t>(Get<K_VALUE>(problemShape_));
    uint64_t localRow = static_cast<uint64_t>(localStart);
    uint64_t expert = static_cast<uint64_t>(expertIdx);
    uint64_t scaleK = Ops::Base::CeilDiv(k, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
    uint64_t nHalf = n / SWIGLU_N_HALF;
    uint64_t scaleNHalf = Ops::Base::CeilDiv(nHalf, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;

    Get<M_VALUE>(problemShape_) = rows;
    Get<IDX_A_OFFSET>(baseOffset_) = localRow * k;
    Get<IDX_B_OFFSET>(baseOffset_) = expert * n * k;
    Get<IDX_A_SCALE_OFFSET>(baseOffset_) = localRow * scaleK;
    Get<IDX_B_SCALE_OFFSET>(baseOffset_) = expert * n * scaleK;
    Get<IDX_RESERVED_OFFSET>(baseOffset_) = 0;
    Get<IDX_C_OFFSET>(baseOffset_) = localRow * nHalf;
    Get<IDX_C_SCALE_OFFSET>(baseOffset_) = localRow * scaleNHalf;
    Get<IDX_FLAG_OFFSET>(baseOffset_) = expert;
    Get<IDX_B2_OFFSET>(baseOffset_) = expert * k * nHalf;
    Get<IDX_B2_SCALE_OFFSET>(baseOffset_) = expert * k * scaleNHalf;
    Get<IDX_Y2_OFFSET>(baseOffset_) = localRow * k;
    Get<IDX_M_OFFSET>(baseOffset_) = localRow;
    return true;
}

template <TemplateMegaMoeTypeClass>
template <bool PublishGmm1Ready>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::UpdateGlobalBuffer(GMMAddrInfo &gmmAddrInfo, uint32_t gmmIndex)
{
    if (gmmIndex == 0) {
        gmmAddrInfo.aGlobal = params_.workspaceInfo.ptrA + Get<IDX_A_OFFSET>(baseOffset_) * sizeof(QuantOutType);
        gmmAddrInfo.aScaleGlobal =
            params_.workspaceInfo.ptrAScale + Get<IDX_A_SCALE_OFFSET>(baseOffset_) * sizeof(QuantScaleOutType);
        if constexpr (g_coreType == AIC) {
            gmmAddrInfo.bGlobal = params_.bGmAddr + Get<IDX_B_OFFSET>(baseOffset_) * sizeof(QuantOutType);
            if constexpr (UseBf16NoQuant) {
                gmmAddrInfo.bScaleGlobal = nullptr;
            } else {
                gmmAddrInfo.bScaleGlobal =
                    params_.bScaleGmAddr + Get<IDX_B_SCALE_OFFSET>(baseOffset_) * sizeof(QuantScaleOutType);
            }
        } else {
            AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> vecBaseOffset{
                Get<IDX_C_OFFSET>(baseOffset_),
                Get<IDX_C_SCALE_OFFSET>(baseOffset_),
                Get<IDX_FLAG_OFFSET>(baseOffset_),
                0L,
                0L,
                0L};
            if constexpr (PublishGmm1Ready) {
                streamingEpilogueOp_.UpdateGlobalAddr(vecBaseOffset);
            } else {
                epilogueOp_.UpdateGlobalAddr(vecBaseOffset);  // GMMSwiglu调用
            }
        }
    } else if constexpr (g_coreType == AIC) {
        gmmAddrInfo.aGlobal = params_.workspaceInfo.ptrA2 + Get<IDX_C_OFFSET>(baseOffset_) * sizeof(QuantOutType);
        gmmAddrInfo.aScaleGlobal =
            params_.workspaceInfo.ptrA2Scale + Get<IDX_C_SCALE_OFFSET>(baseOffset_) * sizeof(QuantScaleOutType);
        gmmAddrInfo.bGlobal = params_.b2GmAddr + Get<IDX_B2_OFFSET>(baseOffset_) * sizeof(QuantOutType);
        if constexpr (UseBf16NoQuant) {
            gmmAddrInfo.bScaleGlobal = nullptr;
        } else {
            gmmAddrInfo.bScaleGlobal =
                params_.b2ScaleGmAddr + Get<IDX_B2_SCALE_OFFSET>(baseOffset_) * sizeof(QuantScaleOutType);
        }
    }
    gmmAddrInfo.flagSwigluToGmm2 =
        (__gm__ int32_t *)params_.workspaceInfo.ptrFlagSwiGluToGmm2 + Get<IDX_FLAG_OFFSET>(baseOffset_) * INT_CACHELINE;
    gmmAddrInfo.dispatchDoneCores = (__gm__ int32_t *)params_.workspaceInfo.ptrDispatchDoneCores +
                                    Get<IDX_FLAG_OFFSET>(baseOffset_) * DISPATCH_DONE_STRIDE;
    gmmAddrInfo.dispatchTileReady = (__gm__ int32_t *)params_.workspaceInfo.ptrDispatchTileReady +
                                    Get<IDX_FLAG_OFFSET>(baseOffset_) *
                                        static_cast<uint64_t>(maxDispatchTilesPerExpert_) * DISPATCH_TILE_READY_STRIDE;
    // Default target; DispatchTokens replaces it with the exact number of
    // non-empty row owners under the runtime full-worker partition.
    gmmAddrInfo.dispatchReadyTarget = static_cast<int32_t>(blockNum_);
    gmmAddrInfo.flagGmm2Ready =
        (__gm__ int32_t *)params_.workspaceInfo.ptrGmm2ReadyFlag + Get<IDX_FLAG_OFFSET>(baseOffset_) * INT_CACHELINE;
    gmmAddrInfo.outputGlobal =
        params_.workspaceInfo.ptrCombineAccum + Get<IDX_A_OFFSET>(baseOffset_) * sizeof(bfloat16_t);
    flagSwigluToGmm2_.SetGlobalBuffer(gmmAddrInfo.flagSwigluToGmm2);
    dispatchDoneCores_.SetGlobalBuffer(gmmAddrInfo.dispatchDoneCores);
    gmm2ReadyFlag_.SetGlobalBuffer(gmmAddrInfo.flagGmm2Ready);
    combineAccumGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(gmmAddrInfo.outputGlobal));
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ResetTokenPerExpert()
{
    if (aivCoreIdx_ != blockAivNum_ - 1) {
        return;
    }
    int32_t num = worldSize_ * Ops::Base::CeilAlign(static_cast<uint32_t>(worldSize_ * expertPerRank_),
                                                    static_cast<uint32_t>(ALIGN_128));
    SyncFunc<AscendC::HardEvent::MTE3_V>();
    LocalTensor<int32_t> tmp{AscendC::TPosition::VECIN, 0, 0};
    Duplicate(tmp, 0, num);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(tokenPerExpertWin_, tmp, num);
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::UnpermuteCombineDedup()
{
    // 优化点：使用bf16累加
    if constexpr (g_coreType == AscendC::AIC) {
        return;
    }
    int32_t coreLen, coreOffset;
    TilingByCore(m_, coreLen, coreOffset, 1);
    GlobalTensor<int32_t> expertIdxTensor;
    expertIdxTensor.SetGlobalBuffer((__gm__ int32_t *)params_.expertIdxGmAddr);
    GlobalTensor<bfloat16_t> partialOutput;
    partialOutput.SetGlobalBuffer((__gm__ bfloat16_t *)params_.peermemInfo.ptrD);
    GlobalTensor<bfloat16_t> output;
    output.SetGlobalBuffer((__gm__ bfloat16_t *)params_.y2GmAddr);

    LocalTensor<bfloat16_t> dataResBf16 = DataResBuf_.Get<bfloat16_t>();
    LocalTensor<int32_t> rankFlag = RankFlagBuf_.Get<int32_t>();
    LocalTensor<bfloat16_t> dataIn0Bf16 = dataResBf16[k_ * 2];
    LocalTensor<bfloat16_t> dataIn1Bf16 = dataResBf16[k_ * 3];
    // The first batch has no preceding output write.  Later batches receive
    // this event after their two-row MTE3 write, so MTE2 can safely overwrite
    // the accumulator rows without racing the previous batch's reader.
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
    // 每批2个token各自累加到独立行, 最后两行合并一次写出, 同步次数减半
    constexpr int32_t UNPERMUTE_BATCH = 2;
    for (int32_t batchStart = coreOffset; batchStart < coreOffset + coreLen; batchStart += UNPERMUTE_BATCH) {
        int32_t batchEnd =
            batchStart + UNPERMUTE_BATCH < coreOffset + coreLen ? batchStart + UNPERMUTE_BATCH : coreOffset + coreLen;
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        for (int32_t tokenIdx = batchStart; tokenIdx < batchEnd; ++tokenIdx) {
            int32_t bufIdx = tokenIdx - batchStart;
            LocalTensor<bfloat16_t> resUb = dataResBf16[static_cast<int64_t>(bufIdx) * k_];

            // 常见worldSize<=32场景直接使用标量bitmask，避免每个token清零并回读rankFlag UB。
            // 更大worldSize继续使用原有UB标记路径。
            uint32_t rankMask = 0;
            if (worldSize_ <= 32) {
                uint32_t fullRankMask = worldSize_ >= 32
                                            ? 0xffffffffU
                                            : (worldSize_ == 0 ? 0U : (1U << static_cast<uint32_t>(worldSize_)) - 1U);
                for (int32_t topKIdx = 0; topKIdx < static_cast<int32_t>(topK_); ++topKIdx) {
                    int32_t expertIdx = expertIdxTensor(tokenIdx * topK_ + topKIdx);
                    int32_t rankIdx = DecodeExpertOwnerRank(expertIdx);
                    if (rankIdx >= 0 && rankIdx < static_cast<int32_t>(worldSize_)) {
                        rankMask |= 1U << static_cast<uint32_t>(rankIdx);
                        if (rankMask == fullRankMask) {
                            break;
                        }
                    }
                }
            } else {
                Duplicate(rankFlag, static_cast<int32_t>(-1), worldSize_);
                SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
                WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
                for (int32_t topKIdx = 0; topKIdx < static_cast<int32_t>(topK_); ++topKIdx) {
                    int32_t expertIdx = expertIdxTensor(tokenIdx * topK_ + topKIdx);
                    int32_t rankIdx = DecodeExpertOwnerRank(expertIdx);
                    if (rankIdx >= 0 && rankIdx < static_cast<int32_t>(worldSize_)) {
                        rankFlag.SetValue(rankIdx, rankIdx);
                    }
                }
            }

            // 找到第一个有效rank并启动首个GM->UB读取。
            int32_t currentRank = 0;
            while (currentRank < static_cast<int32_t>(worldSize_)) {
                bool isValid = worldSize_ <= 32 ? ((rankMask & (1U << static_cast<uint32_t>(currentRank))) != 0)
                                                : (rankFlag.GetValue(currentRank) != -1);
                if (isValid) {
                    break;
                }
                ++currentRank;
            }
            int32_t pingpongId = 0;
            if (currentRank < static_cast<int32_t>(worldSize_)) {
                // First valid rank directly initializes the accumulator.  This
                // removes Duplicate(0) and the first vector Add while preserving
                // the ascending-rank BF16 accumulation order.
                DataCopy(resUb, partialOutput[(static_cast<int64_t>(currentRank) * m_ + tokenIdx) * k_], k_);
                SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);

                int32_t nextRank = currentRank + 1;
                while (nextRank < static_cast<int32_t>(worldSize_)) {
                    bool isValid = worldSize_ <= 32 ? ((rankMask & (1U << static_cast<uint32_t>(nextRank))) != 0)
                                                    : (rankFlag.GetValue(nextRank) != -1);
                    if (isValid) {
                        break;
                    }
                    ++nextRank;
                }
                currentRank = nextRank;
                if (currentRank < static_cast<int32_t>(worldSize_)) {
                    WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
                    DataCopy(dataIn0Bf16, partialOutput[(static_cast<int64_t>(currentRank) * m_ + tokenIdx) * k_], k_);
                    SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                }
                // Queue the second owner before joining the first copy so MTE2
                // setup and the scalar owner scan are not exposed as a bubble.
                WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
                PipeBarrier<PIPE_V>();
            } else {
                // Defensive fallback for malformed/empty routing input.
                // Unlike the normal direct-MTE2 path, Duplicate writes from V,
                // so join the previous batch's MTE3 reader on the matching pipe.
                SyncFunc<AscendC::HardEvent::MTE3_V>();
                Duplicate(resUb, static_cast<bfloat16_t>(0.0), k_);
            }

            // 当前rank做Vector累加时，MTE2预取下一rank到另一块UB。
            while (currentRank < static_cast<int32_t>(worldSize_)) {
                auto currentEvent = pingpongId == 0 ? EVENT_ID0 : EVENT_ID1;
                auto currentData = pingpongId == 0 ? dataIn0Bf16 : dataIn1Bf16;
                WaitFlag<AscendC::HardEvent::MTE2_V>(currentEvent);

                int32_t nextRank = currentRank + 1;
                while (nextRank < static_cast<int32_t>(worldSize_)) {
                    bool isValid = worldSize_ <= 32 ? ((rankMask & (1U << static_cast<uint32_t>(nextRank))) != 0)
                                                    : (rankFlag.GetValue(nextRank) != -1);
                    if (isValid) {
                        break;
                    }
                    ++nextRank;
                }
                if (nextRank < static_cast<int32_t>(worldSize_)) {
                    int32_t nextPingpongId = 1 - pingpongId;
                    auto nextEvent = nextPingpongId == 0 ? EVENT_ID0 : EVENT_ID1;
                    auto nextData = nextPingpongId == 0 ? dataIn0Bf16 : dataIn1Bf16;
                    WaitFlag<AscendC::HardEvent::V_MTE2>(nextEvent);
                    DataCopy(nextData, partialOutput[(static_cast<int64_t>(nextRank) * m_ + tokenIdx) * k_], k_);
                    SetFlag<AscendC::HardEvent::MTE2_V>(nextEvent);
                }

                PipeBarrier<PIPE_V>();
                Add(resUb, resUb, currentData, k_);
                PipeBarrier<PIPE_V>();
                SetFlag<AscendC::HardEvent::V_MTE2>(currentEvent);
                currentRank = nextRank;
                pingpongId = 1 - pingpongId;
            }

            // 共享专家：不带 topk 权重，直接累加进该 token 的累加器（AIC 的 GMM2 结果）。
            // 等所有 AIC 核发布该专家的完成标志后再读，避免读到未写完的行。
            if (sharedExpertNum_ > 0U) {
                GlobalTensor<bfloat16_t> sharedOutGm;
                sharedOutGm.SetGlobalBuffer(
                    reinterpret_cast<__gm__ bfloat16_t *>(params_.workspaceInfo.ptrSharedOutput));
                for (uint32_t s = 0U; s < sharedExpertNum_; ++s) {
                    __gm__ int32_t *sharedReadyAddr =
                        reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.ptrSharedReadyFlag) +
                        (sharedExpertNum_ + s) * INT_CACHELINE;
                    uint32_t sharedPollMisses = 0U;
                    int64_t sharedPollDelay = 100;
                    while (AscendC::ReadGmByPassDCache(sharedReadyAddr) < static_cast<int32_t>(blockNum_)) {
                        GmSpinBackoffStep(sharedPollMisses, sharedPollDelay);
                    }
                    // dataIn0Bf16 是路由 pingpong 的复用缓冲，它的 V->MTE2 release credit
                    // 由 EVENT_ID0 承载（prologue 种子 Set、路由预取 Wait、路由末次 Add 后
                    // Set）。共享块必须接回同一条链：先 Wait 取回 credit 再复用，用完再
                    // Set 还给下一轮。原先借用 EVENT_ID2 自建一条 credit 链，等于伪造了
                    // “缓冲空闲”的凭证，会与路由最后一个 Add(resUb, resUb, dataIn0Bf16)
                    // 并发读写同一块 UB，产生 bf16 撕裂（表现为 max_abs 离群且逐迭代非确定）。
                    WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
                    SyncFunc<AscendC::HardEvent::MTE3_V>();
                    DataCopy(dataIn0Bf16,
                             sharedOutGm[static_cast<int64_t>(s) * static_cast<int64_t>(m_) * static_cast<int64_t>(k_) +
                                         static_cast<int64_t>(tokenIdx) * static_cast<int64_t>(k_)],
                             k_);
                    SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
                    WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
                    PipeBarrier<PIPE_V>();
                    Add(resUb, resUb, dataIn0Bf16, k_);
                    PipeBarrier<PIPE_V>();
                    SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
                }
            }
        }

        // 批内所有token累加完成后, 两行合并一次写出
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(output[static_cast<int64_t>(batchStart) * k_], dataResBf16,
                 static_cast<int64_t>(batchEnd - batchStart) * k_);
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
    WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline int32_t MegaMoe<TemplateMegaMoeTypeFunc>::PrepareExpertPipeline()
{
    // One mixed-core barrier is insufficient for all-rank cumsum visibility
    // on A5.  The optional third barrier publishes the runtime expert order.
    SyncAll<false>();
    SyncAll<false>();
    if constexpr (UseCoreWave) {
        // The real all-rank histogram is now visible.  Reuse the existing
        // optional schedule-publication phase to clear only tile-ready slots
        // addressed by this launch; no additional mixed-core barrier is
        // introduced.  The same histogram then selects identity vs hot-first
        // order, and the third barrier publishes both products.
        if (expertPerRank_ >= EXPERT_SCHEDULE_MIN_ACTIVE_EXPERTS) {
            // The expert plan itself was already built and published during
            // AllGather (BuildExpertOrder sources the raw window slices, not
            // the reduced cumsum), so this phase only clears the tile-ready
            // slots addressed by this launch and the third barrier publishes
            // those resets before either producer runs.
            ResetActiveDispatchTileReady();
            SyncAll<false>();
            LoadExpertScheduleMode();
        }
    }
    int32_t rankBaseRow = GetLocalExpertStartRow();
    WriteExpertTokenNumsOut();

    // AIC_WAIT_ALLGATHER: Tik 在 Process() 调本函数前, Tok 留此 (SyncAll 等 AIV AllGather 完成处)。
    if constexpr (g_coreType == AscendC::AIC) {
        TIMER_BLOCK(timer.template Tok<Overwrite>(AscendTimer::AIC_WAIT_ALLGATHER));
    }
    return rankBaseRow;
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ProcessInterleavedGmmPipeline(TPipe *firstPipe,
                                                                                       int32_t rankBaseRow)
{
    static_assert(UseCoreWave, "Interleaved GMM stages require the core-wave specialization");
    GMMAddrInfo gmmAddrInfo;

    if constexpr (g_coreType == AscendC::AIV) {
        if (subBlockIdx_ == 1U) {
            preOffset_ = rankBaseRow;
            baseOffset_ = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
            DispatchExperts(gmmAddrInfo, 0U, expertPerRank_, rankBaseRow);
            // DispatchTokens joins every active worker at every expert.  Once
            // the final traversal returns, logical block0 can publish the
            // immutable-posLen gate with one store instead of a counter storm.
            PipeBarrier<PIPE_ALL>();
            if (blockIdx_ == 0U) {
                WriteGmByPassDCache(chainLenReadyFlagPtr_, static_cast<int32_t>(blockNum_));
            }
            firstPipe->Destroy();
            // Dispatch CopyTempBuf_ and the AIC GMM2 Fixpipe path share this
            // physical AIV1 UB bank.  Publish ownership only after Destroy
            // drains the final Dispatch copy; AIC consumes the credit once
            // immediately before its first GMM2 context.
            NotifyCube(GMM2_UB_READY_FLAG);
        } else {
            using Gmm1Context = MegaMoeCatlassImpl::Gmm1Context<QuantOutType, QuantOutType, bfloat16_t,
                                                                QuantScaleOutType, QuantScaleOutType, false>;
            Gmm1Context gmm1Context(Get<N_VALUE>(problemShape_), Get<K_VALUE>(problemShape_));
            uint32_t gmm1StartBlockIdx = 0U;
            bool identityIncremental = !expertScheduleEnabled_;
            preOffset_ = rankBaseRow;
            baseOffset_ = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
            for (uint32_t scheduleIdx = 0U; scheduleIdx < expertPerRank_; ++scheduleIdx) {
                uint32_t expertIdx = GetScheduledExpert(scheduleIdx);
                bool hasGroup =
                    identityIncremental ? UpdateIdentityGroupParams(expertIdx) : SetGroupParams(expertIdx, rankBaseRow);
                if (!hasGroup) {
                    continue;
                }
                UpdateGlobalBuffer<true>(gmmAddrInfo, 0U);
                MegaMoeCatlassImpl::GroupMatmulSwigluQuant<QuantOutType, QuantOutType, bfloat16_t, QuantScaleOutType,
                                                           QuantScaleOutType>(
                    gmm1Context, streamingEpilogueOp_, problemShape_, gmmAddrInfo, gmm1StartBlockIdx,
                    gmm1VecSetSyncCom_, timer, expertIdx);
            }
        }

        // Both AIV sub-blocks may enter at different times.  The compile-time
        // gate wait below joins final Dispatch metadata, while per-expert
        // GMM2 readiness retains the producer/Combine overlap.
        MegaMoeCombineImpl::CombineTokensParallel<bfloat16_t, true, true, true>(
            rankId_, combineAccumGmGlobal_, params_, expandedRowIdx_, distProbs_, tokenPerExpertWin_,
            expertTokenCumsum_, firstTokenPosPtr_, stepIdxPtr_, posLenPtr_, chainLenReadyFlagPtr_, posDonePtr_,
            reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.ptrGmm2ReadyFlag),
            reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.ptrExpertOrder), expertScheduleEnabled_, 0U,
            expertPerRank_, GetMGroupsPerWave(), blockAivNum_, aivCoreIdx_, timer);
        return;
    }

    if constexpr (g_coreType == AscendC::AIC) {
        using Gmm1Context = MegaMoeCatlassImpl::Gmm1Context<QuantOutType, QuantOutType, bfloat16_t, QuantScaleOutType,
                                                            QuantScaleOutType, true>;
        using Gmm2Context = MegaMoeCatlassImpl::Gmm2Context<QuantOutType, QuantOutType, bfloat16_t, QuantScaleOutType,
                                                            QuantScaleOutType>;
        uint32_t gmm1StartBlockIdx = 0U;
        uint32_t gmm2StartBlockIdx = 0U;
        bool identityIncremental = !expertScheduleEnabled_;
        const uint32_t mGroupsPerWave = GetMGroupsPerWave();
        preOffset_ = rankBaseRow;
        baseOffset_ = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

        // The retained experiment uses one whole GMM1 grid followed
        // immediately by one whole GMM2 grid per expert. EndSync below only
        // closes this
        // AIC's final cube/vector credit; it is not a whole-grid publication
        // barrier.  Keep GMM2's per-expert ready-counter wait enabled so no
        // GMM2 core can read A2/A2Scale until every GMM1 epilogue tile has
        // completed its MTE3_S publication.
        for (uint32_t scheduleIdx = 0U; scheduleIdx < expertPerRank_; ++scheduleIdx) {
            uint32_t expertIdx = GetScheduledExpert(scheduleIdx);
            bool hasGroup =
                identityIncremental ? UpdateIdentityGroupParams(expertIdx) : SetGroupParams(expertIdx, rankBaseRow);
            if (!hasGroup) {
                continue;
            }
            UpdateGlobalBuffer<true>(gmmAddrInfo, 0U);
            // Gmm1Context/Gmm2Context are constructed and destroyed per expert,
            // never coexisting across the loop.  The Blaze GMM1 BlockMmad ctor
            // sets MTE1_MTE2-family hardware flags (IDs overlapping the catlass
            // 0-3 range) plus SetMMLayoutTransform(true); its dtor waits all of
            // them out.  The catlass Gmm2Context ctor runs pipe.Destroy(),
            // which clears the hardware flag state those IDs hold.  With both
            // contexts alive at once, that clear wipes Blaze's still-set flags
            // and the next Blaze WaitFlag<MTE1_MTE2>(0) dead-waits into an
            // aicore trap (PIPE_ALL cannot restore software flag state).  The
            // generic and complete-stage paths already honor this
            // "Blaze/catlass must not coexist" rule via scoped lifetimes.
            {
                Gmm1Context gmm1Context(Get<N_VALUE>(problemShape_), Get<K_VALUE>(problemShape_));
                MegaMoeCatlassImpl::GroupMatmulSwigluQuant<QuantOutType, QuantOutType, bfloat16_t, QuantScaleOutType,
                                                           QuantScaleOutType>(
                    gmm1Context, streamingEpilogueOp_, problemShape_, gmmAddrInfo, gmm1StartBlockIdx,
                    gmm1VecSetSyncCom_, timer, expertIdx);
            }

            // Do not carry GMM1's final AIC<->AIV credit through an unrelated
            // GMM2 grid.  The next expert would otherwise consume that credit
            // only after GMM2 has reused the AIC pipeline, which traps on the
            // dense K16 second-wave shapes.  The epilogue publishes the GMM1
            // ready counter before NotifyCube, so this also makes the whole
            // expert's GMM2 input visible without a global barrier.
            EndSync();

            // Route the ownership wait to this mixed core's AIV1. The first
            // active expert consumes Dispatch's final credit; every later
            // expert consumes the previous expert's Combine credit. Running
            // GMM1 above the wait keeps GMM1(e)/Combine(e-1) overlapped, while
            // forbidding the next CATLASS context from reinitializing AIV1 UB
            // before this core has drained its prior Combine range.
            WaitForVector(GMM2_UB_READY_FLAG + FLAG_ID_MAX);

            uint32_t expertMGroups =
                Ops::Base::CeilDiv(static_cast<uint32_t>(Get<M_VALUE>(problemShape_)), EXPERT_SCHEDULE_M_TILE_ROWS);
            UpdateGlobalBuffer(gmmAddrInfo, 1U);
            bool fitsMGroupReadyBudget = expertMGroups <= mGroupsPerWave;
            bool allowLongMGroupReady = expertScheduleEnabled_ && fitsMGroupReadyBudget;
            {
                Gmm2Context gmm2Context(Get<K_VALUE>(problemShape_),
                                        Get<N_VALUE>(problemShape_) / MegaMoeCatlassImpl::SWIGLU_N_HALF);
                bool useMGroupReady =
                    MegaMoeCatlassImpl::GroupMatmul2<QuantOutType, QuantOutType, bfloat16_t, QuantScaleOutType,
                                                     QuantScaleOutType, true, true>(
                        gmm2Context, params_, problemShape_, gmmAddrInfo, gmm2StartBlockIdx, expertIdx, 0U, true,
                        fitsMGroupReadyBudget, allowLongMGroupReady, combineAccumGm_, timer);
                if (!useMGroupReady) {
                    SyncFunc<AscendC::HardEvent::FIX_S>();
                    AtomicAdd(const_cast<__gm__ int32_t *>(gmm2ReadyFlag_.GetPhyAddr()), int32_t(1));
                }
            }
            // Return the physical AIV1 UB bank only after Gmm2Context and its
            // Resource/TPipe have been fully destroyed. The paired AIV1 waits
            // for this release before touching its Combine LocalTensors.
            NotifyVector(GMM2_UB_RELEASE_FLAG + FLAG_ID_MAX);
        }
        // The producer emits one Dispatch credit plus one credit per active
        // expert. The loop consumes one per active expert, so exactly one is
        // left here: the final expert's Combine credit, or Dispatch's credit
        // on an empty-owner rank. Drain it before a later launch reuses the
        // hardware flag.
        WaitForVector(GMM2_UB_READY_FLAG + FLAG_ID_MAX);
        EndSync();
    }
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::WaitForGmm1ExpertReady(const GMMAddrInfo &gmmAddrInfo,
                                                                                uint32_t rows, bool &waited)
{
    if (waited) {
        return;
    }
    uint32_t outputN = static_cast<uint32_t>(Get<N_VALUE>(problemShape_)) / MegaMoeCatlassImpl::SWIGLU_N_HALF;
    uint32_t readyTarget = Ops::Base::CeilDiv(rows, MegaMoeCatlassImpl::L1_TILE_M) *
                           Ops::Base::CeilDiv(outputN, MegaMoeCatlassImpl::L1_TILE_N);
    uint32_t pollMisses = 0U;
    int64_t pollDelayCycles = 100;
    while (AscendC::ReadGmByPassDCache(gmmAddrInfo.flagSwigluToGmm2) < static_cast<int32_t>(readyTarget)) {
        GmSpinBackoffStep(pollMisses, pollDelayCycles);
    }
    waited = true;
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline CoreWaveGmm2Plan MegaMoe<TemplateMegaMoeTypeFunc>::PrepareCoreWaveGmm2Plan(int32_t rankBaseRow)
{
    // Immutable metadata only: prepare before the GMM1 context drains, but
    // acquire no GMM2 resource/event or local-memory view until after it.
    uint32_t nTiles = Ops::Base::CeilDiv(k_, MegaMoeCatlassImpl::L1_TILE_N);
    uint32_t tailStripeUnits =
        Ops::Base::CeilDiv(k_ - (nTiles - 1U) * MegaMoeCatlassImpl::L1_TILE_N, MegaMoeCatlassImpl::L1_TILE_N / 2U);
    // The unchanged sparse chunker uses actual 128-column work units.
    uint32_t groupUnits = 2U * (nTiles - 1U) + tailStripeUnits;
    uint32_t totalMGroups = 0U;
    uint32_t totalFullMGroups = 0U;
    if (expertPerRank_ >= EXPERT_SCHEDULE_MIN_ACTIVE_EXPERTS) {
        __gm__ int32_t *planAddr = const_cast<__gm__ int32_t *>(expertOrderScratch_.GetPhyAddr());
        uint32_t headerBase = 1U + 2U * expertPerRank_;
        totalMGroups =
            static_cast<uint32_t>(ReadGmByPassDCache(planAddr + headerBase + EXPERT_PLAN_HEADER_TOTAL_MGROUPS));
        totalFullMGroups =
            static_cast<uint32_t>(ReadGmByPassDCache(planAddr + headerBase + EXPERT_PLAN_HEADER_TOTAL_FULL_MGROUPS));
    } else {
        for (uint32_t scheduleIdx = 0U; scheduleIdx < expertPerRank_; ++scheduleIdx) {
            uint32_t expertIdx = GetScheduledExpert(scheduleIdx);
            int32_t localStart = 0;
            int32_t rows = 0;
            if (GetExpertRange(expertIdx, rankBaseRow, localStart, rows)) {
                totalMGroups += Ops::Base::CeilDiv(static_cast<uint32_t>(rows), EXPERT_SCHEDULE_M_TILE_ROWS);
                totalFullMGroups += static_cast<uint32_t>(rows) / EXPERT_SCHEDULE_M_TILE_ROWS;
            }
        }
    }

    // Preserve which launches enter the dense producer/consumer protocol.
    // Dense launches now distribute M x N tiles, rather than assigning an
    // entire M-group's N sweep to one AIC. Sparse chunking is unchanged.
    bool useDenseTileGrid = blockNum_ > 0U && totalMGroups >= blockNum_ && nTiles > 1U;
    if (!useDenseTileGrid) {
        return {nTiles, groupUnits, totalMGroups, totalFullMGroups, false, false, false};
    }
    uint32_t gmm2K = static_cast<uint32_t>(Get<N_VALUE>(problemShape_) / MegaMoeCatlassImpl::SWIGLU_N_HALF);
    // This is the EXISTING consumer capability gate, not a request for a
    // Full-A compute arena. Keep it paired with CanUseFullAGmm2StripeReady
    // on AIV; changing the L1 policy must not silently change ready formats.
    bool useFullAReadyProtocol =
        Ops::Base::CeilDiv(gmm2K, MegaMoeCatlassImpl::L1_TILE_K) == MegaMoeCatlassImpl::FULL_A_L1_K_STAGES;
    bool publishStripeReady = false;
    if constexpr (UseOwnerChainCombine && UseK16OwnerChain) {
        publishStripeReady = useFullAReadyProtocol && CanUseFullAGmm2StripeReady(k_, gmm2K, totalMGroups, blockNum_);
    } else if constexpr (!UseOwnerChainCombine) {
        publishStripeReady = expertPerRank_ >= EXPERT_SCHEDULE_MIN_ACTIVE_EXPERTS &&
                             ShouldPackCombineRouteMetadata(worldSize_, m_, topK_, maxOutputSize_, blockNum_) &&
                             CanUseExpertMajorGmm2StripeReady(k_, gmm2K, totalMGroups, blockNum_);
    }
    return {nTiles,
            groupUnits,
            totalMGroups,
            totalFullMGroups,
            useDenseTileGrid,
            useFullAReadyProtocol,
            publishStripeReady};
}

template <TemplateMegaMoeTypeClass>
template <bool UseFullAReadyProtocol>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ProcessGmm2MnTiles(int32_t rankBaseRow,
                                                                            uint32_t totalFullMGroups, uint32_t nTiles,
                                                                            uint32_t gmm2StartBlockIdx,
                                                                            bool publishStripeReady)
{
    // Keep one ordinary A2/B2 context for the complete GMM2 stage. Different
    // AICs now own the N tiles of a group, so a whole-A resident arena no
    // longer provides the old cross-N reuse. Publication traits are separate
    // from L1 staging: retain every consumer's existing lane/mask contract.
    using TileGmm2Context =
        MegaMoeCatlassImpl::Gmm2Context<QuantOutType, QuantOutType, bfloat16_t, QuantScaleOutType, QuantScaleOutType,
                                        UseFullAReadyProtocol && UseRaggedGmm2StripeMask, !UseOwnerChainCombine>;
    TileGmm2Context gmm2Context(Get<K_VALUE>(problemShape_),
                                Get<N_VALUE>(problemShape_) / MegaMoeCatlassImpl::SWIGLU_N_HALF,
                                gmm1VecSetSyncCom_ != 0, true);
    // The context now owns the final epilogue and Dispatch credits, including
    // the no-tile case. Do not consume the epilogue credit again via EndSync.
    gmm1VecSetSyncCom_ = 0;
    if constexpr (UseOwnerChainCombine && (!UseFullAReadyProtocol || !UseK16OwnerChain)) {
        publishStripeReady = false;
    }

    GMMAddrInfo gmmAddrInfo;
    uint32_t fullGroupOrdinal = 0U;
    uint32_t tailGroupOrdinal = 0U;
    for (uint32_t scheduleIdx = 0U; scheduleIdx < expertPerRank_; ++scheduleIdx) {
        uint32_t expertIdx = GetScheduledExpert(scheduleIdx);
        if (!SetGroupParams(expertIdx, rankBaseRow)) {
            continue;
        }
        UpdateGlobalBuffer(gmmAddrInfo, 1);
        uint32_t expertRows = static_cast<uint32_t>(Get<M_VALUE>(problemShape_));
        uint32_t expertMGroups = Ops::Base::CeilDiv(expertRows, EXPERT_SCHEDULE_M_TILE_ROWS);
        bool waitedForGmm1Ready = false;
        for (uint32_t mGroup = 0U; mGroup < expertMGroups; ++mGroup) {
            uint32_t groupRows = expertRows - mGroup * EXPERT_SCHEDULE_M_TILE_ROWS;
            // Preserve the existing full/tail ordinal pools. Flattening the
            // two pools separately avoids pinning full rows and short tails
            // to different core classes across repeated two-group experts.
            uint32_t groupOrdinal =
                groupRows >= EXPERT_SCHEDULE_M_TILE_ROWS ? fullGroupOrdinal++ : totalFullMGroups + tailGroupOrdinal++;
            // Flatten (group, N tile), then rotate over the runtime AIC grid.
            // Widen before multiplication; uint32 group/tile counts plus the
            // runtime core phase fit in uint64_t.
            uint32_t groupPhase =
                static_cast<uint32_t>((static_cast<uint64_t>(groupOrdinal) * nTiles + gmm2StartBlockIdx) % blockNum_);
            uint32_t firstTile = blockIdx_ >= groupPhase ? blockIdx_ - groupPhase : blockIdx_ + blockNum_ - groupPhase;
            if (firstTile >= nTiles) {
                continue;
            }
            WaitForGmm1ExpertReady(gmmAddrInfo, expertRows, waitedForGmm1Ready);
            __gm__ int32_t *groupReadyAddr =
                gmmAddrInfo.dispatchTileReady + static_cast<uint64_t>(mGroup) * DISPATCH_TILE_READY_STRIDE;
            for (uint32_t nTile = firstTile; nTile < nTiles; nTile += blockNum_) {
                // A one-tile interval also flushes the helper's job-end
                // publication. Never pass a strided interval: its ready
                // lanes/mask must describe only this AIC's completed tile.
                MegaMoeCatlassImpl::GroupMatmul2MGroupStripe<QuantOutType, QuantOutType, bfloat16_t, QuantScaleOutType,
                                                             QuantScaleOutType>(
                    gmm2Context, problemShape_, gmmAddrInfo, mGroup, nTile, nTile + 1U, combineAccumGm_, timer,
                    expertIdx, publishStripeReady ? groupReadyAddr + GMM2_STRIPE_READY_OFFSET : nullptr);
                if (!publishStripeReady) {
                    SyncFunc<AscendC::HardEvent::FIX_S>();
                }
                // Multiple AICs contribute to this counter. Each unique tile
                // contributes exactly one vote AFTER its output is visible;
                // the unchanged consumer completes at precisely nTiles.
                AtomicAdd(groupReadyAddr + GMM2_READY_OFFSET, int32_t(1));
            }
        }
    }
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ProcessCoreWavePipeline(TPipe *firstPipe, int32_t rankBaseRow)
{
    static_assert(UseCoreWave, "Core-wave pipeline must not be instantiated by the generic tiling key");
    GMMAddrInfo gmmAddrInfo;

    if constexpr (g_coreType == AscendC::AIV) {
        if (subBlockIdx_ == 1U) {
            preOffset_ = rankBaseRow;
            baseOffset_ = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
            // Keep Dispatch ahead of GMM1, but finish its remote payload
            // traffic before the AIC stage reaches GMM2.  This preserves the
            // GMM1(e)-Dispatch(e+1) overlap without making GMM2 input reads
            // compete with later-expert Dispatch stores on GM/L2/NoC.
            if constexpr (UseOwnerChainCombine) {
                // Owner Combine needs a compact chain descriptor in addition
                // to the fanout plan.  Unsupported descriptor widths or a
                // runtime histogram without enough reuse retain both sides of
                // the proven expert-major protocol; the ready publication
                // counter below tells every AIV which protocol was selected.
                if (DispatchFanoutWorkProfitable(rankBaseRow) &&
                    CanPackTokenOwnerDescriptor(expertPerRank_, topK_, maxOutputSize_)) {
                    TIMER_BLOCK(timer.Tik(AscendTimer::AIV_DISPATCH, 0U));
                    DispatchTokensByOwner(rankBaseRow);
                } else {
                    DispatchExperts(gmmAddrInfo, 0U, expertPerRank_, rankBaseRow);
                }
            } else {
                if (DispatchFanoutWorkProfitable(rankBaseRow)) {
                    // Build the inverse route plan with the common SIMT
                    // producer, then retain the single token-owner fanout.
                    // Accelerating the shared startup dependency advances all
                    // GMM1 groups without adding a serial expert prefix.
                    TIMER_BLOCK(timer.Tik(AscendTimer::AIV_DISPATCH, 0U));
                    DispatchTokensByOwner(rankBaseRow);
                } else {
                    // Resource limits, insufficient proven reuse, and tiny
                    // work all retain the proven expert-major path.
                    DispatchExperts(gmmAddrInfo, 0U, expertPerRank_, rankBaseRow);
                }
            }
            // Join the final payload/descriptor work without involving AIC.
            // lane0 retains the owner-plan selection result; lane1 is one
            // launch-wide Dispatch completion counter.  Every paired AIC gets
            // a hardware credit only after all AIV1 workers have arrived.
            PipeBarrier<PIPE_ALL>();
            firstPipe->Destroy();
            AtomicAdd(chainLenReadyFlagPtr_ + CORE_WAVE_DISPATCH_READY_OFFSET, int32_t(1));
            uint32_t pollMisses = 0U;
            int64_t pollDelayCycles = 100;
            while (ReadGmByPassDCache(chainLenReadyFlagPtr_ + CORE_WAVE_DISPATCH_READY_OFFSET) <
                   static_cast<int32_t>(blockNum_)) {
                GmSpinBackoffStep(pollMisses, pollDelayCycles);
            }
            NotifyCube(GMM2_UB_READY_FLAG);
        } else {
            using Gmm1Context = MegaMoeCatlassImpl::Gmm1Context<QuantOutType, QuantOutType, bfloat16_t,
                                                                QuantScaleOutType, QuantScaleOutType, false>;
            Gmm1Context gmm1Context(Get<N_VALUE>(problemShape_), Get<K_VALUE>(problemShape_));
            uint32_t gmm1StartBlockIdx = 0U;
            bool identityIncremental = !expertScheduleEnabled_;
            preOffset_ = rankBaseRow;
            baseOffset_ = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
            // Complete-stage execution no longer has a wave boundary between
            // experts.  Invoke each independent M grid once and let CATLASS
            // distribute all of its tiles, avoiding repeated endpoint reads,
            // cursor planning, and slice setup on every AIV0.
            for (uint32_t scheduleIdx = 0U; scheduleIdx < expertPerRank_; ++scheduleIdx) {
                uint32_t expertIdx = GetScheduledExpert(scheduleIdx);
                bool hasGroup =
                    identityIncremental ? UpdateIdentityGroupParams(expertIdx) : SetGroupParams(expertIdx, rankBaseRow);
                if (!hasGroup) {
                    continue;
                }
                UpdateGlobalBuffer<true>(gmmAddrInfo, 0);
                MegaMoeCatlassImpl::GroupMatmulSwigluQuant<QuantOutType, QuantOutType, bfloat16_t, QuantScaleOutType,
                                                           QuantScaleOutType, false, true, true>(
                    gmm1Context, streamingEpilogueOp_, problemShape_, gmmAddrInfo, gmm1StartBlockIdx,
                    gmm1VecSetSyncCom_, timer, expertIdx);
            }
        }

        // Keep Combine's immutable metadata contract on the vector side, but
        // do not hold AIC GMM2 behind unrelated AIV epilogue tail work.
        SyncAll<true>();

        // C6 keeps C5's global M-group scheduler and restores the producer /
        // consumer overlap: every AIC owner publishes an M-group only after
        // all of its Fixpipe stores are visible, and both AIV sub-blocks start
        // consuming independent ready groups immediately.  There is no second
        // complete-GMM2 barrier on the critical path.
        if constexpr (UseOwnerChainCombine) {
            // BuildDispatchFanoutPlan publishes one vote per AIV1 only when
            // the complete owner descriptor protocol was selected. The final
            // Dispatch counter and vector-only barrier above make this a
            // stable launch-wide decision without another workspace field.
            bool ownerDescriptorsReady = ReadGmByPassDCache(chainLenReadyFlagPtr_) == static_cast<int32_t>(blockNum_);
            if (ownerDescriptorsReady &&
                MegaMoeCombineImpl::CanUseOwnerReadyCombine<UseK16OwnerChain>(params_, blockAivNum_)) {
                MegaMoeCombineImpl::CombineTokensByOwnerReadyChunks<bfloat16_t, UseK16OwnerChain,
                                                                    UseRaggedGmm2StripeMask>(
                    rankId_, combineAccumGmGlobal_, params_, distProbs_, tokenChainDescriptorsPtr_, posLenPtr_,
                    reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.ptrDispatchTileReady),
                    reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.ptrExpertOrder), expertScheduleEnabled_,
                    blockAivNum_, aivCoreIdx_, timer);
            } else {
                MegaMoeCombineImpl::CombineTokensParallel<bfloat16_t, true, false, false, true>(
                    rankId_, combineAccumGmGlobal_, params_, expandedRowIdx_, distProbs_, tokenPerExpertWin_,
                    expertTokenCumsum_, firstTokenPosPtr_, stepIdxPtr_, posLenPtr_, chainLenReadyFlagPtr_, posDonePtr_,
                    reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.ptrGmm2ReadyFlag),
                    reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.ptrExpertOrder), expertScheduleEnabled_,
                    0U, expertPerRank_, GetMGroupsPerWave(), blockAivNum_, aivCoreIdx_, timer);
            }
        } else {
            MegaMoeCombineImpl::CombineTokensParallel<bfloat16_t, true, false, false, true, false, true, true>(
                rankId_, combineAccumGmGlobal_, params_, expandedRowIdx_, distProbs_, tokenPerExpertWin_,
                expertTokenCumsum_, firstTokenPosPtr_, stepIdxPtr_, posLenPtr_, chainLenReadyFlagPtr_, posDonePtr_,
                reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.ptrGmm2ReadyFlag),
                reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.ptrExpertOrder), expertScheduleEnabled_, 0U,
                expertPerRank_, GetMGroupsPerWave(), blockAivNum_, aivCoreIdx_, timer);
        }
        PipeBarrier<PIPE_ALL>();
        return;
    }

    if constexpr (g_coreType == AscendC::AIC) {
        uint32_t gmm1StartBlockIdx = 0U;
        bool identityIncremental = !expertScheduleEnabled_;
        preOffset_ = rankBaseRow;
        baseOffset_ = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        const uint64_t stagePreOffset = preOffset_;
        const uint64_t stageInitialM = Get<M_VALUE>(problemShape_);
        const BlockOffset stageBaseOffset = baseOffset_;
        const bool stageIdentityIncremental = identityIncremental;

        CoreWaveGmm2Plan gmm2Plan;
        // Each independent expert grid is launched once inside the complete
        // stage.  CATLASS already distributes arbitrary M across all AICs;
        // retaining the old wave cursor here only repeated GM endpoint reads
        // and slice setup without introducing a synchronization boundary.
        {
            using Gmm1Context = MegaMoeCatlassImpl::Gmm1Context<QuantOutType, QuantOutType, bfloat16_t,
                                                                QuantScaleOutType, QuantScaleOutType, true>;
            Gmm1Context gmm1Context(Get<N_VALUE>(problemShape_), Get<K_VALUE>(problemShape_));
            for (uint32_t scheduleIdx = 0U; scheduleIdx < expertPerRank_; ++scheduleIdx) {
                uint32_t expertIdx = GetScheduledExpert(scheduleIdx);
                bool hasGroup =
                    identityIncremental ? UpdateIdentityGroupParams(expertIdx) : SetGroupParams(expertIdx, rankBaseRow);
                if (!hasGroup) {
                    continue;
                }
                UpdateGlobalBuffer<true>(gmmAddrInfo, 0);
                MegaMoeCatlassImpl::GroupMatmulSwigluQuant<QuantOutType, QuantOutType, bfloat16_t, QuantScaleOutType,
                                                           QuantScaleOutType, true, true, true>(
                    gmm1Context, streamingEpilogueOp_, problemShape_, gmmAddrInfo, gmm1StartBlockIdx,
                    gmm1VecSetSyncCom_, timer, expertIdx);
            }
            gmm2Plan = PrepareCoreWaveGmm2Plan(rankBaseRow);
        }

        // Do not queue the final AIV0/AIV1 credits before GMM2's Resource
        // initialization: its PIPE_ALL would also hold back independent
        // input loads/MMAD. Each selected context takes these credits below
        // and arms them on PIPE_FIX after its first work Timer::Tik. Expert
        // input readiness and output-before-ready publication stay unchanged.

        preOffset_ = stagePreOffset;
        Get<M_VALUE>(problemShape_) = stageInitialM;
        baseOffset_ = stageBaseOffset;
        identityIncremental = stageIdentityIncremental;
        // Continue the physical-core phase accumulated by GMM1 for a real
        // hot/cold schedule.  This avoids assigning both complete stages the
        // same persistent tail cores.  Identity order starts GMM2 from core
        // zero, with no host-modeled distribution-specific phase.
        uint32_t gmm2StartBlockIdx = expertScheduleEnabled_ ? gmm1StartBlockIdx : 0U;
        {
            const uint32_t nTiles = gmm2Plan.nTiles;
            const uint32_t groupUnits = gmm2Plan.groupUnits;
            const uint32_t totalMGroups = gmm2Plan.totalMGroups;
            const uint32_t totalFullMGroups = gmm2Plan.totalFullMGroups;
            const bool useDenseTileGrid = gmm2Plan.useDenseTileGrid;
            if (!useDenseTileGrid) {
                // Sparse grids assign different N tiles of one M-group to
                // different AICs, so no core can reuse A across N.  Retain the
                // ordinary two-stage CATLASS policy there instead of paying
                // the resident policy's 480-KiB L1 footprint without a hit.
                using SparseGmm2Context = MegaMoeCatlassImpl::Gmm2Context<QuantOutType, QuantOutType, bfloat16_t,
                                                                          QuantScaleOutType, QuantScaleOutType>;
                SparseGmm2Context gmm2Context(Get<K_VALUE>(problemShape_),
                                              Get<N_VALUE>(problemShape_) / MegaMoeCatlassImpl::SWIGLU_N_HALF,
                                              gmm1VecSetSyncCom_ != 0, true);
                gmm1VecSetSyncCom_ = 0;
                uint32_t sparseFullOrdinal = 0U;
                uint32_t sparseTailOrdinal = 0U;
                // Chunked sparse jobs: split each group's N stripes into the
                // fewest contiguous chunks that still spread roughly two
                // chunk waves across the physical AICs.  One context call
                // then sweeps a whole chunk with no per-tile Fixpipe drain,
                // the ready vote is paid once per chunk at its exact tile
                // width (the group still becomes ready at precisely nTiles
                // votes), and consecutive stripes of one M-group share their
                // activation lines in L2.  Two waves bound the straggler
                // penalty at half a chunk; fewer would leave cores idle
                // behind one long chunk, more would re-grow the per-job
                // control cost this path exists to avoid.  All inputs are
                // runtime quantities - no shape or SKU constant.
                uint32_t sparseGroups = totalMGroups > 0U ? totalMGroups : 1U;
                uint32_t chunksPerGroup = Ops::Base::CeilDiv(2U * blockNum_, sparseGroups);
                chunksPerGroup = chunksPerGroup < nTiles ? chunksPerGroup : nTiles;
                chunksPerGroup = chunksPerGroup > 0U ? chunksPerGroup : 1U;
                // Chunk boundaries are laid out in 128-column units so a
                // partial final stripe is not over-counted, then rounded to
                // whole stripes by each stripe's starting unit.  A chunk
                // whose unit range rounds to zero stripes is skipped: its
                // stripes belong to a neighbour and the votes still sum to
                // exactly nTiles.
                uint32_t baseChunkUnits = groupUnits / chunksPerGroup;
                uint32_t extraUnitChunks = groupUnits % chunksPerGroup;
                for (uint32_t scheduleIdx = 0U; scheduleIdx < expertPerRank_; ++scheduleIdx) {
                    uint32_t expertIdx = GetScheduledExpert(scheduleIdx);
                    if (!SetGroupParams(expertIdx, rankBaseRow)) {
                        continue;
                    }
                    UpdateGlobalBuffer(gmmAddrInfo, 1);
                    uint32_t expertMGroups = Ops::Base::CeilDiv(static_cast<uint32_t>(Get<M_VALUE>(problemShape_)),
                                                                EXPERT_SCHEDULE_M_TILE_ROWS);
                    bool waitedForGmm1Ready = false;
                    for (uint32_t mGroup = 0U; mGroup < expertMGroups; ++mGroup) {
                        uint32_t groupRows =
                            static_cast<uint32_t>(Get<M_VALUE>(problemShape_)) - mGroup * EXPERT_SCHEDULE_M_TILE_ROWS;
                        groupRows = groupRows < EXPERT_SCHEDULE_M_TILE_ROWS ? groupRows : EXPERT_SCHEDULE_M_TILE_ROWS;
                        uint32_t groupOrdinal = groupRows == EXPERT_SCHEDULE_M_TILE_ROWS
                                                    ? sparseFullOrdinal++
                                                    : totalFullMGroups + sparseTailOrdinal++;
                        // Flattened chunk ordinals rotate owners across every
                        // AIC (count-balanced within one job), which also
                        // repairs the degenerate old phase mapping when
                        // nTiles was a multiple of the AIC count.
                        uint32_t chunkUnitBegin = 0U;
                        uint32_t chunkBegin = 0U;
                        for (uint32_t chunkIdx = 0U; chunkIdx < chunksPerGroup; ++chunkIdx) {
                            uint32_t chunkUnitEnd =
                                chunkUnitBegin + baseChunkUnits + (chunkIdx < extraUnitChunks ? 1U : 0U);
                            uint32_t chunkEnd = (chunkUnitEnd + 1U) / 2U;
                            chunkEnd = chunkEnd < nTiles ? chunkEnd : nTiles;
                            uint32_t chunkTiles = chunkEnd - chunkBegin;
                            uint32_t owner = (groupOrdinal * chunksPerGroup + chunkIdx + gmm2StartBlockIdx) % blockNum_;
                            if (owner == blockIdx_ && chunkTiles > 0U) {
                                WaitForGmm1ExpertReady(gmmAddrInfo, static_cast<uint32_t>(Get<M_VALUE>(problemShape_)),
                                                       waitedForGmm1Ready);
                                MegaMoeCatlassImpl::GroupMatmul2MGroupStripe<QuantOutType, QuantOutType, bfloat16_t,
                                                                             QuantScaleOutType, QuantScaleOutType>(
                                    gmm2Context, problemShape_, gmmAddrInfo, mGroup, chunkBegin, chunkEnd,
                                    combineAccumGm_, timer, expertIdx);
                                // The ready counter represents completed N
                                // stripes.  FIX_S makes this chunk's stores
                                // visible before its exact-width vote, so
                                // the group still becomes ready at precisely
                                // nTiles accumulated votes.
                                SyncFunc<AscendC::HardEvent::FIX_S>();
                                AtomicAdd(gmmAddrInfo.dispatchTileReady +
                                              static_cast<uint64_t>(mGroup) * DISPATCH_TILE_READY_STRIDE +
                                              GMM2_READY_OFFSET,
                                          int32_t(chunkTiles));
                            }
                            chunkUnitBegin = chunkUnitEnd;
                            chunkBegin = chunkEnd;
                        }
                    }
                }
                // Every sparse chunk has already published its own completion
                // after FIX_S, so no full-stage barrier remains here.
                return;
            }
            const bool publishStripeReady = gmm2Plan.publishStripeReady;
            if (gmm2Plan.useFullAReadyProtocol) {
                ProcessGmm2MnTiles<true>(rankBaseRow, totalFullMGroups, nTiles, gmm2StartBlockIdx, publishStripeReady);
            } else {
                ProcessGmm2MnTiles<false>(rankBaseRow, totalFullMGroups, nTiles, gmm2StartBlockIdx, publishStripeReady);
            }
            // The context is gone after every owned tile has published its
            // completion vote. Kernel block completion drains the
            // remaining AIC work without serializing AIV Combine behind it.
            return;
        }
        {
            using Gmm2Context = MegaMoeCatlassImpl::Gmm2Context<QuantOutType, QuantOutType, bfloat16_t,
                                                                QuantScaleOutType, QuantScaleOutType>;
            Gmm2Context gmm2Context(Get<K_VALUE>(problemShape_),
                                    Get<N_VALUE>(problemShape_) / MegaMoeCatlassImpl::SWIGLU_N_HALF);
            const uint32_t mGroupsPerWave = GetMGroupsPerWave();
            for (uint32_t scheduleIdx = 0U; scheduleIdx < expertPerRank_; ++scheduleIdx) {
                uint32_t expertIdx = GetScheduledExpert(scheduleIdx);
                bool hasGroup =
                    identityIncremental ? UpdateIdentityGroupParams(expertIdx) : SetGroupParams(expertIdx, rankBaseRow);
                if (!hasGroup) {
                    continue;
                }
                uint32_t expertMGroups =
                    Ops::Base::CeilDiv(static_cast<uint32_t>(Get<M_VALUE>(problemShape_)), EXPERT_SCHEDULE_M_TILE_ROWS);
                UpdateGlobalBuffer(gmmAddrInfo, 1);
                // Producer and consumer share the same wave-capacity gate.
                // Very long experts therefore keep the full-ready fallback
                // even though their CATLASS grid is now one whole call.
                bool fitsMGroupReadyBudget = expertMGroups <= mGroupsPerWave;
                bool allowLongMGroupReady = expertScheduleEnabled_ && fitsMGroupReadyBudget;
                bool useMGroupReady =
                    MegaMoeCatlassImpl::GroupMatmul2<QuantOutType, QuantOutType, bfloat16_t, QuantScaleOutType,
                                                     QuantScaleOutType, true, false>(
                        gmm2Context, params_, problemShape_, gmmAddrInfo, gmm2StartBlockIdx, expertIdx, 0U, true,
                        fitsMGroupReadyBudget, allowLongMGroupReady, combineAccumGm_, timer);
                if (!useMGroupReady) {
                    SyncFunc<AscendC::HardEvent::FIX_S>();
                    AtomicAdd(const_cast<__gm__ int32_t *>(gmm2ReadyFlag_.GetPhyAddr()), int32_t(1));
                }
            }
        }
    }
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::CrossRankSyncInWorldSize()
{
    __gm__ int32_t *syncRank = (__gm__ int32_t *)(params_.peermemInfo.ptrBase);
    __gm__ int32_t *syncCount = (__gm__ int32_t *)(params_.peermemInfo.ptrBase + 48 * 1024);
    uint32_t count = static_cast<uint32_t>(ReadGmByPassDCache(syncCount + aivCoreIdx_ * 16)) + 1U;
    for (int i = aivCoreIdx_; i < worldSize_; i += blockAivNum_) {
        __gm__ int32_t *syncRemoteAddr = (__gm__ int32_t *)(winRankAddr_[i]) + rankId_ * 16;
        WriteGmByPassDCache(syncRemoteAddr, static_cast<int32_t>(count));
        auto syncCheck = syncRank + i * 16;
        GmSignalWaitBarrier(syncCheck, count);
    }
    // SyncAll opens with PipeBarrier<PIPE_ALL>() on every participating core
    // (AscendC SyncAllImpl), and this tail is AIV-only where SyncAll<true> is
    // a real sync -- so no explicit barrier is needed before it.  Adding one
    // compiles to back-to-back BAR.ALL with nothing in between.
    SyncAll<true>();
    WriteGmByPassDCache(syncCount + aivCoreIdx_ * 16, static_cast<int32_t>(count));
}

template <TemplateMegaMoeTypeClass>
template <bool EnableBf16L1Reuse>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ProcessGenericGmm2(int32_t rankBaseRow)
{
    static_assert(!EnableBf16L1Reuse || UseBf16NoQuant, "BF16 L1 reuse must not change the MXFP compute policy");
    // Construct one context for the whole stage, including empty experts.
    // The existing first-output handoff and final epilogue drain stay put.
    using Gmm2Context = MegaMoeCatlassImpl::Gmm2Context<QuantOutType, QuantOutType, bfloat16_t, QuantScaleOutType,
                                                        QuantScaleOutType, false, false, EnableBf16L1Reuse>;
    Gmm2Context gmm2Context(Get<K_VALUE>(problemShape_),
                            Get<N_VALUE>(problemShape_) / MegaMoeCatlassImpl::SWIGLU_N_HALF, false, true);
    GMMAddrInfo gmmAddrInfo;
    bool gmm1Sync = false;
    for (uint32_t scheduleIdx = 0; scheduleIdx < expertPerRank_; scheduleIdx++) {
        uint32_t expertIdx = GetScheduledExpert(scheduleIdx);
        bool hasGroup =
            expertScheduleEnabled_ ? SetGroupParams(expertIdx, rankBaseRow) : UpdateIdentityGroupParams(expertIdx);
        if (!hasGroup) {
            continue;
        }
        UpdateGlobalBuffer(gmmAddrInfo, 1);
        MegaMoeCatlassImpl::GroupMatmul2<QuantOutType, QuantOutType, bfloat16_t, QuantScaleOutType, QuantScaleOutType>(
            gmm2Context, params_, problemShape_, gmmAddrInfo, startBlockIdx_, expertIdx, 0U, true, false, false,
            combineAccumGm_, timer);
        if (!gmm1Sync) {
            EndSync();
            gmm1Sync = true;
        }
        SyncFunc<AscendC::HardEvent::FIX_S>();
        AtomicAdd((__gm__ int32_t *)(gmm2ReadyFlag_.GetPhyAddr()), int32_t(1));
    }
    // 兜底排干：共享 GMM1（阶段 3.8）把末尾 credit 并入 gmm1VecSetSyncCom_ 交给
    // 路由链消费；本核若一个路由专家都没执行（所有本地 token 都路由去了远端
    // rank），上面的循环内 EndSync 不会发生，这里无条件补一次——账本非零时
    // 排干（防 credit 泄漏到下一次 launch），已排干时为空操作。
    EndSync();
}

// 共享专家输入准备：把本卡量化输出 a0（行内 data+scale 交错）去交错成 GMM1 需要的
// 分离布局（连续 data 区 + 连续 scale 区）。a0 的行步长与 scale 行内偏移由
// GetA0RowStride / GetA0ScaleOffset 决定。
// BF16 no-quant 为空操作：x 本身就是 [m,k] 连续 BF16 行（host 要求 H%16==0，行步长
// 恰为 2H 字节），与共享 GMM1 的 RowMajor A 布局完全一致，直接读 x 即可，既无需
// 去交错也无需任何拷贝；数据可见性由调用点之后的混核屏障（路由排序屏障与
// PrepareExpertPipeline 的两道屏障）向 AIC 发布，无需独立就绪标志。
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::PrepareSharedExpertInput(TPipe *tPipe)
{
    if constexpr (g_coreType == AscendC::AIC) {
        return;
    }
    if (sharedExpertNum_ == 0U) {
        return;
    }
    if constexpr (UseBf16NoQuant) {
        return;
    }
    const int64_t a0RowStride = GetA0RowStride(static_cast<int64_t>(k_));
    const int64_t a0ScaleOffset = GetA0ScaleOffset(static_cast<int64_t>(k_));
    // 每行 CeilDiv(k,64)*MXFP_MULTI_BASE_SIZE 个 E8M0 scale（每 32 元素一个），与
    // Mxfp8DynamicQuant 的产出（CeilDiv(k,32)）及 Gmm1Context::scaleK 的消费一致。
    const int64_t scaleCols =
        Ops::Base::CeilDiv(static_cast<int64_t>(k_), static_cast<int64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;

    TBuf<> sharedRowBuf;
    tPipe->InitBuffer(sharedRowBuf,
                      static_cast<uint32_t>(Ops::Base::CeilAlign(a0RowStride, static_cast<int64_t>(ALIGN_32))));
    LocalTensor<uint8_t> rowLocal = sharedRowBuf.Get<uint8_t>();

    GlobalTensor<uint8_t> a0Gm;
    GlobalTensor<uint8_t> sharedInputGm;
    GlobalTensor<uint8_t> sharedScaleGm;
    a0Gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(params_.peermemInfo.ptrA0));
    sharedInputGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(params_.workspaceInfo.ptrSharedInputData));
    sharedScaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(params_.workspaceInfo.ptrSharedInputScale));

    int32_t coreLen = 0;
    int32_t coreOffset = 0;
    TilingByCore(m_, coreLen, coreOffset, 1);
    const int32_t coreEnd = coreOffset + coreLen;
    DataCopyPadExtParams<uint8_t> padParams{false, 0U, 0U, 0U};
    for (int32_t row = coreOffset; row < coreEnd; ++row) {
        // 整行（含 scale）一次读入，避免两次读取同一 cacheline。
        DataCopyPad(rowLocal, a0Gm[static_cast<int64_t>(row) * a0RowStride],
                    {1U, static_cast<uint32_t>(a0RowStride), 0U, 0U, 0U}, padParams);
        SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        DataCopyPad(sharedInputGm[static_cast<int64_t>(row) * k_], rowLocal,
                    {1U, static_cast<uint32_t>(k_), 0U, 0U, 0U});
        DataCopyPad(sharedScaleGm[static_cast<int64_t>(row) * scaleCols], rowLocal[a0ScaleOffset],
                    {1U, static_cast<uint32_t>(scaleCols), 0U, 0U, 0U});
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
}

// 共享专家 GMM1（+ AIV 侧 SwiGLU epilogue）：本卡全部 token × 共享权重，
// 无 Dispatch/Combine。在 PrepareExpertPipeline 的两道混核屏障之后、路由
// gmm1Context 构造之前调用（generic 流水）：屏障已把 allgather 结果与 sub0 的
// UB 暂存区释放发布给全核，fixpipe CopyOut 可直接写配对 AIV 的 UB 低址区。
// AIC 跑矩阵乘 + CopyOut，sub0 就地执行 SwiGLU 量化 epilogue 并发布
// flagSwigluToGmm2（行 [0,S)）供 GMM2 等待；sub1 不参与共享链，随屏障放行
// 直接进入 Dispatch——与 AIC 的共享 GMM1 并发执行。共享链必须整体先于路由
// Gmm1Context（同为 Blaze，共用 MTE1_MTE2 族硬件 flag，不可共存）。
// BF16 no-quant：QuantOutType = bfloat16_t 使 Gmm1Context/epilogue 自动走 BF16 特化
// （无 MX scale 操作数）；A 直接读本卡 x（布局天然匹配），权重步长按 2 字节/元素。
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ProcessSharedGmm1()
{
    if (sharedExpertNum_ == 0U) {
        return;
    }
    const uint64_t scaleK =
        Ops::Base::CeilDiv(static_cast<uint64_t>(k_), static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
    const uint64_t scaleSharedK =
        Ops::Base::CeilDiv(static_cast<uint64_t>(sharedK_), static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) *
        MXFP_MULTI_BASE_SIZE;
    constexpr uint64_t sharedElemBytes = UseBf16NoQuant ? static_cast<uint64_t>(sizeof(bfloat16_t)) : 1U;
    const uint64_t weight1Stride = static_cast<uint64_t>(sharedN_) * static_cast<uint64_t>(k_) * sharedElemBytes;
    AscendC::Shape<int64_t, int64_t, int64_t, int64_t> sharedShape{
        static_cast<int64_t>(m_), static_cast<int64_t>(sharedN_), static_cast<int64_t>(k_), 0};

    GMMAddrInfo gmmAddrInfo;
    gmmAddrInfo.dispatchDoneCores = nullptr;
    gmmAddrInfo.dispatchTileReady = nullptr;
    gmmAddrInfo.dispatchReadyTarget = 0;
    gmmAddrInfo.flagGmm2Ready =
        reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.ptrSharedReadyFlag) + sharedExpertNum_ * INT_CACHELINE;
    // 无需任何输入就绪/CopyOut 门控标志：调用点在 PrepareExpertPipeline 的两道
    // 混核屏障之后，共享输入数据（阶段 1 写入）与 sub0 的 UB 释放均已对 AIC 可见。

    using Gmm1Context = MegaMoeCatlassImpl::Gmm1Context<QuantOutType, QuantOutType, bfloat16_t, QuantScaleOutType,
                                                        QuantScaleOutType, g_coreType == AscendC::AIC>;
    Gmm1Context gmm1Context(sharedN_, k_);
    uint32_t gmm1BlockIdx = 0U;
    // 独立的 cube/vector credit 台账：共享链在本 launch 里先于路由 GMM1 执行，
    // 此时 gmm1VecSetSyncCom_ 仍为 0，但共享链结束后路由链要继续用它——用局部
    // 台账把共享链的 credit 生命周期封闭在本函数内，与路由链互不串账。
    int32_t sharedVecSync = 0;
    for (uint32_t s = 0U; s < sharedExpertNum_; ++s) {
        gmmAddrInfo.aGlobal = UseBf16NoQuant ? params_.aGmAddr : params_.workspaceInfo.ptrSharedInputData;
        gmmAddrInfo.aScaleGlobal = UseBf16NoQuant ? nullptr : params_.workspaceInfo.ptrSharedInputScale;
        gmmAddrInfo.bGlobal = sharedWeight1GmAddr_ + static_cast<uint64_t>(s) * weight1Stride;
        gmmAddrInfo.bScaleGlobal =
            UseBf16NoQuant
                ? nullptr
                : sharedWeightScales1GmAddr_ + static_cast<uint64_t>(s) * static_cast<uint64_t>(sharedN_) * scaleK;
        // 每个共享专家一个 SwiGLU 完成标志行（AIC -> AIV）。
        gmmAddrInfo.flagSwigluToGmm2 =
            reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.ptrSharedReadyFlag) + s * INT_CACHELINE;
        if constexpr (g_coreType == AscendC::AIV) {
            // 首槽是本次调用的 activation 基偏移：MXFP8 下为字节偏移（data+scale
            // 分离区），BF16 下为元素偏移（BlockEpilogueSwigluBf16 按 bfloat16_t*
            // 寻址）；BF16 无 scale 区，次槽保持 0。
            AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> sharedVecOffset{
                static_cast<int64_t>(s) * static_cast<int64_t>(m_) * static_cast<int64_t>(sharedK_),
                UseBf16NoQuant
                    ? 0
                    : static_cast<int64_t>(s) * static_cast<int64_t>(m_) * static_cast<int64_t>(scaleSharedK),
                static_cast<int64_t>(s),
                0L,
                0L,
                0L};
            sharedEpilogueOp_.UpdateGlobalAddr(sharedVecOffset);
        }
        // 显式给全调度参数：共享链没有任何 dispatch tile（A 直接来自本卡量化结果），
        // 若沿用 SkipDispatchTileWait 的默认 false，AIC 会对空的 dispatchTileReady
        // 做逐 tile 行计数自旋，直接 271 硬错。PublishGmm1Ready=true 让 AIV 的
        // epilogue 发布就绪供 GMM2 等待；EnableResidentAPrefix 保持 false。
        // UseSharedTiming=true: 打点记入 AIC_SHARE_GMM1/AIV_SHARE_GMM1_EPILOGUE，
        // iter 传共享专家编号 s，与路由专家的 per-expert 行互不覆盖。
        MegaMoeCatlassImpl::GroupMatmulSwigluQuant<QuantOutType, QuantOutType, bfloat16_t, QuantScaleOutType,
                                                   QuantScaleOutType, g_coreType == AscendC::AIC, true, false, true,
                                                   true>(gmm1Context, sharedEpilogueOp_, sharedShape, gmmAddrInfo,
                                                         gmm1BlockIdx, sharedVecSync, timer, s);
    }
    // 账本跨链传递（而非本地排干）：共享链最后一个 AIV NotifyCube credit 不在
    // 此等待，而是并入路由链的 gmm1VecSetSyncCom_——路由 GMM1 的首个 tile 先
    // 发射 GM->L1/MMAD（不碰 UB），再在首个 CopyOut 前 WaitForVector 消费该
    // credit。于是本核的路由 GMM1 矩阵乘与配对 AIV 仍在执行的共享 SwiGLU 尾部
    // 并行；无共享 tile 的核 sharedVecSync==0，路由 GMM1 首个 tile 不等待直接
    // 开始。两种错误写法：在此排干（AIC 空等共享 epilogue 收尾，路由 GMM1 整
    // 体推迟一个尾段）；直接丢弃（泄漏 credit 被路由 tile1 提前消费，配对 AIV
    // 仍在读 tile0 时 UB 被复用）。路由专家全空的核由 ProcessGenericGmm2 末尾
    // 的无条件 EndSync 兜底排干，杜绝跨 launch 泄漏。
    if constexpr (g_coreType == AscendC::AIC) {
        gmm1VecSetSyncCom_ = sharedVecSync;
    }
    // 轮转游标续接：把共享链累计的调度相位（Σ共享 coreLoops mod blockNum，
    // AIC/AIV 两侧跑同一套调度算术、天然同值，fixpipe 交接的 tile 顺序保持
    // 一致）交给路由 GMM1。路由首个专家的 tile 从该相位起转——刚背过共享
    // tile 的前段核被排到本次轮转的尾部，tile 优先落到没跑共享专家的核；
    // 共享 tile 数 ≥ 核数时（每核都摊到共享活）任意相位等价，退化安全。
    startBlockIdx_ = gmm1BlockIdx;
}

// 共享专家 GMM2：仅 AIC，紧跟本核路由 GMM2（ProcessGenericGmm2）之后立即执行。
// GroupMatmul2 内部按 flagSwigluToGmm2 等待 AIV 的 epilogue（该标志在
// AllGatherData 排空后由共享 epilogue 更早发布，此处通常已就绪），
// gmm1ReadyTarget = 0 表示自动取该 GMM 的 tile 总数，因此这里无需额外同步。
// 结果写入 ptrSharedOutput，完成后由 GMM2 标志通知 Unpermute 累加。
// BF16 no-quant：QuantOutType = bfloat16_t 自动选择 BF16 pingpong block（无 MX scale
// 操作数），A/B 基址按 2 字节/元素步长推进。
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ProcessSharedGmm2()
{
    if (sharedExpertNum_ == 0U) {
        return;
    }
    if constexpr (g_coreType == AscendC::AIV) {
        return;  // GMM2 只在 AIC 上执行
    }
    const uint64_t scaleSharedK =
        Ops::Base::CeilDiv(static_cast<uint64_t>(sharedK_), static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) *
        MXFP_MULTI_BASE_SIZE;
    constexpr uint64_t sharedElemBytes = UseBf16NoQuant ? static_cast<uint64_t>(sizeof(bfloat16_t)) : 1U;
    const uint64_t weight2Stride = static_cast<uint64_t>(k_) * static_cast<uint64_t>(sharedK_) * sharedElemBytes;
    AscendC::Shape<int64_t, int64_t, int64_t, int64_t> sharedShape{
        static_cast<int64_t>(m_), static_cast<int64_t>(sharedN_), static_cast<int64_t>(k_), 0};

    GMMAddrInfo gmmAddrInfo;
    gmmAddrInfo.dispatchDoneCores = nullptr;
    gmmAddrInfo.dispatchTileReady = nullptr;
    gmmAddrInfo.dispatchReadyTarget = 0;

    using Gmm2Context =
        MegaMoeCatlassImpl::Gmm2Context<QuantOutType, QuantOutType, bfloat16_t, QuantScaleOutType, QuantScaleOutType>;
    // 不带任何 pending credit：generic 流水的 NotifyCube(GMM2_UB_READY_FLAG) 只在
    // dispatch 排空后发布一次、且已被路由 GMM2 的 context 消费；generic combine
    // (PublishAiv1UbOwnership=false) 不再发布。带 ubOwnershipPending 会在 Arm() 里
    // 死等一个永不出现的 credit。AIC 与 AIV 的会合无需屏障：共享 epilogue 在
    // AllGatherData 排空后即完成并发布 flagSwigluToGmm2，远早于路由 GMM2 结束。
    Gmm2Context gmm2Context(static_cast<uint32_t>(k_), sharedK_, false, false);
    GlobalTensor<bfloat16_t> sharedOutputGm;
    // 轮转游标提到循环外，并以路由 GMM2（ProcessGenericGmm2）留下的 startBlockIdx_
    // 为初值：S 个共享专家的 GMM2 tile 逐专家错开相位摊到不同核。此前游标在循环
    // 内每专家重置 0，S 个专家的 tile 反复压在同一批前置核上。
    uint32_t gmm2BlockIdx = startBlockIdx_;
    for (uint32_t s = 0U; s < sharedExpertNum_; ++s) {
        gmmAddrInfo.aGlobal =
            params_.workspaceInfo.ptrSharedActivation +
            static_cast<uint64_t>(s) * static_cast<uint64_t>(m_) * static_cast<uint64_t>(sharedK_) * sharedElemBytes;
        gmmAddrInfo.aScaleGlobal = UseBf16NoQuant
                                       ? nullptr
                                       : params_.workspaceInfo.ptrSharedActivationScale +
                                             static_cast<uint64_t>(s) * static_cast<uint64_t>(m_) * scaleSharedK;
        gmmAddrInfo.bGlobal = sharedWeight2GmAddr_ + static_cast<uint64_t>(s) * weight2Stride;
        gmmAddrInfo.bScaleGlobal =
            UseBf16NoQuant
                ? nullptr
                : sharedWeightScales2GmAddr_ + static_cast<uint64_t>(s) * static_cast<uint64_t>(k_) * scaleSharedK;
        gmmAddrInfo.outputGlobal =
            params_.workspaceInfo.ptrSharedOutput +
            static_cast<uint64_t>(s) * static_cast<uint64_t>(m_) * static_cast<uint64_t>(k_) * sizeof(bfloat16_t);
        // 该共享专家的 epilogue 就绪标志（GroupMatmul2 内部等待）。
        gmmAddrInfo.flagSwigluToGmm2 =
            reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.ptrSharedReadyFlag) + s * INT_CACHELINE;
        gmmAddrInfo.flagGmm2Ready = reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.ptrSharedReadyFlag) +
                                    (sharedExpertNum_ + s) * INT_CACHELINE;
        sharedOutputGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(gmmAddrInfo.outputGlobal));
        // EnableMGroupReady/WaitForGmm1Ready/EnableBf16L1Reuse 保持默认值
        // (false/true/false)；UseSharedTiming=true: 打点记入 AIC_SHARE_GMM2，
        // iter 传共享专家编号 s。
        MegaMoeCatlassImpl::GroupMatmul2<QuantOutType, QuantOutType, bfloat16_t, QuantScaleOutType, QuantScaleOutType,
                                         false, true, false, true>(gmm2Context, params_, sharedShape, gmmAddrInfo,
                                                                   gmm2BlockIdx, s, 0U, true, false, false,
                                                                   sharedOutputGm, timer);
        // 发布该共享专家的 GMM2 完成标志，供 Unpermute 累加等待。
        AscendC::SyncFunc<AscendC::HardEvent::FIX_S>();
        AscendC::AtomicAdd(reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.ptrSharedReadyFlag) +
                               (sharedExpertNum_ + s) * INT_CACHELINE,
                           int32_t(1));
    }
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::Process()
{
    int64_t oriOverflowMode = GetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>();
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(0);
    // 1.本卡数据量化 (BF16 no-quant 无需量化, x 在 FirstBuffInit 之后原样写入 a0)
    if constexpr (g_coreType == AscendC::AIV) {
        TIMER_BLOCK(timer.Tik());
        if constexpr (!UseBf16NoQuant) {
            TPipe quantPipe;
            Mxfp8DynamicQuant(&quantPipe);
            quantPipe.Destroy();
        }
        TIMER_BLOCK(timer.template Tok<Overwrite>(AscendTimer::AIV_QUANT));
        TIMER_BLOCK(timer.Tik());
        // 共享专家输入准备：放在这里是为了复用 firstPipe 创建之前的 UB 窗口——
        // 阶段 2 之后 firstPipe 的 buffer 一直存活，那时再新建 TPipe 会重复分配 UB。
        // 只写数据、不发布就绪标志：数据可见性由之后的混核屏障（SyncAll<true> 与
        // PrepareExpertPipeline 的两道屏障）向 AIC 的共享 GMM1 发布。
        if (sharedExpertNum_ > 0U) {
            TPipe sharedInputPipe;
            PrepareSharedExpertInput(&sharedInputPipe);
            sharedInputPipe.Destroy();
        }
    }
    // 2. 本卡ExpertIdx排序，得到ExpandedRowIdx、TokenPerExpert
    moe_init_routing_v3(params_.expertIdxGmAddr, params_.peermemInfo.offsetExpandedRowIdx + localExpandedRowIdxSt_,
                        params_.peermemInfo.ptrTokenPerExpert + localTokenPerExpertSt_,
                        params_.workspaceInfo.ptrInitRoutingWorkspace, &params_.tilingData->moeInitRoutingTilingData,
                        params_.tilingData->moeInitRoutingTilingData.tilingKey, rankId_, expertPerRank_);

    if constexpr (g_coreType == AscendC::AIV) {
        TIMER_BLOCK(timer.template Tok<Overwrite>(AscendTimer::AIV_INIT_ROUTING));
    }

    SyncAll<true>();

    // AIV_RESET_DATA: Tik 在 SyncAll 之后 (排除 routing 同步等待)。
    if constexpr (g_coreType == AscendC::AIV) {
        TIMER_BLOCK(timer.Tik());
    }

    // reset Flag、TokenPos、cumsum
    ResetData();

    if constexpr (g_coreType == AscendC::AIV) {
        TIMER_BLOCK(timer.template Tok<Overwrite>(AscendTimer::AIV_RESET_DATA));
    }

    // buffer申请 & workspace Init
    TPipe firstPipe;
    FirstBuffInit(&firstPipe);
    if constexpr (UseBf16NoQuant) {
        // Reuses the copy arena; must precede AllGatherData's publication.
        CopyBf16TokensToA0();
    }

    // 3.将本卡ExpertProbs、ExpandedRowIdx、TokenPerExpert广播到其他卡
    // AIV_ALLGATHER: Tik/Tok 紧贴 AllGatherData (排除 FirstBuffInit)。
    if constexpr (g_coreType == AscendC::AIV) {
        TIMER_BLOCK(timer.Tik());
    }
    AllGatherData();
    if constexpr (g_coreType == AscendC::AIV) {
        TIMER_BLOCK(timer.template Tok<Overwrite>(AscendTimer::AIV_ALLGATHER));
    }

    if (subBlockIdx_ == 0) {
        PipeBarrier<PIPE_ALL>();
        firstPipe.Destroy();
    }

    // 4. Dispatch/GMM1 使用同一专家顺序，内部 gmm1SwigluQuant 按基本块流水。
    GMMAddrInfo gmmAddrInfo;
    int32_t rankBaseRow = 0;
    // AIC_WAIT_ALLGATHER: Tik 紧贴 PrepareExpertPipeline (排除 AIC prologue)。
    if constexpr (g_coreType == AscendC::AIC) {
        TIMER_BLOCK(timer.Tik());
    }
    if constexpr (UseCoreWave) {
        rankBaseRow = PrepareExpertPipeline();
        ProcessCoreWavePipeline(&firstPipe, rankBaseRow);
    } else {
        using Gmm1Context = MegaMoeCatlassImpl::Gmm1Context<QuantOutType, QuantOutType, bfloat16_t, QuantScaleOutType,
                                                            QuantScaleOutType, g_coreType == AscendC::AIC>;
        rankBaseRow = PrepareExpertPipeline();
        // 3.8 共享专家 GMM1（generic 流水）：两道混核屏障已把 allgather 结果与
        // sub0 的 UB 暂存区释放发布给全核。AIC 在此启动共享 GMM1（矩阵乘 +
        // fixpipe CopyOut），sub0 就地执行 SwiGLU 量化 epilogue（发布
        // flagSwigluToGmm2 行 [0,S)），两者按 WaitForCube/NotifyCube 逐 tile 交接；
        // sub1 不参与共享链，随屏障放行直接进入 Dispatch——与 AIC 的共享 GMM1
        // 并发执行。共享链末尾 credit 并入 gmm1VecSetSyncCom_：路由 GMM1 首个
        // tile 先算后等，其 MMAD 与共享 SwiGLU 尾部并行；无共享 tile 的核账本
        // 为 0，路由 GMM1 不等待直接开始。共享链必须整体先于下面的路由
        // gmm1Context 构造：两者同为 Blaze 上下文，共用 MTE1_MTE2 族硬件
        // flag，同时存活会互相抹除已置位的 flag 状态（见
        // ProcessInterleavedGmmPipeline 处的 coexistence 注释）。
        if (sharedExpertNum_ > 0U) {
            // 尾部把共享链的轮转相位续接进 startBlockIdx_，路由 GMM1 从该相位
            // 起转（tile 优先落到没跑共享专家的核，见 ProcessSharedGmm1 注释）。
            ProcessSharedGmm1();
        } else {
            startBlockIdx_ = 0;
        }
        // 路由 gmm1Context 构造后移到屏障/共享链之后——上下文不共存约束换掉了
        // 原先 "hidden under the AIV all-gather" 的摆放；其构造开销落在共享链
        // 之后、首个路由专家之前，由 sub1 已在进行的 Dispatch 流水掩盖。
        Gmm1Context gmm1Context(Get<N_VALUE>(problemShape_), Get<K_VALUE>(problemShape_));
        preOffset_ = rankBaseRow;
        baseOffset_ = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        // Sparse-decode consumes GMM1 input only after every dispatch worker
        // has joined: one gate on the dispatch-complete publication replaces
        // all per-tile readiness spins.  AIV1 stores blockNum_ through
        // chainLenReadyFlagPtr_ after its final per-expert join (below), and
        // both sides derive the target from the same GetBlockNum() launch
        // constant - no AIV-private field is read here.
        if constexpr (UseSparseDecode) {
            if constexpr (g_coreType == AscendC::AIC) {
                uint32_t pollMisses = 0U;
                int64_t pollDelayCycles = 100;
                while (AscendC::ReadGmByPassDCache(chainLenReadyFlagPtr_) < static_cast<int32_t>(blockNum_)) {
                    GmSpinBackoffStep(pollMisses, pollDelayCycles);
                }
            }
        }
        for (uint32_t scheduleIdx = 0; scheduleIdx < expertPerRank_; scheduleIdx++) {
            uint32_t expertIdx = GetScheduledExpert(scheduleIdx);
            bool hasGroup =
                expertScheduleEnabled_ ? SetGroupParams(expertIdx, rankBaseRow) : UpdateIdentityGroupParams(expertIdx);
            if (!hasGroup) {
                continue;
            }
            UpdateGlobalBuffer(gmmAddrInfo, 0);
            if constexpr (g_coreType == AscendC::AIV) {
                if (subBlockIdx_ == 1) {
                    // Tik 紧贴 DispatchTokens; Tok 在其内部 (自旋之前), 见 :961。
                    TIMER_BLOCK(timer.Tik(AscendTimer::AIV_DISPATCH, expertIdx));
                    // The generic path never installs a runtime expert plan;
                    // preserve identity ownership and avoid per-expert phase
                    // arithmetic on this latency-oriented path.
                    DispatchTokens(gmmAddrInfo, expertIdx, false);
                }
            }
            MegaMoeCatlassImpl::GroupMatmulSwigluQuant<QuantOutType, QuantOutType, bfloat16_t, QuantScaleOutType,
                                                       QuantScaleOutType, g_coreType == AscendC::AIC, !UseCoreWave,
                                                       false, UseSparseDecode>(gmm1Context, epilogueOp_, problemShape_,
                                                                               gmmAddrInfo, startBlockIdx_,
                                                                               gmm1VecSetSyncCom_, timer, expertIdx);
        }
    }

    if constexpr (!UseCoreWave) {
        if constexpr (g_coreType == AscendC::AIV) {
            if (subBlockIdx_ == 1) {
                PipeBarrier<PIPE_ALL>();
                // Every Dispatch expert joins its active workers before the
                // next expert starts.  After the final traversal posLen is the
                // immutable local chain length, so publish one completion
                // store and reuse it instead of rescanning all routed rows on
                // AIV0 with per-token atomics.
                if (blockIdx_ == 0U) {
                    WriteGmByPassDCache(chainLenReadyFlagPtr_, static_cast<int32_t>(blockNum_));
                }
                firstPipe.Destroy();
                // Hand the shared AIV1 UB bank to GMM2 only after all generic
                // Dispatch copies have drained.
                NotifyCube(GMM2_UB_READY_FLAG);
            }
        }
    }

    if constexpr (!UseCoreWave) {
        // 5. GMM2 + Combine (与Dispatch/GMM1相同的expert顺序, 无per-expert SyncAll)
        //    AIC: generic路径保留完整expert ready；M-group ready仅在core-wave中启用。
        //    AIV: 单次调用CombineTokensParallel, 每个ready区间内按连续行分核处理token。
        //         每个任务等待对应ready flag和posDone[T]==k (本token前序步完成)
        //         逐expert消费无SyncAll，所有AIV核可并行流转，
        //         仅等待必要的GMM2 ready和本token前序依赖。
        preOffset_ = rankBaseRow;
        baseOffset_ = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        if constexpr (g_coreType == AscendC::AIC) {
            if constexpr (UseBf16NoQuant) {
                if (CanUseBf16Gmm2L1Reuse(Get<K_VALUE>(problemShape_), Get<N_VALUE>(problemShape_) / SWIGLU_N_HALF)) {
                    ProcessGenericGmm2<true>(rankBaseRow);
                } else {
                    ProcessGenericGmm2<false>(rankBaseRow);
                }
            } else {
                ProcessGenericGmm2<false>(rankBaseRow);
            }
            // 5.5 共享专家 GMM2：本核路由 GMM2 执行完毕后立即开始。A 输入就绪
            // （flagSwigluToGmm2 行 [0,S)）已由阶段 3.8 的共享 epilogue 更早发布，
            // GroupMatmul2 内部按 tile 计数等待；完成标志（行 [S,2S)）供阶段 7 的
            // Unpermute 累加消费。与 combine 并行运行的关系同路由 GMM2 尾部一致
            // （fixpipe 直写 GM 的 ptrSharedOutput，不占用 AIV 的 UB）。
            if (sharedExpertNum_ > 0U) {
                ProcessSharedGmm2();
            }
        }

        if constexpr (g_coreType == AscendC::AIV) {
            // AIV: 按与GMM2一致的expert顺序处理combine，每核负责连续行段，
            // 并等待 gmm2ReadyFlag[e] 和 posDone[T]==k。
            // AIV_COMBINE 在 CombineTokensParallel 内部 per-expert 打点。
            // Sparse-decode stays below one M-group wave and spreads its rows
            // across most local experts, so no (core, expert) segment reaches
            // the predecessor pipeline's amortization depth; its event/flag
            // choreography compiles out and the fallback SyncFunc protocol
            // (identical accumulation order) is used instead.
            MegaMoeCombineImpl::CombineTokensParallel<bfloat16_t, false, false, false, false, UseSparseDecode>(
                rankId_, combineAccumGmGlobal_, params_, expandedRowIdx_, distProbs_, tokenPerExpertWin_,
                expertTokenCumsum_, firstTokenPosPtr_, stepIdxPtr_, posLenPtr_, chainLenReadyFlagPtr_, posDonePtr_,
                reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.ptrGmm2ReadyFlag),
                reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.ptrExpertOrder), expertScheduleEnabled_, 0U,
                expertPerRank_, 0U, blockAivNum_, aivCoreIdx_, timer);
        }
    }

    // Join every Combine writer before clearing its routing metadata or
    // publishing a rank-ready signal. Paired UB credits and handshake-owner
    // completion alone do not cover the other AIVs' remote stores.
    if constexpr (g_coreType == AscendC::AIV) {
        SyncAll<true>();
    }

    // 共享专家三段已穿插进路由流水线（generic 流水）：GMM1+SwiGLU 在阶段 3.8
    // （屏障后、路由 GMM1 之前）与 sub1 的 Dispatch 并发，GMM2 在阶段 5.5 紧跟
    // 本核路由 GMM2，此处无共享专家工作。Unpermute（阶段 7）按 flagGmm2Ready
    // 行 [S,2S) 等待共享输出。

    // 7. unpermute & 清理win区上放置token的位置
    if constexpr (g_coreType == AIV) {
        // Only buffer setup/reset is timed here; synchronization stays outside.
        TIMER_BLOCK(timer.Tik());
        TPipe secondPipe;
        SecondBuffInit(&secondPipe);
        ResetTokenPerExpert();
        TIMER_BLOCK(timer.template Tok<Overwrite>(AscendTimer::AIV_POST_COMBINE));
        // The last AIV clears the shared histogram asynchronously. Join that
        // write before any handshake lane publishes rank-ready: a faster peer
        // may otherwise start its next call and have its new histogram slice
        // overwritten by this call's delayed reset.
        SyncAll<true>();
        CrossRankSyncInWorldSize();
        TIMER_BLOCK(timer.Tik());
        UnpermuteCombineDedup();
        TIMER_BLOCK(timer.template Tok<Overwrite>(AscendTimer::AIV_UNPERMUTE));
        PipeBarrier<PIPE_ALL>();
        secondPipe.Destroy();
    }
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(oriOverflowMode);

    TIMER_BLOCK(timerAll.template Tok<Overwrite>());
    TIMER_BLOCK(if (timer.GetTAddr() != nullptr) { AscendC::SyncAll<false>(); });
}

}  // namespace MegaMoeImpl
#endif
