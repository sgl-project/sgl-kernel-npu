/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: FusedDeepMoe operator kernel function header file, for a3
 * Create: 2025-07-19
 * Note:
 * History: 2025-07-19 create FusedDeepMoe operator kernel function header file, for a3
 */
#ifndef FUSED_DEEP_MOE_H
#define FUSED_DEEP_MOE_H

#include "lib/matmul_intf.h"
#include <kernel_operator.h>

#include "catlass_a5/catlass.hpp"
#include "catlass_a5/arch/arch.hpp"
#include "fused_deep_moe/gemm/block/block_mmad.h"
#include "catlass_a5/gemm/block/block_swizzle.hpp"
#include "fused_deep_moe/gemm/dispatch_policy.h"
#include "fused_deep_moe/gemm/kernel/dispatch_mx_gmm1_swiglu.h"
#include "fused_deep_moe/gemm/kernel/mx_gmm2_cast_combine.h"
#include "catlass_a5/gemm/gemm_type.hpp"
#include "catlass_a5/layout/layout.hpp"
#include "catlass_a5/status.hpp"
#include "catlass_a5/gemm/device/device_gemm.hpp"
#include "tla_a5/layout.hpp"
#include "fused_deep_moe/epilogue/block/block_epilogue.h"
#include "fused_deep_moe/epilogue/dispatch_policy.h"

#include "fused_deep_moe_a5_tiling.h"
#include "fused_deep_moe/raw_distributed/cam_moe_distribute_dispatch.h"
#include "fused_deep_moe_a5_base.h"

using namespace Cam;
using namespace Catlass;
using namespace tla;

using ElementC = float;
using ElementMxScale = fp8_e8m0_t;
using ElementGroupList = int64_t;

using Gmm1L1TileShape = Shape<Int<GMM1_L1M>,Int<GMM1_L1N>,Int<GMM1_L1K>>;
using Gmm1L0TileShape = Shape<Int<GMM1_L1M>,Int<GMM1_L1N>,Int<GMM1_L0K>>;
using Gmm1EpilogueTileShape = MatrixShape<GMM1_EPIM, GMM1_L1N>;
using Gmm1BlockScheduler = typename Catlass::Gemm::Block::GemmIdentityBlockSwizzle<GMM1_SWIZZLE_OFFSET, GMM1_SWIZZLE_DIRECTION>;

using Gmm2L1TileShape = Shape<Int<GMM2_L1M>,Int<GMM2_L1N>,Int<GMM2_L1K>>;
using Gmm2L0TileShape = Shape<Int<GMM2_L1M>,Int<GMM2_L1N>,Int<GMM2_L0K>>;
using Gmm2EpilogueTileShape = MatrixShape<GMM2_EPIM, GMM2_L1N>;
using Gmm2BlockScheduler = typename Catlass::Gemm::Block::GemmIdentityBlockSwizzle<GMM2_SWIZZLE_OFFSET, GMM2_SWIZZLE_DIRECTION>;


// AIV Dispatch:    X -> A
// AIC GMM/MM:      A -> Swap
// AIV Epilogue:    Swap -> Swiglu
// AIV Quantzie:    Swiglu -> D
// 1. Swap use the same space with Swiglu, when fused memory
// 2. Epilogue just do silu for the left
// 3. n of routed and shared experts can be different 
template<TemplateMC2TypeClass, class ElementA, class ElementB, class L1TileShape, class L0TileShape, class EpilogueTileShape, class BlockScheduler, bool transB = false>
CATLASS_DEVICE void DispatchMxGmm1SwigluQuantFunc(
    // routed expert, grouped matmul
    Catlass::GemmCoord routedProblemShape, uint32_t problemCount, GM_ADDR gmGroupList,
    GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmAScale, GM_ADDR gmBScale, GM_ADDR gmSwapSpace, GM_ADDR gmSwigluOut, GM_ADDR gmD, GM_ADDR gmDScale,
    // shared expert, matmul
    Catlass::GemmCoord sharedProblemShape,
    GM_ADDR gmShareA, GM_ADDR gmShareB, GM_ADDR gmShareAScale, GM_ADDR gmShareBScale, GM_ADDR gmShareSwapSpace, GM_ADDR gmShareSwigluOut, GM_ADDR gmShareD, GM_ADDR gmShareDScale,
    // dispatch and quant, when EXEC_FLAG_DEEP_FUSE. 
    GM_ADDR gmX, GM_ADDR gmExpertIds, GM_ADDR xActiveMask, GM_ADDR gmMoeSmoothScales, GM_ADDR gmShareSmoothScales, GM_ADDR gmExpandIdx, GM_ADDR gmEpSendCount, GM_ADDR gmExpertTokenNums,
    const FusedDeepMoeInfo &fusedDeepMoeInfo)
{
    static_assert((std::is_same_v<ElementA, float8_e5m2_t> ||
        std::is_same_v<ElementA, float8_e4m3_t> || std::is_same_v<ElementA, float4_e2m1x2_t> ||
        std::is_same_v<ElementA, float4_e1m2x2_t>) &&
        (std::is_same_v<ElementB, float8_e5m2_t> || std::is_same_v<ElementB, float8_e4m3_t> ||
            std::is_same_v<ElementB, float4_e2m1x2_t> || std::is_same_v<ElementB, float4_e1m2x2_t>) &&
        std::is_same_v<ElementMxScale, float8_e8m0_t>,
        "ElementA and ElementB must be float8_e5m2_t, float8_e4m3_t, float4_e2m1x2_t, or float4_e1m2x2_t, ElementMxScale must be float8_e8m0_t");

    uint32_t m = routedProblemShape.m();
    uint32_t n = routedProblemShape.n();
    uint32_t shareN = sharedProblemShape.n();
    uint32_t k = routedProblemShape.k();
    uint32_t groupCount = problemCount;
    uint32_t mxScaleK = CeilDiv<Catlass::MX_SCALE_GROUP_NUM>(k);

    using LayoutTagA = Catlass::layout::RowMajor;
    using LayoutTagB = std::conditional_t<transB, Catlass::layout::ColumnMajor, std::conditional_t<WEIGHT_NZ, Catlass::layout::zN, Catlass::layout::RowMajor>>;
    using LayoutTagMxScale = Catlass::layout::RowMajor;
    using LayoutTagC = Catlass::layout::RowMajor;

    using ArchTag = Catlass::Arch::Ascend950;
    constexpr bool enableUnitFlag = true;
    using DispatchPolicy = Catlass::Gemm::MmadMxWithCallback<ArchTag, enableUnitFlag>;

    auto layoutA = tla::MakeLayout<ElementA, LayoutTagA>(m, k);
    auto layoutB = tla::MakeLayout<ElementB, LayoutTagB>(k, n);
    auto layoutShareB = tla::MakeLayout<ElementB, LayoutTagB>(k, shareN);
    auto layoutMxScaleA = tla::MakeMxScaleLayout<ElementMxScale, LayoutTagA, false>(m, mxScaleK);
    auto layoutMxScaleB = tla::MakeMxScaleLayout<ElementMxScale, LayoutTagMxScale, true>(mxScaleK, n);
    auto layoutShareMxScaleB = tla::MakeMxScaleLayout<ElementMxScale, LayoutTagMxScale, true>(mxScaleK, shareN);
    auto layoutC = tla::MakeLayout<ElementC, LayoutTagC>(m, n);
    auto layoutShareC = tla::MakeLayout<ElementC, LayoutTagC>(m, shareN);

    using TileCopy = Catlass::Gemm::Tile::PackedMxTileCopyTla<
        ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementMxScale, decltype(layoutMxScaleA), ElementMxScale,
        decltype(layoutMxScaleB), ElementC, LayoutTagC, void>;
    using BlockMmad = Catlass::Gemm::Block::BlockMmadTla<
        DispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, void, TileCopy>;
    using EpilogueDispatchPolicy = Catlass::Epilogue::EpilogueAtlasA5SiluHalf<1>;
    // using EpilogueTileShape = MatrixShape<64, 256>;
    using BlockEpilogue = Epilogue::Block::BlockEpilogue<EpilogueDispatchPolicy, ElementC, ExpandXType, ElementC, EpilogueTileShape>;

    // using BlockScheduler = typename Catlass::Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
    // kernel level
    using MatmulKernel = Catlass::Gemm::Kernel::DispatchMxGmm1Swiglu<TemplateMC2TypeFunc, BlockMmad,
            BlockEpilogue, BlockScheduler, ElementGroupList>;
    typename MatmulKernel::Params params{routedProblemShape, groupCount, gmGroupList,
        gmA, layoutA, gmB, layoutB, gmAScale, layoutMxScaleA, gmBScale, layoutMxScaleB, gmSwapSpace/*ptrC*/, layoutC, gmSwigluOut, gmD, gmDScale,
        sharedProblemShape,
        gmShareA, gmShareB, layoutShareB, gmShareAScale, gmShareBScale, layoutShareMxScaleB, gmShareSwapSpace/*ptrShareC*/, layoutShareC, gmShareSwigluOut, gmShareD, gmShareDScale,
        gmX, gmExpertIds, xActiveMask, gmMoeSmoothScales, gmShareSmoothScales, gmExpandIdx, gmEpSendCount, gmExpertTokenNums,
        fusedDeepMoeInfo
    };

    MatmulKernel kernel;
    kernel(params);
}

// AIC GMM/MM:          A -> Swap
// AIV Epilogue:        Swap -> D / IPC
// AIV Combine Recv:    IPC -> output
template<TemplateMC2TypeClass, class ElementA, class ElementB, class L1TileShape, class L0TileShape, class EpilogueTileShape, class BlockScheduler, bool transB = false>
CATLASS_DEVICE void MxGmm2CastCombineFunc(
    // routed expert, grouped matmul
    Catlass::GemmCoord routedProblemShape, uint32_t problemCount, GM_ADDR gmGroupList,
    GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmAScale, GM_ADDR gmBScale, GM_ADDR gmSwapSpace, GM_ADDR gmD,
    // shared expert, matmul
    Catlass::GemmCoord sharedProblemShape,
    GM_ADDR gmShareA, GM_ADDR gmShareB, GM_ADDR gmShareAScale, GM_ADDR gmShareBScale, GM_ADDR gmShareSwapSpace, GM_ADDR gmShareD,
    void *combiner)
{
    static_assert((std::is_same_v<ElementA, float8_e5m2_t> ||
        std::is_same_v<ElementA, float8_e4m3_t> || std::is_same_v<ElementA, float4_e2m1x2_t> ||
        std::is_same_v<ElementA, float4_e1m2x2_t>) &&
        (std::is_same_v<ElementB, float8_e5m2_t> || std::is_same_v<ElementB, float8_e4m3_t> ||
            std::is_same_v<ElementB, float4_e2m1x2_t> || std::is_same_v<ElementB, float4_e1m2x2_t>) &&
        std::is_same_v<ElementMxScale, float8_e8m0_t>,
        "ElementA and ElementB must be float8_e5m2_t, float8_e4m3_t, float4_e2m1x2_t, or float4_e1m2x2_t, ElementMxScale must be float8_e8m0_t");

    uint32_t m = routedProblemShape.m();
    uint32_t n = routedProblemShape.n();
    uint32_t k = routedProblemShape.k();
    uint32_t shareK = sharedProblemShape.k();
    uint32_t groupCount = problemCount;
    uint32_t mxScaleK = CeilDiv<Catlass::MX_SCALE_GROUP_NUM>(k);
    uint32_t mxShareScaleK = CeilDiv<Catlass::MX_SCALE_GROUP_NUM>(shareK);

    using LayoutTagA = Catlass::layout::RowMajor;
    using LayoutTagB = std::conditional_t<transB, Catlass::layout::ColumnMajor, std::conditional_t<WEIGHT_NZ, Catlass::layout::zN, Catlass::layout::RowMajor>>;
    using LayoutTagMxScale = Catlass::layout::RowMajor;
    using LayoutTagC = Catlass::layout::RowMajor;

    using ArchTag = Catlass::Arch::Ascend950;
    constexpr bool enableUnitFlag = true;
    using DispatchPolicy = Catlass::Gemm::MmadMxWithCallback<ArchTag, enableUnitFlag>;

    auto layoutA = tla::MakeLayout<ElementA, LayoutTagA>(m, k);
    auto layoutShareA = tla::MakeLayout<ElementA, LayoutTagA>(m, shareK);
    auto layoutB = tla::MakeLayout<ElementB, LayoutTagB>(k, n);
    auto layoutShareB = tla::MakeLayout<ElementB, LayoutTagB>(shareK, n);
    auto layoutMxScaleA = tla::MakeMxScaleLayout<ElementMxScale, LayoutTagA, false>(m, mxScaleK);
    auto layoutShareMxScaleA = tla::MakeMxScaleLayout<ElementMxScale, LayoutTagA, false>(m, mxShareScaleK);
    auto layoutMxScaleB = tla::MakeMxScaleLayout<ElementMxScale, LayoutTagMxScale, true>(mxScaleK, n);
    auto layoutShareMxScaleB = tla::MakeMxScaleLayout<ElementMxScale, LayoutTagMxScale, true>(mxShareScaleK, n);
    auto layoutC = tla::MakeLayout<ElementC, LayoutTagC>(m, n);

    using TileCopy = Catlass::Gemm::Tile::PackedMxTileCopyTla<
        ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementMxScale, decltype(layoutMxScaleA), ElementMxScale,
        decltype(layoutMxScaleB), ElementC, LayoutTagC, void>;
    using BlockMmad = Catlass::Gemm::Block::BlockMmadTla<
        DispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, void, TileCopy>;
    // using BlockEpilogue = void;
    using EpilogueDispatchPolicy = Catlass::Epilogue::EpilogueAtlasA5CastCombine<EXEC_FLAG>;
    // using EpilogueTileShape = MatrixShape<64, 256>;
    using BlockEpilogue = Epilogue::Block::BlockEpilogue<EpilogueDispatchPolicy, ElementC, ExpandXType, EpilogueTileShape>;

    // using BlockScheduler = typename Catlass::Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
    // kernel level
    using MatmulKernel = Catlass::Gemm::Kernel::MxGmm2CastCombine<TemplateMC2TypeFunc, BlockMmad,
            BlockEpilogue, BlockScheduler, ElementGroupList>;
    typename MatmulKernel::Params params{routedProblemShape, groupCount, gmGroupList,
        gmA, layoutA, gmB, layoutB, gmAScale, layoutMxScaleA, gmBScale, layoutMxScaleB, gmSwapSpace/*ptrC*/, layoutC, gmD,
        sharedProblemShape,
        gmShareA, layoutShareA, gmShareB, layoutShareB, gmShareAScale, layoutShareMxScaleA, gmShareBScale, layoutShareMxScaleB, gmShareSwapSpace/*ptrShareC*/, gmShareD,
        combiner
    };

    MatmulKernel kernel;
    kernel(params);
}


template <TemplateMC2TypeClass>
class FusedDeepMoe {
public:
    __aicore__ inline FusedDeepMoe(){};
    __aicore__ inline void Init(
        // input
        GM_ADDR x, GM_ADDR expert_ids, GM_ADDR gmm1_weight, GM_ADDR gmm1_weight_scale,
        GM_ADDR gmm2_weight, GM_ADDR gmm2_weight_scale, GM_ADDR expert_scales,
        GM_ADDR share_gmm1_weight, GM_ADDR share_gmm1_weight_scale,
        GM_ADDR share_gmm2_weight, GM_ADDR share_gmm2_weight_scale,
        GM_ADDR expert_smooth_scales, GM_ADDR share_smooth_scales, GM_ADDR x_active_mask,
        // output
        GM_ADDR output, GM_ADDR share_output, GM_ADDR expertTokenNums,
        // system
        GM_ADDR workspaceGM, AscendC::TPipe *pipe, const FusedDeepMoeTilingData *tilingData);
    __aicore__ inline void Process();

private:
    GM_ADDR gmX_;
    GM_ADDR gmexpertIds_;
    GM_ADDR gmWeight1_;
    GM_ADDR gmScale1_;
    GM_ADDR gmWeight2_;
    GM_ADDR gmScale2_;
    GM_ADDR gmOutput_;

    GM_ADDR gmShareWeight1_;
    GM_ADDR gmShareWeight1Scale_;
    GM_ADDR gmShareWeight2_;
    GM_ADDR gmShareWeight2Scale_;
    GM_ADDR gmShareOutput_;
    GM_ADDR gmExpertTokenNums_;
    GM_ADDR workspaceGM_;
    GM_ADDR gmSmoothScales_;
    GM_ADDR gmShareSmoothScales_;
    GM_ADDR gmexpertScales_;
    GM_ADDR xActiveMask_;

    uint32_t maxTokenNum_{0};
    uint32_t shareGmm1OutputDim_{0};
    uint32_t gmm1OutputDim_{0};
    uint32_t tokenHiddenSize_{0};
    uint32_t groupCount_{0};
    uint32_t gmm2OutputDim_{0};
    uint32_t shareGmm2InputDim_{0};
    uint32_t gmm2InputDim_{0};
    uint32_t globalRankId_{0};
    uint32_t winSizePerRank_{0};
    uint32_t blockDim_{0};
    uint32_t epRankSize_{0};
    uint32_t moeExpertNumPerRank_{0};
    uint32_t globalBs_{0};
    uint32_t bs_{0};
    uint32_t maxBs_{0};
    uint32_t topK_{0};

    AscendC::TPipe *tpipe_{nullptr};
    const FusedDeepMoeTilingData *tilingData_;
};

template <TemplateMC2TypeClass>
__aicore__ inline void FusedDeepMoe<TemplateMC2TypeFunc>::Init(
    // input
    GM_ADDR x, GM_ADDR expert_ids, GM_ADDR gmm1_weight, GM_ADDR gmm1_weight_scale,
    GM_ADDR gmm2_weight, GM_ADDR gmm2_weight_scale, GM_ADDR expert_scales,
    GM_ADDR share_gmm1_weight, GM_ADDR share_gmm1_weight_scale,
    GM_ADDR share_gmm2_weight, GM_ADDR share_gmm2_weight_scale,
    GM_ADDR expert_smooth_scales, GM_ADDR share_smooth_scales, GM_ADDR x_active_mask,
    // output
    GM_ADDR output, GM_ADDR share_output, GM_ADDR expertTokenNums,
    // system
    GM_ADDR workspaceGM, AscendC::TPipe *pipe, const FusedDeepMoeTilingData *tilingData)
{
    tpipe_ = pipe;
    blockDim_ = AscendC::GetBlockNum();

    gmSmoothScales_ = expert_smooth_scales;
    gmShareSmoothScales_ = share_smooth_scales;
    gmX_ = x;
    gmexpertIds_ = expert_ids;
    gmWeight1_ = gmm1_weight;
    gmScale1_ = gmm1_weight_scale;
    gmWeight2_ = gmm2_weight;
    gmScale2_ = gmm2_weight_scale;
    gmOutput_ = output;
    gmShareWeight1_ = share_gmm1_weight;
    gmShareWeight1Scale_ = share_gmm1_weight_scale;
    gmShareWeight2_ = share_gmm2_weight;
    gmShareWeight2Scale_ = share_gmm2_weight_scale;
    gmShareOutput_ = share_output;
    gmExpertTokenNums_ = expertTokenNums;
#ifdef DEBUG_SPACE
    workspaceGM_ = share_smooth_scales;
#else
    workspaceGM_ = workspaceGM;
#endif
    gmexpertScales_ = expert_scales;
    xActiveMask_ = x_active_mask;
    tilingData_ = tilingData;
    epRankSize_ = tilingData->fusedDeepMoeInfo.epRankSize;
    moeExpertNumPerRank_ = tilingData->fusedDeepMoeInfo.moeExpertNumPerRank;
    globalBs_ = tilingData->fusedDeepMoeInfo.globalBs;
    bs_ = tilingData->fusedDeepMoeInfo.bs;
    topK_ = tilingData->fusedDeepMoeInfo.k;
    maxBs_ = globalBs_ / epRankSize_;

    maxTokenNum_ = maxBs_ * epRankSize_ * (topK_ < moeExpertNumPerRank_ ? topK_ : moeExpertNumPerRank_);
    shareGmm1OutputDim_ = tilingData->fusedDeepMoeInfo.shareGmm1HLen;
    gmm1OutputDim_ = tilingData->fusedDeepMoeInfo.gmm1HLen;
    tokenHiddenSize_ = tilingData->fusedDeepMoeInfo.h;
    groupCount_ = tilingData->fusedDeepMoeInfo.moeExpertNumPerRank;
    gmm2OutputDim_ = tokenHiddenSize_;
    shareGmm2InputDim_ = shareGmm1OutputDim_ / 2;
    gmm2InputDim_ = gmm1OutputDim_ / 2;
}

template <TemplateMC2TypeClass>
__aicore__ inline void FusedDeepMoe<TemplateMC2TypeFunc>::Process()
{
    using ElementA = WeightType;
    using ElementB = WeightType;
    GemmCoord gmm1ProblemShape{maxTokenNum_, gmm1OutputDim_, tokenHiddenSize_};
    GemmCoord shareGmm1ProblemShape{bs_, shareGmm1OutputDim_, tokenHiddenSize_};

    GemmCoord gmm2ProblemShape{maxTokenNum_, gmm2OutputDim_, gmm2InputDim_};
    GemmCoord shareGmm2ProblemShape{bs_, gmm2OutputDim_, shareGmm2InputDim_};

    // Use precomputed workspace offsets from tilingData, no need to recompute here
    GM_ADDR gmShareX1 = workspaceGM_ + tilingData_->workSpaceOffset.shareX1TokenOffset;
    GM_ADDR gmShareX1Scale = workspaceGM_ + tilingData_->workSpaceOffset.shareX1ScaleOffset;
    GM_ADDR gmShareMm1SwapSpace = workspaceGM_ + tilingData_->workSpaceOffset.shareMm1SwapSpaceOffset;
    GM_ADDR gmShareSwigluOut = workspaceGM_ + tilingData_->workSpaceOffset.shareSwigluOffset;
    GM_ADDR gmShareX2 = workspaceGM_ + tilingData_->workSpaceOffset.shareX2TokenOffset;
    GM_ADDR gmShareX2Scale = workspaceGM_ + tilingData_->workSpaceOffset.shareX2ScaleOffset;
    GM_ADDR gmShareMm2SwapSpace = workspaceGM_ + tilingData_->workSpaceOffset.shareMm2SwapSpaceOffset;

    GM_ADDR gmX1 = workspaceGM_ + tilingData_->workSpaceOffset.x1TokenOffset;
    GM_ADDR gmX1Scale = workspaceGM_ + tilingData_->workSpaceOffset.x1ScaleOffset;
    GM_ADDR gmGmm1SwapSpace = workspaceGM_ + tilingData_->workSpaceOffset.gmm1SwapSpaceOffset;
    GM_ADDR gmSwigluOut = workspaceGM_ + tilingData_->workSpaceOffset.swigluOffset;
    GM_ADDR gmX2 = workspaceGM_ + tilingData_->workSpaceOffset.x2TokenOffset;
    GM_ADDR gmX2Scale = workspaceGM_ + tilingData_->workSpaceOffset.x2ScaleOffset;
    GM_ADDR gmGmm2SwapSpace = workspaceGM_ + tilingData_->workSpaceOffset.gmm2SwapSpaceOffset;

    GM_ADDR gmGmm2DepOut = workspaceGM_ + tilingData_->workSpaceOffset.y2TokenOffset;
    GM_ADDR gmGroupList = workspaceGM_ + tilingData_->workSpaceOffset.groupListOffset;
    GM_ADDR gmExpandIdx = workspaceGM_ + tilingData_->workSpaceOffset.expandIdxOffset;
    GM_ADDR gmEpSendCount = workspaceGM_ + tilingData_->workSpaceOffset.epSendCountOffset;
    GM_ADDR gmReserved = workspaceGM_ + tilingData_->workSpaceOffset.reservedOffset;

    if constexpr ((EXEC_FLAG & EXEC_FLAG_DEEP_FUSE) == 0) {
        if constexpr (g_coreType == AscendC::AIV) {
            AscendC::TPipe tpipe;
            MoeDistributeDispatchImpl::CamMoeDistributeDispatch<ExpandXType, int8_t, false, true,
                                static_cast<bool>(EXEC_FLAG & EXEC_FLAG_SMOOTH_QUANT), false, EXEC_FLAG> dispatcher;
            dispatcher.Init(gmX_, gmexpertIds_, gmSmoothScales_, gmShareSmoothScales_, xActiveMask_, gmShareX1, gmX1,
                            gmShareX1Scale, gmX1Scale, gmExpandIdx, gmGroupList, gmEpSendCount, gmExpertTokenNums_,
                            nullptr, nullptr, &tpipe, tilingData_);
            dispatcher.Process();
            tpipe.Destroy();
            icache_preload(8);
        }

        AscendC::PipeBarrier<PIPE_ALL>();
        Arch::CrossCoreFlag gmm1AivFinished{0};
        if constexpr (g_coreType == AscendC::AIV) {
            Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
            Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(gmm1AivFinished);
        } else {
            Arch::CrossCoreWaitFlag(gmm1AivFinished);
        }
    }
    DispatchMxGmm1SwigluQuantFunc<TemplateMC2TypeFunc, ElementA, ElementB, Gmm1L1TileShape, Gmm1L0TileShape, Gmm1EpilogueTileShape, Gmm1BlockScheduler>(
        gmm1ProblemShape, groupCount_, gmGroupList,
        gmX1, gmWeight1_, gmX1Scale, gmScale1_, gmGmm1SwapSpace, gmSwigluOut, gmX2, gmX2Scale,
        shareGmm1ProblemShape,
        gmShareX1, gmShareWeight1_, gmShareX1Scale, gmShareWeight1Scale_, gmShareMm1SwapSpace, gmShareSwigluOut, gmShareX2, gmShareX2Scale,
        gmX_, gmexpertIds_, xActiveMask_, gmSmoothScales_, gmShareSmoothScales_, gmExpandIdx, gmEpSendCount, gmExpertTokenNums_,
        tilingData_->fusedDeepMoeInfo);
    AscendC::PipeBarrier<PIPE_ALL>();
    Arch::CrossCoreFlag gmm1AivFinished{0};
    if constexpr (g_coreType == AscendC::AIV) {
        Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
        Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(gmm1AivFinished);
    } else {
        Arch::CrossCoreWaitFlag(gmm1AivFinished);
    }
    MoeDistributeCombineImpl::CamMoeDistributeCombine<TemplateMC2TypeFunc> combiner;
    if (g_coreType == AscendC::AIV) {
        combiner.Init(gmGmm2DepOut, gmexpertIds_, gmExpandIdx, gmEpSendCount, nullptr, gmexpertScales_, xActiveMask_,
                      gmOutput_, nullptr, nullptr, tilingData_);
    }
    MxGmm2CastCombineFunc<TemplateMC2TypeFunc, ElementA, ElementB, Gmm2L1TileShape, Gmm2L0TileShape, Gmm2EpilogueTileShape, Gmm2BlockScheduler>(
        gmm2ProblemShape, groupCount_, gmGroupList,
        gmX2, gmWeight2_, gmX2Scale, gmScale2_, gmGmm2SwapSpace, gmGmm2DepOut,
        shareGmm2ProblemShape,
        gmShareX2, gmShareWeight2_, gmShareX2Scale, gmShareWeight2Scale_, gmShareMm2SwapSpace, gmShareOutput_,
        &combiner);
}
#endif  // FUSED_DEEP_MOE_H
