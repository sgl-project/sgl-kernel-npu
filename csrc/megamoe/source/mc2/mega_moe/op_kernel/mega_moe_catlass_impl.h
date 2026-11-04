/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 */
#ifndef MEGA_MOE_CATLASS_IMPL_H
#define MEGA_MOE_CATLASS_IMPL_H

#include "kernel_operator.h"

#include "kernel_tiling/kernel_tiling.h"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"
#include "catlass/gemm/kernel/mx_matmul_tla.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "kernel_operator_list_tensor_intf.h"
#include "lib/matmul_intf.h"
#include "tensor_api/tensor.h"
#include "block_mmad_moe.h"
#include "block_mmad_bf16_moe.h"
#include "block_epilogue_swiglu_mx_quant.h"
#include "block_epilogue_swiglu_bf16.h"
#include "mega_moe_base.h"
#include "tile_copy.hpp"
#include "mega_moe_combine_send.h"
#include "AscendTimerV2.hpp"
#include "AscendTimerV2_device.hpp"

using namespace AscendC;
using namespace Catlass;

namespace MegaMoeCatlassImpl {

using ArchTag = Arch::Ascend950;
constexpr static bool enableUnitFlag = true;
using DispatchPolicy = Gemm::MmadMx<ArchTag, enableUnitFlag>;
// The generic resident path keeps three K=256 A payload stages while sweeping
// the output-N stripes of one M-group.
constexpr uint32_t RESIDENT_A_L1_K_STAGES = 3U;
using ResidentADispatchPolicy =
    Gemm::MmadMx<ArchTag, enableUnitFlag, 16U, 1U, true, RESIDENT_A_L1_K_STAGES, 2U, 2U, 2U>;
// K16 GMM2 has K=1536: six K=256 stages cover the complete 256-row A tile.
// A scale factor of one avoids reserving a 32-KiB scale arena per stage, so
// six A stages plus one B stage occupy 462 KiB and fit Ascend950's 512-KiB L1.
// The one-stage B specialization refills its released slot while the current
// MMAD runs, retaining GM->L1/compute overlap without evicting resident A.
using FullAResidentDispatchPolicy = Gemm::MmadMx<ArchTag, enableUnitFlag, 1U, 1U, true, 6U, 1U, 2U, 2U>;
constexpr uint32_t FULL_A_L1_K_STAGES = GMM2_FULL_A_K_STAGES;
using L1TileShape = tla::Shape<tla::Int<256>, tla::Int<256>, tla::Int<256>>;
using L0TileShape = tla::Shape<tla::Int<256>, tla::Int<256>, tla::Int<128>>;
// BF16 (no-quant) elements are twice the FP8 width. The L1 256^3 tile still
// fits two A + two B stages (4 x 128 KiB = 512 KiB); the L0 K tile shrinks
// to 64 so two ping-pong stages fit the 64-KiB L0A/L0B buffers.
using L0TileShapeBf16 = tla::Shape<tla::Int<256>, tla::Int<256>, tla::Int<64>>;
constexpr uint32_t GMM_L0_TILE_K = 128U;
constexpr uint32_t GMM_L0_TILE_K_BF16 = 64U;
using Bf16DispatchPolicy = Gemm::MmadPingpong<ArchTag, enableUnitFlag>;
using Bf16ResidentDispatchPolicy = Gemm::MmadPingpong<ArchTag, enableUnitFlag, false, 1U, true>;
using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;
constexpr uint32_t ALIGN32 = 32U;
constexpr uint32_t L1_TILE_M = 256;
constexpr uint32_t L1_TILE_N = 256;
constexpr uint32_t L1_TILE_K = 256;
static_assert(L1_TILE_M == EXPERT_SCHEDULE_M_TILE_ROWS && L1_TILE_N == GMM2_READY_N_TILE_COLUMNS &&
                  L1_TILE_K == BF16_GMM2_L1_K_COLUMNS &&
                  Bf16ResidentDispatchPolicy::L1A_STAGES == BF16_GMM2_L1_A_STAGES,
              "BF16 reuse gates must match the resident block's actual L1 geometry");
constexpr uint32_t FULL_A_L1_DATA_STAGE_BYTES = L1_TILE_M * L1_TILE_K;
constexpr uint32_t FULL_A_L1_SCALE_STAGE_BYTES = L1_TILE_M * L1_TILE_K / Catlass::MX_SCALE_GROUP_NUM;
constexpr uint32_t FULL_A_L1_BYTES = FULL_A_L1_K_STAGES * (FULL_A_L1_DATA_STAGE_BYTES + FULL_A_L1_SCALE_STAGE_BYTES) +
                                     (FULL_A_L1_DATA_STAGE_BYTES + FULL_A_L1_SCALE_STAGE_BYTES);
static_assert(FULL_A_L1_BYTES == 462U * 1024U, "K16 A6+B1 MX arena must occupy exactly 462 KiB");
static_assert(FULL_A_L1_BYTES <= ArchTag::L1_SIZE, "K16 Full-A MX arena must fit Ascend950 L1");
static_assert(L1_TILE_M == EXPERT_SCHEDULE_M_TILE_ROWS, "Dispatch tile-ready rows must match the GMM1 M tile");
static_assert(L1_TILE_N == GMM2_READY_N_TILE_COLUMNS, "GMM2 M-group ready votes must match the GMM2 N tile");
constexpr uint32_t SWIGLU_N_HALF = 2;
constexpr uint32_t MAX_SINGLE_MN = 256 * 256;
constexpr uint32_t BUFALIGN = (MAX_SINGLE_MN + 31U) / ALIGN32 * ALIGN32 * sizeof(bfloat16_t);
constexpr uint32_t MAX_SINGLE_MN_ALIGN32_NUM = (MAX_SINGLE_MN + 31U) / ALIGN32 * ALIGN32;

struct Gmm1ContextBase {
    __aicore__ inline Gmm1ContextBase(uint32_t nValue, uint32_t kValue)
        : n(nValue),
          k(kValue),
          outputN(nValue / SWIGLU_N_HALF),
          blockNum(GetBlockNum()),
          subBlockIdx(GetSubBlockIdx()),
          blockIdx(GetBlockIdx() / GetTaskRation())
    {}

    uint32_t n;
    uint32_t k;
    uint32_t outputN;
    uint32_t blockNum;
    uint32_t subBlockIdx;
    uint32_t blockIdx;
};

template <typename ElementA, typename ElementB, typename ElementC, typename ElementMxScaleA, typename ElementMxScaleB,
          bool IsAic>
struct Gmm1Context;

template <typename ElementA, typename ElementB, typename ElementC, typename ElementMxScaleA, typename ElementMxScaleB>
struct Gmm1Context<ElementA, ElementB, ElementC, ElementMxScaleA, ElementMxScaleB, false> : public Gmm1ContextBase {
    __aicore__ inline Gmm1Context(uint32_t nValue, uint32_t kValue) : Gmm1ContextBase(nValue, kValue) {}
};

template <typename ElementA, typename ElementB, typename ElementC, typename ElementMxScaleA, typename ElementMxScaleB>
struct Gmm1Context<ElementA, ElementB, ElementC, ElementMxScaleA, ElementMxScaleB, true> : public Gmm1ContextBase {
    static constexpr bool TRANS_A = false;
    static constexpr bool TRANS_B = true;
    static constexpr uint32_t C0_SIZE_A = AuxGetC0Size<ElementA>();
    static constexpr uint32_t C0_SIZE_B = AuxGetC0Size<ElementB>();
    static constexpr uint32_t C0_SIZE_C = AuxGetC0Size<ElementC>();
    static constexpr uint32_t C0_SIZE_SCALE = 2U;

    using LayoutA = Std::conditional_t<TRANS_A, Te::DNExtLayoutPtn, Te::NDExtLayoutPtn>;
    using LayoutB = Std::conditional_t<TRANS_B, Te::DNExtLayoutPtn, Te::NDExtLayoutPtn>;
    using LayoutBias = Te::NDExtLayoutPtn;
    using LayoutC = Te::NDExtLayoutPtn;
    using MakeLayoutA = Te::FrameLayoutFormat<LayoutA, Std::Int<C0_SIZE_A>>;
    using MakeLayoutB = Te::FrameLayoutFormat<LayoutB, Std::Int<C0_SIZE_B>>;
    using MakeLayoutScaleA =
        Std::conditional_t<TRANS_A, Te::FrameLayoutFormat<Te::ScaleADNLayoutPtn, Std::Int<C0_SIZE_SCALE>>,
                           Te::FrameLayoutFormat<Te::ScaleANDLayoutPtn, Std::Int<C0_SIZE_SCALE>>>;
    using MakeLayoutScaleB =
        Std::conditional_t<TRANS_B, Te::FrameLayoutFormat<Te::ScaleBDNLayoutPtn, Std::Int<C0_SIZE_SCALE>>,
                           Te::FrameLayoutFormat<Te::ScaleBNDLayoutPtn, Std::Int<C0_SIZE_SCALE>>>;
    using MakeLayoutC = Te::FrameLayoutFormat<LayoutC, Std::Int<C0_SIZE_C>>;
    using LayoutBType = decltype(MakeLayoutB{}(uint32_t{}, uint32_t{}));
    using LayoutScaleBType = decltype(MakeLayoutScaleB{}(uint32_t{}, uint32_t{}));
    using LayoutBiasType = decltype(Te::MakeFrameLayout<LayoutBias>(uint32_t{}, uint32_t{}));
    using LayoutCType = decltype(MakeLayoutC{}(uint32_t{}, uint32_t{}));
    using BiasType = float;
    using BlazeDispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<>;
    // BF16 (no-quant) elements resolve to BlockMmadBf16Moe through the
    // BlockMmadMxMoe routing specialization in block_mmad_bf16_moe.h.
    static constexpr bool IS_BF16 = Std::is_same_v<ElementA, bfloat16_t>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmadMxMoe<BlazeDispatchPolicy, ElementA, LayoutA, ElementB, LayoutB,
                                                         ElementC, LayoutC, BiasType, LayoutBias>;

    __aicore__ inline Gmm1Context(uint32_t nValue, uint32_t kValue)
        : Gmm1ContextBase(nValue, kValue),
          scaleK(CeilDiv(kValue, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE),
          layoutB(MakeLayoutB{}(kValue, nValue)),
          layoutScaleB(MakeLayoutScaleB{}(CeilDiv(kValue, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE, nValue)),
          layoutBias(Te::MakeFrameLayout<LayoutBias>(1U, nValue)),
          layoutC(MakeLayoutC{}(L1_TILE_M, L1_TILE_N))
    {}

    uint32_t scaleK;
    LayoutBType layoutB;
    LayoutScaleBType layoutScaleB;
    LayoutBiasType layoutBias;
    LayoutCType layoutC;
    BlockMmad blockMmad;
    typename BlockMmad::L1Params l1Params{.kL1 = L1_TILE_K, .scaleKL1 = L1_TILE_K * 2U, .l1BufNum = 2U};
    typename BlockMmad::BlockShape l0TileShape{L1_TILE_M, L1_TILE_N, IS_BF16 ? GMM_L0_TILE_K_BF16 : GMM_L0_TILE_K, 0};

    template <bool EnableResidentAPrefix>
    __aicore__ inline void InitGeometryOnce(uint32_t mValue)
    {
        // BlockMmad::Init derives its geometry from K and the fixed L0/L1
        // shapes, not from this expert's M.  Keep the first expert's complete
        // shape for the API, but do not rebuild the identical L1 offsets and
        // resident-prefix capacity gates for every subsequent expert.  A
        // policy change within one context still performs a fresh Init.
        constexpr int32_t requestedPolicy = EnableResidentAPrefix ? 1 : 0;
        if (initializedResidentPolicy != requestedPolicy) {
            typename BlockMmad::ProblemShape matmulShape{mValue, n, k, 0};
            blockMmad.template Init<EnableResidentAPrefix>(matmulShape, l0TileShape, l1Params, false, false);
            initializedResidentPolicy = requestedPolicy;
        }
    }

    int32_t initializedResidentPolicy = -1;
};

// A stage handoff protects Fixpipe output, not an already-ready expert's
// independent GM->L1/MMAD work. Record credits while constructing the context;
// arm them only after Resource initialization and the first Timer::Tik, both
// of which may issue PIPE_ALL. No new credit is created by this object.
struct Gmm2OutputHandoff {
    __aicore__ inline Gmm2OutputHandoff(bool finalEpiloguePending = false, bool ubOwnershipPending = false)
        : pending_((finalEpiloguePending ? 1U : 0U) | (ubOwnershipPending ? 2U : 0U))
    {}

    Gmm2OutputHandoff(const Gmm2OutputHandoff &) = delete;
    Gmm2OutputHandoff &operator=(const Gmm2OutputHandoff &) = delete;

    __aicore__ inline void Arm()
    {
        if constexpr (g_coreType == AscendC::AIC) {
            if ((pending_ & 1U) != 0U) {
                WaitForVector();
            }
            if ((pending_ & 2U) != 0U) {
                WaitForVector(GMM2_UB_READY_FLAG + FLAG_ID_MAX);
            }
            pending_ = 0U;
        }
    }

    __aicore__ inline ~Gmm2OutputHandoff()
    {
        if constexpr (g_coreType == AscendC::AIC) {
            if (pending_ != 0U) {
                // A core with no GMM2 tile must still consume its launch's
                // credits. No tile output/ready fence will drain them for us.
                Arm();
                SyncFunc<AscendC::HardEvent::FIX_S>();
            }
        }
    }

private:
    uint32_t pending_;
};

template <typename ElementA, typename ElementB, typename ElementC, typename ElementMxScaleA, typename ElementMxScaleB,
          bool UseRaggedStripeMask = false, bool UsePairedStripeRelease = false, bool EnableBf16L1Reuse = false>
struct Gmm2Context {
    // Ready representation belongs to the consumer protocol, not the L1
    // staging policy. Scatter-N producers retain the existing lane/mask
    // format while using the ordinary non-resident A2/B2 compute pipeline.
    static constexpr bool USE_RAGGED_STRIPE_MASK = UseRaggedStripeMask;
    static constexpr bool USE_PAIRED_STRIPE_RELEASE = UsePairedStripeRelease;
    static_assert(!UseRaggedStripeMask || !UsePairedStripeRelease,
                  "paired stripe release and ragged mask publication are exclusive");
    using LayoutA = decltype(tla::MakeLayout<ElementA, layout::RowMajor>(uint32_t{}, uint32_t{}));
    using LayoutB = decltype(tla::MakeLayout<ElementB, layout::ColumnMajor>(uint32_t{}, uint32_t{}));
    using LayoutMxScaleA =
        decltype(tla::MakeMxScaleLayout<ElementMxScaleA, layout::RowMajor, false>(uint32_t{}, uint32_t{}));
    using LayoutMxScaleB =
        decltype(tla::MakeMxScaleLayout<ElementMxScaleB, layout::ColumnMajor, true>(uint32_t{}, uint32_t{}));
    using LayoutC = decltype(tla::MakeLayout<ElementC, layout::RowMajor>(uint32_t{}, uint32_t{}));
    // BF16 no-quant (generic pipeline only) uses the CATLASS BF16 pingpong
    // block, optionally reusing its L1 payload; no MX scale arena, L0 K64.
    static constexpr bool IS_BF16 = Std::is_same_v<ElementA, bfloat16_t>;
    static_assert(!EnableBf16L1Reuse || (IS_BF16 && !UseRaggedStripeMask && !UsePairedStripeRelease),
                  "BF16 L1 reuse retains the generic whole-expert ready protocol");
    using TileCopy =
        Std::conditional_t<IS_BF16,
                           Gemm::Tile::PackedTileCopyTla<ArchTag, ElementA, layout::RowMajor, ElementB,
                                                         layout::ColumnMajor, ElementC, layout::RowMajor, void>,
                           Gemm::Tile::PackedMxTileCopyTla<
                               ArchTag, ElementA, layout::RowMajor, ElementB, layout::ColumnMajor, ElementMxScaleA,
                               LayoutMxScaleA, ElementMxScaleB, LayoutMxScaleB, ElementC, layout::RowMajor, void>>;
    using BlockMmad = Gemm::Block::BlockMmadTla<
        Std::conditional_t<IS_BF16,
                           Std::conditional_t<EnableBf16L1Reuse, Bf16ResidentDispatchPolicy, Bf16DispatchPolicy>,
                           DispatchPolicy>,
        L1TileShape, Std::conditional_t<IS_BF16, L0TileShapeBf16, L0TileShape>, ElementA, ElementB, ElementC, void,
        TileCopy>;

    __aicore__ inline Gmm2Context(uint32_t nValue, uint32_t kValue, bool finalEpiloguePending = false,
                                  bool ubOwnershipPending = false)
        : n(nValue),
          k(kValue),
          scaleK(CeilDiv(kValue, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE),
          blockNum(GetBlockNum()),
          blockIdx(GetBlockIdx() / GetTaskRation()),
          layoutB(tla::MakeLayout<ElementB, layout::ColumnMajor>(kValue, nValue)),
          layoutMxScaleB(tla::MakeMxScaleLayout<ElementMxScaleB, layout::ColumnMajor, true>(
              CeilDiv(kValue, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE, nValue)),
          blockMmad(resource),
          outputHandoff(finalEpiloguePending, ubOwnershipPending)
    {}

    uint32_t n;
    uint32_t k;
    uint32_t scaleK;
    uint32_t blockNum;
    uint32_t blockIdx;
    LayoutB layoutB;
    LayoutMxScaleB layoutMxScaleB;
    Arch::Resource<ArchTag> resource;
    BlockMmad blockMmad;
    // Declared last: drain an empty core's pending credits before the compute
    // context/resource is destroyed. Active cores arm once before output.
    Gmm2OutputHandoff outputHandoff;
};

template <typename ElementA, typename ElementB, typename ElementC, typename ElementMxScaleA, typename ElementMxScaleB,
          bool UseFullA = false, bool UseRaggedStripeMask = false, bool UsePairedStripeRelease = false>
struct Gmm2ResidentAContext {
    static_assert(!UseRaggedStripeMask || UseFullA, "ragged stripe publication requires the Full-A producer");
    static constexpr bool USE_RAGGED_STRIPE_MASK = UseRaggedStripeMask;
    static constexpr bool USE_PAIRED_STRIPE_RELEASE = UsePairedStripeRelease;
    static_assert(!UsePairedStripeRelease || !UseRaggedStripeMask,
                  "paired per-stripe release and ragged mask publication are exclusive");
    using LayoutA = decltype(tla::MakeLayout<ElementA, layout::RowMajor>(uint32_t{}, uint32_t{}));
    using LayoutB = decltype(tla::MakeLayout<ElementB, layout::ColumnMajor>(uint32_t{}, uint32_t{}));
    using LayoutMxScaleA =
        decltype(tla::MakeMxScaleLayout<ElementMxScaleA, layout::RowMajor, false>(uint32_t{}, uint32_t{}));
    using LayoutMxScaleB =
        decltype(tla::MakeMxScaleLayout<ElementMxScaleB, layout::ColumnMajor, true>(uint32_t{}, uint32_t{}));
    using LayoutC = decltype(tla::MakeLayout<ElementC, layout::RowMajor>(uint32_t{}, uint32_t{}));
    using TileCopy = Gemm::Tile::PackedMxTileCopyTla<ArchTag, ElementA, layout::RowMajor, ElementB, layout::ColumnMajor,
                                                     ElementMxScaleA, LayoutMxScaleA, ElementMxScaleB, LayoutMxScaleB,
                                                     ElementC, layout::RowMajor, void>;
    using ResidentPolicy = Std::conditional_t<UseFullA, FullAResidentDispatchPolicy, ResidentADispatchPolicy>;
    using BlockMmad = Gemm::Block::BlockMmadTla<ResidentPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC,
                                                void, TileCopy>;

    __aicore__ inline Gmm2ResidentAContext(uint32_t nValue, uint32_t kValue)
        : n(nValue),
          k(kValue),
          scaleK(CeilDiv(kValue, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE),
          blockNum(GetBlockNum()),
          blockIdx(GetBlockIdx() / GetTaskRation()),
          layoutB(tla::MakeLayout<ElementB, layout::ColumnMajor>(kValue, nValue)),
          layoutMxScaleB(tla::MakeMxScaleLayout<ElementMxScaleB, layout::ColumnMajor, true>(
              CeilDiv(kValue, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE, nValue)),
          blockMmad(resource)
    {}

    uint32_t n;
    uint32_t k;
    uint32_t scaleK;
    uint32_t blockNum;
    uint32_t blockIdx;
    LayoutB layoutB;
    LayoutMxScaleB layoutMxScaleB;
    Arch::Resource<ArchTag> resource;
    BlockMmad blockMmad;
    // Preserve the stripe helper's common context interface. Retained
    // resident callers keep their existing externally managed handoff.
    Gmm2OutputHandoff outputHandoff;
};

// Execute an explicit M-group/N-stripe interval. Scatter-N scheduling passes
// one tile per call; sparse scheduling can still pass a contiguous chunk.
// Every supported context issues its output Fixpipe store before returning
// from blockMmad, so FIX_S can release exactly the completed N stripes.
template <typename ElementA, typename ElementB, typename ElementC, typename ElementMxScaleA, typename ElementMxScaleB,
          typename Gmm2ContextType>
__aicore__ inline void GroupMatmul2MGroupStripe(Gmm2ContextType &context,
                                                const AscendC::Shape<int64_t, int64_t, int64_t, int64_t> &problemShape,
                                                const GMMAddrInfo &gmmAddrInfo, uint32_t mGroupIdx, uint32_t nTileBegin,
                                                uint32_t nTileEnd, GlobalTensor<ElementC> combineAccumGm,
                                                AscendTimerDevice &timer, uint32_t expertIdx,
                                                __gm__ int32_t *stripeReadyAddr = nullptr)
{
    uint32_t m = Get<M_VALUE>(problemShape);
    uint32_t n = context.n;
    uint32_t k = context.k;
    uint32_t mLoc = mGroupIdx * L1_TILE_M;
    uint32_t nTileCount = CeilDiv(n, L1_TILE_N);
    if (mLoc >= m || nTileBegin >= nTileCount) {
        return;
    }
    nTileEnd = nTileEnd < nTileCount ? nTileEnd : nTileCount;
    if (nTileBegin >= nTileEnd) {
        return;
    }

    uint32_t actualM = m - mLoc;
    actualM = actualM < L1_TILE_M ? actualM : L1_TILE_M;
    auto layoutA = tla::MakeLayout<ElementA, layout::RowMajor>(m, k);
    auto layoutMxScaleA = tla::MakeMxScaleLayout<ElementMxScaleA, layout::RowMajor, false>(m, context.scaleK);
    auto layoutC = tla::MakeLayout<ElementC, layout::RowMajor>(m, n);

    GlobalTensor<ElementA> aGlobalTensor;
    GlobalTensor<ElementB> bGlobalTensor;
    GlobalTensor<ElementMxScaleA> aScaleGlobalTensor;
    GlobalTensor<ElementMxScaleB> bScaleGlobalTensor;
    aGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(gmmAddrInfo.aGlobal));
    bGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(gmmAddrInfo.bGlobal));
    aScaleGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ ElementMxScaleA *>(gmmAddrInfo.aScaleGlobal));
    bScaleGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ ElementMxScaleB *>(gmmAddrInfo.bScaleGlobal));
    // Several M-groups share B while all N stripes of this job share A.
    aGlobalTensor.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_NORMAL);
    bGlobalTensor.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_NORMAL);

    auto tensorA = tla::MakeTensor(aGlobalTensor, layoutA, Arch::PositionGM{});
    auto tensorB = tla::MakeTensor(bGlobalTensor, context.layoutB, Arch::PositionGM{});
    auto tensorMxScaleA = tla::MakeTensor(aScaleGlobalTensor, layoutMxScaleA, Arch::PositionGM{});
    auto tensorMxScaleB = tla::MakeTensor(bScaleGlobalTensor, context.layoutMxScaleB, Arch::PositionGM{});
    auto tensorC = tla::MakeTensor(combineAccumGm, layoutC, Arch::PositionGM{});
    auto tensorBlockA = GetTile(tensorA, tla::MakeCoord(mLoc, 0U), tla::MakeShape(actualM, k));
    auto tensorBlockMxScaleA =
        GetTile(tensorMxScaleA, tla::MakeCoord(mLoc, 0U), tla::MakeShape(actualM, CeilDiv(k, MXFP_SCALE_GROUP_NUM)));

    // Only the corresponding stripe-publication specialization retains these
    // pending counters; ordinary per-tile/whole-group contexts fold them out.
    uint32_t pendingReadyMask = 0U;
    uint32_t publishedReadyMask = 0U;
    uint32_t pendingReadyTiles = 0U;
    TIMER_BLOCK(timer.Tik(AscendTimer::AIC_GMM2, expertIdx));
    context.outputHandoff.Arm();
    for (uint32_t nTile = nTileBegin; nTile < nTileEnd; ++nTile) {
        uint32_t nLoc = nTile * L1_TILE_N;
        uint32_t actualN = n - nLoc;
        actualN = actualN < L1_TILE_N ? actualN : L1_TILE_N;
        GemmCoord actualShape{actualM, actualN, k};
        auto tensorBlockB = GetTile(tensorB, tla::MakeCoord(0U, nLoc), tla::MakeShape(k, actualN));
        auto tensorBlockMxScaleB = GetTile(tensorMxScaleB, tla::MakeCoord(0U, nLoc),
                                           tla::MakeShape(CeilDiv(k, MXFP_SCALE_GROUP_NUM), actualN));
        auto tensorBlockC = GetTile(tensorC, tla::MakeCoord(mLoc, nLoc), tla::MakeShape(actualM, actualN));
        if constexpr (Gmm2ContextType::IS_BF16) {
            // BF16 no-quant uses the same stripe scheduler but no MX scale
            // operands. The scale views above remain type-compatible aliases
            // so the shared address/layout setup stays unchanged.
            context.blockMmad(tensorBlockA, tensorBlockB, tensorBlockC, actualShape);
        } else {
            context.blockMmad(tensorBlockA, tensorBlockB, tensorBlockC, actualShape, tensorBlockMxScaleA,
                              tensorBlockMxScaleB);
        }
        if (stripeReadyAddr != nullptr) {
            if constexpr (Gmm2ContextType::USE_PAIRED_STRIPE_RELEASE) {
                ++pendingReadyTiles;
                if (pendingReadyTiles == 2U || nTile + 1U == nTileEnd) {
                    // Publish only this job's completed tiles. A scheduler
                    // split may bisect a 512-column batch: flush at job end,
                    // and let the consumer require both independent lanes.
                    SyncFunc<AscendC::HardEvent::FIX_S>();
                    for (uint32_t readyTile = nTile + 1U - pendingReadyTiles; readyTile <= nTile; ++readyTile) {
                        WriteGmByPassDCache(stripeReadyAddr + readyTile, int32_t(1));
                    }
                    pendingReadyTiles = 0U;
                }
            } else if constexpr (Gmm2ContextType::USE_RAGGED_STRIPE_MASK) {
                pendingReadyMask |= 1U << nTile;
                ++pendingReadyTiles;
                if (ShouldPublishRaggedGmm2Batch(nTile, nTileEnd, nTileCount, pendingReadyTiles)) {
                    // Release only completed columns. All stores in this
                    // batch precede the mask, and the final job tile always
                    // flushes before its whole-group completion vote.
                    SyncFunc<AscendC::HardEvent::FIX_S>();
                    __gm__ int32_t *maskAddr = stripeReadyAddr + GMM2_STRIPE_MASK_OFFSET - GMM2_STRIPE_READY_OFFSET;
                    if (nTileBegin == 0U && nTileEnd == nTileCount) {
                        // Ordinary resident groups have exactly one writer.
                        publishedReadyMask |= pendingReadyMask;
                        WriteGmByPassDCache(maskAddr, static_cast<int32_t>(publishedReadyMask));
                    } else {
                        // Split jobs own disjoint N bits. Each bit is added
                        // once, so integer addition is an exact atomic OR;
                        // there can be no carry or lost concurrent update.
                        AtomicAdd(maskAddr, static_cast<int32_t>(pendingReadyMask));
                    }
                    pendingReadyMask = 0U;
                    pendingReadyTiles = 0U;
                }
            } else {
                // FIX_S is the producer-release fence: a consumer that observes
                // lane t may immediately read exactly this output stripe from GM.
                // Every (expert, M-group, stripe) has one scheduler owner, so the
                // lane is a non-atomic monotonic 0 -> 1 publication.
                SyncFunc<AscendC::HardEvent::FIX_S>();
                WriteGmByPassDCache(stripeReadyAddr + nTile, int32_t(1));
            }
        }
    }
    TIMER_BLOCK(timer.template Tok<Span>(AscendTimer::AIC_GMM2, expertIdx));
}

// GroupMatmulSwigluQuant - GMM1 + SwiGLU + Quant
// UseSharedTiming: 共享专家链路记录到 AIC_SHARE_GMM1/AIV_SHARE_GMM1_EPILOGUE 槽位，
// 与路由专家的 per-expert 行互不覆盖；iter 由调用方传共享专家编号。
template <typename ElementA, typename ElementB, typename ElementC, typename ElementMxScaleA, typename ElementMxScaleB,
          bool IsAic, bool PublishGmm1Ready, bool EnableResidentAPrefix = false, bool SkipDispatchTileWait = false,
          bool UseSharedTiming = false>
__aicore__ inline void GroupMatmulSwigluQuant(
    Gmm1Context<ElementA, ElementB, ElementC, ElementMxScaleA, ElementMxScaleB, IsAic> &context,
    BlockEpilogueSwigluMxQuant<L1TileShape, ElementA, ElementC, ElementMxScaleA, ElementMxScaleB, true, false,
                               PublishGmm1Ready> &epilogueOp,
    const AscendC::Shape<int64_t, int64_t, int64_t, int64_t> &problemShape, const GMMAddrInfo &gmmAddrInfo,
    uint32_t &startBlockIdx, int32_t &vecSetSyncCom, AscendTimerDevice &timer, uint32_t expertIdx)
{
    static_assert(IsAic == (g_coreType == AscendC::AIC), "GMM1 context must match the compiled core type");
    // 分槽打点：AIC 分支只用 gmm1Timing，AIV 分支只用 gmm1EpilogueTiming，
    // [[maybe_unused]] 兼顾另一分支与 ENABLE_TIMER=0 的编译。
    [[maybe_unused]] constexpr AscendTimer::DynamicTimingType gmm1Timing =
        UseSharedTiming ? AscendTimer::AIC_SHARE_GMM1 : AscendTimer::AIC_GMM1;
    [[maybe_unused]] constexpr AscendTimer::DynamicTimingType gmm1EpilogueTiming =
        UseSharedTiming ? AscendTimer::AIV_SHARE_GMM1_EPILOGUE : AscendTimer::AIV_GMM1_EPILOGUE;
    // Resident-A prefixing is confined to the AIC MMAD branch, while GMM1
    // readiness is published by the AIV epilogue after its MTE3 stores.  The
    // two policies are independent and may be enabled together by complete
    // staging when GMM2 consumes precise per-expert readiness.
    uint32_t m = Get<M_VALUE>(problemShape);
    uint32_t k = context.k;
    uint32_t outputN = context.outputN;

    epilogueOp.UpdateNextProblem({m, outputN, k, 0});

    uint32_t blockNum = context.blockNum;
    uint32_t subBlockIdx = context.subBlockIdx;
    uint32_t blockIdx = context.blockIdx;
    BlockScheduler scheduler({m, outputN, k}, MakeCoord(L1_TILE_M, L1_TILE_N));
    uint32_t coreLoops = scheduler.GetCoreLoops();
    uint32_t startLoopIdx = (blockIdx < startBlockIdx ? blockIdx + blockNum : blockIdx) - startBlockIdx;

    if constexpr (IsAic) {
        using Context = Gmm1Context<ElementA, ElementB, ElementC, ElementMxScaleA, ElementMxScaleB, true>;
        using MakeLayoutA = typename Context::MakeLayoutA;
        using MakeLayoutScaleA = typename Context::MakeLayoutScaleA;
        using BiasType = typename Context::BiasType;
        uint32_t scaleK = context.scaleK;
        auto layoutA = MakeLayoutA{}(m, k);
        auto layoutScaleA = MakeLayoutScaleA{}(m, scaleK);
        context.template InitGeometryOnce<EnableResidentAPrefix>(m);

        auto tensorUbFirst = Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(0), context.layoutC);
        auto tensorUbSecond = Te::MakeTensor(
            Te::MakeMemPtr<Te::Location::UB, ElementC>(MAX_SINGLE_MN_ALIGN32_NUM * sizeof(ElementC)), context.layoutC);
        auto tensorA = Te::MakeTensor(
            Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementA *>(gmmAddrInfo.aGlobal)), layoutA);
        auto tensorB =
            Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementB *>(gmmAddrInfo.bGlobal)),
                           context.layoutB);
        bool w1StreamedOnce = CeilDiv(m, L1_TILE_M) == 1U;
        // Single-M-tile experts stream every W1 byte exactly once (each B
        // tile is consumed by one AIC and never revisited), so caching the
        // weight stream only evicts useful lines; bypass L2 for it.  Experts
        // with several M tiles re-read W1 once per M tile and keep the
        // normal allocation policy.  Same policy as GMM2's single-M-group
        // weight gate below and the Blaze qbmm precedent (which disables
        // the MX scale stream together with the weight data).
        if (w1StreamedOnce) {
            tensorB.SetL2CacheHint(Te::CacheMode::CACHE_MODE_DISABLE);
        }
        auto tensorMxScaleA = Te::MakeTensor(
            Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementMxScaleA *>(gmmAddrInfo.aScaleGlobal)),
            layoutScaleA);
        auto tensorMxScaleB = Te::MakeTensor(
            Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementMxScaleB *>(gmmAddrInfo.bScaleGlobal)),
            context.layoutScaleB);
        // BF16 no-quant has no weight scales: the scale views are never read.
        if constexpr (!Context::IS_BF16) {
            if (w1StreamedOnce) {
                tensorMxScaleB.SetL2CacheHint(Te::CacheMode::CACHE_MODE_DISABLE);
            }
        }
        auto tensorBias = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ BiasType *>(0UL)),
                                         context.layoutBias);

        // The first weight half computes into private L0C before waiting for
        // the previous vector consumer.  Only CopyOut touches the shared UB,
        // so GM->L1/MMAD can cover the AIC<->AIV credit tail safely.
        const bool gmm1HasLoop = startLoopIdx < coreLoops;
        int32_t readyTileIdx = -1;
        for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += blockNum) {
            GemmCoord blockCoord = scheduler.GetBlockCoord(loopIdx);
            GemmCoord actualShape = scheduler.GetActualBlockShape(blockCoord);
            uint32_t mLoc = blockCoord.m() * L1_TILE_M;
            uint32_t nLoc = blockCoord.n() * L1_TILE_N;
            uint32_t kLoc = blockCoord.k() * L1_TILE_K;

            // Sparse-decode gates the whole GMM1 phase on one
            // dispatch-complete flag before this loop, so its per-tile row
            // counters are never published and the spin compiles out.
            if constexpr (!SkipDispatchTileWait) {
                int32_t tileIdx = static_cast<int32_t>(mLoc / L1_TILE_M);
                if (tileIdx != readyTileIdx) {
                    int32_t readyTarget = static_cast<int32_t>(mLoc + L1_TILE_M > m ? m - mLoc : L1_TILE_M);
                    __gm__ int32_t *readyAddr = gmmAddrInfo.dispatchTileReady +
                                                static_cast<int64_t>(tileIdx) * DISPATCH_TILE_READY_STRIDE +
                                                DISPATCH_READY_OFFSET;
                    uint32_t pollMisses = 0U;
                    int64_t pollDelayCycles = 100;
                    while (readyTarget != AscendC::ReadGmByPassDCache(readyAddr)) {
                        GmSpinBackoffStep(pollMisses, pollDelayCycles);
                    }
                    readyTileIdx = tileIdx;
                }
            }

            if (loopIdx == startLoopIdx) {
                TIMER_BLOCK(timer.Tik(gmm1Timing, expertIdx));
            }
            auto tensorBlockA =
                tensorA.Slice(Te::MakeCoord(mLoc, kLoc), Te::MakeShape(actualShape.m(), actualShape.k()));
            auto tensorBlockMxScaleA =
                tensorMxScaleA.Slice(Te::MakeCoord(mLoc, kLoc / MXFP_SCALE_GROUP_NUM),
                                     Te::MakeShape(actualShape.m(), CeilDiv(actualShape.k(), MXFP_SCALE_GROUP_NUM)));
            auto tensorBlockUbFirst =
                tensorUbFirst.Slice(Te::MakeCoord(0, 0), Te::MakeShape(actualShape.m(), actualShape.n()));
            auto tensorBlockUbSecond =
                tensorUbSecond.Slice(Te::MakeCoord(0, 0), Te::MakeShape(actualShape.m(), actualShape.n()));
            typename Context::BlockMmad::BlockShape singleShape{actualShape.m(), actualShape.n(), actualShape.k(), 0};

            for (uint32_t weightBlock = 0; weightBlock < SWIGLU_N_HALF; ++weightBlock) {
                auto tensorBlockB = tensorB.Slice(Te::MakeCoord(kLoc, nLoc + weightBlock * outputN),
                                                  Te::MakeShape(actualShape.k(), actualShape.n()));
                auto tensorBlockMxScaleB = tensorMxScaleB.Slice(
                    Te::MakeCoord(kLoc / MXFP_SCALE_GROUP_NUM, nLoc + weightBlock * outputN),
                    Te::MakeShape(CeilDiv(actualShape.k(), MXFP_SCALE_GROUP_NUM), actualShape.n()));
                auto tensorL0C = context.blockMmad.template ComputeMaybeWithResidentAPrefix<EnableResidentAPrefix>(
                    tensorBlockA, tensorBlockB, tensorBlockMxScaleA, tensorBlockMxScaleB, tensorBias, singleShape,
                    weightBlock != 0U);
                if (weightBlock == 0U && vecSetSyncCom) {
                    WaitForVector();
                }
                context.blockMmad.CopyOut(weightBlock == 0U ? tensorBlockUbFirst : tensorBlockUbSecond, tensorL0C);
            }
            NotifyVector();
            vecSetSyncCom = 1;
        }
        if (gmm1HasLoop) {
            // Preserve the first start and actual last end across submissions.
            TIMER_BLOCK(timer.template Tok<Span>(gmm1Timing, expertIdx));
        }
    } else {
        // GMM1 的 AIV 侧 epilogue (swiglu+quant), per-expert, 与 AIC_GMM1 对称。
        // timer (干活区间口径): Tik 在首片 WaitForCube 之后, Tok 在循环后; 仅 sub0, 空核不打点。
        const bool epiHasLoop = subBlockIdx == 0 && startLoopIdx < coreLoops;
        for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += blockNum) {
            GemmCoord blockCoord = scheduler.GetBlockCoord(loopIdx);
            GemmCoord actualShape = scheduler.GetActualBlockShape(blockCoord);
            uint32_t mLoc = blockCoord.m() * L1_TILE_M;
            uint32_t nLoc = blockCoord.n() * L1_TILE_N;
            if (subBlockIdx == 0) {
                Std::tuple<int64_t, int64_t, int64_t, int64_t> epilogueShape{actualShape.m(), actualShape.n(), 0, 0};
                Std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> epilogueOffset{
                    mLoc * outputN + nLoc,
                    mLoc * CeilDiv(outputN, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE +
                        CeilDiv(nLoc, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE,
                    0,
                    0,
                    0,
                    0};
                WaitForCube();
                if (loopIdx == startLoopIdx) {
                    TIMER_BLOCK(timer.Tik(gmm1EpilogueTiming, expertIdx));
                }
                AscendC::SetCtrlSpr<60, 60>(0);
                epilogueOp(epilogueShape, epilogueOffset);
                NotifyCube();
            }
            vecSetSyncCom = 1;
        }
        if (epiHasLoop) {
            TIMER_BLOCK(timer.template Tok<Span>(gmm1EpilogueTiming, expertIdx));
        }
    }
    startBlockIdx = (startBlockIdx + coreLoops) % blockNum;
}

// GroupMatmul2 - GMM2矩阵乘法直接写入workspace(ptrCombineAccum)
// UseSharedTiming: 共享专家链路记录到 AIC_SHARE_GMM2 槽位，与路由专家的
// per-expert 行互不覆盖；iter 由调用方传共享专家编号。
template <typename ElementA, typename ElementB, typename ElementC, typename ElementMxScaleA, typename ElementMxScaleB,
          bool EnableMGroupReady = false, bool WaitForGmm1Ready = true, bool EnableBf16L1Reuse = false,
          bool UseSharedTiming = false>
__aicore__ inline bool GroupMatmul2(Gmm2Context<ElementA, ElementB, ElementC, ElementMxScaleA, ElementMxScaleB, false,
                                                false, EnableBf16L1Reuse> &context,
                                    const Params &params,
                                    const AscendC::Shape<int64_t, int64_t, int64_t, int64_t> &problemShape,
                                    const GMMAddrInfo &gmmAddrInfo, uint32_t &startBlockIdx, uint32_t expertIdx,
                                    uint32_t gmm1ReadyTarget, bool coversWholeExpert, bool allowMGroupReady,
                                    bool allowLongMGroupReady, GlobalTensor<ElementC> combineAccumGm,
                                    AscendTimerDevice &timer)
{
    static_assert(!EnableBf16L1Reuse || !EnableMGroupReady,
                  "BF16 N-local scheduling does not change the ready representation");
    [[maybe_unused]] constexpr AscendTimer::DynamicTimingType gmm2Timing =
        UseSharedTiming ? AscendTimer::AIC_SHARE_GMM2 : AscendTimer::AIC_GMM2;
    uint32_t m = Get<M_VALUE>(problemShape);
    uint32_t n = context.n;  // GMM2输出维度 = k_
    uint32_t k = context.k;  // GMM2输入维度 = n_/2

    uint32_t blockNum = context.blockNum;
    uint32_t blockIdx = context.blockIdx;

    GlobalTensor<int32_t> flagSwigluToGmm2;
    if constexpr (WaitForGmm1Ready) {
        flagSwigluToGmm2.SetGlobalBuffer((__gm__ int32_t *)gmmAddrInfo.flagSwigluToGmm2);
    }

    uint32_t scaleK = context.scaleK;
    auto layoutA = tla::MakeLayout<ElementA, layout::RowMajor>(m, k);
    auto layoutMxScaleA = tla::MakeMxScaleLayout<ElementMxScaleA, layout::RowMajor, false>(m, scaleK);
    auto layoutC = tla::MakeLayout<ElementC, layout::RowMajor>(m, n);

    GlobalTensor<ElementA> aGlobalTensor;
    GlobalTensor<ElementB> bGlobalTensor;
    GlobalTensor<ElementMxScaleA> aScaleGlobalTensor;
    GlobalTensor<ElementMxScaleB> bScaleGlobalTensor;

    aGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(gmmAddrInfo.aGlobal));
    bGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(gmmAddrInfo.bGlobal));
    aScaleGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ ElementMxScaleA *>(gmmAddrInfo.aScaleGlobal));
    bScaleGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ ElementMxScaleB *>(gmmAddrInfo.bScaleGlobal));

    // Matrix A or Matrix B does not have duplicate data reads. Setting L2 Cache to Disable,
    // data reads will bypass L2 Cache.
    if (coversWholeExpert && CeilDiv(m, L1_TILE_M) == 1) {
        bGlobalTensor.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    } else {
        bGlobalTensor.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_NORMAL);
    }
    if (CeilDiv(n, L1_TILE_N) == 1) {
        aGlobalTensor.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    } else {
        aGlobalTensor.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_NORMAL);
    }

    auto tensorA = tla::MakeTensor(aGlobalTensor, layoutA, Arch::PositionGM{});
    auto tensorB = tla::MakeTensor(bGlobalTensor, context.layoutB, Arch::PositionGM{});
    auto tensorMxScaleA = tla::MakeTensor(aScaleGlobalTensor, layoutMxScaleA, Arch::PositionGM{});
    auto tensorMxScaleB = tla::MakeTensor(bScaleGlobalTensor, context.layoutMxScaleB, Arch::PositionGM{});
    auto tensorC = tla::MakeTensor(combineAccumGm, layoutC, Arch::PositionGM{});

    BlockScheduler scheduler({m, n, k}, MakeCoord(L1_TILE_M, L1_TILE_N));
    uint32_t coreLoops = scheduler.GetCoreLoops();
    uint32_t startLoopIdx = (blockIdx < startBlockIdx ? blockIdx + blockNum : blockIdx) - startBlockIdx;
    Bf16Gmm2NLocalTile nLocal;
    if constexpr (EnableBf16L1Reuse) {
        nLocal = GetBf16Gmm2NLocalTile(m, n, blockNum, startLoopIdx);
    }
    uint32_t readyWaveCount = 0U;
    if constexpr (EnableMGroupReady) {
        readyWaveCount = allowMGroupReady ? GetGmm2ReadyWaveCount(m, n, blockNum, allowLongMGroupReady) : 0U;
    }
    bool publishReadyPerTile = readyWaveCount >= 2U;
    uint32_t readyNTiles = GetGmm2NTileCount(n);
    uint32_t readyPrefixGroups = GetGmm2ReadyPrefixGroupCount(m, n, blockNum, readyWaveCount);
    uint32_t readyPrefixRows = readyPrefixGroups * EXPERT_SCHEDULE_M_TILE_ROWS;
    uint32_t readyPrefixLoops = readyPrefixGroups * readyNTiles;
    bool schedulerUsesSuffix = false;
    if (readyPrefixGroups != 0U) {
        // Reuse the ordinary scheduler object instead of keeping two extra
        // scheduler states live across every GMM2 tile.  A ready prefix never
        // exceeds one physical wave, so an active core changes from the
        // prefix view to the suffix view at most once.
        schedulerUsesSuffix = startLoopIdx >= readyPrefixLoops;
        uint32_t schedulerRows = schedulerUsesSuffix ? m - readyPrefixRows : readyPrefixRows;
        scheduler.Update({schedulerRows, n, k}, MakeCoord(L1_TILE_M, L1_TILE_N));
    }

    // timer (干活区间口径): 首片 flagSwigluToGmm2 自旋在 Tik 之前 (段内无逐片等待),
    // Tok 在循环后; 片间零 timer 屏障; 空核不打点。
    if (startLoopIdx < coreLoops) {
        if constexpr (WaitForGmm1Ready) {
            uint32_t targetLoops = gmm1ReadyTarget;
            if (targetLoops == 0U) {
                BlockScheduler gmmBlockScheduler({m, k, n}, MakeCoord(L1_TILE_M, L1_TILE_N));
                targetLoops = gmmBlockScheduler.GetCoreLoops();
            }
            __gm__ int32_t *flagValueAddr = (__gm__ int32_t *)flagSwigluToGmm2.GetPhyAddr();
            // Use a monotonic lower-bound predicate so a producer that has
            // already advanced the counter cannot strand a late consumer.
            uint32_t pollMisses = 0U;
            int64_t pollDelayCycles = 100;
            while (AscendC::ReadGmByPassDCache(flagValueAddr) < static_cast<int32_t>(targetLoops)) {
                GmSpinBackoffStep(pollMisses, pollDelayCycles);
            }
        }
        TIMER_BLOCK(timer.Tik(gmm2Timing, expertIdx));
        context.outputHandoff.Arm();
    }
    for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += blockNum) {
        uint32_t orderedLoopIdx = loopIdx;
        uint32_t mGroupBase = 0U;
        if (readyPrefixGroups != 0U) {
            bool useSuffix = loopIdx >= readyPrefixLoops;
            if (useSuffix && !schedulerUsesSuffix) {
                scheduler.Update({m - readyPrefixRows, n, k}, MakeCoord(L1_TILE_M, L1_TILE_N));
                schedulerUsesSuffix = true;
            }
            if (useSuffix) {
                orderedLoopIdx -= readyPrefixLoops;
                mGroupBase = readyPrefixGroups;
            }
        }
        GemmCoord blockCoord;
        if constexpr (EnableBf16L1Reuse) {
            if (nLocal.tileCount != 0U) {
                blockCoord = GemmCoord{nLocal.mGroup, nLocal.nBegin++, 0U};
            } else {
                blockCoord = scheduler.GetBlockCoord(orderedLoopIdx);
            }
        } else {
            blockCoord = scheduler.GetBlockCoord(orderedLoopIdx);
        }
        GemmCoord actualShape = scheduler.GetActualBlockShape(blockCoord);
        blockCoord = GemmCoord{blockCoord.m() + mGroupBase, blockCoord.n(), blockCoord.k()};

        uint32_t mLoc = blockCoord.m() * L1_TILE_M;
        uint32_t nLoc = blockCoord.n() * L1_TILE_N;
        uint32_t kLoc = blockCoord.k() * L1_TILE_K;

        auto tensorBlockA =
            GetTile(tensorA, tla::MakeCoord(mLoc, kLoc), tla::MakeShape(actualShape.m(), actualShape.k()));
        auto tensorBlockMxScaleA =
            GetTile(tensorMxScaleA, tla::MakeCoord(mLoc, kLoc / MXFP_SCALE_GROUP_NUM),
                    tla::MakeShape(actualShape.m(), CeilDiv(actualShape.k(), MXFP_SCALE_GROUP_NUM)));
        auto tensorBlockC =
            GetTile(tensorC, tla::MakeCoord(mLoc, nLoc), tla::MakeShape(actualShape.m(), actualShape.n()));
        auto tensorBlockB =
            GetTile(tensorB, tla::MakeCoord(kLoc, nLoc), tla::MakeShape(actualShape.k(), actualShape.n()));
        auto tensorBlockMxScaleB =
            GetTile(tensorMxScaleB, tla::MakeCoord(kLoc / MXFP_SCALE_GROUP_NUM, nLoc),
                    tla::MakeShape(CeilDiv(actualShape.k(), MXFP_SCALE_GROUP_NUM), actualShape.n()));
        if constexpr (Gmm2Context<ElementA, ElementB, ElementC, ElementMxScaleA, ElementMxScaleB>::IS_BF16) {
            // BF16 no-quant: plain BF16 x BF16 MMAD, the scale views are unused.
            context.blockMmad(tensorBlockA, tensorBlockB, tensorBlockC, actualShape);
        } else {
            context.blockMmad(tensorBlockA, tensorBlockB, tensorBlockC, actualShape, tensorBlockMxScaleA,
                              tensorBlockMxScaleB);
        }
        if constexpr (EnableMGroupReady) {
            if (publishReadyPerTile) {
                // Multi-wave experts may give one AIC several logical tiles.
                // Vote after every Fixpipe store so each M-group receives
                // exactly one vote per N tile and can stream independently.
                SyncFunc<AscendC::HardEvent::FIX_S>();
                __gm__ int32_t *readyAddr = gmmAddrInfo.dispatchTileReady +
                                            static_cast<int64_t>(blockCoord.m()) * DISPATCH_TILE_READY_STRIDE +
                                            GMM2_READY_OFFSET;
                AtomicAdd(readyAddr, int32_t(1));
            }
        }
    }
    if (startLoopIdx < coreLoops) {
        TIMER_BLOCK(timer.template Tok<Span>(gmm2Timing, expertIdx));
    }
    if constexpr (EnableMGroupReady) {
        if (readyWaveCount == 1U && startLoopIdx < coreLoops) {
            // A single-wave expert gives every active AIC exactly one tile,
            // so the established loop-after publication remains unchanged.
            // Multi-wave paths already published every tile in the loop.
            // An active
            // AIC owns exactly the tile at startLoopIdx.  Resolve the M-group
            // outside the MMAD loop so fallback code generation keeps the
            // cumulative-C4 loop body unchanged and does not carry another
            // live scalar.
            GemmCoord completedBlockCoord = scheduler.GetBlockCoord(startLoopIdx);
            // Publish only after this tile's Fixpipe store is globally
            // visible.  GMM2 owns the second ready cacheline, so its aggregate
            // counter cannot race a late GMM1 waiter on the Dispatch line.
            SyncFunc<AscendC::HardEvent::FIX_S>();
            __gm__ int32_t *readyAddr = gmmAddrInfo.dispatchTileReady +
                                        static_cast<int64_t>(completedBlockCoord.m()) * DISPATCH_TILE_READY_STRIDE +
                                        GMM2_READY_OFFSET;
            AtomicAdd(readyAddr, int32_t(1));
        }
    }
    startBlockIdx = (startBlockIdx + coreLoops) % blockNum;
    return readyWaveCount != 0U;
}

}  // namespace MegaMoeCatlassImpl

#endif
