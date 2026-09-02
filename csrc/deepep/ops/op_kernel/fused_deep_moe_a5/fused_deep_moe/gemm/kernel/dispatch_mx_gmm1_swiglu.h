#ifndef CATLASS_GEMM_KERNEL_DISPATCH_MX_GMM1_SWIGLU_H
#define CATLASS_GEMM_KERNEL_DISPATCH_MX_GMM1_SWIGLU_H

#include "ascendc/basic_api/interface/kernel_operator_list_tensor_intf.h"
#include "catlass/catlass.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/detail/callback.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"
#include "catlass/epilogue/tile/tile_swizzle.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"

#include "../../../fused_deep_moe_a5_profile.h"
#include "dynamic_mx_quant.h"
#include "../../../fused_deep_moe_a5_base.h"
#include "../../fused_deep_moe_utils.h"
#include "../../../fused_deep_moe_a5_tiling.h"

constexpr uint32_t STATE_OFFSET = 512;
constexpr uint64_t WIN_STATE_OFFSET = 512 * 1024;
constexpr uint64_t STATE_WIN_OFFSET = 900 * 1024;
constexpr uint64_t GROUP_TOKEN_NUM_OFFSET = FusedDeepMoeSync::GROUP_TOKEN_NUM_OFFSET;
constexpr uint64_t SOFT_SYNC_OFFSET = 964 * 1024;
constexpr uint64_t SHARE_QUANT_SOFT_SYNC_OFFSET = 1000 * 1024;
constexpr uint32_t SELF_STATE_OFFSET = 256 * 1024;
constexpr uint32_t SUM_TMP_TENSOR_SIZE = 1024;
constexpr uint32_t UB_ALIGN = 32;
constexpr uint32_t TOKEN_EXTRA_SPACE = 512;
constexpr uint32_t INT32_COUNT_PER_BLOCK = 8;
constexpr int64_t REDUCE_SUM_WORK_SIZE = 4096;  // 最大支持64k-fp32累加
constexpr int32_t SUB_AIV_NUM = 2;
constexpr int32_t ODD_EVEN_BASE = 2;
constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t GATHER_SECOND_NUM = 2;
constexpr uint32_t TOKEN_READY_FLAG_INDEX = 0;
constexpr uint32_t TOKEN_RESERVED_FLAG_INDEX = 1;
static_assert(TOKEN_RESERVED_FLAG_INDEX == TOKEN_READY_FLAG_INDEX + 1);
constexpr uint8_t DISPATCH_SEND_PRIVATE_FORMAT_V1 = 1;
constexpr uint8_t DISPATCH_RECV_PRIVATE_FORMAT_V1 = 1;
constexpr uint32_t GROUP_INFO_SIZE = FusedDeepMoeSync::GROUP_INFO_SIZE;
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
template <TemplateMC2TypeClass, class BlockMmad_, class BlockEpilogue_, class BlockScheduler_, class ElementGroupList_>
class DispatchMxGmm1Swiglu
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
        GM_ADDR gmX2ReadyState;
        FusedDeepMoeProfileWriter *profile;

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
        uint64_t weightExpertStrideBytes;
        // Methods
        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(GemmCoord const &problemShape_, uint32_t problemCount_, GM_ADDR ptrGroupList_, GM_ADDR ptrA_,
               LayoutA const &layoutA_, GM_ADDR ptrB_, LayoutB const &layoutB_, GM_ADDR ptrMxScaleA_,
               LayoutMxScaleA layoutMxScaleA_, GM_ADDR ptrMxScaleB_, LayoutMxScaleB layoutMxScaleB_, GM_ADDR ptrC_,
               LayoutC const &layoutC_, GM_ADDR gmSwigluOut_, GM_ADDR ptrX2_, GM_ADDR gmX2Scale_,
               GemmCoord const &shareProblemShape_, GM_ADDR ptrShareA_, GM_ADDR ptrShareB_,
               LayoutB const &layoutShareB_, GM_ADDR ptrShareMxScaleA_, GM_ADDR ptrShareMxScaleB_,
               LayoutMxScaleB layoutShareMxScaleB_, GM_ADDR ptrShareC_, LayoutC const &layoutShareC_,
               GM_ADDR gmShareSwigluOut_, GM_ADDR ptrShareX2_, GM_ADDR gmShareX2Scale_, GM_ADDR gmX_,
               GM_ADDR gmExpertIds_, GM_ADDR gmXActiveMask_, GM_ADDR gmMoeSmoothScales_, GM_ADDR gmShareSmoothScales_,
               GM_ADDR gmExpandIdx_, GM_ADDR gmEpSendCount_, GM_ADDR gmExpertTokenNums_, GM_ADDR gmX2ReadyState_,
               const FusedDeepMoeInfo &fusedDeepMoeInfo, FusedDeepMoeProfileWriter *profile_)
            : problemShape(problemShape_),
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
              gmSwigluOut(reinterpret_cast<__gm__ ElementC *>(gmSwigluOut_)),
              ptrX2(reinterpret_cast<__gm__ ElementA *>(ptrX2_)),
              gmX2Scale(reinterpret_cast<__gm__ ElementMxScaleA *>(gmX2Scale_)),
              shareProblemShape(shareProblemShape_),
              ptrShareA(reinterpret_cast<__gm__ ElementA *>(ptrShareA_)),
              ptrShareB(reinterpret_cast<__gm__ ElementB *>(ptrShareB_)),
              layoutShareB(layoutShareB_),
              ptrShareMxScaleA(reinterpret_cast<__gm__ ElementMxScaleA *>(ptrShareMxScaleA_)),
              ptrShareMxScaleB(reinterpret_cast<__gm__ ElementMxScaleB *>(ptrShareMxScaleB_)),
              layoutShareMxScaleB(layoutShareMxScaleB_),
              ptrShareC(reinterpret_cast<__gm__ ElementC *>(ptrShareC_)),
              layoutShareC(layoutShareC_),
              gmShareSwigluOut(reinterpret_cast<__gm__ ElementC *>(gmShareSwigluOut_)),
              ptrShareX2(reinterpret_cast<__gm__ ElementA *>(ptrShareX2_)),
              gmShareX2Scale(reinterpret_cast<__gm__ ElementMxScaleA *>(gmShareX2Scale_)),
              gmX(gmX_),
              gmExpertIds(gmExpertIds_),
              gmXActiveMask(gmXActiveMask_),
              gmMoeSmoothScales(gmMoeSmoothScales_),
              gmShareSmoothScales(gmShareSmoothScales_),
              gmExpandIdx(gmExpandIdx_),
              gmEpSendCount(gmEpSendCount_),
              gmExpertTokenNums(gmExpertTokenNums_),
              gmX2ReadyState(gmX2ReadyState_),
              profile(profile_),
              epRankSize(fusedDeepMoeInfo.epRankSize),
              epRankId(fusedDeepMoeInfo.epRankId),
              moeExpertNum(fusedDeepMoeInfo.moeExpertNum),
              moeExpertNumPerRank(fusedDeepMoeInfo.moeExpertNumPerRank),
              quantMode(fusedDeepMoeInfo.quantMode),
              globalBs(fusedDeepMoeInfo.globalBs),
              bs(fusedDeepMoeInfo.bs),
              topK(fusedDeepMoeInfo.k),
              tokenLen(fusedDeepMoeInfo.h),
              shareN(fusedDeepMoeInfo.shareGmm1HLen),
              weightExpertStrideBytes(fusedDeepMoeInfo.gmm1WeightExpertStrideBytes)
        {}
    };

    // Methods
    CATLASS_DEVICE
    DispatchMxGmm1Swiglu()
    {
        aiCoreGroupNum = AscendC::GetBlockNum();
        subBlockNum = AscendC::GetSubBlockNum();
        aiCoreGroupIdx = AscendC::GetBlockIdx() / subBlockNum;
        aicNum = aiCoreGroupNum;
        aivNum = aiCoreGroupNum * SUB_AIV_NUM;  // 1C2V
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
            return;
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
    CATLASS_DEVICE void operator()(Params const &params);
    __aicore__ inline void WaitGroupTokenNumReady(AscendC::GlobalTensor<int32_t> &groupTokenNumStateTensor,
                                                  uint32_t expected)
    {
        while (true) {
            if (FlushAndGetValue<int32_t>(groupTokenNumStateTensor, 0) == static_cast<int32_t>(expected)) {
                break;
            }
            SPIN_WAIT_CYCLES();
        }
    }

    CATLASS_DEVICE
    void NotifySharedX2Ready(uint32_t groupIdx, uint32_t counterIndex)
    {
        AscendC::GlobalTensor<int32_t> readyTensor;
        readyTensor.SetGlobalBuffer((__gm__ int32_t *)(statusDataSpaceGm + GROUP_TOKEN_NUM_OFFSET) +
                                    groupIdx * GROUP_INFO_SIZE + counterIndex);
        AscendC::LocalTensor<int32_t> notifyLocalTensor = resource.ubBuf.template GetBufferByByte<int32_t>(0);
        AscendC::Duplicate(notifyLocalTensor, static_cast<int32_t>(0), INT32_COUNT_PER_BLOCK);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
        notifyLocalTensor.SetValue(0, 1);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
        AscendC::SetAtomicAdd<int32_t>();
        AscendC::DataCopy(readyTensor, notifyLocalTensor, INT32_COUNT_PER_BLOCK);
        AscendC::SetAtomicNone();
        AscendC::PipeBarrier<PIPE_MTE3>();
    }

    CATLASS_DEVICE
    void NotifyRoutedX2Ready(uint32_t groupIdx)
    {
        AscendC::GlobalTensor<int32_t> readyTensor;
        readyTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(
            gmX2ReadyState + static_cast<uint64_t>(groupIdx) * FusedDeepMoeSync::X2_READY_SLOT_SIZE));
        AscendC::LocalTensor<int32_t> notifyLocalTensor =
            resource.ubBuf.template GetBufferByByte<int32_t>(ArchTag::UB_SIZE - UB_BLOCK_SIZE);
        AscendC::Duplicate(notifyLocalTensor, static_cast<int32_t>(0), INT32_COUNT_PER_BLOCK);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
        notifyLocalTensor.SetValue(FusedDeepMoeSync::X2_READY_COUNTER_INDEX, 1);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
        AscendC::PipeBarrier<PIPE_MTE3>();
        AscendC::SetAtomicAdd<int32_t>();
        AscendC::DataCopy(readyTensor, notifyLocalTensor, INT32_COUNT_PER_BLOCK);
        AscendC::PipeBarrier<PIPE_MTE3>();
        AscendC::SetAtomicNone();
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    CATLASS_DEVICE
    void CleanRoutedX2ReadyState()
    {
        AscendC::LocalTensor<int32_t> zeroLocalTensor =
            resource.ubBuf.template GetBufferByByte<int32_t>(ArchTag::UB_SIZE - UB_BLOCK_SIZE);
        AscendC::Duplicate(zeroLocalTensor, static_cast<int32_t>(0), INT32_COUNT_PER_BLOCK);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        // On AIV, GetBlockIdx() already identifies the logical AIV worker.
        // Do not multiply by the AIC/AIV subblock count: that skips slots and
        // leaves some ready counters uncleared before the next dispatch.
        uint32_t workerIdx = AscendC::GetBlockIdx();
        uint32_t workerCount = AscendC::GetBlockNum();
        for (uint32_t slotIdx = workerIdx; slotIdx < problemCount; slotIdx += workerCount) {
            AscendC::GlobalTensor<int32_t> readyTensor;
            readyTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(
                gmX2ReadyState + static_cast<uint64_t>(slotIdx) * FusedDeepMoeSync::X2_READY_SLOT_SIZE));
            AscendC::DataCopy(readyTensor, zeroLocalTensor, INT32_COUNT_PER_BLOCK);
        }
        AscendC::PipeBarrier<PIPE_MTE3>();
    }

    __aicore__ inline GM_ADDR GetWindStateAddrByRankId(int64_t rankId)
    {
        return Mc2Kernel::GetBaseWindStateAddrByRankId(winContext_, rankId, epRankId) + dataState * WIN_STATE_OFFSET;
    }

    __aicore__ inline GM_ADDR GetWindAddrByRankId(int64_t rankId)
    {
        return Mc2Kernel::GetBaseWindAddrByRankId(winContext_, rankId, epRankId) + winDataSizeOffset +
               rankId * OPT_RANK_OFFSET;
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const &params)
    {
        AscendC::ICachePreLoad(1);
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
        aicSetFunc = {reinterpret_cast<__gm__ int32_t *>(statusDataSpaceGm + SOFT_SYNC_OFFSET),
                      static_cast<int32_t>(AscendC::GetBlockIdx())};
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
                shareQuantTokenStateTensor.SetGlobalBuffer(
                    (__gm__ int32_t *)(statusDataSpaceGm + SHARE_QUANT_SOFT_SYNC_OFFSET));
                uint32_t expected = waitFlagCount * vToCFlag;
                WaitGroupTokenNumReady(shareQuantTokenStateTensor, expected);
            }
            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += aicNum) {
                GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
                GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);

                auto tensorBlockA =
                    GetTile(tensorA, tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.k() * L1_TILE_K),
                            tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));

                auto tensorBlockB =
                    GetTile(tensorB, tla::MakeCoord(blockCoord.k() * L1_TILE_K, blockCoord.n() * L1_TILE_N),
                            tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));

                auto tensorBlockC =
                    GetTile(tensorC, tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.n() * L1_TILE_N),
                            tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));

                auto tensorBlockMxScaleA =
                    GetTile(tensorMxScaleA,
                            tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.k() * L1_TILE_K / MX_SCALE_GROUP_NUM),
                            tla::MakeShape(actualBlockShape.m(), CeilDiv<MX_SCALE_GROUP_NUM>(actualBlockShape.k())));

                auto tensorBlockMxScaleB =
                    GetTile(tensorMxScaleB,
                            tla::MakeCoord(blockCoord.k() * L1_TILE_K / MX_SCALE_GROUP_NUM, blockCoord.n() * L1_TILE_N),
                            tla::MakeShape(CeilDiv<MX_SCALE_GROUP_NUM>(actualBlockShape.k()), actualBlockShape.n()));

                blockMmad(tensorBlockA, tensorBlockB, tensorBlockC, actualBlockShape, tensorBlockMxScaleA,
                          tensorBlockMxScaleB);
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
            int64_t mxScaleAlignedK =
                static_cast<int64_t>(CeilDiv<MX_BASEK_FACTOR>(params.problemShape.k()) * MX_SCALE_COPY_GROUP_NUM);

            int64_t totalM = 0;
            auto tensorA = tla::MakeTensor(gmA, params.layoutA, Arch::PositionGM{});
            auto tensorC = tla::MakeTensor(gmC, params.layoutC, Arch::PositionGM{});

            AscendC::GlobalTensor<int32_t> groupTokenNumStateTensor;
            for (uint32_t groupIdx = 0; groupIdx < params.problemCount; ++groupIdx) {
                uint64_t profStart = 0;
                gmMxScaleA.SetGlobalBuffer(params.ptrMxScaleA + gmGroupOffsetMxScaleA);
                if constexpr (EXEC_FLAG & EXEC_FLAG_TENSOR_LIST) {
                    gmB.SetGlobalBuffer(gmBlistTensorDesc.GetDataPtr<ElementB>(groupIdx));
                    gmMxScaleB.SetGlobalBuffer(gmBScalelistTensorDesc.GetDataPtr<ElementMxScaleB>(groupIdx));
                } else {
                    if (params.weightExpertStrideBytes != 0U) {
                        auto *weightBase =
                            reinterpret_cast<__gm__ uint8_t *>(gmBlistTensorDesc.GetDataPtr<ElementB>(0));
                        gmB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(
                            weightBase + static_cast<uint64_t>(groupIdx) * params.weightExpertStrideBytes));
                    } else {
                        gmB.SetGlobalBuffer(gmBlistTensorDesc.GetDataPtr<ElementB>(0) + gmGroupOffsetB);
                    }
                    gmMxScaleB.SetGlobalBuffer(gmBScalelistTensorDesc.GetDataPtr<ElementMxScaleB>(0) +
                                               gmGroupOffsetMxScaleB);
                }
                if constexpr (EXEC_FLAG & EXEC_FLAG_DEEP_FUSE) {
                    groupTokenNumStateTensor.SetGlobalBuffer(
                        (__gm__ int32_t *)(statusDataSpaceGm + GROUP_TOKEN_NUM_OFFSET) + groupIdx * GROUP_INFO_SIZE);
                    // wait AIV recv needed tokens
                    uint32_t expected = actualRecvCoreNumPerGroup * vToCFlag;
                    WaitGroupTokenNumReady(groupTokenNumStateTensor, expected);
                    callbackAfterFixpipe();
                    currentM = groupTokenNumStateTensor.GetValue(GROUP_TOKEN_COUNT);
                } else {
                    currentM = (groupIdx == 0) ? groupList.GetValue(groupIdx)
                                               : (groupList.GetValue(groupIdx) - groupList.GetValue(groupIdx - 1));
                }
                if (params.profile != nullptr) {
                    profStart = params.profile->Now();
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

                    auto tensorBlockA = GetTile(
                        tensorA, tla::MakeCoord(totalM + blockCoord.m() * L1_TILE_M, blockCoord.k() * L1_TILE_K),
                        tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));

                    auto tensorBlockB =
                        GetTile(tensorB, tla::MakeCoord(blockCoord.k() * L1_TILE_K, blockCoord.n() * L1_TILE_N),
                                tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));

                    auto tensorBlockC = GetTile(
                        tensorC, tla::MakeCoord(totalM + blockCoord.m() * L1_TILE_M, blockCoord.n() * L1_TILE_N),
                        tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));

                    auto tensorBlockMxScaleA = GetTile(
                        tensorMxScaleA,
                        tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.k() * L1_TILE_K / MX_SCALE_GROUP_NUM),
                        tla::MakeShape(actualBlockShape.m(), CeilDiv<MX_SCALE_GROUP_NUM>(actualBlockShape.k())));

                    auto tensorBlockMxScaleB = GetTile(
                        tensorMxScaleB,
                        tla::MakeCoord(blockCoord.k() * L1_TILE_K / MX_SCALE_GROUP_NUM, blockCoord.n() * L1_TILE_N),
                        tla::MakeShape(CeilDiv<MX_SCALE_GROUP_NUM>(actualBlockShape.k()), actualBlockShape.n()));

                    blockMmad(tensorBlockA, tensorBlockB, tensorBlockC, actualBlockShape, tensorBlockMxScaleA,
                              tensorBlockMxScaleB);
                    callbackAfterFixpipe();
                }
                totalM += inGroupProblemShape.m();

                if constexpr (!(EXEC_FLAG & EXEC_FLAG_TENSOR_LIST)) {
                    if (params.weightExpertStrideBytes == 0U) {
                        if constexpr (AscendC::Std::is_one_of_v<ElementB, float4_e2m1x2_t, float4_e1m2x2_t>) {
                            gmGroupOffsetB += std::is_same_v<LayoutB, layout::ColumnMajor>
                                                  ? CeilDiv<2>(inGroupProblemShape.k()) * inGroupProblemShape.n()
                                                  : CeilDiv<2>(inGroupProblemShape.n()) * inGroupProblemShape.k();
                        } else {
                            gmGroupOffsetB += inGroupProblemShape.k() * inGroupProblemShape.n();
                        }
                    }
                    gmGroupOffsetMxScaleB += mxScaleAlignedK * inGroupProblemShape.n();
                }
                gmGroupOffsetMxScaleA += inGroupProblemShape.m() * mxScaleAlignedK;

                startCoreIdx = (startCoreIdx + coreLoops) % aicNum;
                if (params.profile != nullptr) {
                    params.profile->Record(FusedDeepMoeProfileStage::Gmm1, groupIdx, profStart, params.profile->Now());
                }
            }

            if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
                blockMmad.template SynchronizeBlock<decltype(tensorC)>();
            }
        }

        AscendC::PipeBarrier<PIPE_ALL>();
        if constexpr (!(EXEC_FLAG & EXEC_FLAG_DEEP_FUSE)) {
            AscendC::SyncAll<false>();
        }
    }

    CATLASS_DEVICE
    void QuantDynamicMx(AscendC::LocalTensor<ElementA> &outLocal, AscendC::LocalTensor<XType> &inLocal,
                        AscendC::LocalTensor<float> &tokenF32LT, uint32_t quantLength, uint32_t mxScaleNumPerToken)
    {
        __ubuf__ XType *srcAddr = (__ubuf__ XType *)inLocal.GetPhyAddr();
        __ubuf__ uint16_t *maxExpAddr = (__ubuf__ uint16_t *)tokenF32LT.GetPhyAddr();
        __ubuf__ uint16_t *halfScaleLocalAddr = (__ubuf__ uint16_t *)tokenF32LT[mxScaleNumPerToken].GetPhyAddr();
        __ubuf__ int8_t *outLocalAddr = (__ubuf__ int8_t *)outLocal.GetPhyAddr();
        __ubuf__ uint16_t *mxScaleLocalAddr = (__ubuf__ uint16_t *)outLocal[quantLength].GetPhyAddr();

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

        AscendC::Duplicate(statusTensor_, (int32_t)0, expertCntUp * INT32_COUNT_PER_BLOCK);
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
                offset = (epRankId + (rankIndex) % moeExpertNumPerRank * epRankSize) * stateOffset;
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
        AscendC::LocalTensor<int32_t> yInt32Tensor =
            (yInt8Tensor[tokenLength].template ReinterpretCast<ElementMxScaleA>())[x1MxScaleNum]
                .template ReinterpretCast<int32_t>();
        if constexpr (EXEC_FLAG & EXEC_FLAG_SMOOTH_QUANT) {
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
    uint32_t SendToMoeExprt(GM_ADDR gmX, GM_ADDR gmExpandIdx, GM_ADDR gmMoeSmoothScales)
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
            return 0U;
        }
        AscendC::Duplicate(expertCountTensor, (int32_t)0, expertIdsCnt);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(1);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(1);

        AscendC::GlobalTensor<XType> srcWinGMTensor;
        srcWinGMTensor.SetGlobalBuffer((__gm__ XType *)gmX);
        AscendC::GlobalTensor<float> moeSmoothScaleGMTensor;

        if constexpr (EXEC_FLAG & EXEC_FLAG_SMOOTH_QUANT) {
            moeSmoothScaleGMTensor.SetGlobalBuffer((__gm__ float *)gmMoeSmoothScales);
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
                GM_ADDR rankGM = (__gm__ uint8_t *)(GetWindAddrByRankId(tempRankId) +
                                                    (expertPerSizeOnWin * (epRankId * moeExpertNumPerRank +
                                                                           dstExpertId % moeExpertNumPerRank)) +
                                                    hCommuSize * curExpertCnt);
                dstWinGMTensor.SetGlobalBuffer((__gm__ ElementA *)rankGM);

                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventId);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
                AscendC::DataCopy(xInTensor[index], srcWinGMTensor[tokenIndex / axisK * tokenLength], tokenLength);
                if constexpr (EXEC_FLAG & EXEC_FLAG_SMOOTH_QUANT) {
                    AscendC::PipeBarrier<PIPE_MTE2>();
                    AscendC::DataCopy(moeSmoothScaleTensor[index], moeSmoothScaleGMTensor[dstExpertId * tokenLength],
                                      tokenLength);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventId);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventId);
                QuantToken(xInTensor[index], moeSmoothScaleTensor[index], yInt8Tensor[index], ubOffset);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventId);

                AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventId);

                AscendC::DataCopy(dstWinGMTensor, yInt8Tensor[index], tokenLength);
                AscendC::PipeBarrier<PIPE_MTE3>();
                AscendC::DataCopy(dstWinGMTensor[tokenLength], yInt8Tensor[index][tokenLength],
                                  MxByte2Count<ElementA>(scaleFlagSize));
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
        AscendC::DataCopyExtParams expertIdsCntParams = {1U, static_cast<uint32_t>(sendTokenNum * sizeof(uint32_t)), 0U,
                                                         0U, 0U};
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
        AscendC::DataCopyPad(expandIdxGMTensor, expertCountTensor, expertIdsCntParams);
        return sendValidTokenIndex;
    }

    CATLASS_DEVICE void SendCoreFunc(GM_ADDR gmX, GM_ADDR gmExpertIds, GM_ADDR gmMoeSmoothScales, GM_ADDR gmExpandIdx,
                                     GM_ADDR gmXActiveMask, FusedDeepMoeProfileWriter *profile)
    {
        uint64_t profDispatchSendStart = 0;
        if (profile != nullptr) {
            profDispatchSendStart = profile->Now();
        }
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
            if constexpr (EXEC_FLAG & EXEC_FLAG_SMOOTH_QUANT) {
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
        AscendC::DataCopyExtParams expertIdsCntParams = {1U, static_cast<uint32_t>(expertIdsCnt * sizeof(uint32_t)), 0U,
                                                         0U, 0U};
        AscendC::DataCopyPadExtParams<int32_t> copyPadParams{false, 0U, 0U, 0U};
        AscendC::DataCopyPad(expertIdsTensor_, expertIdsGMTensor_, expertIdsCntParams, copyPadParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);

        CalAndSendTokenCount();
        AscendC::PipeBarrier<PIPE_ALL>();
        sendToMoeAivNum = sendCoreNum;
        uint32_t sendValidTokenCount = SendToMoeExprt(gmX, gmExpandIdx, gmMoeSmoothScales);
        AscendC::PipeBarrier<PIPE_ALL>();
        if (profile != nullptr) {
            auto dispatchSendPayload = Cam::ToProfilePrivatePayloadRaw(Cam::MakeDispatchSendPrivatePayloadV1(
                Cam::PROFILE_PRIVATE_DATA_VALID, DISPATCH_SEND_PRIVATE_FORMAT_V1,
                static_cast<uint64_t>(sendValidTokenCount), static_cast<uint64_t>(hCommuSize)));
            profile->Record(FusedDeepMoeProfileStage::DispatchSend, 0U, profDispatchSendStart, profile->Now(),
                            dispatchSendPayload);
        }
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
        srcXGMTensor.SetGlobalBuffer((__gm__ XType *)gmX);
        AscendC::GlobalTensor<ElementA> dstXInt8GMTensor;
        dstXInt8GMTensor.SetGlobalBuffer((__gm__ ElementA *)gmShareX1Token);
        AscendC::GlobalTensor<ElementMxScaleA> dstXScaleGMTensor;
        dstXScaleGMTensor.SetGlobalBuffer((__gm__ ElementMxScaleA *)gmShareX1Scale);
        AscendC::GlobalTensor<float> shareSmoothScaleGMTensor;
        shareSmoothScaleGMTensor.SetGlobalBuffer((__gm__ float *)gmShareSmoothScales);

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
        if constexpr (EXEC_FLAG & EXEC_FLAG_SMOOTH_QUANT) {
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
        AscendC::DataCopyExtParams dataCopyParamsFloat = {1U, (uint32_t)(x1MxScaleNum * sizeof(ElementMxScaleA)), 0U,
                                                          0U, 0U};
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
            AscendC::DataCopyPad(dstXScaleGMTensor[tokenIndex * x1MxScaleNum], yScaleTensor[index],
                                 dataCopyParamsFloat);
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
            (__gm__ int32_t *)(statusDataSpaceGm + SHARE_QUANT_SOFT_SYNC_OFFSET));
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
            AscendC::DataCopy(statusFp32Tensor_,
                              windowInstatusFp32Tensor_[startStatusIndex * stateOffset / sizeof(float)], intriParams);
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
    void RecvToken(GM_ADDR gmX1, GM_ADDR gmX1Scale, uint32_t startRankId, uint32_t startTokenIdx,
                   uint32_t startTokenIdxInRank, uint32_t recvTokenNum)
    {
        AscendC::DataCopyExtParams dataCopyParamsFloat = {1U, (uint32_t)(x1MxScaleNum * sizeof(ElementMxScaleA)), 0U,
                                                          0U, 0U};
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
            tokGlobalInt32.SetGlobalBuffer(
                (__gm__ int32_t *)(wAddr + currentTokenIdxInRank * hCommuSize + hOutSize + scaleSize));
            expandXOutGlobal.SetGlobalBuffer((__gm__ ElementA *)(gmX1 + currentTokenIdx * hOutSize), tokenLength);
            while (true) {
                AscendC::DataCopy(tmpLocalTensor, tokGlobalInt32, INT32_COUNT_PER_BLOCK);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
                if (tmpLocalTensor.GetValue(TOKEN_READY_FLAG_INDEX) == tokenFlag) {
                    break;
                }
                SPIN_WAIT_CYCLES();
            }
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
            AscendC::DataCopy(xTmpTensor_, tokGlobal, MxByte2Count<ElementA>(hCommuSize));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(0);
            SetValueAndFlush<int32_t>(tokGlobalInt32, TOKEN_READY_FLAG_INDEX, 0);
            AscendC::DataCopyPad(dynamicScalesOutGMTensor_[currentTokenIdx * x1MxScaleNum], xOutFp32Tensor_,
                                 dataCopyParamsFloat);
            AscendC::DataCopy(expandXOutGlobal, xTmpTensor_, tokenLength);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);

            curRecvTokenCount += 1;
            currentTokenIdxInRank += 1;
            currentTokenIdx += 1;
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    }

    CATLASS_DEVICE
    void RecvCoreFunc(GM_ADDR gmX1, GM_ADDR gmX1Scale, GM_ADDR gmEpSendCount, FusedDeepMoeProfileWriter *profile)
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

        AscendC::LocalTensor<int32_t> notifyCubeTensor = resource.ubBuf.template GetBufferByByte<int32_t>(ubOffset);

        ubOffset += CEIL_UP((expertCntUp / recvCoreNum + 1) * sizeof(int32_t));
        reduceSumWorkLocalTensor = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
        ubOffset += REDUCE_SUM_WORK_SIZE;

        RecvCount(ubOffset);

        uint32_t recvExpertNum = expertCntUp;
        uint32_t recvCoreNumPerGroup = recvCoreNum;
        uint32_t recvRankNumPerCore = epRankSize / recvCoreNumPerGroup;
        uint32_t remainderRankNum = epRankSize % recvCoreNumPerGroup;
        uint32_t startCoreIdx = 0;

        uint32_t subUbOffset =
            CEIL_UP(expertCntUp * UB_BLOCK_SIZE) + CEIL_UP(UB_BLOCK_SIZE) + CEIL_UP(expertCntUp * sizeof(float));
        uint32_t preExpertToken = 0;
        for (uint32_t groupId = 0; groupId < localExpertNum; ++groupId) {
            uint64_t profDispatchRecvStart = 0;
            uint64_t profDispatchRecvEnd = 0;
            uint64_t profDispatchRecvNotifyStart = 0;
            uint64_t profDispatchRecvNotifyEnd = 0;
            if (profile != nullptr) {
                profDispatchRecvStart = profile->Now();
            }
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
                RecvToken(gmX1, gmX1Scale, startRankId, startTokenIdx + preExpertToken, startTokenIdxInRank,
                          recvTokenPerCore);
            }
            // recv finish, inform AIC
            AscendC::PipeBarrier<PIPE_ALL>();
            if (profile != nullptr) {
                profDispatchRecvEnd = profile->Now();
                profDispatchRecvNotifyStart = profile->Now();
            }
            uint32_t idleCoreNum = recvCoreNum - useCoreNum;
            bool hasToken = coreTokenCount > 0;
            bool isIdleOwner = recvCoreIdx == 0 && idleCoreNum > 0;
            uint32_t notifyCoreCount = hasToken ? 1U : 0U;
            if (isIdleOwner) {
                notifyCoreCount += idleCoreNum;
            }

            if (notifyCoreCount > 0) {
                for (uint32_t index = 0; index < INT32_COUNT_PER_BLOCK; ++index) {
                    notifyCubeTensor.SetValue(index, 0);
                }
                uint32_t cvFlagContribution = static_cast<uint32_t>(vToCFlag) * notifyCoreCount;
                notifyCubeTensor.SetValue(CV_FLAG_INDEX, static_cast<int32_t>(cvFlagContribution));
                notifyCubeTensor.SetValue(GROUP_ID_INDEX, groupId * notifyCoreCount);
                notifyCubeTensor.SetValue(SELF_COUNT_INDEX, coreTokenCount);
                AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);

                AscendC::GlobalTensor<int32_t> groupTokenNumStateTensor;
                groupTokenNumStateTensor.SetGlobalBuffer(
                    (__gm__ int32_t *)(statusDataSpaceGm + GROUP_TOKEN_NUM_OFFSET));
                AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
                AscendC::SetAtomicAdd<int32_t>();
                AscendC::DataCopy(groupTokenNumStateTensor[groupId * GROUP_INFO_SIZE], notifyCubeTensor,
                                  INT32_COUNT_PER_BLOCK);
                AscendC::SetAtomicNone();
                AscendC::PipeBarrier<PIPE_ALL>();
            }
            if (profile != nullptr) {
                profDispatchRecvNotifyEnd = profile->Now();
            }

            startCoreIdx = (startCoreIdx + currentM) % recvCoreNum;
            preExpertToken += currentM;
            if (profile != nullptr) {
                auto dispatchRecvPayload = Cam::ToProfilePrivatePayloadRaw(Cam::MakeDispatchRecvPrivatePayloadV1(
                    Cam::PROFILE_PRIVATE_DATA_VALID, DISPATCH_RECV_PRIVATE_FORMAT_V1,
                    static_cast<uint64_t>(coreTokenCount)));
                profile->Record(FusedDeepMoeProfileStage::DispatchRecv, groupId, profDispatchRecvStart,
                                profDispatchRecvEnd, dispatchRecvPayload);
                profile->Record(FusedDeepMoeProfileStage::DispatchRecvNotify, groupId, profDispatchRecvNotifyStart,
                                profDispatchRecvNotifyEnd, dispatchRecvPayload);
            }
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
        for (uint32_t index = 1; index < sendCountPerCore; ++index) {
            sendCountsLocalTensor(index) = sendCountsLocalTensor(index - 1) +
                                           statusTensor_.GetValue((sendCountStart + index) * INT32_COUNT_PER_BLOCK + 1);
        }
        AscendC::DataCopyExtParams sendCountDataCopyOutParams = {
            1U, static_cast<uint32_t>(sendCountPerCore * sizeof(int32_t)), 0U, 0U, 0U};
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
        AscendC::DataCopyPad(sendCountsGlobal[sendCountStart], sendCountsLocalTensor, sendCountDataCopyOutParams);
    }

    CATLASS_DEVICE
    void AivInitParams(Params const &params)
    {
        problemCount = params.problemCount;
        gmX2ReadyState = params.gmX2ReadyState;
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
        scaleFlagSize = CEIL(scaleSize + sizeof(int32_t), TOKEN_EXTRA_SPACE) * TOKEN_EXTRA_SPACE;  // scale and flag
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
    void AivOnlySync()
    {
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::SyncAll<true>();
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    CATLASS_DEVICE
    void FinalizeGroupMetaAfterRecv(__gm__ ElementGroupList_ *ptrGroupList, GM_ADDR gmEpSendCount,
                                    GM_ADDR gmExpertTokenNums)
    {
        if (aivNum > 0) {
            uint32_t expertPerCore = localExpertNum / aivNum;
            uint32_t remainExpert = localExpertNum % aivNum;
            uint32_t expertStart = expertPerCore * aivIdx + ((aivIdx < remainExpert) ? aivIdx : remainExpert);
            uint32_t expertCount = expertPerCore + ((aivIdx < remainExpert) ? 1 : 0);
            if (expertCount == 0) {
                return;
            }

            AscendC::GlobalTensor<int64_t> expertTokenNumsOutGMTensor_;
            expertTokenNumsOutGMTensor_.SetGlobalBuffer((__gm__ int64_t *)(ptrGroupList));
            AscendC::GlobalTensor<int32_t> sendCountsGlobal;
            sendCountsGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(gmEpSendCount));
            AscendC::GlobalTensor<int64_t> nonCumSumExpertTokenNumsTensor;
            nonCumSumExpertTokenNumsTensor.SetGlobalBuffer((__gm__ int64_t *)gmExpertTokenNums);

            int64_t metaUbOffset = 0;
            uint32_t groupListLocalBytes =
                static_cast<uint32_t>(CEIL_UP(expertCount * static_cast<uint32_t>(sizeof(int64_t))));
            AscendC::LocalTensor<int64_t> groupListLocalTensor =
                resource.ubBuf.template GetBufferByByte<int64_t>(metaUbOffset);
            metaUbOffset += groupListLocalBytes;
            AscendC::LocalTensor<int64_t> expertTokenNumsLocalTensor =
                resource.ubBuf.template GetBufferByByte<int64_t>(metaUbOffset);
            metaUbOffset += groupListLocalBytes;

            uint32_t prevTokenNum = 0;
            if (expertStart > 0) {
                prevTokenNum =
                    FlushAndGetValue<int32_t>(sendCountsGlobal, (expertStart - 1) * epRankSize + epRankSize - 1);
            }

            for (uint32_t expertOffset = 0; expertOffset < expertCount; ++expertOffset) {
                uint32_t localMoeIndex = expertStart + expertOffset;
                uint32_t tokenNum =
                    FlushAndGetValue<int32_t>(sendCountsGlobal, localMoeIndex * epRankSize + epRankSize - 1);
                groupListLocalTensor.SetValue(expertOffset, static_cast<int64_t>(tokenNum));
                expertTokenNumsLocalTensor.SetValue(expertOffset, static_cast<int64_t>(tokenNum - prevTokenNum));
                prevTokenNum = tokenNum;
            }

            AscendC::DataCopyExtParams copyOutParams = {1U, static_cast<uint32_t>(expertCount * sizeof(int64_t)), 0U,
                                                        0U, 0U};
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
            AscendC::DataCopyPad(expertTokenNumsOutGMTensor_[expertStart], groupListLocalTensor, copyOutParams);
            AscendC::DataCopyPad(nonCumSumExpertTokenNumsTensor[expertStart], expertTokenNumsLocalTensor,
                                 copyOutParams);
        }
    }

    CATLASS_DEVICE
    void UpdateAndCleanInfo(__gm__ ElementGroupList_ *ptrGroupList, GM_ADDR gmEpSendCount, GM_ADDR gmExpertTokenNums)
    {
        (void)ptrGroupList;
        (void)gmEpSendCount;
        (void)gmExpertTokenNums;
        if (isCompCore && AscendC::GetSubBlockIdx() == 0) {
            AscendC::GlobalTensor<int32_t> softSyncTensor;
            softSyncTensor.SetGlobalBuffer((__gm__ int32_t *)(statusDataSpaceGm + SOFT_SYNC_OFFSET));
            AscendC::LocalTensor<int32_t> tmpZeroLocalTensor = resource.ubBuf.template GetBufferByByte<int32_t>(0);
            AscendC::Duplicate(tmpZeroLocalTensor, (int32_t)0, INT32_COUNT_PER_BLOCK);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
            AscendC::DataCopy(softSyncTensor[compCoreIdx * CVSoftSync::SOFT_SYNC_SPACE_SIZE / sizeof(int32_t)],
                              tmpZeroLocalTensor, INT32_COUNT_PER_BLOCK);
        }
        if constexpr (EXEC_FLAG & EXEC_FLAG_DEEP_FUSE) {
            if (aivIdx == aiCoreGroupNum * subBlockNum - 1) {
                AscendC::LocalTensor<int32_t> tmpZeroLocalTensor =
                    resource.ubBuf.template GetBufferByByte<int32_t>(512);
                AscendC::Duplicate(tmpZeroLocalTensor, (int32_t)0, INT32_COUNT_PER_BLOCK);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
                if constexpr (EXEC_FLAG & EXEC_FLAG_SHARED_EXPERT) {
                    AscendC::GlobalTensor<int32_t> shareQuantTokenStateTensor;
                    shareQuantTokenStateTensor.SetGlobalBuffer(
                        (__gm__ int32_t *)(statusDataSpaceGm + SHARE_QUANT_SOFT_SYNC_OFFSET));
                    AscendC::DataCopy(shareQuantTokenStateTensor, tmpZeroLocalTensor, INT32_COUNT_PER_BLOCK);
                }
            }
        }
    }

    CATLASS_DEVICE
    uint32_t GetAssignedLoopCount(uint32_t coreLoops, uint32_t coreNum, uint32_t producerCore, uint32_t startCoreIdx)
    {
        uint32_t startLoopIdx = (producerCore + coreNum - startCoreIdx) % coreNum;
        if (startLoopIdx >= coreLoops) {
            return 0;
        }
        return 1 + (coreLoops - 1 - startLoopIdx) / coreNum;
    }

    CATLASS_DEVICE
    uint32_t GetBlockLoopIdx(uint32_t mLoops, uint32_t nLoops, uint32_t mBlock, uint32_t nBlock)
    {
        if constexpr (GMM1_SWIZZLE_DIRECTION == 0) {
            uint32_t tileBlockIdx = mBlock / GMM1_SWIZZLE_OFFSET;
            uint32_t tileBlockLoop = CEIL(mLoops, GMM1_SWIZZLE_OFFSET);
            uint32_t nRow = GMM1_SWIZZLE_OFFSET;
            if (tileBlockIdx == tileBlockLoop - 1) {
                nRow = mLoops - GMM1_SWIZZLE_OFFSET * tileBlockIdx;
            }
            uint32_t inTileBlockRow = mBlock - tileBlockIdx * GMM1_SWIZZLE_OFFSET;
            uint32_t nInnerIdx = (tileBlockIdx % 2 == 1) ? (nLoops - nBlock - 1) : nBlock;
            return tileBlockIdx * (GMM1_SWIZZLE_OFFSET * nLoops) + nInnerIdx * nRow + inTileBlockRow;
        } else {
            uint32_t tileBlockIdx = nBlock / GMM1_SWIZZLE_OFFSET;
            uint32_t tileBlockLoop = CEIL(nLoops, GMM1_SWIZZLE_OFFSET);
            uint32_t nCol = GMM1_SWIZZLE_OFFSET;
            if (tileBlockIdx == tileBlockLoop - 1) {
                nCol = nLoops - GMM1_SWIZZLE_OFFSET * tileBlockIdx;
            }
            uint32_t mInnerIdx = (tileBlockIdx % 2 == 1) ? (mLoops - mBlock - 1) : mBlock;
            uint32_t nInTileBlock = nBlock - tileBlockIdx * GMM1_SWIZZLE_OFFSET;
            return tileBlockIdx * (GMM1_SWIZZLE_OFFSET * mLoops) + mInnerIdx * nCol + nInTileBlock;
        }
    }

    CATLASS_DEVICE
    int32_t GetProducerCoreForLoop(uint32_t loopIdx, uint32_t coreNum, uint32_t startCoreIdx)
    {
        return static_cast<int32_t>((startCoreIdx + loopIdx) % coreNum);
    }

    CATLASS_DEVICE
    uint32_t GetTargetForLoop(uint32_t completedCount, uint32_t loopIdx, uint32_t coreNum)
    {
        return completedCount + loopIdx / coreNum + 1;
    }

    CATLASS_DEVICE
    void SyncSwigluOutBeforeMul()
    {
        // BlockEpilogue publishes its final GM write with MTE3_V(0). Consume
        // that event before Mul/Quant starts reading swigluOut from GM. The
        // event is then re-armed for the next BlockEpilogue stage (or the
        // BlockEpilogue destructor when this is the final stage).
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
    }

    CATLASS_DEVICE
    void ProcessMulAndQuantFromSwigluOut(__gm__ ElementC *swigluOutAddr, __gm__ ElementA *x2Addr,
                                         __gm__ ElementMxScaleA *x2ScaleAddr, uint32_t maxTokenNum, uint32_t gmm1HLen,
                                         uint32_t gmm2HLen, uint32_t rowOffset, uint32_t leftColOffset,
                                         uint32_t rightColOffset, GemmCoord const &leftActualBlockShape,
                                         GemmCoord const &rightActualBlockShape)
    {
        constexpr uint32_t MUL_TILE_M = 8;
        constexpr uint32_t MUL_TILE_N = L1_TILE_N;
        constexpr uint32_t MX_GROUP_SIZE = 32;
        constexpr uint32_t MUL_TILE_ELEMENT_COUNT = MUL_TILE_M * MUL_TILE_N;
        constexpr uint32_t QUANT_ROW_SCALE_COUNT = MUL_TILE_N / MX_GROUP_SIZE;
        constexpr uint32_t EPILOGUE_UB_SIZE = BlockEpilogue::UB_STAGES * BlockEpilogue::TILE_COUNT *
                                              (sizeof(ElementC) + sizeof(XType) + sizeof(ElementC));
        constexpr uint32_t MUL_UB_SIZE = 2 * MUL_TILE_M * MUL_TILE_N * sizeof(ElementC);
        constexpr uint32_t MUL_BUFFER_SIZE = MUL_TILE_M * MUL_TILE_N * sizeof(ElementC);
        constexpr uint32_t QUANT_INPUT_SIZE = MUL_TILE_ELEMENT_COUNT * sizeof(XType);
        constexpr uint32_t QUANT_OUTPUT_SIZE = MxCount2Byte<ElementA>(MUL_TILE_N);
        constexpr uint32_t QUANT_SCALE_SIZE = CEIL_UP(QUANT_ROW_SCALE_COUNT * sizeof(ElementMxScaleA));
        constexpr uint32_t QUANT_SCRATCH_SIZE = CEIL_UP(QUANT_ROW_SCALE_COUNT * 2 * sizeof(float));
        constexpr uint32_t QUANT_WORKSPACE_SIZE =
            QUANT_INPUT_SIZE + QUANT_OUTPUT_SIZE + QUANT_SCALE_SIZE + QUANT_SCRATCH_SIZE;
        static_assert(L1_TILE_N % MX_GROUP_SIZE == 0, "Routed quant tile must contain complete MX groups");
        static_assert((L1_TILE_N / MX_GROUP_SIZE) % 2 == 0,
                      "Routed quant tile must contain an even number of MX groups per row");
        static_assert(EPILOGUE_UB_SIZE + MUL_UB_SIZE <= ArchTag::UB_SIZE,
                      "Swiglu epilogue and routed mul buffers exceed UB capacity");
        static_assert(QUANT_WORKSPACE_SIZE <= MUL_BUFFER_SIZE,
                      "Routed quant workspace does not fit in the reusable right mul buffer");

        uint32_t actualPairM = leftActualBlockShape.m();
        uint32_t actualPairN = (leftActualBlockShape.n() < rightActualBlockShape.n()) ? leftActualBlockShape.n()
                                                                                      : rightActualBlockShape.n();
        if (actualPairM == 0 || actualPairN == 0) {
            return;
        }

        auto tileShape = MakeCoord(BlockEpilogue::TILE_M, BlockEpilogue::TILE_N);
        Catlass::Epilogue::Tile::EpilogueHorizontalTileSwizzle epilogueTileSwizzle(MakeCoord(actualPairM, actualPairN),
                                                                                   tileShape);
        uint32_t tileLoops = epilogueTileSwizzle.GetLoops();
        uint32_t subblockIdx = AscendC::GetSubBlockIdx();
        uint32_t subblockNum = AscendC::GetSubBlockNum();

        uint32_t ubOffset = EPILOGUE_UB_SIZE;
        AscendC::LocalTensor<ElementC> leftLocalTensor = resource.ubBuf.template GetBufferByByte<ElementC>(ubOffset);
        ubOffset += MUL_BUFFER_SIZE;
        AscendC::LocalTensor<ElementC> rightLocalTensor = resource.ubBuf.template GetBufferByByte<ElementC>(ubOffset);
        // Mul no longer needs the right operand, so quant reuses that 8 KiB region.
        uint32_t quantUbOffset = ubOffset;
        AscendC::LocalTensor<XType> quantInputLocalTensor =
            resource.ubBuf.template GetBufferByByte<XType>(quantUbOffset);
        quantUbOffset += QUANT_INPUT_SIZE;
        AscendC::LocalTensor<ElementA> quantOutputLocalTensor =
            resource.ubBuf.template GetBufferByByte<ElementA>(quantUbOffset);
        quantUbOffset += QUANT_OUTPUT_SIZE + QUANT_SCALE_SIZE;
        AscendC::LocalTensor<float> quantScratchLocalTensor =
            resource.ubBuf.template GetBufferByByte<float>(quantUbOffset);

        AscendC::GlobalTensor<ElementC> gmSwigluOutTensor;
        gmSwigluOutTensor.SetGlobalBuffer(swigluOutAddr);
        AscendC::GlobalTensor<ElementA> gmX2Tensor;
        gmX2Tensor.SetGlobalBuffer(x2Addr);
        AscendC::GlobalTensor<uint8_t> gmX2ScaleTensor;
        gmX2ScaleTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(x2ScaleAddr));

        auto fullLayout = tla::MakeLayout<ElementC, Catlass::layout::RowMajor>(maxTokenNum, gmm1HLen);
        auto fullTensor = tla::MakeTensor(gmSwigluOutTensor, fullLayout, Arch::PositionGM{});

        auto leftBlockTensor = GetTile(fullTensor, tla::MakeCoord(rowOffset, leftColOffset),
                                       tla::MakeShape(actualPairM, leftActualBlockShape.n()));
        auto rightBlockTensor = GetTile(fullTensor, tla::MakeCoord(rowOffset, rightColOffset),
                                        tla::MakeShape(actualPairM, rightActualBlockShape.n()));
        uint32_t mxScaleNumPerToken = CeilDiv(CeilDiv(gmm2HLen, MX_GROUP_SIZE), 2) * 2;

        constexpr int32_t MUL_EVENT_ID = 2;  // Need to be distinguished from the event in blockEpilogue
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(MUL_EVENT_ID);
        for (uint32_t loopIdx = subblockIdx; loopIdx < tileLoops; loopIdx += subblockNum) {
            auto tileCoord = epilogueTileSwizzle.GetTileCoord(loopIdx);
            auto actualTileShape = epilogueTileSwizzle.GetActualTileShape(tileCoord);
            MatrixCoord tileOffsetInBlock = tileCoord * tileShape;
            uint32_t tileOffsetInBlockRow = tileOffsetInBlock.row();
            uint32_t tileOffsetInBlockColumn = tileOffsetInBlock.column();

            for (uint32_t rowInTile = 0; rowInTile < actualTileShape.row(); rowInTile += MUL_TILE_M) {
                uint32_t actualMulM =
                    (actualTileShape.row() - rowInTile < MUL_TILE_M) ? (actualTileShape.row() - rowInTile) : MUL_TILE_M;
                uint32_t actualMulN = actualTileShape.column();
                uint32_t count = actualMulM * actualMulN;
                uint32_t subTileRow = tileOffsetInBlockRow + rowInTile;

                auto leftSubTensor = GetTile(leftBlockTensor, tla::MakeCoord(subTileRow, tileOffsetInBlockColumn),
                                             tla::MakeShape(actualMulM, actualMulN));
                auto rightSubTensor = GetTile(rightBlockTensor, tla::MakeCoord(subTileRow, tileOffsetInBlockColumn),
                                              tla::MakeShape(actualMulM, actualMulN));
                auto layoutUb =
                    tla::MakeLayout(tla::MakeShape(actualMulM, actualMulN), tla::MakeStride(actualMulN, tla::Int<1>{}));
                auto leftUbTensor = tla::MakeTensor(leftLocalTensor, layoutUb, Arch::PositionUB{});
                auto rightUbTensor = tla::MakeTensor(rightLocalTensor, layoutUb, Arch::PositionUB{});

                using CopyGmToUb = typename Catlass::Epilogue::Tile::CopyGm2UbTla<ArchTag, decltype(leftSubTensor),
                                                                                  decltype(leftUbTensor)>;
                CopyGmToUb copyGmToUb;

                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(MUL_EVENT_ID);
                copyGmToUb(leftUbTensor, leftSubTensor);
                copyGmToUb(rightUbTensor, rightSubTensor);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(MUL_EVENT_ID);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(MUL_EVENT_ID);
                AscendC::Mul(leftLocalTensor, leftLocalTensor, rightLocalTensor, count);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(quantInputLocalTensor, leftLocalTensor, AscendC::RoundMode::CAST_RINT, count);
                AscendC::PipeBarrier<PIPE_V>();
                uint32_t mxScaleNumPerRow = actualMulN / MX_GROUP_SIZE;
                uint32_t outputColOffset = leftColOffset + tileOffsetInBlockColumn;
                uint32_t outputScaleColOffset = outputColOffset / MX_GROUP_SIZE;
                AscendC::DataCopyExtParams mxScaleCopyParams = {
                    1U, static_cast<uint32_t>(mxScaleNumPerRow * sizeof(ElementMxScaleA)), 0U, 0U, 0U};
                // UB-to-GM DataCopyPad advances source blocks at 32-byte granularity, so emit each 8-byte scale row
                // from the aligned single-row quant output instead of treating tightly packed rows as a 2D source.
                for (uint32_t rowIdx = 0; rowIdx < actualMulM; ++rowIdx) {
                    if (rowIdx > 0) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(MUL_EVENT_ID);
                    }
                    AscendC::LocalTensor<XType> quantInputRowTensor = quantInputLocalTensor[rowIdx * actualMulN];
                    QuantDynamicMx(quantOutputLocalTensor, quantInputRowTensor, quantScratchLocalTensor, actualMulN,
                                   mxScaleNumPerRow);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(MUL_EVENT_ID);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(MUL_EVENT_ID);

                    AscendC::LocalTensor<uint8_t> mxScaleLocalTensor =
                        quantOutputLocalTensor[actualMulN].template ReinterpretCast<uint8_t>();
                    uint32_t outputRow = rowOffset + subTileRow + rowIdx;
                    AscendC::DataCopy(gmX2Tensor[outputRow * gmm2HLen + outputColOffset], quantOutputLocalTensor,
                                      actualMulN);
                    AscendC::DataCopyPad(gmX2ScaleTensor[outputRow * mxScaleNumPerToken + outputScaleColOffset],
                                         mxScaleLocalTensor, mxScaleCopyParams);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(MUL_EVENT_ID);
                }
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(MUL_EVENT_ID);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(MUL_EVENT_ID);
            }
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(MUL_EVENT_ID);
    }

    CATLASS_DEVICE
    void PostSwigluDynamicQuant(__gm__ ElementC *swigluOutAddr, __gm__ ElementA *x2Addr,
                                __gm__ ElementMxScaleA *x2ScaleAddr, uint32_t tokenNum, uint32_t mmOutDim,
                                uint32_t &startCoreIdx)
    {
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
        AscendC::LocalTensor<ElementC> fp32TokenLocalTensor =
            resource.ubBuf.template GetBufferByByte<ElementC>(ubOffset);
        ubOffset += mmOutDim * sizeof(ElementC);
        AscendC::LocalTensor<XType> bf16TokenLocalTensor = resource.ubBuf.template GetBufferByByte<XType>(ubOffset);
        ubOffset += mmOutDim * sizeof(XType);
        AscendC::LocalTensor<ElementA> fp8TokenLocalTensor =
            resource.ubBuf.template GetBufferByByte<ElementA>(ubOffset);
        ubOffset += quantTokenSize + CEIL_UP(mxScaleNumPerToken * sizeof(ElementMxScaleB));
        AscendC::LocalTensor<uint8_t> mxScaleLocalTensor =
            fp8TokenLocalTensor[quantLength].template ReinterpretCast<uint8_t>();
        AscendC::LocalTensor<ElementC> tokenF32LT = resource.ubBuf.template GetBufferByByte<ElementC>(ubOffset);
        ubOffset += CEIL_UP(mxScaleNumPerToken * 2 * sizeof(float));
        AscendC::DataCopyExtParams mnxScaleParams = {1U, static_cast<uint8_t>(mxScaleNumPerToken * sizeof(uint8_t)), 0U,
                                                     0U, 0U};
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
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const &params)
    {
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
                             (GM_ADDR)params.gmExpandIdx, (GM_ADDR)params.gmXActiveMask, params.profile);
            }
            if (isRecvCore) {
                RecvCoreFunc((GM_ADDR)params.ptrA, (GM_ADDR)params.ptrMxScaleA, (GM_ADDR)params.gmEpSendCount,
                             params.profile);
            }
            CleanRoutedX2ReadyState();
            AivOnlySync();
            FinalizeGroupMetaAfterRecv(params.ptrGroupList, params.gmEpSendCount, params.gmExpertTokenNums);
            AivOnlySync();
        }

        uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t coreNum = AscendC::GetBlockNum();

        AscendC::GlobalTensor<ElementC> gmC;
        AscendC::GlobalTensor<ElementC> gmSwigluOutTensor;
        AscendC::GlobalTensor<ElementC> gmShareSwigluOutTensor;
        uint32_t startCoreIdx = 0;

        // Keep the BlockEpilogue lifetime bounded to the GMM1/Swiglu stage.
        // Its destructor drains the event-0 pipeline before the later
        // global synchronization and status cleanup stages.
        {
            BlockEpilogue blockEpilogue(resource);
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

                    auto tensorBlockC =
                        GetTile(tensorC, tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.n() * L1_TILE_N),
                                tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));

                    auto tensorBlockD =
                        GetTile(tensorD, tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.n() * L1_TILE_N),
                                tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));

                    bool isLeft = (blockCoord.n() * L1_TILE_N < params.shareProblemShape.n() / 2);
                    CheckSyncFlag(reinterpret_cast<__gm__ int32_t *>(statusDataSpaceGm + SOFT_SYNC_OFFSET),
                                  static_cast<int32_t>(compCoreIdx), target);
                    target += 1;
                    blockEpilogue(tensorBlockC, tensorBlockD, actualBlockShape, isLeft);
                }
                startCoreIdx = (startCoreIdx + coreLoops) % coreNum;
            }
            {
                int64_t totalM = 0;
                gmC.SetGlobalBuffer(params.ptrC);
                gmSwigluOutTensor.SetGlobalBuffer(params.gmSwigluOut);
                AscendC::GlobalTensor<ElementGroupList> groupList;
                groupList.SetGlobalBuffer(params.ptrGroupList);

                auto tensorC = tla::MakeTensor(gmC, params.layoutC, Arch::PositionGM{});
                auto tensorD = tla::MakeTensor(gmSwigluOutTensor, params.layoutC, Arch::PositionGM{});
                AscendC::GlobalTensor<int32_t> groupTokenNumStateTensor;
                constexpr uint32_t MAX_PRODUCER_CORE_NUM = 256;
                uint32_t producerCompletedCount[MAX_PRODUCER_CORE_NUM];
                for (uint32_t producerCore = 0; producerCore < MAX_PRODUCER_CORE_NUM; ++producerCore) {
                    producerCompletedCount[producerCore] = 0;
                }
                if constexpr (EXEC_FLAG & EXEC_FLAG_SHARED_EXPERT) {
                    GemmCoord sharedProblemShape{axisBS, params.shareProblemShape.n(), params.shareProblemShape.k()};
                    BlockScheduler sharedBlockScheduler(sharedProblemShape, MakeCoord(L1_TILE_M, L1_TILE_N));
                    uint32_t sharedCoreLoops = sharedBlockScheduler.GetCoreLoops();
                    for (uint32_t producerCore = 0; producerCore < coreNum; ++producerCore) {
                        producerCompletedCount[producerCore] +=
                            GetAssignedLoopCount(sharedCoreLoops, coreNum, producerCore, 0);
                    }
                }

                for (uint32_t groupIdx = 0; groupIdx < params.problemCount; ++groupIdx) {
                    if constexpr (EXEC_FLAG & EXEC_FLAG_DEEP_FUSE) {
                        groupTokenNumStateTensor.SetGlobalBuffer(
                            (__gm__ int32_t *)(statusDataSpaceGm + GROUP_TOKEN_NUM_OFFSET) +
                            groupIdx * GROUP_INFO_SIZE);
                        CheckSyncFlag(reinterpret_cast<__gm__ int32_t *>(statusDataSpaceGm + SOFT_SYNC_OFFSET),
                                      static_cast<int32_t>(compCoreIdx), target);
                        target += 1;
                        currentM = FlushAndGetValue<int32_t>(groupTokenNumStateTensor, GROUP_TOKEN_COUNT);
                        for (uint32_t producerCore = 0; producerCore < coreNum; ++producerCore) {
                            producerCompletedCount[producerCore] += 1;
                        }
                    } else {
                        currentM = (groupIdx == 0) ? groupList.GetValue(groupIdx)
                                                   : (groupList.GetValue(groupIdx) - groupList.GetValue(groupIdx - 1));
                    }
                    GemmCoord inGroupProblemShape{currentM, params.problemShape.n(), params.problemShape.k()};
                    BlockScheduler matmulBlockScheduler(inGroupProblemShape, MakeCoord(L1_TILE_M, L1_TILE_N));
                    uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();
                    uint32_t groupStartCoreIdx = startCoreIdx;
                    uint32_t mLoops = CEIL(currentM, L1_TILE_M);
                    uint32_t nLoops = CEIL(params.problemShape.n(), L1_TILE_N);
                    uint32_t routedHalfN = params.problemShape.n() / 2;

                    uint32_t startLoopIdx;
                    if (coreIdx < startCoreIdx) {
                        startLoopIdx = coreIdx + coreNum - startCoreIdx;
                    } else {
                        startLoopIdx = coreIdx - startCoreIdx;
                    }

                    for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += coreNum) {
                        GemmCoord rightBlockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
                        uint32_t rightColOffset = rightBlockCoord.n() * L1_TILE_N;
                        if (rightColOffset < routedHalfN) {
                            continue;
                        }

                        uint32_t leftColOffset = rightColOffset - routedHalfN;
                        uint32_t leftNBlock = leftColOffset / L1_TILE_N;
                        GemmCoord leftBlockCoord{rightBlockCoord.m(), leftNBlock, 0};
                        GemmCoord rightActualBlockShape = matmulBlockScheduler.GetActualBlockShape(rightBlockCoord);
                        GemmCoord leftActualBlockShape = matmulBlockScheduler.GetActualBlockShape(leftBlockCoord);

                        uint32_t leftLoopIdx = GetBlockLoopIdx(mLoops, nLoops, leftBlockCoord.m(), leftBlockCoord.n());
                        int32_t leftProducerCore = GetProducerCoreForLoop(leftLoopIdx, coreNum, groupStartCoreIdx);
                        int32_t rightProducerCore = GetProducerCoreForLoop(loopIdx, coreNum, groupStartCoreIdx);
                        uint32_t leftTarget =
                            GetTargetForLoop(producerCompletedCount[leftProducerCore], leftLoopIdx, coreNum);
                        uint32_t rightTarget =
                            GetTargetForLoop(producerCompletedCount[rightProducerCore], loopIdx, coreNum);

                        auto tensorLeftBlockC =
                            GetTile(tensorC, tla::MakeCoord(totalM + leftBlockCoord.m() * L1_TILE_M, leftColOffset),
                                    tla::MakeShape(leftActualBlockShape.m(), leftActualBlockShape.n()));
                        auto tensorLeftBlockD =
                            GetTile(tensorD, tla::MakeCoord(totalM + leftBlockCoord.m() * L1_TILE_M, leftColOffset),
                                    tla::MakeShape(leftActualBlockShape.m(), leftActualBlockShape.n()));
                        auto tensorRightBlockC =
                            GetTile(tensorC, tla::MakeCoord(totalM + rightBlockCoord.m() * L1_TILE_M, rightColOffset),
                                    tla::MakeShape(rightActualBlockShape.m(), rightActualBlockShape.n()));
                        auto tensorRightBlockD =
                            GetTile(tensorD, tla::MakeCoord(totalM + rightBlockCoord.m() * L1_TILE_M, rightColOffset),
                                    tla::MakeShape(rightActualBlockShape.m(), rightActualBlockShape.n()));

                        uint64_t profSwigluStart = 0;
                        if (params.profile != nullptr) {
                            profSwigluStart = params.profile->Now();
                        }
                        CheckSyncFlag(reinterpret_cast<__gm__ int32_t *>(statusDataSpaceGm + SOFT_SYNC_OFFSET),
                                      leftProducerCore, leftTarget);
                        blockEpilogue(tensorLeftBlockC, tensorLeftBlockD, leftActualBlockShape, true);
                        CheckSyncFlag(reinterpret_cast<__gm__ int32_t *>(statusDataSpaceGm + SOFT_SYNC_OFFSET),
                                      rightProducerCore, rightTarget);
                        blockEpilogue(tensorRightBlockC, tensorRightBlockD, rightActualBlockShape, false);
                        SyncSwigluOutBeforeMul();

                        if (params.profile != nullptr) {
                            params.profile->Record(FusedDeepMoeProfileStage::Swiglu, groupIdx, profSwigluStart,
                                                   params.profile->Now());
                        }

                        uint64_t profMulQuantStart = 0;
                        if (params.profile != nullptr) {
                            profMulQuantStart = params.profile->Now();
                        }
                        ProcessMulAndQuantFromSwigluOut(params.gmSwigluOut, params.ptrX2, params.gmX2Scale,
                                                        params.problemShape.m(), params.problemShape.n(), routedHalfN,
                                                        totalM + rightBlockCoord.m() * L1_TILE_M, leftColOffset,
                                                        rightColOffset, leftActualBlockShape, rightActualBlockShape);

                        if (params.profile != nullptr) {
                            params.profile->Record(FusedDeepMoeProfileStage::Quant, groupIdx, profMulQuantStart,
                                                   params.profile->Now());
                        }
                    }

                    if constexpr (EXEC_FLAG & EXEC_FLAG_DEEP_FUSE) {
                        NotifyRoutedX2Ready(groupIdx);
                    }

                    totalM += inGroupProblemShape.m();
                    target += GetAssignedLoopCount(coreLoops, coreNum, compCoreIdx, groupStartCoreIdx);
                    for (uint32_t producerCore = 0; producerCore < coreNum; ++producerCore) {
                        producerCompletedCount[producerCore] +=
                            GetAssignedLoopCount(coreLoops, coreNum, producerCore, groupStartCoreIdx);
                    }

                    startCoreIdx = (startCoreIdx + coreLoops) % coreNum;
                }
                AscendC::PipeBarrier<PIPE_ALL>();
            }
        }
        icache_preload(8);
        if constexpr (EXEC_FLAG & EXEC_FLAG_DEEP_FUSE) {
            AscendC::SyncAll<true>();
        } else {
            AscendC::SyncAll<false>();
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        UpdateAndCleanInfo(params.ptrGroupList, params.gmEpSendCount, params.gmExpertTokenNums);
        AscendC::PipeBarrier<PIPE_ALL>();
        startCoreIdx = 0;
        if constexpr (EXEC_FLAG & EXEC_FLAG_SHARED_EXPERT) {
            PostSwigluDynamicQuant(params.gmShareSwigluOut, params.ptrShareX2, params.gmShareX2Scale, axisBS,
                                   params.shareProblemShape.n(), startCoreIdx);
            if constexpr (EXEC_FLAG & EXEC_FLAG_DEEP_FUSE) {
                NotifySharedX2Ready(0, FusedDeepMoeSync::SHARED_X2_DONE_COUNT_INDEX);
            }
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
    uint32_t problemCount{0};
    GM_ADDR gmX2ReadyState{nullptr};

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
    bool isCompCore{false};        // calculate deq_swiglu
    bool isShareQuantCore{false};  // calculate share quant
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

#endif  // (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)

}  // namespace Catlass::Gemm::Kernel

#endif  // CATLASS_GEMM_KERNEL_DISPATCH_MX_GMM1_SWIGLU_H
