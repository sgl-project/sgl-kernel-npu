/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define CATLASS_ARCH 3510

#include "catlass/arch/kda_arch.hpp"
#include "catlass/arch/kda_cross_core_sync.hpp"
#include "catlass/arch/kda_resource.hpp"
#include "catlass/kda_catlass.hpp"
#include "catlass/kda_debug.hpp"
#include "../block/block_scheduler_kda_fwd_h_arch35.hpp"
#include "catlass/epilogue/block/kda_block_epilogue.hpp"
#include "../../epilogue/block/block_epilogue_kda_fwdh_update_arch35.hpp"
#include "../../epilogue/block/block_epilogue_kda_fwdh_vnew_arch35.hpp"
#include "catlass/gemm/block/kda_block_mmad.hpp"
#include "../../../../kernel_utils/block/block_mmad_pingpong_tla_kernel_utils.hpp"
#include "../../../../kernel_utils/block/block_mmad_pingpong_tla_multi.hpp"
#include "../../../../kernel_utils/block/block_mmad_pingpong_tla_preloadA_l1B.hpp"
#include "catlass/gemm/kda_gemm_dispatch_policy.hpp"
#include "catlass/gemm/kda_gemm_type.hpp"
#include "catlass/layout/kda_layout.hpp"
#include "catlass/kda_gemm_coord.hpp"
#include "tla/kda_tensor.hpp"
#include "tla/kda_tla_layout.hpp"
#include "tla/kda_tensor.hpp"

using _0 = tla::Int<0>;
using _1 = tla::Int<1>;
using _2 = tla::Int<2>;
using _4 = tla::Int<4>;
using _8 = tla::Int<8>;
using _16 = tla::Int<16>;
using _32 = tla::Int<32>;
using _64 = tla::Int<64>;
using _128 = tla::Int<128>;
using _256 = tla::Int<256>;
using _512 = tla::Int<512>;
using _1024 = tla::Int<1024>;
using _2048 = tla::Int<2048>;
using _4096 = tla::Int<4096>;
using _8192 = tla::Int<8192>;
using _16384 = tla::Int<16384>;
using _32768 = tla::Int<32768>;
using _65536 = tla::Int<65536>;

#include "kernel_operator.h"
using namespace Catlass;
using namespace tla;

namespace Catlass::Gemm::Kernel {

struct KdaFwdHTileShapes128 {
    using L1TileShape = tla::Shape<_128, _128, _128>;
    using L0TileShape = L1TileShape;
};

struct KdaFwdHTileShapes256 {
    using L1TileShape = tla::Shape<_128, _256, _128>;
    using L0TileShape = tla::Shape<_128, _256, _64>;
};

template <bool KGated, bool ScalarGated, bool UseExp2>
struct KdaFwdHGateTag {
    static constexpr bool value = KGated;
    static constexpr bool scalarGated = ScalarGated;
    static constexpr bool useExp2 = UseExp2;
};

template <typename INPUT_TYPE, typename G_TYPE, typename STATE_TYPE, typename WORKSPACE_TYPE,
          typename TileShapes = KdaFwdHTileShapes128, bool kGated = false, bool scalarGated = true,
          bool useExp2 = false>
class KdaFwdHKernel {
public:
    using ArchTag = Arch::Ascend950;
    using CubeScheduler = typename Catlass::Gemm::Block::BlockSchedulerKdaFwdHCube;
    using VecScheduler = typename Catlass::Gemm::Block::BlockSchedulerKdaFwdHVec;

    using DispatchPolicyTlaMulti = Gemm::MmadPingpongTlaMulti<ArchTag, false, false, 2>;
    using DispatchPolicyTlaTail = Gemm::MmadPingpongTlaMulti<ArchTag, true, false, 1>;
    using DispatchPolicyDirectUb = Common::MmadPingpong<ArchTag, false, false, 2>;
    using DispatchPolicyTlaPreloadAL1B = Gemm::MmadPingpongTlaPreloadAL1B<ArchTag, true>;
    using L1TileShapeVTla = typename TileShapes::L1TileShape;
    using L0TileShapeVTla = typename TileShapes::L0TileShape;

    using WType = Gemm::GemmType<INPUT_TYPE, layout::RowMajor>;
    using HType = Gemm::GemmType<INPUT_TYPE, layout::RowMajor>;
    using VworkType = Gemm::GemmType<WORKSPACE_TYPE, layout::RowMajor>;
    using KType = Gemm::GemmType<INPUT_TYPE, layout::ColumnMajor>;
    using HworkType = Gemm::GemmType<WORKSPACE_TYPE, layout::RowMajor>;
    using VType = Gemm::GemmType<INPUT_TYPE, layout::RowMajor>;
    using GType = Gemm::GemmType<G_TYPE, layout::RowMajor>;
    using UType = Gemm::GemmType<INPUT_TYPE, layout::RowMajor>;
    using FinalStateType = Gemm::GemmType<STATE_TYPE, layout::RowMajor>;
    using VUpdateType = Gemm::GemmType<INPUT_TYPE, layout::zN>;

    // cube 1
    using TileCopyWH = Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, INPUT_TYPE, layout::RowMajor, INPUT_TYPE,
                                                              layout::RowMajor, WORKSPACE_TYPE, layout::RowMajor>;
    using TileCopyWHDirectUb = Common::Tile::PackedTileCopyTlaToUB<ArchTag, INPUT_TYPE, layout::RowMajor, INPUT_TYPE,
                                                                   layout::RowMajor, WORKSPACE_TYPE, layout::RowMajor,
                                                                   void, Gemm::Tile::CopyL0CToUBMode::NO_SPLIT>;
    using BlockMmadWH = Gemm::Block::BlockMmadTla<DispatchPolicyTlaMulti, L1TileShapeVTla, L0TileShapeVTla, INPUT_TYPE,
                                                  INPUT_TYPE, WORKSPACE_TYPE, void, TileCopyWH>;
    using BlockMmadWHTail = Gemm::Block::BlockMmadTla<DispatchPolicyTlaTail, L1TileShapeVTla, L0TileShapeVTla,
                                                      INPUT_TYPE, INPUT_TYPE, WORKSPACE_TYPE, void, TileCopyWH>;
    using BlockMmadWHDirectUb = Common::BlockMmadTla<DispatchPolicyDirectUb, L1TileShapeVTla, L0TileShapeVTla,
                                                     INPUT_TYPE, INPUT_TYPE, WORKSPACE_TYPE, void, TileCopyWHDirectUb>;

    // cube 2
    using TileCopyKV = Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, INPUT_TYPE, layout::ColumnMajor, INPUT_TYPE,
                                                              layout::zN, WORKSPACE_TYPE, layout::RowMajor>;
    using TileCopyKVDirectUb = Common::Tile::PackedTileCopyTlaToUB<ArchTag, INPUT_TYPE, layout::ColumnMajor, INPUT_TYPE,
                                                                   layout::zN, WORKSPACE_TYPE, layout::RowMajor, void,
                                                                   Gemm::Tile::CopyL0CToUBMode::NO_SPLIT>;
    using TileMmadKV = Gemm::Tile::TileMmadTla<ArchTag, INPUT_TYPE, typename TileCopyKV::LayoutTagL1A>;
    using BlockMmadKV = Gemm::Block::BlockMmadTla<DispatchPolicyTlaMulti, L1TileShapeVTla, L0TileShapeVTla, INPUT_TYPE,
                                                  INPUT_TYPE, WORKSPACE_TYPE, void, TileCopyKV>;
    using BlockMmadKVTail = Gemm::Block::BlockMmadTla<DispatchPolicyTlaTail, L1TileShapeVTla, L0TileShapeVTla,
                                                      INPUT_TYPE, INPUT_TYPE, WORKSPACE_TYPE, void, TileCopyKV>;
    using BlockMmadKVDirectUb = Common::BlockMmadTla<DispatchPolicyDirectUb, L1TileShapeVTla, L0TileShapeVTla,
                                                     INPUT_TYPE, INPUT_TYPE, WORKSPACE_TYPE, void, TileCopyKVDirectUb>;

    // vec 1
    using DispatchPolicyKdaFwdHVnew = Epilogue::EpilogueAtlasKdaFwdHVnew;
    using GateTag = KdaFwdHGateTag<kGated, scalarGated, useExp2>;
    using EpilogueKdaFwdHVnew = Epilogue::Block::BlockEpilogue<DispatchPolicyKdaFwdHVnew, VType, GType, UType,
                                                               VworkType, VUpdateType, FinalStateType, GateTag>;

    // vec 2
    using DispatchPolicyKdaFwdHUpdate = Epilogue::EpilogueAtlasKdaFwdHUpdate;
    using EpilogueKdaFwdHUpdate = Epilogue::Block::BlockEpilogue<DispatchPolicyKdaFwdHUpdate, HType, GType, HType,
                                                                 HworkType, FinalStateType, GateTag>;

    using KdaFwdHOffsets = Catlass::Gemm::Block::KdaFwdHOffsets;

    using ElementK = INPUT_TYPE;
    using ElementW = INPUT_TYPE;
    using ElementU = INPUT_TYPE;
    using ElementG = G_TYPE;
    using ElementH = INPUT_TYPE;
    using ElementV = INPUT_TYPE;
    using ElementVUpdate = INPUT_TYPE;
    using ElementVWork = WORKSPACE_TYPE;
    using ElementHWork = WORKSPACE_TYPE;
    using ElementInitialState = STATE_TYPE;
    using ElementFinalState = STATE_TYPE;

    using LayoutW = Catlass::layout::RowMajor;
    using LayoutH = Catlass::layout::RowMajor;
    using LayoutV = Catlass::layout::RowMajor;
    using LayoutK = Catlass::layout::ColumnMajor;
    using LayoutVUpdate = typename VUpdateType::Layout;

    static constexpr uint64_t DIRECT_UB_FREE_FLAG_BEGIN = 1;
    static constexpr uint64_t DIRECT_UB_READY_FLAG_BEGIN = 6;
    static constexpr uint64_t DIRECT_UB_FLAG_STRIDE = 16;
    static constexpr uint32_t DIRECT_UB_STAGES = 2;
    static constexpr uint32_t DIRECT_VEC_NUM = 2;

    uint32_t batch;
    uint32_t seqlen;
    uint32_t kNumHead;
    uint32_t vNumHead;
    uint32_t kHeadDim;
    uint32_t vHeadDim;
    uint32_t chunkSize;
    bool useInitialState;
    bool storeFinalState;
    uint32_t isVariedLen;
    uint32_t shapeBatch;
    uint32_t tokenBatch;
    uint32_t vWorkspaceOffset;
    uint32_t vUpdateWorkspaceOffset;
    uint32_t hWorkspaceOffset;
    uint32_t numSeqWorkspaceOffset;
    uint32_t numChunksWorkspaceOffset;
    uint32_t kDecayWorkspaceOffset;
    bool useDirectFp32Ub;

    AscendC::GlobalTensor<ElementK> gmK;
    AscendC::GlobalTensor<ElementW> gmW;
    AscendC::GlobalTensor<ElementU> gmU;
    AscendC::GlobalTensor<ElementG> gmG;
    AscendC::GlobalTensor<ElementG> gmGk;
    AscendC::GlobalTensor<ElementInitialState> gmInitialState;
    AscendC::GlobalTensor<ElementH> gmH;
    AscendC::GlobalTensor<ElementV> gmV;
    AscendC::GlobalTensor<ElementFinalState> gmFinalState;
    AscendC::GlobalTensor<ElementVWork> gmVWorkspace;
    AscendC::GlobalTensor<ElementV> gmVUpdateWorkspace;
    AscendC::GlobalTensor<ElementHWork> gmHWorkspace;
    AscendC::GlobalTensor<ElementK> gmKDecayWorkspace;

    AscendC::GlobalTensor<int64_t> gmSeqlen;
    AscendC::GlobalTensor<int64_t> gmNumSeq;
    AscendC::GlobalTensor<int64_t> gmNumChunks;

    AscendC::LocalTensor<ElementHWork> ubHUpdatePing;
    AscendC::LocalTensor<ElementHWork> ubHUpdatePong;
    AscendC::LocalTensor<ElementVWork> ubVWorkPing;
    AscendC::LocalTensor<ElementVWork> ubVWorkPong;

    AscendC::LocalTensor<ElementV> l1VUpdatePing;
    AscendC::LocalTensor<ElementV> l1VUpdatePong;

    CubeScheduler cubeBlockScheduler;
    VecScheduler vecBlockScheduler;

    Arch::Resource<ArchTag> resource;

    __aicore__ inline KdaFwdHKernel() {}

    __aicore__ inline void Init(GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR gk, GM_ADDR inital_state,
                                GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR h, GM_ADDR v_new,
                                GM_ADDR final_state, GM_ADDR tiling, GM_ADDR user)
    {
        __gm__ ChunkKdaFwdHRuntimeTiling *__restrict kdaFwdHTilingData =
            reinterpret_cast<__gm__ ChunkKdaFwdHRuntimeTiling *__restrict>(tiling);

        batch = kdaFwdHTilingData->batch;
        seqlen = kdaFwdHTilingData->seqlen;
        kNumHead = kdaFwdHTilingData->kNumHead;
        vNumHead = kdaFwdHTilingData->vNumHead;
        kHeadDim = kdaFwdHTilingData->kHeadDim;
        vHeadDim = kdaFwdHTilingData->vHeadDim;
        chunkSize = kdaFwdHTilingData->chunkSize;
        useInitialState = kdaFwdHTilingData->useInitialState;
        storeFinalState = kdaFwdHTilingData->storeFinalState;
        isVariedLen = kdaFwdHTilingData->isVariedLen;
        shapeBatch = kdaFwdHTilingData->shapeBatch;
        tokenBatch = kdaFwdHTilingData->tokenBatch;
        vWorkspaceOffset = kdaFwdHTilingData->vWorkspaceOffset;
        vUpdateWorkspaceOffset = kdaFwdHTilingData->vUpdateWorkspaceOffset;
        hWorkspaceOffset = kdaFwdHTilingData->hWorkspaceOffset;
        numSeqWorkspaceOffset = kdaFwdHTilingData->numSeqWorkspaceOffset;
        numChunksWorkspaceOffset = kdaFwdHTilingData->numChunksWorkspaceOffset;
        kDecayWorkspaceOffset = kdaFwdHTilingData->kDecayWorkspaceOffset;
        uint64_t effectiveTaskCount = static_cast<uint64_t>(isVariedLen ? tokenBatch : shapeBatch) * vNumHead;
        useDirectFp32Ub = std::is_same<ElementVWork, float>::value && chunkSize <= 64 && kHeadDim == 128 &&
                          vHeadDim == 128 && effectiveTaskCount >= AscendC::GetBlockNum();

        gmK.SetGlobalBuffer((__gm__ ElementK *)k);
        gmW.SetGlobalBuffer((__gm__ ElementW *)w);
        gmU.SetGlobalBuffer((__gm__ ElementU *)u);
        gmG.SetGlobalBuffer((__gm__ ElementG *)(scalarGated ? g : gk));
        gmGk.SetGlobalBuffer((__gm__ ElementG *)(kGated ? gk : g));
        gmInitialState.SetGlobalBuffer((__gm__ ElementInitialState *)inital_state);
        gmH.SetGlobalBuffer((__gm__ ElementH *)h);
        gmV.SetGlobalBuffer((__gm__ ElementV *)v_new);
        gmFinalState.SetGlobalBuffer((__gm__ ElementFinalState *)final_state);
        gmVWorkspace.SetGlobalBuffer((__gm__ ElementVWork *)(user + vWorkspaceOffset));
        gmVUpdateWorkspace.SetGlobalBuffer((__gm__ ElementV *)(user + vUpdateWorkspaceOffset));
        gmHWorkspace.SetGlobalBuffer((__gm__ ElementHWork *)(user + hWorkspaceOffset));
        gmKDecayWorkspace.SetGlobalBuffer((__gm__ ElementK *)(user + kDecayWorkspaceOffset));

        gmSeqlen.SetGlobalBuffer((__gm__ int64_t *)cu_seqlens);
        gmNumSeq.SetGlobalBuffer((__gm__ int64_t *)(user + numSeqWorkspaceOffset));
        gmNumChunks.SetGlobalBuffer((__gm__ int64_t *)(user + numChunksWorkspaceOffset));

        ubHUpdatePing = resource.ubBuf.template GetBufferByByte<ElementHWork>(32 * 1024);
        ubHUpdatePong = resource.ubBuf.template GetBufferByByte<ElementHWork>(96 * 1024);
        ubVWorkPing = resource.ubBuf.template GetBufferByByte<ElementVWork>(32 * 1024);
        ubVWorkPong = resource.ubBuf.template GetBufferByByte<ElementVWork>(96 * 1024);

        l1VUpdatePing = resource.l1Buf.template GetBufferByByte<ElementV>(0);
        l1VUpdatePong = resource.l1Buf.template GetBufferByByte<ElementV>(chunkSize * vHeadDim * sizeof(ElementV));

        if ASCEND_IS_AIC {
            cubeBlockScheduler.Init(cu_seqlens, chunk_indices, tiling, user);
        }

        if ASCEND_IS_AIV {
            vecBlockScheduler.Init(cu_seqlens, chunk_indices, tiling, user);
        }
    }

    template <typename TilingData>
    __aicore__ inline void InitFromData(GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR gk, GM_ADDR inital_state,
                                        GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR h, GM_ADDR v_new,
                                        GM_ADDR final_state, const TilingData &tilingData, GM_ADDR user)
    {
        batch = tilingData.batch;
        seqlen = tilingData.seqlen;
        kNumHead = tilingData.kNumHead;
        vNumHead = tilingData.vNumHead;
        kHeadDim = tilingData.kHeadDim;
        vHeadDim = tilingData.vHeadDim;
        chunkSize = tilingData.chunkSize;
        useInitialState = tilingData.useInitialState;
        storeFinalState = tilingData.storeFinalState;
        isVariedLen = tilingData.isVariedLen;
        shapeBatch = tilingData.shapeBatch;
        tokenBatch = tilingData.tokenBatch;
        vWorkspaceOffset = tilingData.vWorkspaceOffset;
        vUpdateWorkspaceOffset = tilingData.vUpdateWorkspaceOffset;
        hWorkspaceOffset = tilingData.hWorkspaceOffset;
        numSeqWorkspaceOffset = tilingData.numSeqWorkspaceOffset;
        numChunksWorkspaceOffset = tilingData.numChunksWorkspaceOffset;
        kDecayWorkspaceOffset = tilingData.kDecayWorkspaceOffset;
        uint64_t effectiveTaskCount = static_cast<uint64_t>(isVariedLen ? tokenBatch : shapeBatch) * vNumHead;
        useDirectFp32Ub = std::is_same<ElementVWork, float>::value && chunkSize <= 64 && kHeadDim == 128 &&
                          vHeadDim == 128 && effectiveTaskCount >= AscendC::GetBlockNum();

        gmK.SetGlobalBuffer((__gm__ ElementK *)k);
        gmW.SetGlobalBuffer((__gm__ ElementW *)w);
        gmU.SetGlobalBuffer((__gm__ ElementU *)u);
        gmG.SetGlobalBuffer((__gm__ ElementG *)(scalarGated ? g : gk));
        gmInitialState.SetGlobalBuffer((__gm__ ElementInitialState *)inital_state);
        gmH.SetGlobalBuffer((__gm__ ElementH *)h);
        gmV.SetGlobalBuffer((__gm__ ElementV *)v_new);
        gmFinalState.SetGlobalBuffer((__gm__ ElementFinalState *)final_state);
        gmVWorkspace.SetGlobalBuffer((__gm__ ElementVWork *)(user + vWorkspaceOffset));
        gmVUpdateWorkspace.SetGlobalBuffer((__gm__ ElementV *)(user + vUpdateWorkspaceOffset));
        gmHWorkspace.SetGlobalBuffer((__gm__ ElementHWork *)(user + hWorkspaceOffset));
        gmGk.SetGlobalBuffer((__gm__ ElementG *)(kGated ? gk : g));
        gmKDecayWorkspace.SetGlobalBuffer((__gm__ ElementK *)(user + kDecayWorkspaceOffset));
        gmSeqlen.SetGlobalBuffer((__gm__ int64_t *)cu_seqlens);
        gmNumSeq.SetGlobalBuffer((__gm__ int64_t *)(user + numSeqWorkspaceOffset));
        gmNumChunks.SetGlobalBuffer((__gm__ int64_t *)(user + numChunksWorkspaceOffset));

        ubHUpdatePing = resource.ubBuf.template GetBufferByByte<ElementHWork>(32 * 1024);
        ubHUpdatePong = resource.ubBuf.template GetBufferByByte<ElementHWork>(96 * 1024);
        ubVWorkPing = resource.ubBuf.template GetBufferByByte<ElementVWork>(32 * 1024);
        ubVWorkPong = resource.ubBuf.template GetBufferByByte<ElementVWork>(96 * 1024);

        l1VUpdatePing = resource.l1Buf.template GetBufferByByte<ElementV>(0);
        l1VUpdatePong = resource.l1Buf.template GetBufferByByte<ElementV>(chunkSize * vHeadDim * sizeof(ElementV));

        if ASCEND_IS_AIC {
            cubeBlockScheduler.InitFromData(cu_seqlens, chunk_indices, tilingData, user);
        }
        if ASCEND_IS_AIV {
            vecBlockScheduler.InitFromData(cu_seqlens, chunk_indices, tilingData, user);
        }
    }

    // Tail helpers borrow the stream's V_MTE2 free token and restore it before returning.
    __aicore__ inline void ComputeTailVWorkspace(const KdaFwdHOffsets &offsets, uint32_t tailEventId)
    {
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();
        uint32_t rowsPerSubBlock = CeilDiv(offsets.blockTokens, subBlockNum);
        uint32_t rowBegin = subBlockIdx * rowsPerSubBlock;
        uint32_t rowEnd = Min(rowBegin + rowsPerSubBlock, offsets.blockTokens);
        if (rowBegin >= rowEnd) {
            return;
        }
        AscendC::ResetMask();
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(tailEventId);

        if (offsets.vBlockDim == vHeadDim && vHeadDim <= 256) {
            constexpr uint32_t TAIL_H_INPUT_OFFSET = 0;
            constexpr uint32_t TAIL_W_INPUT_OFFSET = 166 * 1024;
            constexpr uint32_t TAIL_OUTPUT_OFFSET = 168 * 1024;
            uint32_t rowCount = rowEnd - rowBegin;
            AscendC::LocalTensor<ElementH> hInputUb =
                resource.ubBuf.template GetBufferByByte<ElementH>(TAIL_H_INPUT_OFFSET);
            AscendC::LocalTensor<ElementW> wInputUb =
                resource.ubBuf.template GetBufferByByte<ElementW>(TAIL_W_INPUT_OFFSET);
            AscendC::LocalTensor<float> outputUb = resource.ubBuf.template GetBufferByByte<float>(TAIL_OUTPUT_OFFSET);
            AscendC::DataCopy(hInputUb, gmH[offsets.hSrcOffset], kHeadDim * offsets.vBlockDim);
            AscendC::DataCopy(wInputUb, gmW[offsets.wOffset + rowBegin * kHeadDim], rowCount * kHeadDim);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(tailEventId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(tailEventId);
            __ubuf__ float *outputAddr = reinterpret_cast<__ubuf__ float *>(outputUb.GetPhyAddr());
            __ubuf__ ElementW *wInputAddr = reinterpret_cast<__ubuf__ ElementW *>(wInputUb.GetPhyAddr());
            __ubuf__ ElementH *hInputAddr = reinterpret_cast<__ubuf__ ElementH *>(hInputUb.GetPhyAddr());
            AscendC::VF_CALL<
                Catlass::Epilogue::Block::detail::ComputeKdaTailVWorkspaceRegbaseDualIssue<ElementW, ElementH>>(
                outputAddr, wInputAddr, hInputAddr, static_cast<uint16_t>(rowCount),
                static_cast<uint16_t>(offsets.vBlockDim), static_cast<uint16_t>(kHeadDim));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(tailEventId);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(tailEventId);
            AscendC::DataCopy(gmVWorkspace[offsets.vWorkOffset + rowBegin * offsets.vBlockDim], outputUb,
                              rowCount * offsets.vBlockDim);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(tailEventId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(tailEventId);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(tailEventId);
            return;
        }

        constexpr uint32_t TAIL_INPUT_OFFSET = 166 * 1024;
        constexpr uint32_t TAIL_FLOAT_OFFSET = 167 * 1024;
        constexpr uint32_t TAIL_ACCUM_OFFSET = 168 * 1024;
        constexpr uint32_t TAIL_WEIGHT_INPUT_OFFSET = 169 * 1024;
        constexpr uint32_t TAIL_WEIGHT_FLOAT_OFFSET = 170 * 1024;
        AscendC::LocalTensor<ElementH> inputUb = resource.ubBuf.template GetBufferByByte<ElementH>(TAIL_INPUT_OFFSET);
        AscendC::LocalTensor<float> floatUb = resource.ubBuf.template GetBufferByByte<float>(TAIL_FLOAT_OFFSET);
        AscendC::LocalTensor<float> accumUb = resource.ubBuf.template GetBufferByByte<float>(TAIL_ACCUM_OFFSET);
        AscendC::LocalTensor<ElementW> weightInputUb =
            resource.ubBuf.template GetBufferByByte<ElementW>(TAIL_WEIGHT_INPUT_OFFSET);
        AscendC::LocalTensor<float> weightFloatUb =
            resource.ubBuf.template GetBufferByByte<float>(TAIL_WEIGHT_FLOAT_OFFSET);

        for (uint32_t tokenRow = rowBegin; tokenRow < rowEnd; ++tokenRow) {
            AscendC::DataCopy(weightInputUb, gmW[offsets.wOffset + tokenRow * kHeadDim], kHeadDim);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(tailEventId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(tailEventId);
            AscendC::Cast(weightFloatUb, weightInputUb, AscendC::RoundMode::CAST_NONE, kHeadDim);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_S>(tailEventId);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(tailEventId);

            AscendC::Duplicate(accumUb, 0.0f, offsets.vBlockDim);
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t kIdx = 0; kIdx < kHeadDim; ++kIdx) {
                AscendC::DataCopy(inputUb, gmH[offsets.hSrcOffset + kIdx * vHeadDim], offsets.vBlockDim);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(tailEventId);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(tailEventId);
                AscendC::Cast(floatUb, inputUb, AscendC::RoundMode::CAST_NONE, offsets.vBlockDim);
                AscendC::PipeBarrier<PIPE_V>();
                float weight = weightFloatUb.GetValue(kIdx);
                AscendC::SetFlag<AscendC::HardEvent::S_V>(tailEventId);
                AscendC::WaitFlag<AscendC::HardEvent::S_V>(tailEventId);
                AscendC::Muls(floatUb, floatUb, weight, offsets.vBlockDim);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(accumUb, accumUb, floatUb, offsets.vBlockDim);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(tailEventId);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(tailEventId);
            }
            AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(tailEventId);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(tailEventId);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(tailEventId);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(tailEventId);
            AscendC::DataCopy(gmVWorkspace[offsets.vWorkOffset + tokenRow * offsets.vBlockDim], accumUb,
                              offsets.vBlockDim);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(tailEventId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(tailEventId);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(tailEventId);
    }

    __aicore__ inline void ComputeTailHWorkspace(const KdaFwdHOffsets &offsets, uint32_t tailEventId)
    {
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();
        uint32_t rowsPerSubBlock = CeilDiv(kHeadDim, subBlockNum);
        uint32_t rowBegin = subBlockIdx * rowsPerSubBlock;
        uint32_t rowEnd = Min(rowBegin + rowsPerSubBlock, kHeadDim);
        AscendC::ResetMask();
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(tailEventId);

        constexpr uint32_t TAIL_INPUT_OFFSET = 166 * 1024;
        constexpr uint32_t TAIL_FLOAT_OFFSET = 167 * 1024;
        constexpr uint32_t TAIL_ACCUM_OFFSET = 168 * 1024;
        constexpr uint32_t TAIL_WEIGHT_INPUT_OFFSET = 169 * 1024;
        constexpr uint32_t TAIL_WEIGHT_FLOAT_OFFSET = 170 * 1024;
        AscendC::LocalTensor<ElementV> inputUb = resource.ubBuf.template GetBufferByByte<ElementV>(TAIL_INPUT_OFFSET);
        AscendC::LocalTensor<float> floatUb = resource.ubBuf.template GetBufferByByte<float>(TAIL_FLOAT_OFFSET);
        AscendC::LocalTensor<float> accumUb = resource.ubBuf.template GetBufferByByte<float>(TAIL_ACCUM_OFFSET);
        AscendC::LocalTensor<ElementK> weightInputUb =
            resource.ubBuf.template GetBufferByByte<ElementK>(TAIL_WEIGHT_INPUT_OFFSET);
        AscendC::LocalTensor<float> weightFloatUb =
            resource.ubBuf.template GetBufferByByte<float>(TAIL_WEIGHT_FLOAT_OFFSET);

        for (uint32_t kRow = rowBegin; kRow < rowEnd; ++kRow) {
            AscendC::Duplicate(accumUb, 0.0f, offsets.vBlockDim);
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t tokenRow = 0; tokenRow < offsets.blockTokens; ++tokenRow) {
                AscendC::DataCopy(weightInputUb, gmKDecayWorkspace[offsets.kDecayWorkOffset + tokenRow * kHeadDim],
                                  kHeadDim);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(tailEventId);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(tailEventId);
                AscendC::Cast(weightFloatUb, weightInputUb, AscendC::RoundMode::CAST_NONE, kHeadDim);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetFlag<AscendC::HardEvent::V_S>(tailEventId);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(tailEventId);
                AscendC::DataCopy(inputUb, gmVUpdateWorkspace[offsets.vWorkOffset + tokenRow * offsets.vBlockDim],
                                  offsets.vBlockDim);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(tailEventId);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(tailEventId);
                AscendC::Cast(floatUb, inputUb, AscendC::RoundMode::CAST_NONE, offsets.vBlockDim);
                AscendC::PipeBarrier<PIPE_V>();
                float weight = weightFloatUb.GetValue(kRow);
                AscendC::SetFlag<AscendC::HardEvent::S_V>(tailEventId);
                AscendC::WaitFlag<AscendC::HardEvent::S_V>(tailEventId);
                AscendC::Muls(floatUb, floatUb, weight, offsets.vBlockDim);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(accumUb, accumUb, floatUb, offsets.vBlockDim);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(tailEventId);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(tailEventId);
                AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(tailEventId);
                AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(tailEventId);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(tailEventId);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(tailEventId);
            AscendC::DataCopy(gmHWorkspace[offsets.hWorkOffset + kRow * offsets.vBlockDim], accumUb, offsets.vBlockDim);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(tailEventId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(tailEventId);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(tailEventId);
    }

    __aicore__ inline void ComputeKdaTailFinalState(const KdaFwdHOffsets &offsets, uint32_t tailEventId)
    {
        static_assert(kGated && !scalarGated && useExp2 && std::is_same<ElementFinalState, float>::value,
                      "The fused tail state path is specific to FP32 KDA state");
        constexpr uint32_t TAIL_STATE_OFFSET = 0;
        constexpr uint32_t TAIL_K_INPUT_OFFSET = 166 * 1024;
        constexpr uint32_t TAIL_V_INPUT_OFFSET = 170 * 1024;
        constexpr uint32_t TAIL_GATE_INPUT_OFFSET = 174 * 1024;
        constexpr uint32_t TAIL_GATE_SCALE_OFFSET = 175 * 1024;

        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();
        uint32_t rowsPerSubBlock = CeilDiv(kHeadDim, subBlockNum);
        uint32_t rowBegin = subBlockIdx * rowsPerSubBlock;
        uint32_t rowEnd = Min(rowBegin + rowsPerSubBlock, kHeadDim);
        if (rowBegin >= rowEnd) {
            return;
        }

        AscendC::ResetMask();
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(tailEventId);
        AscendC::LocalTensor<float> stateUb = resource.ubBuf.template GetBufferByByte<float>(TAIL_STATE_OFFSET);
        AscendC::LocalTensor<ElementK> kInputUb =
            resource.ubBuf.template GetBufferByByte<ElementK>(TAIL_K_INPUT_OFFSET);
        AscendC::LocalTensor<ElementV> vInputUb =
            resource.ubBuf.template GetBufferByByte<ElementV>(TAIL_V_INPUT_OFFSET);
        AscendC::LocalTensor<ElementG> gateInputUb =
            resource.ubBuf.template GetBufferByByte<ElementG>(TAIL_GATE_INPUT_OFFSET);
        AscendC::LocalTensor<float> gateScaleUb =
            resource.ubBuf.template GetBufferByByte<float>(TAIL_GATE_SCALE_OFFSET);

        uint32_t gateOffset = offsets.gkOffset + (offsets.blockTokens - 1) * kHeadDim + rowBegin;
        uint32_t rowCount = rowEnd - rowBegin;
        uint32_t stateElements = rowCount * offsets.vBlockDim;
        uint32_t kElements = offsets.blockTokens * kHeadDim;
        uint32_t vElements = offsets.blockTokens * offsets.vBlockDim;

        if (offsets.isInitialState) {
            if (useInitialState) {
                AscendC::DataCopy(stateUb, gmInitialState[offsets.initialStateOffset + rowBegin * offsets.vBlockDim],
                                  stateElements);
            } else {
                AscendC::Duplicate(stateUb, 0.0f, stateElements);
            }
        } else {
            AscendC::DataCopy(stateUb, gmFinalState[offsets.finalStateOffset + rowBegin * offsets.vBlockDim],
                              stateElements);
        }
        AscendC::DataCopy(kInputUb, gmK[offsets.wkOffset], kElements);
        AscendC::DataCopy(vInputUb, gmV[offsets.uvOffset], vElements);
        AscendC::DataCopy(gateInputUb, gmGk[gateOffset], rowCount);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(tailEventId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(tailEventId);

        __ubuf__ float *stateAddr = reinterpret_cast<__ubuf__ float *>(stateUb.GetPhyAddr());
        __ubuf__ ElementK *kInputAddr = reinterpret_cast<__ubuf__ ElementK *>(kInputUb.GetPhyAddr());
        __ubuf__ ElementV *vInputAddr = reinterpret_cast<__ubuf__ ElementV *>(vInputUb.GetPhyAddr());
        __ubuf__ ElementG *gateInputAddr = reinterpret_cast<__ubuf__ ElementG *>(gateInputUb.GetPhyAddr());
        __ubuf__ float *gateScaleAddr = reinterpret_cast<__ubuf__ float *>(gateScaleUb.GetPhyAddr());
        AscendC::VF_CALL<Catlass::Epilogue::Block::detail::PrepareKGateRegbase<ElementG, true>>(
            gateScaleAddr, gateInputAddr, static_cast<uint16_t>(rowCount));
        AscendC::VF_CALL<
            Catlass::Epilogue::Block::detail::ComputeKdaTailFinalStateRegbaseDualIssue<ElementK, ElementV>>(
            stateAddr, kInputAddr, vInputAddr, gateScaleAddr, static_cast<uint16_t>(rowCount),
            static_cast<uint16_t>(offsets.vBlockDim), static_cast<uint16_t>(offsets.blockTokens),
            static_cast<uint16_t>(kHeadDim), static_cast<uint16_t>(rowBegin));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(tailEventId);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(tailEventId);
        AscendC::DataCopy(gmFinalState[offsets.finalStateOffset + rowBegin * offsets.vBlockDim], stateUb,
                          stateElements);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(tailEventId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(tailEventId);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(tailEventId);
    }

    __aicore__ inline void Process()
    {
        // FwdH can run after another stage in a megakernel. Start its AIC/AIV
        // handshake only after every core has retired the preceding stage.
        AscendC::SyncAll<false>();

        if ASCEND_IS_AIC {
            uint32_t coreIdx = AscendC::GetBlockIdx();
            uint32_t coreNum = vecBlockScheduler.cubeCoreNum;

            BlockMmadWH blockMmadWH(resource,
                                    chunkSize * cubeBlockScheduler.vBlockSize * sizeof(ElementV) * PING_PONG_STAGES);
            BlockMmadKV blockMmadKV(resource,
                                    chunkSize * cubeBlockScheduler.vBlockSize * sizeof(ElementV) * PING_PONG_STAGES);
            BlockMmadWHTail blockMmadWHTail(
                resource, chunkSize * cubeBlockScheduler.vBlockSize * sizeof(ElementV) * PING_PONG_STAGES);
            BlockMmadKVTail blockMmadKVTail(
                resource, chunkSize * cubeBlockScheduler.vBlockSize * sizeof(ElementV) * PING_PONG_STAGES);
            bool useBoundedMmad = isVariedLen || (seqlen % chunkSize != 0);

            auto wLayout =
                tla::MakeLayout<ElementW, LayoutW>(shapeBatch * kNumHead * cubeBlockScheduler.totalTokens, kHeadDim);
            auto hLayout = tla::MakeLayout<ElementH, LayoutH>(
                shapeBatch * vNumHead * cubeBlockScheduler.totalChunks * kHeadDim, vHeadDim);

            auto kLayout =
                tla::MakeLayout<ElementK, LayoutK>(kHeadDim, shapeBatch * kNumHead * cubeBlockScheduler.totalTokens);
            auto hworkLayout = tla::MakeLayout<ElementHWork, LayoutH>(kHeadDim, cubeBlockScheduler.vBlockSize);

            AscendC::SyncAll<false>();
            uint32_t currStage = 0; // 0: C1, 1: C2
            while (cubeBlockScheduler.isRunning) {
                if (currStage == 0) {
                    /* C1: v_work = w @ h[i] */
                    cubeBlockScheduler.InitTasks();
                    if (useDirectFp32Ub) {
                        BlockMmadWHDirectUb blockMmadWHDirectUb(
                            resource, chunkSize * cubeBlockScheduler.vBlockSize * sizeof(ElementV) * PING_PONG_STAGES);
                        for (uint32_t i = 0; i < PING_PONG_STAGES; ++i) {
                            uint32_t streamId = cubeBlockScheduler.GetStreamId(i);
                            const auto &stream = cubeBlockScheduler.GetStream(i);
                            if (cubeBlockScheduler.StreamIsDone(stream)) {
                                continue;
                            }

                            const KdaFwdHOffsets &cube1Offsets = cubeBlockScheduler.GetCurTaskOffsets(stream);
                            Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec2Done[streamId]);
                            if (cube1Offsets.blockTokens < 16) {
                                Arch::CrossCoreSetFlag<0x2, PIPE_MTE2>(cubeBlockScheduler.cube1Done[streamId]);
                                continue;
                            }
                            int64_t cube1OffsetW = cube1Offsets.wOffset;
                            int64_t cube1OffsetH = cube1Offsets.hSrcOffset;
                            auto tensorW = tla::MakeTensor(gmW[cube1OffsetW], wLayout, Catlass::Arch::PositionGM{});
                            auto tensorH = tla::MakeTensor(gmH[cube1OffsetH], hLayout, Catlass::Arch::PositionGM{});
                            GemmCoord cube1Shape{cube1Offsets.blockTokens, cube1Offsets.vBlockDim, kHeadDim};
                            auto tensorBlockW =
                                GetTile(tensorW, tla::MakeCoord(0, 0), tla::MakeShape(cube1Shape.m(), cube1Shape.k()));
                            auto tensorBlockH =
                                GetTile(tensorH, tla::MakeCoord(0, 0), tla::MakeShape(cube1Shape.k(), cube1Shape.n()));

                            auto ubLayout = tla::MakeLayout<ElementVWork, LayoutV>(cube1Shape.m(), cube1Shape.n());
                            auto tensorUbPing = tla::MakeTensor(ubVWorkPing, ubLayout, Catlass::Arch::PositionUB{});
                            auto tensorUbPong = tla::MakeTensor(ubVWorkPong, ubLayout, Catlass::Arch::PositionUB{});
                            using UbTensor = decltype(tensorUbPing);
                            UbTensor tensorUbList[BlockMmadWHDirectUb::MAX_CUBE_VEC_SYNC_NUM];
                            for (uint32_t ubIdx = 0; ubIdx < BlockMmadWHDirectUb::MAX_CUBE_VEC_SYNC_NUM; ++ubIdx) {
                                tensorUbList[ubIdx] = (ubIdx & 1U) ? tensorUbPong : tensorUbPing;
                            }
                            uint32_t ubListId = streamId;
                            uint32_t rowsPerSubBlock = CeilDiv(cube1Shape.m(), DIRECT_VEC_NUM);
                            blockMmadWHDirectUb(tensorBlockW, tensorBlockH, tensorUbList, cube1Shape, rowsPerSubBlock,
                                                0, DIRECT_UB_FREE_FLAG_BEGIN, DIRECT_UB_READY_FLAG_BEGIN, ubListId,
                                                DIRECT_VEC_NUM, DIRECT_UB_STAGES);
                        }
                    } else if (useBoundedMmad) {
                        for (uint32_t i = 0; i < PING_PONG_STAGES; ++i) {
                            uint32_t streamId = cubeBlockScheduler.GetStreamId(i);
                            const auto &stream = cubeBlockScheduler.GetStream(i);
                            if (cubeBlockScheduler.StreamIsDone(stream)) {
                                continue;
                            }

                            const KdaFwdHOffsets &cube1Offsets = cubeBlockScheduler.GetCurTaskOffsets(stream);
                            Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec2Done[streamId]);
                            if (cube1Offsets.blockTokens < 16) {
                                Arch::CrossCoreSetFlag<0x2, PIPE_MTE2>(cubeBlockScheduler.cube1Done[streamId]);
                                continue;
                            }
                            auto vLayout = tla::MakeLayout<ElementVWork, LayoutV>(cube1Offsets.blockTokens,
                                                                                  cube1Offsets.vBlockDim);
                            auto tensorW =
                                tla::MakeTensor(gmW[cube1Offsets.wOffset], wLayout, Catlass::Arch::PositionGM{});
                            auto tensorH =
                                tla::MakeTensor(gmH[cube1Offsets.hSrcOffset], hLayout, Catlass::Arch::PositionGM{});
                            auto tensorV = tla::MakeTensor(gmVWorkspace[cube1Offsets.vWorkOffset], vLayout,
                                                           Catlass::Arch::PositionGM{});
                            GemmCoord cube1Shape{cube1Offsets.blockTokens, cube1Offsets.vBlockDim, kHeadDim};
                            auto tensorBlockW =
                                GetTile(tensorW, tla::MakeCoord(0, 0), tla::MakeShape(cube1Shape.m(), cube1Shape.k()));
                            auto tensorBlockH =
                                GetTile(tensorH, tla::MakeCoord(0, 0), tla::MakeShape(cube1Shape.k(), cube1Shape.n()));
                            auto tensorBlockV =
                                GetTile(tensorV, tla::MakeCoord(0, 0), tla::MakeShape(cube1Shape.m(), cube1Shape.n()));

                            blockMmadWHTail.preSetFlags();
                            blockMmadWHTail(tensorBlockW, tensorBlockH, tensorBlockV, cube1Shape, EmptyClass{}, true);
                            blockMmadWHTail.finalWaitFlags();
                            AscendC::PipeBarrier<PIPE_ALL>();
                            Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeBlockScheduler.cube1Done[streamId]);
                        }
                    } else {
                        blockMmadWH.preSetFlags();
                        for (uint32_t i = 0; i < PING_PONG_STAGES; ++i) {
                            uint32_t streamId = cubeBlockScheduler.GetStreamId(i);
                            const auto &stream = cubeBlockScheduler.GetStream(i);
                            if (cubeBlockScheduler.StreamIsDone(stream)) {
                                continue;
                            }

                            const KdaFwdHOffsets &cube1Offsets = cubeBlockScheduler.GetCurTaskOffsets(stream);
                            Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec2Done[streamId]);
                            if (cube1Offsets.blockTokens < 16) {
                                Arch::CrossCoreSetFlag<0x2, PIPE_MTE2>(cubeBlockScheduler.cube1Done[streamId]);
                                continue;
                            }
                            auto vLayout = tla::MakeLayout<ElementVWork, LayoutV>(cube1Offsets.blockTokens,
                                                                                  cube1Offsets.vBlockDim);
                            int64_t cube1OffsetW = cube1Offsets.wOffset;
                            int64_t cube1OffsetH = cube1Offsets.hSrcOffset;
                            int64_t cube1OffsetVwork = cube1Offsets.vWorkOffset;
                            auto tensorW = tla::MakeTensor(gmW[cube1OffsetW], wLayout, Catlass::Arch::PositionGM{});
                            auto tensorH = tla::MakeTensor(gmH[cube1OffsetH], hLayout, Catlass::Arch::PositionGM{});
                            auto tensorV =
                                tla::MakeTensor(gmVWorkspace[cube1OffsetVwork], vLayout, Catlass::Arch::PositionGM{});
                            GemmCoord cube1Shape{cube1Offsets.blockTokens, cube1Offsets.vBlockDim, kHeadDim};
                            auto tensorBlockW =
                                GetTile(tensorW, tla::MakeCoord(0, 0), tla::MakeShape(cube1Shape.m(), cube1Shape.k()));
                            auto tensorBlockH =
                                GetTile(tensorH, tla::MakeCoord(0, 0), tla::MakeShape(cube1Shape.k(), cube1Shape.n()));
                            auto tensorBlockV =
                                GetTile(tensorV, tla::MakeCoord(0, 0), tla::MakeShape(cube1Shape.m(), cube1Shape.n()));

                            blockMmadWH(tensorBlockW, tensorBlockH, tensorBlockV, cube1Shape);
                            Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeBlockScheduler.cube1Done[streamId]);
                        }
                        blockMmadWH.finalWaitFlags();
                    }
                } else {
                    /* C2: h[i+1] = k.T @ v_work */
                    if (useDirectFp32Ub) {
                        BlockMmadKVDirectUb blockMmadKVDirectUb(
                            resource, chunkSize * cubeBlockScheduler.vBlockSize * sizeof(ElementV) * PING_PONG_STAGES);
                        for (uint32_t i = 0; i < PING_PONG_STAGES; ++i) {
                            uint32_t streamId = cubeBlockScheduler.GetStreamId(i);
                            const auto &stream = cubeBlockScheduler.GetStream(i);
                            if (cubeBlockScheduler.StreamIsDone(stream)) {
                                continue;
                            }
                            const KdaFwdHOffsets &cube2Offsets = cubeBlockScheduler.GetCurTaskOffsets(stream);
                            Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec1Done[streamId]);

                            if (cubeBlockScheduler.NeedProcessStage2(stream)) {
                                if (cube2Offsets.blockTokens < 16) {
                                    Arch::CrossCoreSetFlag<0x2, PIPE_MTE2>(cubeBlockScheduler.cube2Done[streamId]);
                                    continue;
                                }
                                int64_t cube2OffsetK = kGated ? cube2Offsets.kDecayWorkOffset : cube2Offsets.wkOffset;
                                int64_t cube2OffsetVwork = cube2Offsets.vWorkOffset;
                                auto tensorK =
                                    kGated ? tla::MakeTensor(gmKDecayWorkspace[cube2OffsetK], kLayout,
                                                             Catlass::Arch::PositionGM{}) :
                                             tla::MakeTensor(gmK[cube2OffsetK], kLayout, Catlass::Arch::PositionGM{});
                                auto vUpdateLayout = tla::MakeLayout<ElementVUpdate, LayoutVUpdate>(
                                    cube2Offsets.blockTokens, cube2Offsets.vBlockDim);
                                auto tensorVwork = tla::MakeTensor(gmVUpdateWorkspace[cube2OffsetVwork], vUpdateLayout,
                                                                   Catlass::Arch::PositionGM{});
                                GemmCoord cube2Shape{kHeadDim, cube2Offsets.vBlockDim, cube2Offsets.blockTokens};
                                auto tensorBlockK = GetTile(tensorK, tla::MakeCoord(0, 0),
                                                            tla::MakeShape(cube2Shape.m(), cube2Shape.k()));
                                auto tensorBlockVwork = GetTile(tensorVwork, tla::MakeCoord(0, 0),
                                                                tla::MakeShape(cube2Shape.k(), cube2Shape.n()));

                                auto ubLayout = tla::MakeLayout<ElementHWork, LayoutH>(cube2Shape.m(), cube2Shape.n());
                                auto tensorUbPing =
                                    tla::MakeTensor(ubHUpdatePing, ubLayout, Catlass::Arch::PositionUB{});
                                auto tensorUbPong =
                                    tla::MakeTensor(ubHUpdatePong, ubLayout, Catlass::Arch::PositionUB{});
                                using UbTensor = decltype(tensorUbPing);
                                UbTensor tensorUbList[BlockMmadKVDirectUb::MAX_CUBE_VEC_SYNC_NUM];
                                for (uint32_t ubIdx = 0; ubIdx < BlockMmadKVDirectUb::MAX_CUBE_VEC_SYNC_NUM; ++ubIdx) {
                                    tensorUbList[ubIdx] = (ubIdx & 1U) ? tensorUbPong : tensorUbPing;
                                }
                                uint32_t ubListId = streamId;
                                uint32_t rowsPerSubBlock = CeilDiv(cube2Shape.m(), DIRECT_VEC_NUM);
                                blockMmadKVDirectUb(tensorBlockK, tensorBlockVwork, tensorUbList, cube2Shape,
                                                    rowsPerSubBlock, 0, DIRECT_UB_FREE_FLAG_BEGIN,
                                                    DIRECT_UB_READY_FLAG_BEGIN, ubListId, DIRECT_VEC_NUM,
                                                    DIRECT_UB_STAGES);
                            }
                        }
                    } else if (useBoundedMmad) {
                        for (uint32_t i = 0; i < PING_PONG_STAGES; ++i) {
                            uint32_t streamId = cubeBlockScheduler.GetStreamId(i);
                            const auto &stream = cubeBlockScheduler.GetStream(i);
                            if (cubeBlockScheduler.StreamIsDone(stream)) {
                                continue;
                            }
                            const KdaFwdHOffsets &cube2Offsets = cubeBlockScheduler.GetCurTaskOffsets(stream);
                            Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec1Done[streamId]);

                            if (cubeBlockScheduler.NeedProcessStage2(stream)) {
                                if (cube2Offsets.blockTokens < 16) {
                                    Arch::CrossCoreSetFlag<0x2, PIPE_MTE2>(cubeBlockScheduler.cube2Done[streamId]);
                                    continue;
                                }
                                int64_t cube2OffsetK = kGated ? cube2Offsets.kDecayWorkOffset : cube2Offsets.wkOffset;
                                auto tensorK =
                                    kGated ? tla::MakeTensor(gmKDecayWorkspace[cube2OffsetK], kLayout,
                                                             Catlass::Arch::PositionGM{}) :
                                             tla::MakeTensor(gmK[cube2OffsetK], kLayout, Catlass::Arch::PositionGM{});
                                auto vUpdateLayout = tla::MakeLayout<ElementVUpdate, LayoutVUpdate>(
                                    cube2Offsets.blockTokens, cube2Offsets.vBlockDim);
                                auto tensorVwork = tla::MakeTensor(gmVUpdateWorkspace[cube2Offsets.vWorkOffset],
                                                                   vUpdateLayout, Catlass::Arch::PositionGM{});
                                auto tensorHwork = tla::MakeTensor(gmHWorkspace[cube2Offsets.hWorkOffset], hworkLayout,
                                                                   Catlass::Arch::PositionGM{});
                                GemmCoord cube2Shape{kHeadDim, cube2Offsets.vBlockDim, cube2Offsets.blockTokens};
                                auto tensorBlockK = GetTile(tensorK, tla::MakeCoord(0, 0),
                                                            tla::MakeShape(cube2Shape.m(), cube2Shape.k()));
                                auto tensorBlockVwork = GetTile(tensorVwork, tla::MakeCoord(0, 0),
                                                                tla::MakeShape(cube2Shape.k(), cube2Shape.n()));
                                auto tensorBlockHwork = GetTile(tensorHwork, tla::MakeCoord(0, 0),
                                                                tla::MakeShape(cube2Shape.m(), cube2Shape.n()));

                                blockMmadKVTail.preSetFlags();
                                blockMmadKVTail(tensorBlockK, tensorBlockVwork, tensorBlockHwork, cube2Shape,
                                                EmptyClass{}, true);
                                blockMmadKVTail.finalWaitFlags();
                                AscendC::PipeBarrier<PIPE_ALL>();
                            }
                            Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeBlockScheduler.cube2Done[streamId]);
                        }
                    } else {
                        blockMmadKV.preSetFlags();
                        for (uint32_t i = 0; i < PING_PONG_STAGES; ++i) {
                            uint32_t streamId = cubeBlockScheduler.GetStreamId(i);
                            const auto &stream = cubeBlockScheduler.GetStream(i);
                            if (cubeBlockScheduler.StreamIsDone(stream)) {
                                continue;
                            }
                            const KdaFwdHOffsets &cube2Offsets = cubeBlockScheduler.GetCurTaskOffsets(stream);
                            Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec1Done[streamId]);

                            if (cubeBlockScheduler.NeedProcessStage2(stream)) {
                                if (cube2Offsets.blockTokens < 16) {
                                    Arch::CrossCoreSetFlag<0x2, PIPE_MTE2>(cubeBlockScheduler.cube2Done[streamId]);
                                    continue;
                                }
                                // step 3: h[i+1] = k.T @ v_work
                                int64_t cube2OffsetK = kGated ? cube2Offsets.kDecayWorkOffset : cube2Offsets.wkOffset;
                                int64_t cube2OffsetVwork = cube2Offsets.vWorkOffset;
                                auto tensorK =
                                    kGated ? tla::MakeTensor(gmKDecayWorkspace[cube2OffsetK], kLayout,
                                                             Catlass::Arch::PositionGM{}) :
                                             tla::MakeTensor(gmK[cube2OffsetK], kLayout, Catlass::Arch::PositionGM{});
                                auto vUpdateLayout = tla::MakeLayout<ElementVUpdate, LayoutVUpdate>(
                                    cube2Offsets.blockTokens, cube2Offsets.vBlockDim);
                                auto tensorVwork = tla::MakeTensor(gmVUpdateWorkspace[cube2OffsetVwork], vUpdateLayout,
                                                                   Catlass::Arch::PositionGM{});
                                auto tensorHwork = tla::MakeTensor(gmHWorkspace[cube2Offsets.hWorkOffset], hworkLayout,
                                                                   Catlass::Arch::PositionGM{});
                                GemmCoord cube2Shape{kHeadDim, cube2Offsets.vBlockDim, cube2Offsets.blockTokens};
                                auto tensorBlockK = GetTile(tensorK, tla::MakeCoord(0, 0),
                                                            tla::MakeShape(cube2Shape.m(), cube2Shape.k()));
                                auto tensorBlockVwork = GetTile(tensorVwork, tla::MakeCoord(0, 0),
                                                                tla::MakeShape(cube2Shape.k(), cube2Shape.n()));
                                auto tensorBlockHwork = GetTile(tensorHwork, tla::MakeCoord(0, 0),
                                                                tla::MakeShape(cube2Shape.m(), cube2Shape.n()));

                                blockMmadKV(tensorBlockK, tensorBlockVwork, tensorBlockHwork, cube2Shape);
                            }
                            Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeBlockScheduler.cube2Done[streamId]);
                        }
                        blockMmadKV.finalWaitFlags();
                    }
                }
                currStage ^= 0x01;
            }
            Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec2Done[0]);
            Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec2Done[1]);
            if (useDirectFp32Ub) {
                for (uint32_t slot = 0; slot < DIRECT_UB_STAGES; ++slot) {
                    AscendC::CrossCoreWaitFlag<0x4, PIPE_FIX>(DIRECT_UB_FREE_FLAG_BEGIN + slot);
                    AscendC::CrossCoreWaitFlag<0x4, PIPE_FIX>(DIRECT_UB_FREE_FLAG_BEGIN + DIRECT_UB_FLAG_STRIDE + slot);
                }
            }
        }

        if ASCEND_IS_AIV {
            uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
            uint32_t subBlockNum = AscendC::GetSubBlockNum();
            uint32_t coreIdx = AscendC::GetBlockIdx() / subBlockNum;
            uint32_t coreNum = AscendC::GetBlockNum();
            uint32_t taskCount = (isVariedLen ? vecBlockScheduler.tokenBatch : shapeBatch) * vNumHead;
            uint32_t tasksPerCore = taskCount > coreNum ? PING_PONG_STAGES : 1;
            uint32_t taskStride = coreNum * tasksPerCore;
            uint32_t rowsPerSubBlock = (kHeadDim + subBlockNum - 1) / subBlockNum;
            uint32_t rowBegin = subBlockIdx * rowsPerSubBlock;
            uint32_t rowEnd = Min(rowBegin + rowsPerSubBlock, kHeadDim);
            uint32_t hRowsPerTile = (32 * 1024) / (vHeadDim * sizeof(ElementH));
            uint32_t stateRowsPerTile = (64 * 1024) / (vHeadDim * sizeof(ElementInitialState));
            uint32_t rowsPerTile = Min(hRowsPerTile, stateRowsPerTile);
            uint32_t totalChunks = isVariedLen ? vecBlockScheduler.totalChunks : ((seqlen + chunkSize - 1) / chunkSize);
            uint32_t stateBlockSize = kHeadDim * vHeadDim;
            uint32_t pingpongFlag = 1;
            AscendC::LocalTensor<ElementInitialState> stateUbTensorPing =
                resource.ubBuf.template GetBufferByByte<ElementInitialState>(0);
            AscendC::LocalTensor<ElementInitialState> stateUbTensorPong =
                resource.ubBuf.template GetBufferByByte<ElementInitialState>(96 * 1024);
            AscendC::LocalTensor<ElementH> hUbTensorPing = resource.ubBuf.template GetBufferByByte<ElementH>(64 * 1024);
            AscendC::LocalTensor<ElementH> hUbTensorPong =
                resource.ubBuf.template GetBufferByByte<ElementH>(160 * 1024);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
            for (uint32_t slot = 0; slot < tasksPerCore; ++slot) {
                for (uint32_t taskIdx = coreIdx * tasksPerCore + slot; taskIdx < taskCount; taskIdx += taskStride) {
                    uint32_t batchIdx = taskIdx / vNumHead;
                    uint32_t vHeadIdx = taskIdx % vNumHead;
                    uint32_t chunkOffset = isVariedLen ? vecBlockScheduler.GetVarlenChunkOffset(batchIdx) : 0;
                    uint32_t shapeBatchIdx = isVariedLen ? 0 : batchIdx;
                    uint32_t hBaseOffset =
                        (shapeBatchIdx * vNumHead * totalChunks + vHeadIdx * totalChunks + chunkOffset) *
                        stateBlockSize;
                    uint32_t initialStateBaseOffset = taskIdx * stateBlockSize;
                    for (uint32_t rowOffset = rowBegin; rowOffset < rowEnd; rowOffset += rowsPerTile) {
                        uint32_t rowsThisTile = Min(rowsPerTile, rowEnd - rowOffset);
                        uint32_t stateTileElems = rowsThisTile * vHeadDim;
                        uint32_t hOffset = hBaseOffset + rowOffset * vHeadDim;
                        AscendC::LocalTensor<ElementInitialState> stateUbTensor =
                            pingpongFlag ? stateUbTensorPing : stateUbTensorPong;
                        AscendC::LocalTensor<ElementH> hUbTensor = pingpongFlag ? hUbTensorPing : hUbTensorPong;
                        auto eventId = pingpongFlag ? EVENT_ID1 : EVENT_ID0;
                        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
                        if (useInitialState) {
                            uint32_t initialStateOffset = initialStateBaseOffset + rowOffset * vHeadDim;
                            if constexpr (!std::is_same<ElementInitialState, ElementH>::value) {
                                AscendC::DataCopy(stateUbTensor, gmInitialState[initialStateOffset], stateTileElems);
                                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventId);
                                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventId);
                                AscendC::Cast(hUbTensor, stateUbTensor, AscendC::RoundMode::CAST_RINT, stateTileElems);
                                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventId);
                                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventId);
                                AscendC::DataCopy(gmH[hOffset], hUbTensor, stateTileElems);
                            } else {
                                AscendC::DataCopy(stateUbTensor, gmInitialState[initialStateOffset], stateTileElems);
                                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
                                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
                                AscendC::DataCopy(gmH[hOffset], stateUbTensor, stateTileElems);
                            }
                        } else {
                            AscendC::Duplicate(hUbTensor, static_cast<ElementH>(0), stateTileElems);
                            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventId);
                            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventId);
                            AscendC::DataCopy(gmH[hOffset], hUbTensor, stateTileElems);
                        }
                        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
                        pingpongFlag = 1 - pingpongFlag;
                    }
                }
            }
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);

            AscendC::SyncAll<false>();

            if (useDirectFp32Ub) {
                for (uint32_t slot = 0; slot < DIRECT_UB_STAGES; ++slot) {
                    uint64_t flagOffset = slot + subBlockIdx * DIRECT_UB_FLAG_STRIDE;
                    AscendC::CrossCoreSetFlag<0x4, PIPE_V>(DIRECT_UB_FREE_FLAG_BEGIN + flagOffset);
                }
            }
            Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecBlockScheduler.vec2Done[0]);
            Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecBlockScheduler.vec2Done[1]);

            EpilogueKdaFwdHVnew epilogueKdaFwdHVnew(resource);
            EpilogueKdaFwdHUpdate epilogueKdaFwdHUpdate(resource);
            uint32_t pongBaseEvent = 4;

            if (storeFinalState && std::is_same<ElementFinalState, float>::value) {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0); // preset final_state
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + pongBaseEvent);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2); // preset h
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2 + pongBaseEvent);
            } else {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0); // preset h_update
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + pongBaseEvent);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2); // preset h
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2 + pongBaseEvent);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1); // preset u
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1 + pongBaseEvent);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3); // preset g
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3 + pongBaseEvent);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0); // preset h_update
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0 + pongBaseEvent);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2); // preset h
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2 + pongBaseEvent);
            uint32_t currStage = 0; // 0: V1, 1: V2
            bool event0FromMte3[PING_PONG_STAGES] = {false, false};
            bool event2FromMte3[PING_PONG_STAGES] = {
                !(storeFinalState && std::is_same<ElementFinalState, float>::value),
                !(storeFinalState && std::is_same<ElementFinalState, float>::value)};
            while (vecBlockScheduler.isRunning) {
                if (currStage == 0) {
                    /* V1:
                     * gmV = gmU - gmVWorkspace
                     * g_buf = gmG[-1] - gmG
                     * g_buf = exp(g_buf)
                     * gmVWorkspace = g_buf * gmV
                     */
                    vecBlockScheduler.InitTasks();
                    for (uint32_t i = 0; i < PING_PONG_STAGES; ++i) {
                        uint32_t streamId = vecBlockScheduler.GetStreamId(i);
                        const auto &stream = vecBlockScheduler.GetStream(i);
                        if (vecBlockScheduler.StreamIsDone(stream)) {
                            continue;
                        }
                        const KdaFwdHOffsets &vec1Offsets = vecBlockScheduler.GetCurTaskOffsets(stream);
                        AscendC::LocalTensor<ElementV> l1VUpdate = (i == 0) ? l1VUpdatePing : l1VUpdatePong;
                        bool tailVectorPath = vec1Offsets.blockTokens < 16;
                        if (tailVectorPath) {
                            Arch::CrossCoreWaitFlag(vecBlockScheduler.cube1Done[streamId]);
                            ComputeTailVWorkspace(vec1Offsets, EVENT_ID3 + (i == 0 ? 0 : pongBaseEvent));
                        }
                        bool waitWsFromMte3 = storeFinalState && std::is_same<ElementFinalState, float>::value &&
                                              event0FromMte3[streamId];
                        bool useDirectForTask = useDirectFp32Ub && !tailVectorPath;
                        epilogueKdaFwdHVnew(
                            gmV[vec1Offsets.uvOffset], gmVUpdateWorkspace[vec1Offsets.vWorkOffset], l1VUpdate,
                            gmG[vec1Offsets.gOffset], gmU[vec1Offsets.uvOffset], gmVWorkspace[vec1Offsets.vWorkOffset],
                            gmGk[vec1Offsets.gkOffset], gmK[vec1Offsets.wkOffset],
                            gmKDecayWorkspace[vec1Offsets.kDecayWorkOffset], vec1Offsets.blockTokens, kHeadDim,
                            vec1Offsets.vBlockDim, vHeadDim, vecBlockScheduler.cube1Done[streamId],
                            vecBlockScheduler.vec1Done[streamId], vec1Offsets.isInitialState, vec1Offsets.isFinalState,
                            storeFinalState, waitWsFromMte3, (i == 0), tailVectorPath, useDirectForTask,
                            DIRECT_UB_FREE_FLAG_BEGIN, DIRECT_UB_READY_FLAG_BEGIN, DIRECT_UB_FLAG_STRIDE);
                        if (storeFinalState && std::is_same<ElementFinalState, float>::value) {
                            event0FromMte3[streamId] = false;
                        }
                    }
                } else {
                    /* V2: h[i+1] += h_work if i < num_chunks - 1 else None */
                    for (uint32_t i = 0; i < PING_PONG_STAGES; ++i) {
                        uint32_t streamId = vecBlockScheduler.GetStreamId(i);
                        const auto &stream = vecBlockScheduler.GetStream(i);
                        if (vecBlockScheduler.StreamIsDone(stream)) {
                            continue;
                        }
                        const KdaFwdHOffsets &vec2Offsets = vecBlockScheduler.GetCurTaskOffsets(stream);
                        if (vecBlockScheduler.NeedProcessStage2(stream)) {
                            bool tailVectorPath = vec2Offsets.blockTokens < 16;
                            bool fuseTailFinalState = false;
                            if constexpr (kGated && !scalarGated && useExp2 &&
                                          std::is_same<ElementFinalState, float>::value) {
                                fuseTailFinalState = tailVectorPath && vec2Offsets.isFinalState && storeFinalState;
                            }
                            if (tailVectorPath) {
                                Arch::CrossCoreWaitFlag(vecBlockScheduler.cube2Done[streamId]);
                                if (fuseTailFinalState) {
                                    ComputeKdaTailFinalState(vec2Offsets, EVENT_ID3 + (i == 0 ? 0 : pongBaseEvent));
                                } else {
                                    ComputeTailHWorkspace(vec2Offsets, EVENT_ID3 + (i == 0 ? 0 : pongBaseEvent));
                                }
                            }
                            if (!fuseTailFinalState) {
                                if (storeFinalState && std::is_same<ElementFinalState, float>::value) {
                                    if (vec2Offsets.isInitialState && event2FromMte3[streamId]) {
                                        uint32_t event2Id = EVENT_ID2 + (i == 0 ? 0 : pongBaseEvent);
                                        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event2Id);
                                        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event2Id);
                                        event2FromMte3[streamId] = false;
                                    }
                                    // Update always writes the FP32 state through MTE3,
                                    // including intermediate chunks.
                                    event0FromMte3[streamId] = true;
                                    event2FromMte3[streamId] = !vec2Offsets.isFinalState;
                                }
                                epilogueKdaFwdHUpdate(
                                    gmH[vec2Offsets.hDstOffset], gmFinalState[vec2Offsets.finalStateOffset],
                                    gmG[vec2Offsets.gOffset], gmH[vec2Offsets.hSrcOffset],
                                    gmHWorkspace[vec2Offsets.hWorkOffset], gmGk[vec2Offsets.gkOffset],
                                    gmInitialState[vec2Offsets.initialStateOffset], vec2Offsets.blockTokens, kHeadDim,
                                    vec2Offsets.vBlockDim, vHeadDim, vecBlockScheduler.cube2Done[streamId],
                                    vec2Offsets.isInitialState, vec2Offsets.isFinalState, storeFinalState,
                                    useInitialState, (i == 0), tailVectorPath, useDirectFp32Ub && !tailVectorPath,
                                    DIRECT_UB_FREE_FLAG_BEGIN, DIRECT_UB_READY_FLAG_BEGIN, DIRECT_UB_FLAG_STRIDE);
                            }
                        } else {
                            if (!useDirectFp32Ub) {
                                Arch::CrossCoreWaitFlag(vecBlockScheduler.cube2Done[streamId]);
                            }
                        }
                        Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecBlockScheduler.vec2Done[streamId]);
                    }
                }
                currStage ^= 0x01;
            }
            if (storeFinalState && std::is_same<ElementFinalState, float>::value) {
                if (event0FromMte3[0]) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                } else {
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
                }
                if (event0FromMte3[1]) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0 + pongBaseEvent);
                } else {
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + pongBaseEvent);
                }
                if (event2FromMte3[0]) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
                } else {
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
                }
                if (event2FromMte3[1]) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2 + pongBaseEvent);
                } else {
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2 + pongBaseEvent);
                }
            } else {
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0); // preset h_update
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + pongBaseEvent);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2); // preset h
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2 + pongBaseEvent);
            }
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1); // preset u
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1 + pongBaseEvent);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3); // preset g
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3 + pongBaseEvent);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0); // drain h_update
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0 + pongBaseEvent);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2); // drain h
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2 + pongBaseEvent);
        }
    }
};

} // namespace Catlass::Gemm::Kernel
