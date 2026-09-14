/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define CATLASS_ARCH 2201

#include "catlass/arch/kda_arch.hpp"
#include "catlass/arch/kda_cross_core_sync.hpp"
#include "catlass/arch/kda_resource.hpp"
#include "catlass/kda_catlass.hpp"
#include "catlass/kda_debug.hpp"
#include "catlass/epilogue/block/kda_block_epilogue.hpp"
#include "../../epilogue/block/block_epilogue_kda_fwdh_update_arch22.hpp"
#include "../../epilogue/block/block_epilogue_kda_fwdh_vnew_arch22.hpp"
#include "catlass/gemm/block/kda_block_mmad.hpp"
#include "../../../../kernel_utils/block/block_mmad_pingpong_tla_multi.hpp"
#include "../block/block_scheduler_kda_fwd_h_arch22.hpp"
#include "catlass/gemm/kda_gemm_dispatch_policy.hpp"
#include "catlass/gemm/kda_gemm_type.hpp"
#include "catlass/layout/kda_layout.hpp"
#include "catlass/kda_gemm_coord.hpp"
#include "tla/kda_tensor.hpp"
#include "tla/kda_tla_layout.hpp"
#include "tla/kda_tensor.hpp"

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
    using ArchTag = Arch::AtlasA2;
    using CubeScheduler = typename Catlass::Gemm::Block::BlockSchedulerKdaFwdHCube;
    using VecScheduler = typename Catlass::Gemm::Block::BlockSchedulerKdaFwdHVec;

    using DispatchPolicyTla = Gemm::MmadPingpongTlaMulti<ArchTag, true, false>;
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

    // cube 1
    using TileCopyWH = Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, INPUT_TYPE, layout::RowMajor, INPUT_TYPE,
                                                              layout::RowMajor, WORKSPACE_TYPE, layout::RowMajor>;
    using BlockMmadWH = Gemm::Block::BlockMmadTla<DispatchPolicyTla, L1TileShapeVTla, L0TileShapeVTla, INPUT_TYPE,
                                                  INPUT_TYPE, WORKSPACE_TYPE, void, TileCopyWH>;

    // cube 2
    using TileCopyKV = Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, INPUT_TYPE, layout::ColumnMajor, INPUT_TYPE,
                                                              layout::RowMajor, WORKSPACE_TYPE, layout::RowMajor>;
    using BlockMmadKV = Gemm::Block::BlockMmadTla<DispatchPolicyTla, L1TileShapeVTla, L0TileShapeVTla, INPUT_TYPE,
                                                  INPUT_TYPE, WORKSPACE_TYPE, void, TileCopyKV>;

    // vec 1
    using DispatchPolicyKdaFwdHVnew = Epilogue::EpilogueAtlasKdaFwdHVnew;
    using GateTag = KdaFwdHGateTag<kGated, scalarGated, useExp2>;
    using EpilogueKdaFwdHVnew = Epilogue::Block::BlockEpilogue<DispatchPolicyKdaFwdHVnew, VType, GType, UType,
                                                               VworkType, FinalStateType, GateTag>;

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
    using ElementVWork = WORKSPACE_TYPE;
    using ElementHWork = WORKSPACE_TYPE;
    using ElementInitialState = STATE_TYPE;
    using ElementFinalState = STATE_TYPE;

    using LayoutW = Catlass::layout::RowMajor;
    using LayoutH = Catlass::layout::RowMajor;
    using LayoutV = Catlass::layout::RowMajor;
    using LayoutK = Catlass::layout::ColumnMajor;

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

    AscendC::GlobalTensor<ElementK> gmK;
    AscendC::GlobalTensor<ElementW> gmW;
    AscendC::GlobalTensor<ElementU> gmU;
    AscendC::GlobalTensor<ElementG> gmG;
    AscendC::GlobalTensor<ElementInitialState> gmInitialState;
    AscendC::GlobalTensor<ElementH> gmH;
    AscendC::GlobalTensor<ElementV> gmV;
    AscendC::GlobalTensor<ElementFinalState> gmFinalState;
    AscendC::GlobalTensor<ElementVWork> gmVWorkspace;
    AscendC::GlobalTensor<ElementV> gmVUpdateWorkspace;
    AscendC::GlobalTensor<ElementHWork> gmHWorkspace;

    AscendC::GlobalTensor<ElementG> gmGk;
    AscendC::GlobalTensor<ElementK> gmKDecayWorkspace;

    AscendC::GlobalTensor<int64_t> gmSeqlen;
    AscendC::GlobalTensor<int64_t> gmNumSeq;
    AscendC::GlobalTensor<int64_t> gmNumChunks;

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

        if ASCEND_IS_AIC {
            cubeBlockScheduler.InitFromData(cu_seqlens, chunk_indices, tilingData, user);
        }
        if ASCEND_IS_AIV {
            vecBlockScheduler.InitFromData(cu_seqlens, chunk_indices, tilingData, user);
        }
    }

    template <typename Element>
    __aicore__ inline float LoadScalarAsFloat(AscendC::GlobalTensor<Element> tensor, uint32_t offset) const
    {
        Element value = tensor.GetValue(offset);
        if constexpr (std::is_same<Element, bfloat16_t>::value) {
            return AscendC::ToFloat(value);
        }
        return static_cast<float>(value);
    }

    // Borrow the stream's V_MTE2 free token and restore it before returning.
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

        constexpr uint32_t TAIL_INPUT_OFFSET = 166 * 1024;
        constexpr uint32_t TAIL_FLOAT_OFFSET = 167 * 1024;
        constexpr uint32_t TAIL_ACCUM_OFFSET = 168 * 1024;
        AscendC::LocalTensor<ElementH> inputUb = resource.ubBuf.template GetBufferByByte<ElementH>(TAIL_INPUT_OFFSET);
        AscendC::LocalTensor<float> floatUb = resource.ubBuf.template GetBufferByByte<float>(TAIL_FLOAT_OFFSET);
        AscendC::LocalTensor<float> accumUb = resource.ubBuf.template GetBufferByByte<float>(TAIL_ACCUM_OFFSET);

        for (uint32_t tokenRow = rowBegin; tokenRow < rowEnd; ++tokenRow) {
            AscendC::Duplicate(accumUb, 0.0f, offsets.vBlockDim);
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t kIdx = 0; kIdx < kHeadDim; ++kIdx) {
                AscendC::DataCopy(inputUb, gmH[offsets.hSrcOffset + kIdx * vHeadDim], offsets.vBlockDim);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(tailEventId);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(tailEventId);
                AscendC::Cast(floatUb, inputUb, AscendC::RoundMode::CAST_NONE, offsets.vBlockDim);
                // inputUb is reused by the next kIdx iteration.  PIPE_V and
                // PIPE_MTE2 run independently, so a PIPE_V barrier alone does
                // not stop the next DataCopy from overwriting inputUb while
                // Cast is still reading it.  Queue a V -> MTE2 dependency
                // before that next copy, matching ComputeHWorkspaceVector().
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(tailEventId);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(tailEventId);
                AscendC::PipeBarrier<PIPE_V>();
                float weight = LoadScalarAsFloat(gmW, offsets.wOffset + tokenRow * kHeadDim + kIdx);
                AscendC::Muls(floatUb, floatUb, weight, offsets.vBlockDim);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(accumUb, accumUb, floatUb, offsets.vBlockDim);
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(tailEventId);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(tailEventId);
            AscendC::DataCopy(gmVWorkspace[offsets.vWorkOffset + tokenRow * offsets.vBlockDim], accumUb,
                              offsets.vBlockDim);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(tailEventId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(tailEventId);
            // The MTE3 copy above still reads accumUb.  The next token row
            // starts by clearing the same buffer on PIPE_V, so protect that
            // reuse explicitly; MTE3_MTE2 does not order MTE3 against PIPE_V.
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(tailEventId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(tailEventId);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(tailEventId);
    }

    __aicore__ inline void ComputeHWorkspaceVector(const KdaFwdHOffsets &offsets, uint32_t eventId)
    {
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();
        uint32_t rowsPerSubBlock = CeilDiv(kHeadDim, subBlockNum);
        uint32_t rowBegin = subBlockIdx * rowsPerSubBlock;
        uint32_t rowEnd = Min(rowBegin + rowsPerSubBlock, kHeadDim);

        constexpr uint32_t MAX_K_ROWS = 64;
        constexpr uint32_t ACCUM_OFFSET = 0;
        constexpr uint32_t WEIGHT_BROADCAST_OFFSET = 32 * 1024;
        constexpr uint32_t WEIGHT_INPUT_OFFSET = 64 * 1024;
        constexpr uint32_t WEIGHT_FLOAT_OFFSET = 65 * 1024;
        constexpr uint32_t V_INPUT_OFFSET = 66 * 1024;
        constexpr uint32_t V_FLOAT_OFFSET = 67 * 1024;
        constexpr uint32_t PRODUCT_OFFSET = 68 * 1024;
        constexpr uint32_t BROADCAST_SCRATCH_OFFSET = 96 * 1024;

        AscendC::LocalTensor<float> accumUb = resource.ubBuf.template GetBufferByByte<float>(ACCUM_OFFSET);
        AscendC::LocalTensor<float> weightBroadcastUb =
            resource.ubBuf.template GetBufferByByte<float>(WEIGHT_BROADCAST_OFFSET);
        AscendC::LocalTensor<ElementK> weightInputUb =
            resource.ubBuf.template GetBufferByByte<ElementK>(WEIGHT_INPUT_OFFSET);
        AscendC::LocalTensor<float> weightFloatUb = resource.ubBuf.template GetBufferByByte<float>(WEIGHT_FLOAT_OFFSET);
        AscendC::LocalTensor<ElementV> vInputUb = resource.ubBuf.template GetBufferByByte<ElementV>(V_INPUT_OFFSET);
        AscendC::LocalTensor<float> vFloatUb = resource.ubBuf.template GetBufferByByte<float>(V_FLOAT_OFFSET);
        AscendC::LocalTensor<float> productUb = resource.ubBuf.template GetBufferByByte<float>(PRODUCT_OFFSET);
        AscendC::LocalTensor<uint8_t> broadcastScratchUb =
            resource.ubBuf.template GetBufferByByte<uint8_t>(BROADCAST_SCRATCH_OFFSET);

        AscendC::ResetMask();
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventId);

        for (uint32_t kTileStart = rowBegin; kTileStart < rowEnd; kTileStart += MAX_K_ROWS) {
            uint32_t rowCount = Min(MAX_K_ROWS, rowEnd - kTileStart);
            uint32_t accumElements = rowCount * offsets.vBlockDim;
            AscendC::Duplicate(accumUb, 0.0f, accumElements);
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t tokenRow = 0; tokenRow < offsets.blockTokens; ++tokenRow) {
                AscendC::DataCopy(weightInputUb, gmK[offsets.wkOffset + tokenRow * kHeadDim], kHeadDim);
                AscendC::DataCopy(vInputUb, gmV[offsets.uvOffset + tokenRow * vHeadDim], offsets.vBlockDim);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventId);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventId);
                AscendC::Cast(weightFloatUb, weightInputUb, AscendC::RoundMode::CAST_NONE, kHeadDim);
                AscendC::Cast(vFloatUb, vInputUb, AscendC::RoundMode::CAST_NONE, offsets.vBlockDim);
                AscendC::PipeBarrier<PIPE_V>();

                uint32_t dstShape[2] = {rowCount, offsets.vBlockDim};
                uint32_t srcShape[2] = {rowCount, 1};
                AscendC::Broadcast<float, 2, 1>(weightBroadcastUb, weightFloatUb[kTileStart], dstShape, srcShape,
                                                broadcastScratchUb);
                AscendC::PipeBarrier<PIPE_V>();

                for (uint32_t localRow = 0; localRow < rowCount; ++localRow) {
                    uint32_t rowOffset = localRow * offsets.vBlockDim;
                    AscendC::Mul<float>(productUb, vFloatUb, weightBroadcastUb[rowOffset], offsets.vBlockDim);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add<float>(accumUb[rowOffset], accumUb[rowOffset], productUb, offsets.vBlockDim);
                    AscendC::PipeBarrier<PIPE_V>();
                }
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventId);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventId);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventId);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventId);
            AscendC::DataCopy(gmHWorkspace[offsets.hWorkOffset + kTileStart * offsets.vBlockDim], accumUb,
                              accumElements);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventId);
    }

    __aicore__ inline void PresetVectorPipelineEvents()
    {
        constexpr uint32_t pongBaseEvent = 4;
        if (storeFinalState && std::is_same<ElementFinalState, float>::value) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + pongBaseEvent);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2 + pongBaseEvent);
        } else {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + pongBaseEvent);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2 + pongBaseEvent);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1 + pongBaseEvent);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3 + pongBaseEvent);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0 + pongBaseEvent);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2 + pongBaseEvent);
    }

    __aicore__ inline void DrainVectorPipelineEvents(const bool (&event0FromMte3)[PING_PONG_STAGES],
                                                     const bool (&event2FromMte3)[PING_PONG_STAGES])
    {
        constexpr uint32_t pongBaseEvent = 4;
        if (storeFinalState && std::is_same<ElementFinalState, float>::value) {
            for (uint32_t streamId = 0; streamId < PING_PONG_STAGES; ++streamId) {
                uint32_t eventOffset = streamId * pongBaseEvent;
                if (event0FromMte3[streamId]) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0 + eventOffset);
                } else {
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + eventOffset);
                }
                if (event2FromMte3[streamId]) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2 + eventOffset);
                } else {
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2 + eventOffset);
                }
            }
        } else {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + pongBaseEvent);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2 + pongBaseEvent);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1 + pongBaseEvent);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3 + pongBaseEvent);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0 + pongBaseEvent);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2 + pongBaseEvent);
    }

    __aicore__ inline void Process()
    {
        if (isVariedLen) {
            AscendC::SyncAll<false>();
        }

        if ASCEND_IS_AIC {
            uint32_t coreIdx = AscendC::GetBlockIdx();
            uint32_t coreNum = AscendC::GetBlockNum();

            auto wLayout =
                tla::MakeLayout<ElementW, LayoutW>(shapeBatch * kNumHead * cubeBlockScheduler.totalTokens, kHeadDim);
            auto hLayout = tla::MakeLayout<ElementH, LayoutH>(
                shapeBatch * vNumHead * cubeBlockScheduler.totalChunks * kHeadDim, vHeadDim);
            auto vLayout = tla::MakeLayout<ElementVWork, LayoutV>(coreNum * chunkSize * PING_PONG_STAGES,
                                                                  cubeBlockScheduler.vBlockSize);
            auto kLayout =
                tla::MakeLayout<ElementK, LayoutK>(kHeadDim, shapeBatch * kNumHead * cubeBlockScheduler.totalTokens);
            auto vworkLayout = tla::MakeLayout<ElementV, LayoutV>(coreNum * chunkSize * PING_PONG_STAGES,
                                                                  cubeBlockScheduler.vBlockSize);
            auto hworkLayout = tla::MakeLayout<ElementHWork, LayoutH>(coreNum * kHeadDim * PING_PONG_STAGES,
                                                                      cubeBlockScheduler.vBlockSize);

            uint32_t taskWaveCount = cubeBlockScheduler.GetTaskWaveCount();
            for (uint32_t waveIdx = 0; waveIdx < taskWaveCount; ++waveIdx) {
                BlockMmadWH blockMmadWH(resource);
                BlockMmadKV blockMmadKV(resource);
                BlockMmadWH blockMmadWHTail(resource);
                BlockMmadKV blockMmadKVTail(resource);
                AscendC::SyncAll<false>();
                cubeBlockScheduler.InitTaskWave(waveIdx);
                uint32_t currStage = 0; // 0: C1, 1: C2
                while (cubeBlockScheduler.isRunning) {
                    if (currStage == 0) {
                        /* C1: v_work = w @ h[i] */
                        cubeBlockScheduler.InitTasks();
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
                            if (cube1Offsets.blockTokens < chunkSize) {
                                blockMmadWHTail.preSetFlags();
                                blockMmadWHTail(tensorBlockW, tensorBlockH, tensorBlockV, cube1Shape, EmptyClass{},
                                                true);
                                blockMmadWHTail.finalWaitFlags();
                            } else {
                                blockMmadWH.preSetFlags();
                                blockMmadWH(tensorBlockW, tensorBlockH, tensorBlockV, cube1Shape);
                                blockMmadWH.finalWaitFlags();
                            }
                            AscendC::PipeBarrier<PIPE_ALL>();
                            Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeBlockScheduler.cube1Done[streamId]);
                        }
                    } else {
                        /* C2: h[i+1] = k.T @ v_work */
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
                                auto tensorK =
                                    tla::MakeTensor(gmK[cube2Offsets.wkOffset], kLayout, Catlass::Arch::PositionGM{});
                                auto tensorVwork = tla::MakeTensor(gmVUpdateWorkspace[cube2Offsets.vWorkOffset],
                                                                   vworkLayout, Catlass::Arch::PositionGM{});
                                auto tensorHwork = tla::MakeTensor(gmHWorkspace[cube2Offsets.hWorkOffset], hworkLayout,
                                                                   Catlass::Arch::PositionGM{});
                                GemmCoord cube2Shape{kHeadDim, cube2Offsets.vBlockDim, cube2Offsets.blockTokens};
                                auto tensorBlockK = GetTile(tensorK, tla::MakeCoord(0, 0),
                                                            tla::MakeShape(cube2Shape.m(), cube2Shape.k()));
                                auto tensorBlockVwork = GetTile(tensorVwork, tla::MakeCoord(0, 0),
                                                                tla::MakeShape(cube2Shape.k(), cube2Shape.n()));
                                auto tensorBlockHwork = GetTile(tensorHwork, tla::MakeCoord(0, 0),
                                                                tla::MakeShape(cube2Shape.m(), cube2Shape.n()));
                                if (cube2Offsets.blockTokens < chunkSize) {
                                    blockMmadKVTail.preSetFlags();
                                    blockMmadKVTail(tensorBlockK, tensorBlockVwork, tensorBlockHwork, cube2Shape,
                                                    EmptyClass{}, true);
                                    blockMmadKVTail.finalWaitFlags();
                                } else {
                                    blockMmadKV.preSetFlags();
                                    blockMmadKV(tensorBlockK, tensorBlockVwork, tensorBlockHwork, cube2Shape);
                                    blockMmadKV.finalWaitFlags();
                                }
                                AscendC::PipeBarrier<PIPE_ALL>();
                            }
                            Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeBlockScheduler.cube2Done[streamId]);
                        }
                    }
                    currStage ^= 0x01;
                }
                Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec2Done[0]);
                Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec2Done[1]);
            }
        }

        if ASCEND_IS_AIV {
            uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
            uint32_t subBlockNum = AscendC::GetSubBlockNum();
            uint32_t coreIdx = AscendC::GetBlockIdx() / subBlockNum;
            uint32_t coreNum = AscendC::GetBlockNum();
            uint32_t taskCount = (isVariedLen ? vecBlockScheduler.tokenBatch : shapeBatch) * vNumHead;
            uint32_t rowsPerSubBlock = (kHeadDim + subBlockNum - 1) / subBlockNum;
            uint32_t rowBegin = subBlockIdx * rowsPerSubBlock;
            uint32_t rowEnd = Min(rowBegin + rowsPerSubBlock, kHeadDim);
            uint32_t hRowsPerTile = (32 * 1024) / (vHeadDim * sizeof(ElementH));
            uint32_t stateRowsPerTile = (64 * 1024) / (vHeadDim * sizeof(ElementInitialState));
            uint32_t rowsPerTile = Min(hRowsPerTile, stateRowsPerTile);
            uint32_t totalChunks = isVariedLen ? vecBlockScheduler.totalChunks : ((seqlen + chunkSize - 1) / chunkSize);
            uint32_t stateBlockSize = kHeadDim * vHeadDim;
            AscendC::LocalTensor<ElementInitialState> stateUbTensorPing =
                resource.ubBuf.template GetBufferByByte<ElementInitialState>(0);
            AscendC::LocalTensor<ElementInitialState> stateUbTensorPong =
                resource.ubBuf.template GetBufferByByte<ElementInitialState>(96 * 1024);
            AscendC::LocalTensor<ElementH> hUbTensorPing = resource.ubBuf.template GetBufferByByte<ElementH>(64 * 1024);
            AscendC::LocalTensor<ElementH> hUbTensorPong =
                resource.ubBuf.template GetBufferByByte<ElementH>(160 * 1024);
            uint32_t taskWaveCount = vecBlockScheduler.GetTaskWaveCount();
            for (uint32_t waveIdx = 0; waveIdx < taskWaveCount; ++waveIdx) {
                EpilogueKdaFwdHVnew epilogueKdaFwdHVnew(resource);
                EpilogueKdaFwdHUpdate epilogueKdaFwdHUpdate(resource);
                uint32_t taskIdx = waveIdx * coreNum + coreIdx;
                uint32_t pingpongFlag = 1;
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
                if (taskIdx < taskCount) {
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
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);

                AscendC::SyncAll<false>();
                vecBlockScheduler.InitTaskWave(waveIdx);
                Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecBlockScheduler.vec2Done[0]);
                Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecBlockScheduler.vec2Done[1]);
                PresetVectorPipelineEvents();
                uint32_t currStage = 0; // 0: V1, 1: V2
                bool waitStageFence = false;
                bool event0FromMte3[PING_PONG_STAGES] = {false, false};
                bool event2FromMte3[PING_PONG_STAGES] = {
                    !(storeFinalState && std::is_same<ElementFinalState, float>::value),
                    !(storeFinalState && std::is_same<ElementFinalState, float>::value)};
                constexpr uint32_t pongEventOffset = 4;
                while (vecBlockScheduler.isRunning) {
                    if (waitStageFence) {
                        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
                    }
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
                            bool tailVectorPath = vec1Offsets.blockTokens < 16;
                            if (tailVectorPath) {
                                // For a short tail Cube does not calculate W @ H,
                                // but cube1Done still orders this task after the
                                // previous chunk's V2 update of gmH.  Wait before
                                // the software W @ H reads gmH, not afterwards in
                                // the epilogue.
                                Arch::CrossCoreWaitFlag(vecBlockScheduler.cube1Done[streamId]);
                                ComputeTailVWorkspace(vec1Offsets,
                                                      EVENT_ID3 + (streamId == 0 ? 0 : pongEventOffset));
                            }
                            bool waitWsFromMte3 = storeFinalState && std::is_same<ElementFinalState, float>::value &&
                                                  event0FromMte3[streamId];
                            epilogueKdaFwdHVnew(gmV[vec1Offsets.uvOffset], gmVUpdateWorkspace[vec1Offsets.vWorkOffset],
                                                gmG[vec1Offsets.gOffset], gmU[vec1Offsets.uvOffset],
                                                gmVWorkspace[vec1Offsets.vWorkOffset], vec1Offsets.blockTokens,
                                                vec1Offsets.vBlockDim, vHeadDim, vecBlockScheduler.cube1Done[streamId],
                                                vecBlockScheduler.vec1Done[streamId], vec1Offsets.isInitialState,
                                                vec1Offsets.isFinalState, storeFinalState, waitWsFromMte3,
                                                (streamId == 0), tailVectorPath);
                            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
                            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
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
                                bool vectorC2Path = vec2Offsets.blockTokens < 16;
                                if (vectorC2Path) {
                                    Arch::CrossCoreWaitFlag(vecBlockScheduler.cube2Done[streamId]);
                                    ComputeHWorkspaceVector(vec2Offsets,
                                                            EVENT_ID3 + (streamId == 0 ? 0 : pongEventOffset));
                                }
                                if (storeFinalState && std::is_same<ElementFinalState, float>::value) {
                                    event0FromMte3[streamId] = true;
                                    event2FromMte3[streamId] = !vec2Offsets.isFinalState;
                                }
                                // step 4:  h[i+1] += h_work if i < num_chunks - 1 else None
                                epilogueKdaFwdHUpdate(
                                    gmH[vec2Offsets.hDstOffset], gmFinalState[vec2Offsets.finalStateOffset],
                                    gmG[vec2Offsets.gOffset], gmH[vec2Offsets.hSrcOffset],
                                    gmHWorkspace[vec2Offsets.hWorkOffset], gmGk[vec2Offsets.gkOffset],
                                    gmInitialState[vec2Offsets.initialStateOffset], vec2Offsets.blockTokens, kHeadDim,
                                    vec2Offsets.vBlockDim, vHeadDim, vecBlockScheduler.cube2Done[streamId],
                                    vec2Offsets.isInitialState, vec2Offsets.isFinalState, storeFinalState,
                                    useInitialState, (streamId == 0), vectorC2Path);
                                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
                                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
                            } else {
                                Arch::CrossCoreWaitFlag(vecBlockScheduler.cube2Done[streamId]);
                            }
                            Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecBlockScheduler.vec2Done[streamId]);
                        }
                    }
                    waitStageFence = vecBlockScheduler.isRunning;
                    if (waitStageFence) {
                        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
                    }
                    currStage ^= 0x01;
                }

                DrainVectorPipelineEvents(event0FromMte3, event2FromMte3);
            }
        }
    }
};

} // namespace Catlass::Gemm::Kernel
