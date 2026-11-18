/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../op_kernel/chunk_kda_fwd_tiling_data.h"
#include "arch35/chunk_kda_fwd_tiling_impl.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>
#include <cstring>

namespace optiling {
namespace {
constexpr uint64_t KDA_ALIGN = 512;
constexpr uint64_t KDA_SOLVE_SCRATCH_SLOTS = 5;
constexpr uint64_t KDA_SOLVE_PIPELINE_DEPTH = 4;
constexpr uint64_t KDA_SCORE_QUEUE_SLOTS = 4;
constexpr uint64_t KDA_SCORE_SCRATCH_PLANES = 3;
constexpr uint64_t KDA_GDN_PIPELINE_DEPTH = 2;

constexpr int64_t MAX_KDA_K_DIM = 256;
constexpr int64_t MAX_KDA_HEAD_NUM = 128;
constexpr int64_t MAX_KDA_VARLEN_SEQUENCES = 1024;

uint64_t AlignWorkspace(uint64_t bytes)
{
    return (bytes + KDA_ALIGN - 1) / KDA_ALIGN * KDA_ALIGN;
}

uint64_t AllocateWorkspace(uint64_t &cursor, uint64_t bytes)
{
    const uint64_t offset = AlignWorkspace(cursor);
    cursor = offset + bytes;
    return offset;
}

static void ComputeChunkKdaFwdTilingData(int64_t batch, int64_t seqlen, int64_t hNum, int64_t hvNum, int64_t kDim,
                                         int64_t vDim, int64_t chunkSize, int64_t seqNum, int64_t totalChunks,
                                         int64_t inputRank, double scale, double lowerBound, bool hasInitialState,
                                         bool isVarLen, bool safeGate, bool sequenceMajor, bool useGateInKernel,
                                         bool hasALog, bool hasDtBias, int64_t gateDataType, bool storeFinalState,
                                         bool storeGk, bool storeW, bool storeU, bool storeQG, bool storeKg,
                                         bool storeVNew, bool storeH, uint32_t blockDim,
                                         const optiling::arch35::ChunkKdaFwdArch35Options &arch35Options,
                                         int64_t betaDataType, optiling::ChunkKdaFwdTilingData &td)
{
    std::memset(&td, 0, sizeof(td));

    td.batch = batch;
    td.seqNum = seqNum;
    td.qHeadNum = hNum;
    td.vHeadNum = hvNum;
    td.seqlen = seqlen;
    td.kHeadDim = kDim;
    td.vHeadDim = vDim;
    td.chunkSize = chunkSize;
    td.totalChunks = totalChunks;
    td.inputRank = inputRank;
    td.scale = static_cast<float>(scale);
    td.lowerBound = static_cast<float>(lowerBound);
    td.hasInitialState = hasInitialState;
    td.isVarLen = isVarLen;
    td.safeGate = safeGate;
    td.inputSequenceMajor = sequenceMajor;
    td.useGateInKernel = useGateInKernel;
    td.hasALog = hasALog;
    td.hasDtBias = hasDtBias;
    td.computeGateInPrepare = arch35Options.computeGateInPrepare;
    td.fusePostWu = arch35Options.fusePostWu;
    td.fusePostWuIntoFwdH = arch35Options.fusePostWuIntoFwdH;
    td.useDenseFwdH = arch35Options.useDenseFwdH;
    td.storeFinalState = storeFinalState;
    td.storeGk = storeGk;
    td.storeW = storeW;
    td.storeU = storeU;
    td.storeQG = storeQG;
    td.storeKg = storeKg;
    td.storeVNew = storeVNew;
    td.storeH = storeH;
    td.gateDataType = gateDataType;
    td.gateUsedCoreNum = static_cast<int64_t>(blockDim) * 2;
    td.prepareUsedCoreNum = blockDim;
    td.postWuUsedCoreNum = blockDim;
    td.outputUsedCoreNum = blockDim;
    td.betaDataType = betaDataType;

    const uint64_t dataBytes = 2;  // direct-launch currently only supports bf16 q/k/v
    const uint64_t tokenHeads = static_cast<uint64_t>(batch) * hvNum * seqlen;
    const uint64_t kTensorBytes = tokenHeads * kDim * dataBytes;
    const uint64_t vTensorBytes = tokenHeads * vDim * dataBytes;
    const uint64_t gkBytes = tokenHeads * kDim * sizeof(float);
    const uint64_t stateElements = static_cast<uint64_t>(seqNum) * hvNum * kDim * vDim;
    const uint64_t hChunkCount =
        isVarLen ? static_cast<uint64_t>(totalChunks) : static_cast<uint64_t>(batch) * totalChunks;
    const uint64_t hBytes = hChunkCount * hvNum * kDim * vDim * dataBytes;

    uint64_t cursor = 0;
    td.gkStorageOffset = storeGk ? 0 : AllocateWorkspace(cursor, gkBytes);
    td.finalStateStorageOffset = storeFinalState ? 0 : AllocateWorkspace(cursor, stateElements * sizeof(float));
    td.wStorageOffset = storeW ? 0 : AllocateWorkspace(cursor, kTensorBytes);
    td.uStorageOffset = storeU ? 0 : AllocateWorkspace(cursor, vTensorBytes);
    td.qgStorageOffset = storeQG ? 0 : AllocateWorkspace(cursor, kTensorBytes);
    td.kgStorageOffset = storeKg ? 0 : AllocateWorkspace(cursor, kTensorBytes);
    const uint64_t vNewStorageBytes = arch35Options.useDenseFwdH && !storeVNew
                                          ? static_cast<uint64_t>(batch) * hvNum * chunkSize * vDim * dataBytes
                                          : vTensorBytes;
    td.vNewStorageOffset = storeVNew ? 0 : AllocateWorkspace(cursor, vNewStorageBytes);
    const uint64_t hStorageBytes =
        arch35Options.useDenseFwdH && !storeH ? static_cast<uint64_t>(batch) * hvNum * kDim * vDim * dataBytes : hBytes;
    td.hStorageOffset = storeH ? 0 : AllocateWorkspace(cursor, hStorageBytes);
    td.qgScaledOffset = AllocateWorkspace(cursor, kTensorBytes);

    const uint64_t matrixBytes = tokenHeads * chunkSize * sizeof(float);
    td.prepareAqkFp32Offset = AllocateWorkspace(cursor, matrixBytes);
    td.prepareAkkFp32Offset = AllocateWorkspace(cursor, matrixBytes);
    td.prepareScratchOffset = AlignWorkspace(cursor);
    const uint64_t solveDepth = safeGate ? KDA_SOLVE_PIPELINE_DEPTH : 1;
    const uint64_t solveBytes =
        static_cast<uint64_t>(blockDim) * solveDepth * KDA_SOLVE_SCRATCH_SLOTS * chunkSize * chunkSize * sizeof(float);
    const uint64_t scoreBytes = static_cast<uint64_t>(blockDim) * KDA_SCORE_QUEUE_SLOTS * KDA_SCORE_SCRATCH_PLANES *
                                chunkSize * kDim * dataBytes;
    cursor = td.prepareScratchOffset + AlignWorkspace(solveBytes) + scoreBytes;

    td.postWuScratchOffset = AlignWorkspace(cursor);
    if (!arch35Options.fusePostWu && !arch35Options.fusePostWuIntoFwdH) {
        // PostWU addresses W scratch by padded chunk index, not packed token index.
        const uint64_t paddedTokenHeads = static_cast<uint64_t>(batch) * hvNum * totalChunks * chunkSize;
        cursor = td.postWuScratchOffset + paddedTokenHeads * kDim * sizeof(float);
    }

    td.fwdHWorkspaceBaseOffset = AlignWorkspace(cursor);
    uint64_t fwdHCursor = 0;
    td.vWorkspaceOffset = AllocateWorkspace(
        fwdHCursor, static_cast<uint64_t>(blockDim) * chunkSize * vDim * sizeof(float) * KDA_GDN_PIPELINE_DEPTH);
    td.vUpdateWorkspaceOffset = AllocateWorkspace(
        fwdHCursor, static_cast<uint64_t>(blockDim) * chunkSize * vDim * sizeof(float) * KDA_GDN_PIPELINE_DEPTH);
    td.kDecayWorkspaceOffset = AllocateWorkspace(
        fwdHCursor, static_cast<uint64_t>(blockDim) * chunkSize * kDim * sizeof(float) * KDA_GDN_PIPELINE_DEPTH);
    td.hWorkspaceOffset = AllocateWorkspace(
        fwdHCursor, static_cast<uint64_t>(blockDim) * kDim * vDim * sizeof(float) * KDA_GDN_PIPELINE_DEPTH);
    const uint64_t tokenBatch = isVarLen ? static_cast<uint64_t>(seqNum) : 1;
    td.numSeqWorkspaceOffset = AllocateWorkspace(fwdHCursor, (tokenBatch + 1) * sizeof(int64_t));
    td.numChunksWorkspaceOffset = AllocateWorkspace(fwdHCursor, (tokenBatch + 1) * sizeof(int64_t));
    cursor = td.fwdHWorkspaceBaseOffset + AlignWorkspace(fwdHCursor);

    td.outputScratchOffset = AllocateWorkspace(cursor, 2 * tokenHeads * vDim * sizeof(float));
}

}  // namespace

static ge::graphStatus TilingChunkKdaFwd(gert::TilingContext *context)
{
    const auto *attrs = context->GetAttrs();
    if (!attrs || !context->GetPlatformInfo()) return ge::GRAPH_FAILED;
    const auto *batch = attrs->GetInt(0);
    const auto *seqlen = attrs->GetInt(1);
    const auto *q_heads = attrs->GetInt(2);
    const auto *v_heads = attrs->GetInt(3);
    const auto *k_dim = attrs->GetInt(4);
    const auto *v_dim = attrs->GetInt(5);
    const auto *chunk_size = attrs->GetInt(6);
    const auto *seq_num = attrs->GetInt(7);
    const auto *total_chunks = attrs->GetInt(8);
    const auto *input_rank = attrs->GetInt(9);
    const auto *scale = attrs->GetFloat(10);
    const auto *lower_bound = attrs->GetFloat(11);
    const auto *sequence_major = attrs->GetBool(12);
    const auto *safe_gate = attrs->GetBool(13);
    const auto *use_gate_in_kernel = attrs->GetBool(14);
    const auto *store_final_state = attrs->GetBool(15);
    const auto *store_gk = attrs->GetBool(16);
    const auto *store_w = attrs->GetBool(17);
    const auto *store_u = attrs->GetBool(18);
    const auto *store_qg = attrs->GetBool(19);
    const auto *store_kg = attrs->GetBool(20);
    const auto *store_v_new = attrs->GetBool(21);
    const auto *store_h = attrs->GetBool(22);
    if (!batch || !seqlen || !q_heads || !v_heads || !k_dim || !v_dim || !chunk_size || !seq_num || !total_chunks ||
        !input_rank || !scale || !lower_bound || !sequence_major || !safe_gate || !use_gate_in_kernel ||
        !store_final_state || !store_gk || !store_w || !store_u || !store_qg || !store_kg || !store_v_new || !store_h)
        return ge::GRAPH_FAILED;
    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const uint32_t physicalCoreNum = std::max<uint32_t>(platform.GetCoreNumAic(), 1);
    const bool isAscend950 = platform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND950;
    const bool isVarLen = context->GetOptionalInputDesc(8) != nullptr;
    const bool hasALog = context->GetOptionalInputDesc(5) != nullptr;
    const bool hasDtBias = context->GetOptionalInputDesc(6) != nullptr;
    const bool hasInitialState = context->GetOptionalInputDesc(7) != nullptr;
    if (!context->GetInputDesc(3) || !context->GetInputDesc(4)) return ge::GRAPH_FAILED;
    const bool gIsFp32 = context->GetInputDesc(3)->GetDataType() == ge::DT_FLOAT;
    const int64_t gateDataType = gIsFp32 ? 2 : 1;
    const int64_t betaDataType = context->GetInputDesc(4)->GetDataType() == ge::DT_FLOAT ? 0 : 1;
    const uint64_t fwdHTaskCount = static_cast<uint64_t>(isVarLen ? *seq_num : *batch) * *v_heads;
    const uint32_t blockDim =
        isAscend950 ? static_cast<uint32_t>(std::min<uint64_t>(physicalCoreNum, std::max<uint64_t>(fwdHTaskCount, 1)))
                    : physicalCoreNum;
    const auto options = arch35::ConfigureChunkKdaFwdArch35(isAscend950, true, gIsFp32, hasALog, *use_gate_in_kernel,
                                                            *safe_gate, isVarLen, *seqlen, *v_heads, *chunk_size,
                                                            *k_dim, *v_dim, *store_qg, *store_v_new, *store_h);
    ChunkKdaFwdTilingData td;
    ComputeChunkKdaFwdTilingData(*batch, *seqlen, *q_heads, *v_heads, *k_dim, *v_dim, *chunk_size, *seq_num,
                                 *total_chunks, *input_rank, *scale, *lower_bound, hasInitialState, isVarLen,
                                 *safe_gate, *sequence_major, *use_gate_in_kernel, hasALog, hasDtBias, gateDataType,
                                 *store_final_state, *store_gk, *store_w, *store_u, *store_qg, *store_kg, *store_v_new,
                                 *store_h, blockDim, options, betaDataType, td);
    const uint64_t tokenHeads = static_cast<uint64_t>(*batch) * *v_heads * *seqlen;
    const uint64_t userWorkspace = td.outputScratchOffset + 2 * tokenHeads * *v_dim * sizeof(float);
    auto *workspace = context->GetWorkspaceSizes(1);
    auto *raw = context->GetRawTilingData();
    if (!workspace || !raw || raw->GetCapacity() < sizeof(td)) return ge::GRAPH_FAILED;
    std::memcpy(raw->GetData(), &td, sizeof(td));
    raw->SetDataSize(sizeof(td));
    workspace[0] = platform.GetLibApiWorkSpaceSize() + AlignWorkspace(userWorkspace);
    context->SetBlockDim(blockDim);
    context->SetScheduleMode(1);
    context->SetTilingKey(0);
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(SglChunkKdaFwd).Tiling(TilingChunkKdaFwd);
}  // namespace optiling
