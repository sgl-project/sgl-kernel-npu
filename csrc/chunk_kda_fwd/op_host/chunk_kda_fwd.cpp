/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file chunk_kda_fwd.cpp
 * \brief host-side direct-launch implementation of chunk_kda_fwd
 *
 * Port of the aclnn ChunkKdaFwd front-end. Instead of building an aclnn executor
 * graph (transpose/reshape + ADD_TO_LAUNCHER_LIST_AICORE), the layout conversions
 * are done with plain torch ops and the kernel is launched directly through
 * EXEC_KERNEL_CMD. The tiling computation mirrors Tiling4ChunkKdaFwd and the
 * serialized tiling struct is copied GM->stack inside the kernel entry.
 */

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <tuple>
#include <vector>

#include "acl/acl.h"

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/DeviceUtils.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "tiling/platform/platform_ascendc.h"

#include "defines.h"
#include "torch_helper.h"
#include "aclrtlaunch_chunk_kda_fwd.h"

#include "../op_kernel/chunk_kda_fwd_tiling_data.h"
#include "arch35/chunk_kda_fwd_tiling_impl.h"

namespace sglang {
namespace npu_kernel {

constexpr uint32_t PADDING_BYTE = 32U;

constexpr uint64_t KDA_ALIGN = 512;
constexpr uint64_t KDA_SOLVE_SCRATCH_SLOTS = 5;
constexpr uint64_t KDA_SOLVE_PIPELINE_DEPTH = 4;
constexpr uint64_t KDA_SCORE_QUEUE_SLOTS = 4;
constexpr uint64_t KDA_SCORE_SCRATCH_PLANES = 3;
constexpr uint64_t KDA_GDN_PIPELINE_DEPTH = 2;

constexpr int64_t MAX_KDA_K_DIM = 256;
constexpr int64_t MAX_KDA_HEAD_NUM = 128;
constexpr int64_t MAX_KDA_VARLEN_SEQUENCES = 1024;


enum class KdaFwdLayout {
    BSND,
    BNSD,
    TND,
    NTD,
};

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

struct KdaShapeInfo {
    bool isRank3 = false;
    int64_t batch = 0;
    int64_t seqlen = 0;
    int64_t hNum = 0;
    int64_t hvNum = 0;
    int64_t kDim = 0;
    int64_t vDim = 0;
    int64_t seqNum = 0;
    int64_t totalChunks = 0;
};

bool ParseLayout(const std::string &layout, KdaFwdLayout &parsed)
{
    if (layout == "BSND") {
        parsed = KdaFwdLayout::BSND;
    } else if (layout == "BNSD") {
        parsed = KdaFwdLayout::BNSD;
    } else if (layout == "TND") {
        parsed = KdaFwdLayout::TND;
    } else if (layout == "NTD") {
        parsed = KdaFwdLayout::NTD;
    } else {
        return false;
    }
    return true;
}

int64_t Dim(const at::Tensor &tensor, size_t idx)
{
    return tensor.size(static_cast<int64_t>(idx));
}

bool SameShape(const at::Tensor &lhs, const at::Tensor &rhs)
{
    return lhs.dim() == rhs.dim() && lhs.sizes() == rhs.sizes();
}

bool HasShape(const at::Tensor &tensor, std::initializer_list<int64_t> expected)
{
    if (tensor.dim() != static_cast<int64_t>(expected.size())) {
        return false;
    }
    size_t idx = 0;
    for (int64_t dim : expected) {
        if (tensor.size(static_cast<int64_t>(idx++)) != dim) {
            return false;
        }
    }
    return true;
}

int64_t CountChunks(const c10::optional<at::Tensor> &cuSeqlens, int64_t seqlen, int64_t chunkSize)
{
    if (!cuSeqlens.has_value()) {
        return (seqlen + chunkSize - 1) / chunkSize;
    }
    const at::Tensor cu = cuSeqlens->cpu().contiguous();
    const int64_t *cuPtr = reinterpret_cast<const int64_t *>(cu.data_ptr());
    const int64_t n = cu.numel();
    int64_t chunks = 0;
    for (int64_t idx = 0; idx + 1 < n; ++idx) {
        chunks += (cuPtr[idx + 1] - cuPtr[idx] + chunkSize - 1) / chunkSize;
    }
    return chunks;
}

void ResolveShapeInfo(const at::Tensor &q, const at::Tensor &v, const c10::optional<at::Tensor> &beta,
                      const c10::optional<at::Tensor> &g, KdaFwdLayout layout, int64_t chunkSize,
                      const c10::optional<at::Tensor> &cuSeqlens, KdaShapeInfo &info)
{
    info.isRank3 = layout == KdaFwdLayout::TND || layout == KdaFwdLayout::NTD;
    const int64_t tensorRank = info.isRank3 ? 3 : 4;
    const int64_t betaRank = info.isRank3 ? 2 : 3;
    TORCH_CHECK(q.dim() == tensorRank && v.dim() == tensorRank && g->dim() == tensorRank &&
                    beta->dim() == betaRank,
                "chunk_kda_fwd: q/k/v/g and beta ranks must match layout: rank3/rank2 for TND/NTD, "
                "rank4/rank3 for BSND/BNSD.");

    if (layout == KdaFwdLayout::TND) {
        info.batch = 1;
        info.seqlen = Dim(q, 0);
        info.hNum = Dim(q, 1);
        info.kDim = Dim(q, 2);
        info.hvNum = Dim(v, 1);
        info.vDim = Dim(v, 2);
        TORCH_CHECK(HasShape(v, {info.seqlen, info.hvNum, info.vDim}) &&
                        HasShape(*g, {info.seqlen, info.hvNum, info.kDim}) &&
                        HasShape(*beta, {info.seqlen, info.hvNum}),
                    "chunk_kda_fwd: TND expects v/g/beta as [T,HV,V], [T,HV,K], [T,HV].");
    } else if (layout == KdaFwdLayout::NTD) {
        info.batch = 1;
        info.hNum = Dim(q, 0);
        info.seqlen = Dim(q, 1);
        info.kDim = Dim(q, 2);
        info.hvNum = Dim(v, 0);
        info.vDim = Dim(v, 2);
        TORCH_CHECK(HasShape(v, {info.hvNum, info.seqlen, info.vDim}) &&
                        HasShape(*g, {info.hvNum, info.seqlen, info.kDim}) &&
                        HasShape(*beta, {info.hvNum, info.seqlen}),
                    "chunk_kda_fwd: NTD expects v/g/beta as [HV,T,V], [HV,T,K], [HV,T].");
    } else if (layout == KdaFwdLayout::BSND) {
        info.batch = Dim(q, 0);
        info.seqlen = Dim(q, 1);
        info.hNum = Dim(q, 2);
        info.kDim = Dim(q, 3);
        info.hvNum = Dim(v, 2);
        info.vDim = Dim(v, 3);
        TORCH_CHECK(HasShape(v, {info.batch, info.seqlen, info.hvNum, info.vDim}) &&
                        HasShape(*g, {info.batch, info.seqlen, info.hvNum, info.kDim}) &&
                        HasShape(*beta, {info.batch, info.seqlen, info.hvNum}),
                    "chunk_kda_fwd: BSND expects v/g/beta as [B,T,HV,V], [B,T,HV,K], [B,T,HV].");
    } else {  // BNSD
        info.batch = Dim(q, 0);
        info.hNum = Dim(q, 1);
        info.seqlen = Dim(q, 2);
        info.kDim = Dim(q, 3);
        info.hvNum = Dim(v, 1);
        info.vDim = Dim(v, 3);
        TORCH_CHECK(HasShape(v, {info.batch, info.hvNum, info.seqlen, info.vDim}) &&
                        HasShape(*g, {info.batch, info.hvNum, info.seqlen, info.kDim}) &&
                        HasShape(*beta, {info.batch, info.hvNum, info.seqlen}),
                    "chunk_kda_fwd: BNSD expects v/g/beta as [B,HV,T,V], [B,HV,T,K], [B,HV,T].");
    }

    info.seqNum = cuSeqlens.has_value() ? cuSeqlens->numel() - 1 : info.batch;
    info.totalChunks = CountChunks(cuSeqlens, info.seqlen, chunkSize);
}

void ComputeTilingData(int64_t batch, int64_t seqlen, int64_t hNum, int64_t hvNum, int64_t kDim,
                       int64_t vDim, int64_t chunkSize, int64_t seqNum, int64_t totalChunks,
                       int64_t inputRank, double scale, double lowerBound, bool hasInitialState,
                       bool isVarLen, bool safeGate, bool sequenceMajor, bool useGateInKernel,
                       bool hasALog, bool hasDtBias, int64_t gateDataType,
                       bool storeFinalState, bool storeGk, bool storeW,
                       bool storeU, bool storeQG, bool storeKg, bool storeVNew, bool storeH,
                       uint32_t blockDim, const optiling::arch35::ChunkKdaFwdArch35Options &arch35Options,
                       int64_t betaDataType, ChunkKdaFwd::ChunkKdaFwdTilingData &td)
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
    td.finalStateStorageOffset =
        storeFinalState ? 0 : AllocateWorkspace(cursor, stateElements * sizeof(float));
    td.wStorageOffset = storeW ? 0 : AllocateWorkspace(cursor, kTensorBytes);
    td.uStorageOffset = storeU ? 0 : AllocateWorkspace(cursor, vTensorBytes);
    td.qgStorageOffset = storeQG ? 0 : AllocateWorkspace(cursor, kTensorBytes);
    td.kgStorageOffset = storeKg ? 0 : AllocateWorkspace(cursor, kTensorBytes);
    const uint64_t vNewStorageBytes =
        arch35Options.useDenseFwdH && !storeVNew
            ? static_cast<uint64_t>(batch) * hvNum * chunkSize * vDim * dataBytes
            : vTensorBytes;
    td.vNewStorageOffset = storeVNew ? 0 : AllocateWorkspace(cursor, vNewStorageBytes);
    const uint64_t hStorageBytes =
        arch35Options.useDenseFwdH && !storeH
            ? static_cast<uint64_t>(batch) * hvNum * kDim * vDim * dataBytes
            : hBytes;
    td.hStorageOffset = storeH ? 0 : AllocateWorkspace(cursor, hStorageBytes);
    td.qgScaledOffset = AllocateWorkspace(cursor, kTensorBytes);

    const uint64_t matrixBytes = tokenHeads * chunkSize * sizeof(float);
    td.prepareAqkFp32Offset = AllocateWorkspace(cursor, matrixBytes);
    td.prepareAkkFp32Offset = AllocateWorkspace(cursor, matrixBytes);
    td.prepareScratchOffset = AlignWorkspace(cursor);
    const uint64_t solveDepth = safeGate ? KDA_SOLVE_PIPELINE_DEPTH : 1;
    const uint64_t solveBytes =
        static_cast<uint64_t>(blockDim) * solveDepth * KDA_SOLVE_SCRATCH_SLOTS * chunkSize * chunkSize *
        sizeof(float);
    const uint64_t scoreBytes = static_cast<uint64_t>(blockDim) * KDA_SCORE_QUEUE_SLOTS *
                                KDA_SCORE_SCRATCH_PLANES * chunkSize * kDim * dataBytes;
    cursor = td.prepareScratchOffset + AlignWorkspace(solveBytes) + scoreBytes;

    td.postWuScratchOffset = AlignWorkspace(cursor);
    if (!arch35Options.fusePostWu && !arch35Options.fusePostWuIntoFwdH) {
        cursor = td.postWuScratchOffset + tokenHeads * kDim * sizeof(float);
    }

    td.fwdHWorkspaceBaseOffset = AlignWorkspace(cursor);
    uint64_t fwdHCursor = 0;
    td.vWorkspaceOffset = AllocateWorkspace(
        fwdHCursor, static_cast<uint64_t>(blockDim) * chunkSize * vDim * sizeof(float) *
                        KDA_GDN_PIPELINE_DEPTH);
    td.vUpdateWorkspaceOffset = AllocateWorkspace(
        fwdHCursor, static_cast<uint64_t>(blockDim) * chunkSize * vDim * sizeof(float) *
                        KDA_GDN_PIPELINE_DEPTH);
    td.kDecayWorkspaceOffset = AllocateWorkspace(
        fwdHCursor, static_cast<uint64_t>(blockDim) * chunkSize * kDim * sizeof(float) *
                        KDA_GDN_PIPELINE_DEPTH);
    td.hWorkspaceOffset = AllocateWorkspace(
        fwdHCursor, static_cast<uint64_t>(blockDim) * kDim * vDim * sizeof(float) *
                        KDA_GDN_PIPELINE_DEPTH);
    const uint64_t tokenBatch = isVarLen ? static_cast<uint64_t>(seqNum) : 1;
    td.numSeqWorkspaceOffset = AllocateWorkspace(fwdHCursor, (tokenBatch + 1) * sizeof(int64_t));
    td.numChunksWorkspaceOffset = AllocateWorkspace(fwdHCursor, (tokenBatch + 1) * sizeof(int64_t));
    cursor = td.fwdHWorkspaceBaseOffset + AlignWorkspace(fwdHCursor);

    td.outputScratchOffset = AllocateWorkspace(cursor, 2 * tokenHeads * vDim * sizeof(float));
}

std::tuple<at::Tensor, c10::optional<at::Tensor>, c10::optional<at::Tensor>, at::Tensor, at::Tensor,
           c10::optional<at::Tensor>, c10::optional<at::Tensor>, c10::optional<at::Tensor>,
           c10::optional<at::Tensor>, c10::optional<at::Tensor>, c10::optional<at::Tensor>>
chunk_kda_fwd(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, const at::Tensor &g,
              const at::Tensor &beta, const c10::optional<at::Tensor> &aLog,
              const c10::optional<at::Tensor> &dtBias, const c10::optional<at::Tensor> &initialState,
              const c10::optional<at::Tensor> &cuSeqlens,
              const c10::optional<at::Tensor> &chunkIndices, const std::string &layout, double scale,
              int64_t chunkSize, bool safeGate, double lowerBound, bool useGateInKernel,
              bool stateVFirst, bool outputFinalState, bool outputGk, bool outputW, bool outputU,
              bool outputQG, bool outputKg, bool outputVNew, bool outputH)
{
    TORCH_CHECK(q.defined() && k.defined() && v.defined() && g.defined() && beta.defined(),
                "chunk_kda_fwd: q, k, v, g and beta must be defined");

    TORCH_CHECK(chunkSize == 64 || chunkSize == 128, "chunk_kda_fwd: chunkSize must be 64 or 128");

    KdaFwdLayout parsedLayout;
    TORCH_CHECK(ParseLayout(layout, parsedLayout), "chunk_kda_fwd: layout must be BSND, BNSD, TND or NTD");

    const bool isRank3 = parsedLayout == KdaFwdLayout::TND || parsedLayout == KdaFwdLayout::NTD;

    // dtype checks: q/k/v must be bf16 (direct-launch restriction), g/beta fp32 or bf16.
    TORCH_CHECK(q.scalar_type() == at::kBFloat16 && k.scalar_type() == at::kBFloat16 &&
                    v.scalar_type() == at::kBFloat16,
                "chunk_kda_fwd: q, k and v must be bfloat16 (direct-launch currently supports bf16)");
    const at::ScalarType gateType = g.scalar_type();
    TORCH_CHECK(gateType == at::kFloat || gateType == at::kBFloat16,
                "chunk_kda_fwd: g must be float32 or bfloat16");
    const at::ScalarType betaType = beta.scalar_type();
    TORCH_CHECK(betaType == at::kFloat || betaType == at::kBFloat16,
                "chunk_kda_fwd: beta must be float32 or bfloat16");
    const bool gIsFp32 = gateType == at::kFloat;
    const int64_t gateDataType = gIsFp32 ? 2 : (gateType == at::kBFloat16 ? 1 : 0);
    const int64_t betaDataType = betaType == at::kFloat ? 0 : 1;

    if (aLog.has_value() && aLog->defined()) {
        TORCH_CHECK(aLog->scalar_type() == at::kFloat, "chunk_kda_fwd: a_log must be float32");
    }
    if (dtBias.has_value() && dtBias->defined()) {
        TORCH_CHECK(dtBias->scalar_type() == at::kFloat, "chunk_kda_fwd: dt_bias must be float32");
    }
    if (initialState.has_value() && initialState->defined()) {
        TORCH_CHECK(initialState->scalar_type() == at::kFloat,
                    "chunk_kda_fwd: initial_state must be float32");
    }
    if (useGateInKernel) {
        TORCH_CHECK(aLog.has_value() && aLog->defined(),
                    "chunk_kda_fwd: a_log is required when use_gate_in_kernel is true");
    }

    KdaShapeInfo info;
    ResolveShapeInfo(q, v, beta, g, parsedLayout, chunkSize, cuSeqlens, info);

    TORCH_CHECK(info.hNum > 0 && info.hvNum >= info.hNum && info.hvNum % info.hNum == 0,
                "chunk_kda_fwd: H and HV must be positive, HV must be a multiple of H");
    TORCH_CHECK(info.hNum <= MAX_KDA_HEAD_NUM && info.hvNum <= MAX_KDA_HEAD_NUM,
                "chunk_kda_fwd: H and HV must be <= 128");
    TORCH_CHECK(info.kDim >= 16 && info.kDim <= MAX_KDA_K_DIM && info.kDim % 16 == 0 &&
                    info.vDim >= 16 && info.vDim <= 256 && info.vDim % 16 == 0,
                "chunk_kda_fwd: K/V must be multiples of 16, K <= 256 and V <= 256");
    TORCH_CHECK(SameShape(q, k) && q.dim() == k.dim(),
                "chunk_kda_fwd: q and k must have identical shape");
    TORCH_CHECK(info.seqlen > 0 && info.totalChunks > 0, "chunk_kda_fwd: invalid sequence length");
    if (cuSeqlens.has_value() && cuSeqlens->defined()) {
        TORCH_CHECK(cuSeqlens->scalar_type() == at::kLong, "chunk_kda_fwd: cu_seqlens must be int64");
        TORCH_CHECK(cuSeqlens->numel() >= 2, "chunk_kda_fwd: cu_seqlens must contain at least [0, T]");
        TORCH_CHECK(!isRank3 || info.batch == 1, "chunk_kda_fwd: rank4 varlen requires B=1");
        TORCH_CHECK(info.seqNum <= MAX_KDA_VARLEN_SEQUENCES,
                    "chunk_kda_fwd: varlen supports at most 1024 sequences");
        const at::Tensor cuCpu = cuSeqlens->cpu().contiguous();
        const int64_t *cuPtr = reinterpret_cast<const int64_t *>(cuCpu.data_ptr());
        TORCH_CHECK(cuPtr[0] == 0, "chunk_kda_fwd: cu_seqlens[0] must be 0");
        TORCH_CHECK(cuPtr[cuCpu.numel() - 1] == info.seqlen,
                    "chunk_kda_fwd: cu_seqlens last element must equal the sequence length");
        for (int64_t idx = 0; idx + 1 < cuCpu.numel(); ++idx) {
            TORCH_CHECK(cuPtr[idx] <= cuPtr[idx + 1],
                        "chunk_kda_fwd: cu_seqlens must be nondecreasing");
        }
    }
    if (chunkIndices.has_value() && chunkIndices->defined()) {
        TORCH_CHECK(cuSeqlens.has_value() && cuSeqlens->defined(),
                    "chunk_kda_fwd: chunk_indices requires cu_seqlens");
        TORCH_CHECK(chunkIndices->scalar_type() == at::kLong, "chunk_kda_fwd: chunk_indices must be int64");
        TORCH_CHECK(chunkIndices->numel() == info.totalChunks * 2,
                    "chunk_kda_fwd: chunk_indices must contain one (seq_id, chunk_id) pair per chunk");
    }
    if (useGateInKernel && safeGate) {
        TORCH_CHECK(lowerBound >= -5.0 && lowerBound < 0.0,
                    "chunk_kda_fwd: lower_bound must be in [-5, 0) when safe_gate is true");
    }
    if (initialState.has_value() && initialState->defined()) {
        const bool valid = stateVFirst
                               ? HasShape(*initialState, {info.seqNum, info.hvNum, info.vDim, info.kDim})
                               : HasShape(*initialState, {info.seqNum, info.hvNum, info.kDim, info.vDim});
        TORCH_CHECK(valid,
                    "chunk_kda_fwd: initial_state must be [N,HV,K,V] (state_v_first=false) or "
                    "[N,HV,V,K] (state_v_first=true)");
    }

    const auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    const uint32_t physicalCoreNum = std::max<uint32_t>(ascendcPlatform->GetCoreNumAic(), 1);
    const bool isAscend950 =
        ascendcPlatform->GetSocVersion() == platform_ascendc::SocVersion::ASCEND950;
    const bool isVarLen = cuSeqlens.has_value() && cuSeqlens->defined();
    const uint64_t fwdHTaskCount = static_cast<uint64_t>(isVarLen ? info.seqNum : info.batch) *
                                   info.hvNum;
    const uint32_t blockDim =
        isAscend950 ? static_cast<uint32_t>(
                          std::min<uint64_t>(physicalCoreNum, std::max<uint64_t>(fwdHTaskCount, 1))) :
                      physicalCoreNum;

    const bool hasALog = aLog.has_value() && aLog->defined();
    const bool hasDtBias = dtBias.has_value() && dtBias->defined();
    const bool hasInitialState = initialState.has_value() && initialState->defined();

    const auto arch35Options = optiling::arch35::ConfigureChunkKdaFwdArch35(
        isAscend950, true /* qIsBf16 */, gIsFp32, hasALog, useGateInKernel, safeGate, isVarLen,
        info.seqlen, info.hvNum, chunkSize, info.kDim, info.vDim, outputQG, outputVNew, outputH);

    const bool storeFinalState = outputFinalState;
    const bool storeGk = outputGk;
    const bool storeW = outputW;
    const bool storeU = outputU;
    const bool storeQG = outputQG;
    const bool storeKg = outputKg;
    const bool storeVNew = outputVNew;
    const bool storeH = outputH;

    ChunkKdaFwd::ChunkKdaFwdTilingData tilingData;
    ComputeTilingData(info.batch, info.seqlen, info.hNum, info.hvNum, info.kDim, info.vDim, chunkSize,
                      info.seqNum, info.totalChunks, info.isRank3 ? 3 : 4, scale, lowerBound,
                      hasInitialState, isVarLen, safeGate, parsedLayout == KdaFwdLayout::BSND,
                      useGateInKernel, hasALog, hasDtBias, gateDataType, storeFinalState,
                      storeGk, storeW, storeU, storeQG, storeKg, storeVNew, storeH, blockDim,
                      arch35Options, betaDataType, tilingData);

    // ---- layout conversion to the kernel's input convention ----
    at::Tensor qHead, kHead, vHead, gHead, betaHead;
    if (parsedLayout == KdaFwdLayout::BSND) {
        qHead = q.contiguous();
        kHead = k.contiguous();
        vHead = v.contiguous();
        gHead = g.contiguous();
        betaHead = beta.transpose(1, 2).contiguous();  // (B, S, HV) -> (B, HV, S)
    } else if (parsedLayout == KdaFwdLayout::BNSD) {
        qHead = q.contiguous();
        kHead = k.contiguous();
        vHead = v.contiguous();
        gHead = g.contiguous();
        betaHead = beta.contiguous();
    } else if (parsedLayout == KdaFwdLayout::TND) {
        qHead = q.transpose(0, 1).contiguous().reshape({1, info.hNum, info.seqlen, info.kDim});
        kHead = k.transpose(0, 1).contiguous().reshape({1, info.hNum, info.seqlen, info.kDim});
        vHead = v.transpose(0, 1).contiguous().reshape({1, info.hvNum, info.seqlen, info.vDim});
        gHead = g.transpose(0, 1).contiguous().reshape({1, info.hvNum, info.seqlen, info.kDim});
        betaHead = beta.transpose(0, 1).contiguous().reshape({1, info.hvNum, info.seqlen});
    } else {  // NTD
        qHead = q.reshape({1, info.hNum, info.seqlen, info.kDim});
        kHead = k.reshape({1, info.hNum, info.seqlen, info.kDim});
        vHead = v.reshape({1, info.hvNum, info.seqlen, info.vDim});
        gHead = g.reshape({1, info.hvNum, info.seqlen, info.kDim});
        betaHead = beta.reshape({1, info.hvNum, info.seqlen});
    }

    at::Tensor initStateContig;
    void *initStatePtr = nullptr;
    if (hasInitialState) {
        at::Tensor init = *initialState;
        if (stateVFirst) {
            init = init.transpose(-1, -2).contiguous();  // (N, HV, V, K) -> (N, HV, K, V)
        }
        initStateContig = init.contiguous();
        initStatePtr = initStateContig.data_ptr();
    }

    void *aLogPtr = nullptr;
    if (hasALog) {
        aLogPtr = aLog->contiguous().data_ptr();
    }
    void *dtBiasPtr = nullptr;
    if (hasDtBias) {
        dtBiasPtr = dtBias->contiguous().data_ptr();
    }

    at::Tensor cuSeqlensContig, chunkIndicesContig;
    void *cuSeqlensPtr = nullptr;
    void *chunkIndicesPtr = nullptr;
    if (cuSeqlens.has_value() && cuSeqlens->defined()) {
        cuSeqlensContig = cuSeqlens->contiguous();
        cuSeqlensPtr = cuSeqlensContig.data_ptr();
    }
    if (chunkIndices.has_value() && chunkIndices->defined()) {
        chunkIndicesContig = chunkIndices->contiguous();
        chunkIndicesPtr = chunkIndicesContig.data_ptr();
    }

    // ---- outputs (kernel internal layouts, then reshaped/transposed for the user) ----
    const int64_t b = info.batch;
    const int64_t s = info.seqlen;
    const int64_t hv = info.hvNum;
    const int64_t kd = info.kDim;
    const int64_t vd = info.vDim;
    const int64_t c = chunkSize;

    at::Tensor attnOut = at::empty({b, s, hv, vd}, q.options());
    at::Tensor aqkOut = at::empty({b, hv, s, c}, q.options());
    at::Tensor akkOut = at::empty({b, hv, s, c}, q.options());

    at::Tensor finalStateOut;
    void *finalStatePtr = nullptr;
    if (storeFinalState) {
        finalStateOut = at::empty({info.seqNum, hv, kd, vd},
                                  at::TensorOptions().dtype(at::kFloat).device(q.device()));
        finalStatePtr = finalStateOut.data_ptr();
    }

    at::Tensor gkOut;
    void *gkPtr = nullptr;
    if (storeGk) {
        gkOut = at::empty({b, hv, s, kd},
                          at::TensorOptions().dtype(at::kFloat).device(q.device()));
        gkPtr = gkOut.data_ptr();
    }

    at::Tensor wOut;
    void *wPtr = nullptr;
    if (storeW) {
        wOut = at::empty({b, hv, s, kd}, q.options());
        wPtr = wOut.data_ptr();
    }

    at::Tensor uOut;
    void *uPtr = nullptr;
    if (storeU) {
        uOut = at::empty({b, hv, s, vd}, q.options());
        uPtr = uOut.data_ptr();
    }

    at::Tensor qgOut;
    void *qgPtr = nullptr;
    if (storeQG) {
        qgOut = at::empty({b, hv, s, kd}, q.options());
        qgPtr = qgOut.data_ptr();
    }

    at::Tensor kgOut;
    void *kgPtr = nullptr;
    if (storeKg) {
        kgOut = at::empty({b, hv, s, kd}, q.options());
        kgPtr = kgOut.data_ptr();
    }

    at::Tensor vNewOut;
    void *vNewPtr = nullptr;
    if (storeVNew) {
        vNewOut = at::empty({b, hv, s, vd}, q.options());
        vNewPtr = vNewOut.data_ptr();
    }

    at::Tensor hOutInternal;
    void *hPtr = nullptr;
    if (storeH) {
        // kernel-internal h layout is (B, HV, NC, K, V); reshaped below for the user.
        hOutInternal = at::empty({b, hv, info.totalChunks, kd, vd}, q.options());
        hPtr = hOutInternal.data_ptr();
    }

    // ---- workspace + tiling ----
    int32_t tilingSize = (static_cast<int32_t>(sizeof(ChunkKdaFwd::ChunkKdaFwdTilingData)) +
                          static_cast<int32_t>(PADDING_BYTE) - 1) /
                         PADDING_BYTE * PADDING_BYTE;

    auto cpuTiling = at::empty({tilingSize}, at::kByte);
    std::memcpy(cpuTiling.data_ptr(), &tilingData, sizeof(ChunkKdaFwd::ChunkKdaFwdTilingData));
    at::Tensor tilingTensor = TorchNpuHelper::CopyTensorHostToDevice(cpuTiling);

    const uint64_t tokenHeads = static_cast<uint64_t>(b) * hv * s;
    const uint64_t totalUserWorkspace = tilingData.outputScratchOffset + 2 * tokenHeads * vd * sizeof(float);
    const uint64_t sysytemWorkspaceBytes = static_cast<int64_t>(ascendcPlatform->GetLibApiWorkSpaceSize());
    const int64_t totalWorkspaceBytes =
        sysytemWorkspaceBytes + static_cast<int64_t>(AlignWorkspace(totalUserWorkspace));
    auto workspaceTensor =
        at::empty({totalWorkspaceBytes}, at::TensorOptions().dtype(at::kByte).device(q.device()));

    EXEC_KERNEL_CMD(chunk_kda_fwd, blockDim, qHead, kHead, vHead, gHead, betaHead, aLogPtr, dtBiasPtr,
                    initStatePtr, cuSeqlensPtr, chunkIndicesPtr, attnOut, finalStatePtr, gkPtr, aqkOut,
                    akkOut, wPtr, uPtr, qgPtr, kgPtr, vNewPtr, hPtr, workspaceTensor, tilingTensor);

    // ---- post-process outputs to the aclnn-documented user layouts ----
    at::Tensor attn = attnOut;
    at::Tensor aqk = aqkOut;
    at::Tensor akk = akkOut;
    if (isRank3) {
        attn = attnOut.reshape({s, hv, vd});
        aqk = aqkOut.reshape({hv, s, c});
        akk = akkOut.reshape({hv, s, c});
    }

    c10::optional<at::Tensor> finalState = c10::nullopt;
    if (storeFinalState) {
        at::Tensor fs = finalStateOut;
        if (stateVFirst) {
            fs = fs.transpose(-1, -2).contiguous();  // (N, HV, K, V) -> (N, HV, V, K)
        }
        finalState = fs;
    }

    c10::optional<at::Tensor> gk = c10::nullopt;
    if (storeGk) {
        gk = isRank3 ? gkOut.reshape({hv, s, kd}) : gkOut;
    }

    c10::optional<at::Tensor> w = c10::nullopt;
    if (storeW) {
        w = isRank3 ? wOut.reshape({hv, s, kd}) : wOut;
    }

    c10::optional<at::Tensor> u = c10::nullopt;
    if (storeU) {
        u = isRank3 ? uOut.reshape({hv, s, vd}) : uOut;
    }

    c10::optional<at::Tensor> qg = c10::nullopt;
    if (storeQG) {
        qg = isRank3 ? qgOut.reshape({hv, s, kd}) : qgOut;
    }

    c10::optional<at::Tensor> kg = c10::nullopt;
    if (storeKg) {
        kg = isRank3 ? kgOut.reshape({hv, s, kd}) : kgOut;
    }

    c10::optional<at::Tensor> vNew = c10::nullopt;
    if (storeVNew) {
        vNew = isRank3 ? vNewOut.reshape({hv, s, vd}) : vNewOut;
    }

    c10::optional<at::Tensor> h = c10::nullopt;
    if (storeH) {
        // (B, HV, NC, K, V) -> (B, NC, HV, K, V) [or state_v_first: (B, NC, HV, V, K)]
        at::Tensor hView = hOutInternal.permute({0, 2, 1, 3, 4});
        if (stateVFirst) {
            hView = hOutInternal.permute({0, 2, 1, 4, 3});
        }
        at::Tensor hOut = hView.contiguous();
        if (isRank3) {
            hOut = hOut.reshape({info.totalChunks, hv, stateVFirst ? vd : kd, stateVFirst ? kd : vd});
        }
        h = hOut;
    }

    return std::make_tuple(attn, finalState, gk, aqk, akk, w, u, qg, kg, vNew, h);
}

}  // namespace npu_kernel
}  // namespace sglang
