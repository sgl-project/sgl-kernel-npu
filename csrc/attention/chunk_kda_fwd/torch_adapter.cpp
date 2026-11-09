/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_runtime.h"
#include "aclnn_sgl_chunk_kda_fwd.h"
#include <array>
#include <limits>
#include <torch/library.h>
#include <tuple>
#include <vector>

namespace {
constexpr int64_t MAX_KDA_K_DIM = 256;
constexpr int64_t MAX_KDA_HEAD_NUM = 128;
constexpr int64_t MAX_KDA_VARLEN_SEQUENCES = 1024;
enum class KdaFwdLayout {
    BSND,
    BNSD,
    TND,
    NTD,
};

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
    TORCH_CHECK(q.dim() == tensorRank && v.dim() == tensorRank && g->dim() == tensorRank && beta->dim() == betaRank,
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

std::tuple<at::Tensor, c10::optional<at::Tensor>, c10::optional<at::Tensor>, at::Tensor, at::Tensor,
           c10::optional<at::Tensor>, c10::optional<at::Tensor>, c10::optional<at::Tensor>, c10::optional<at::Tensor>,
           c10::optional<at::Tensor>, c10::optional<at::Tensor>>
chunk_kda_fwd(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, const at::Tensor &g,
              const at::Tensor &beta, const c10::optional<at::Tensor> &aLog, const c10::optional<at::Tensor> &dtBias,
              const c10::optional<at::Tensor> &initialState, const c10::optional<at::Tensor> &cuSeqlens,
              const c10::optional<at::Tensor> &chunkIndices, const std::string &layout, double scale, int64_t chunkSize,
              bool safeGate, double lowerBound, bool useGateInKernel, bool stateVFirst, bool outputFinalState,
              bool outputGk, bool outputW, bool outputU, bool outputQG, bool outputKg, bool outputVNew, bool outputH)
{
    TORCH_CHECK(q.defined() && k.defined() && v.defined() && g.defined() && beta.defined(),
                "chunk_kda_fwd: q, k, v, g and beta must be defined");

    TORCH_CHECK(chunkSize == 64 || chunkSize == 128, "chunk_kda_fwd: chunkSize must be 64 or 128");

    KdaFwdLayout parsedLayout;
    TORCH_CHECK(ParseLayout(layout, parsedLayout), "chunk_kda_fwd: layout must be BSND, BNSD, TND or NTD");

    const bool isRank3 = parsedLayout == KdaFwdLayout::TND || parsedLayout == KdaFwdLayout::NTD;

    // dtype checks: q/k/v must be bf16 (direct-launch restriction), g/beta fp32 or bf16.
    TORCH_CHECK(
        q.scalar_type() == at::kBFloat16 && k.scalar_type() == at::kBFloat16 && v.scalar_type() == at::kBFloat16,
        "chunk_kda_fwd: q, k and v must be bfloat16 (direct-launch currently supports bf16)");
    const at::ScalarType gateType = g.scalar_type();
    TORCH_CHECK(gateType == at::kFloat || gateType == at::kBFloat16, "chunk_kda_fwd: g must be float32 or bfloat16");
    const at::ScalarType betaType = beta.scalar_type();
    TORCH_CHECK(betaType == at::kFloat || betaType == at::kBFloat16, "chunk_kda_fwd: beta must be float32 or bfloat16");
    const bool gIsFp32 = gateType == at::kFloat;

    if (aLog.has_value() && aLog->defined()) {
        TORCH_CHECK(aLog->scalar_type() == at::kFloat, "chunk_kda_fwd: a_log must be float32");
    }
    if (dtBias.has_value() && dtBias->defined()) {
        TORCH_CHECK(dtBias->scalar_type() == at::kFloat, "chunk_kda_fwd: dt_bias must be float32");
    }
    if (initialState.has_value() && initialState->defined()) {
        TORCH_CHECK(initialState->scalar_type() == at::kFloat, "chunk_kda_fwd: initial_state must be float32");
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
    TORCH_CHECK(info.kDim >= 16 && info.kDim <= MAX_KDA_K_DIM && info.kDim % 16 == 0 && info.vDim >= 16 &&
                    info.vDim <= 256 && info.vDim % 16 == 0,
                "chunk_kda_fwd: K/V must be multiples of 16, K <= 256 and V <= 256");
    TORCH_CHECK(SameShape(q, k) && q.dim() == k.dim(), "chunk_kda_fwd: q and k must have identical shape");
    TORCH_CHECK(info.seqlen > 0 && info.totalChunks > 0, "chunk_kda_fwd: invalid sequence length");
    if (cuSeqlens.has_value() && cuSeqlens->defined()) {
        TORCH_CHECK(cuSeqlens->scalar_type() == at::kLong, "chunk_kda_fwd: cu_seqlens must be int64");
        TORCH_CHECK(cuSeqlens->numel() >= 2, "chunk_kda_fwd: cu_seqlens must contain at least [0, T]");
        TORCH_CHECK(!isRank3 || info.batch == 1, "chunk_kda_fwd: rank4 varlen requires B=1");
        TORCH_CHECK(info.seqNum <= MAX_KDA_VARLEN_SEQUENCES, "chunk_kda_fwd: varlen supports at most 1024 sequences");
        const at::Tensor cuCpu = cuSeqlens->cpu().contiguous();
        const int64_t *cuPtr = reinterpret_cast<const int64_t *>(cuCpu.data_ptr());
        TORCH_CHECK(cuPtr[0] == 0, "chunk_kda_fwd: cu_seqlens[0] must be 0");
        TORCH_CHECK(cuPtr[cuCpu.numel() - 1] == info.seqlen,
                    "chunk_kda_fwd: cu_seqlens last element must equal the sequence length");
        for (int64_t idx = 0; idx + 1 < cuCpu.numel(); ++idx) {
            TORCH_CHECK(cuPtr[idx] <= cuPtr[idx + 1], "chunk_kda_fwd: cu_seqlens must be nondecreasing");
        }
    }
    if (chunkIndices.has_value() && chunkIndices->defined()) {
        TORCH_CHECK(cuSeqlens.has_value() && cuSeqlens->defined(), "chunk_kda_fwd: chunk_indices requires cu_seqlens");
        TORCH_CHECK(chunkIndices->scalar_type() == at::kLong, "chunk_kda_fwd: chunk_indices must be int64");
        TORCH_CHECK(chunkIndices->numel() == info.totalChunks * 2,
                    "chunk_kda_fwd: chunk_indices must contain one (seq_id, chunk_id) pair per chunk");
    }
    if (useGateInKernel && safeGate) {
        TORCH_CHECK(lowerBound >= -5.0 && lowerBound < 0.0,
                    "chunk_kda_fwd: lower_bound must be in [-5, 0) when safe_gate is true");
    }
    if (initialState.has_value() && initialState->defined()) {
        const bool valid = stateVFirst ? HasShape(*initialState, {info.seqNum, info.hvNum, info.vDim, info.kDim})
                                       : HasShape(*initialState, {info.seqNum, info.hvNum, info.kDim, info.vDim});
        TORCH_CHECK(valid,
                    "chunk_kda_fwd: initial_state must be [N,HV,K,V] (state_v_first=false) or "
                    "[N,HV,V,K] (state_v_first=true)");
    }

    const bool hasALog = aLog.has_value() && aLog->defined();
    const bool hasDtBias = dtBias.has_value() && dtBias->defined();
    const bool hasInitialState = initialState.has_value() && initialState->defined();

    const bool storeFinalState = outputFinalState;
    const bool storeGk = outputGk;
    const bool storeW = outputW;
    const bool storeU = outputU;
    const bool storeQG = outputQG;
    const bool storeKg = outputKg;
    const bool storeVNew = outputVNew;
    const bool storeH = outputH;

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
    if (hasInitialState) {
        at::Tensor init = *initialState;
        if (stateVFirst) {
            init = init.transpose(-1, -2).contiguous();  // (N, HV, V, K) -> (N, HV, K, V)
        }
        initStateContig = init.contiguous();
    }

    at::Tensor aLogTensor, dtBiasTensor;
    if (hasALog) {
        aLogTensor = aLog->contiguous();
    }
    if (hasDtBias) {
        dtBiasTensor = dtBias->contiguous();
    }

    at::Tensor cuSeqlensContig, chunkIndicesContig;
    if (cuSeqlens.has_value() && cuSeqlens->defined()) {
        cuSeqlensContig = cuSeqlens->contiguous();
    }
    if (chunkIndices.has_value() && chunkIndices->defined()) {
        chunkIndicesContig = chunkIndices->contiguous();
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
    if (storeFinalState) {
        finalStateOut = at::empty({info.seqNum, hv, kd, vd}, at::TensorOptions().dtype(at::kFloat).device(q.device()));
    }

    at::Tensor gkOut;
    if (storeGk) {
        gkOut = at::empty({b, hv, s, kd}, at::TensorOptions().dtype(at::kFloat).device(q.device()));
    }

    at::Tensor wOut;
    if (storeW) {
        wOut = at::empty({b, hv, s, kd}, q.options());
    }

    at::Tensor uOut;
    if (storeU) {
        uOut = at::empty({b, hv, s, vd}, q.options());
    }

    at::Tensor qgOut;
    if (storeQG) {
        qgOut = at::empty({b, hv, s, kd}, q.options());
    }

    at::Tensor kgOut;
    if (storeKg) {
        kgOut = at::empty({b, hv, s, kd}, q.options());
    }

    at::Tensor vNewOut;
    if (storeVNew) {
        vNewOut = at::empty({b, hv, s, vd}, q.options());
    }

    at::Tensor hOutInternal;
    if (storeH) {
        // kernel-internal h layout is (B, HV, NC, K, V); reshaped below for the user.
        hOutInternal = at::empty({b, hv, info.totalChunks, kd, vd}, q.options());
    }

    static auto query = reinterpret_cast<decltype(&aclnnSglChunkKdaFwdGetWorkspaceSize)>(
        sgl_cann::op_api_symbol("aclnnSglChunkKdaFwdGetWorkspaceSize"));
    static auto execute =
        reinterpret_cast<decltype(&aclnnSglChunkKdaFwd)>(sgl_cann::op_api_symbol("aclnnSglChunkKdaFwd"));
    auto desc0 = qHead.defined() ? sgl_cann::tensor_descriptor(qHead, ACL_BF16) : std::shared_ptr<aclTensor>{};
    auto desc1 = kHead.defined() ? sgl_cann::tensor_descriptor(kHead, ACL_BF16) : std::shared_ptr<aclTensor>{};
    auto desc2 = vHead.defined() ? sgl_cann::tensor_descriptor(vHead, ACL_BF16) : std::shared_ptr<aclTensor>{};
    auto desc3 = gHead.defined() ? sgl_cann::tensor_descriptor(gHead, (gIsFp32 ? ACL_FLOAT : ACL_BF16))
                                 : std::shared_ptr<aclTensor>{};
    auto desc4 = betaHead.defined()
                     ? sgl_cann::tensor_descriptor(betaHead, (betaType == at::kFloat ? ACL_FLOAT : ACL_BF16))
                     : std::shared_ptr<aclTensor>{};
    auto desc5 =
        aLogTensor.defined() ? sgl_cann::tensor_descriptor(aLogTensor, ACL_FLOAT) : std::shared_ptr<aclTensor>{};
    auto desc6 =
        dtBiasTensor.defined() ? sgl_cann::tensor_descriptor(dtBiasTensor, ACL_FLOAT) : std::shared_ptr<aclTensor>{};
    auto desc7 = initStateContig.defined() ? sgl_cann::tensor_descriptor(initStateContig, ACL_FLOAT)
                                           : std::shared_ptr<aclTensor>{};
    auto desc8 = cuSeqlensContig.defined() ? sgl_cann::tensor_descriptor(cuSeqlensContig, ACL_INT64)
                                           : std::shared_ptr<aclTensor>{};
    auto desc9 = chunkIndicesContig.defined() ? sgl_cann::tensor_descriptor(chunkIndicesContig, ACL_INT64)
                                              : std::shared_ptr<aclTensor>{};
    auto desc10 = attnOut.defined() ? sgl_cann::tensor_descriptor(attnOut, ACL_BF16) : std::shared_ptr<aclTensor>{};
    auto desc11 =
        finalStateOut.defined() ? sgl_cann::tensor_descriptor(finalStateOut, ACL_FLOAT) : std::shared_ptr<aclTensor>{};
    auto desc12 = gkOut.defined() ? sgl_cann::tensor_descriptor(gkOut, ACL_FLOAT) : std::shared_ptr<aclTensor>{};
    auto desc13 = aqkOut.defined() ? sgl_cann::tensor_descriptor(aqkOut, ACL_BF16) : std::shared_ptr<aclTensor>{};
    auto desc14 = akkOut.defined() ? sgl_cann::tensor_descriptor(akkOut, ACL_BF16) : std::shared_ptr<aclTensor>{};
    auto desc15 = wOut.defined() ? sgl_cann::tensor_descriptor(wOut, ACL_BF16) : std::shared_ptr<aclTensor>{};
    auto desc16 = uOut.defined() ? sgl_cann::tensor_descriptor(uOut, ACL_BF16) : std::shared_ptr<aclTensor>{};
    auto desc17 = qgOut.defined() ? sgl_cann::tensor_descriptor(qgOut, ACL_BF16) : std::shared_ptr<aclTensor>{};
    auto desc18 = kgOut.defined() ? sgl_cann::tensor_descriptor(kgOut, ACL_BF16) : std::shared_ptr<aclTensor>{};
    auto desc19 = vNewOut.defined() ? sgl_cann::tensor_descriptor(vNewOut, ACL_BF16) : std::shared_ptr<aclTensor>{};
    auto desc20 =
        hOutInternal.defined() ? sgl_cann::tensor_descriptor(hOutInternal, ACL_BF16) : std::shared_ptr<aclTensor>{};
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    TORCH_CHECK(query(desc0.get(), desc1.get(), desc2.get(), desc3.get(), desc4.get(), desc5.get(), desc6.get(),
                      desc7.get(), desc8.get(), desc9.get(), info.batch, info.seqlen, info.hNum, info.hvNum, info.kDim,
                      info.vDim, chunkSize, info.seqNum, info.totalChunks, info.isRank3 ? 3 : 4, scale, lowerBound,
                      parsedLayout == KdaFwdLayout::BSND, safeGate, useGateInKernel, storeFinalState, storeGk, storeW,
                      storeU, storeQG, storeKg, storeVNew, storeH, desc10.get(), desc11.get(), desc12.get(),
                      desc13.get(), desc14.get(), desc15.get(), desc16.get(), desc17.get(), desc18.get(), desc19.get(),
                      desc20.get(), &workspaceSize, &executor) == ACL_SUCCESS,
                "aclnnSglChunkKdaFwdGetWorkspaceSize failed: ", aclGetRecentErrMsg());
    TORCH_CHECK(workspaceSize <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                "ACLNN workspace size exceeds the Tensor allocation limit");
    auto workspace = at::empty({static_cast<int64_t>(workspaceSize)}, q.options().dtype(at::kByte));
    auto stream = c10_npu::getCurrentNPUStream();
    std::array<at::Tensor, 21> tensors{qHead,       kHead,         vHead,           gHead,           betaHead,
                                       aLogTensor,  dtBiasTensor,  initStateContig, cuSeqlensContig, chunkIndicesContig,
                                       attnOut,     finalStateOut, gkOut,           aqkOut,          akkOut,
                                       wOut,        uOut,          qgOut,           kgOut,           vNewOut,
                                       hOutInternal};
    for (const auto &tensor : tensors) {
        if (tensor.defined()) tensor.record_stream(stream);
    }
    workspace.record_stream(stream);
    std::array<std::shared_ptr<aclTensor>, 21> descriptors{desc0,  desc1,  desc2,  desc3,  desc4,  desc5,  desc6,
                                                           desc7,  desc8,  desc9,  desc10, desc11, desc12, desc13,
                                                           desc14, desc15, desc16, desc17, desc18, desc19, desc20};
    auto aclStream = stream.stream(false);
    auto launch = [tensors, descriptors, workspace, workspaceSize, executor, aclStream]() -> int {
        TORCH_CHECK(
            execute(workspaceSize ? workspace.data_ptr() : nullptr, workspaceSize, executor, aclStream) == ACL_SUCCESS,
            "aclnnSglChunkKdaFwd failed: ", aclGetRecentErrMsg());
        return ACL_SUCCESS;
    };
    at_npu::native::OpCommand::RunOpApi("aclnnSglChunkKdaFwd", launch);

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

}  // namespace

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def(
        "chunk_kda_fwd(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, "
        "Tensor? a_log=None, Tensor? dt_bias=None, Tensor? initial_state=None, "
        "Tensor? cu_seqlens=None, Tensor? chunk_indices=None, "
        "str layout='BSND', float scale=1.0, int chunk_size=64, "
        "bool safe_gate=False, float lower_bound=-5.0, bool use_gate_in_kernel=False, "
        "bool state_v_first=False, bool output_final_state=True, bool output_gk=False, "
        "bool output_w=False, bool output_u=False, bool output_qg=False, "
        "bool output_kg=False, bool output_v_new=False, bool output_h=False) "
        "-> (Tensor, Tensor?, Tensor?, Tensor, Tensor, Tensor?, Tensor?, Tensor?, Tensor?, Tensor?, Tensor?)");
}
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("chunk_kda_fwd", TORCH_FN(chunk_kda_fwd));
}
