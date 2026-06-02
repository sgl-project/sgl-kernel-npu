/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <iostream>

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
#include "aclrtlaunch_recurrent_gated_delta_rule.h"

namespace sglang {
namespace npu_kernel {

HOST_API at::Tensor recurrent_gated_delta_rule(at::Tensor &mix_qkv, at::Tensor &recurrent_state, at::Tensor &beta,
                                               double scale, at::Tensor &actual_seq_lengths,
                                               at::Tensor &ssm_state_indices, int64_t nk, int64_t nv,
                                               c10::optional<at::Tensor> intermediate_state_opt,
                                               c10::optional<at::Tensor> cache_indices_opt,
                                               c10::optional<at::Tensor> num_accepted_tokens_opt,
                                               c10::optional<at::Tensor> g_opt, c10::optional<at::Tensor> gk_opt)
{
    TORCH_CHECK(mix_qkv.defined(), "MixQKV tensor must be defined");
    TORCH_CHECK(recurrent_state.defined(), "State tensor must be defined");

    TORCH_CHECK(mix_qkv.dim() == 3, "MixQKV must be 3-dimensional (B, S, D)");
    TORCH_CHECK(recurrent_state.dim() == 4, "State must be 4-dimensional (N, nv, dv, dk)");

    int64_t b = mix_qkv.size(0);
    int64_t s = mix_qkv.size(1);
    int64_t d = mix_qkv.size(2);
    int t = b * s;
    int64_t dv = recurrent_state.size(2);
    int64_t dk = recurrent_state.size(3);

    int64_t expectedD = 2 * nk * dk + nv * dv;
    TORCH_CHECK(d == expectedD, "mix_qkv width mismatch. Expected: " + std::to_string(expectedD) +
                                    ", Got: " + std::to_string(d) + ". Formula: D = nv*dv + 2*nk*dk, where nv=" +
                                    std::to_string(nv) + ", dv(or State.size(2))=" + std::to_string(dv) +
                                    ", nk=" + std::to_string(nk) + ", dk(or State.size(3))=" + std::to_string(dk));

    TORCH_CHECK(recurrent_state.size(1) == nv, "State third dimension must match nv");

    TORCH_CHECK(beta.dim() == 3, "Beta must be 3-dimensional (B, S, nv)");
    TORCH_CHECK(beta.size(0) == b, "Beta batch size must match MixQKV");
    TORCH_CHECK(beta.size(1) == s, "Beta sequence length must match MixQKV");
    TORCH_CHECK(beta.size(2) == nv, "Beta last dimension must equal nv");

    TORCH_CHECK(ssm_state_indices.dim() == 2, "ssm_state_indices must be 2-dimensional (B, S)");
    TORCH_CHECK(ssm_state_indices.size(0) == b, "ssm_state_indices batch size must match MixQKV");
    TORCH_CHECK(ssm_state_indices.size(1) == s, "ssm_state_indices sequence length must match MixQKV");

    uint64_t ubSize{0UL};
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    uint32_t coreNum = ascendcPlatform->GetCoreNum();

    int devidx = mix_qkv.device().index();
    c10_npu::set_device(devidx);

    // =================================Calculate the size of UB===================================
    const int64_t MAX_MTP = 8;
    const int64_t ALIGN_SIZE = 16;

    auto ceilAlign = [](int64_t value, int64_t align) { return (value + align - 1) & ~(align - 1); };

    auto ceilDiv = [](int64_t dividend, int64_t divisor) { return (dividend + divisor - 1) / divisor; };

    int64_t aNv = ceilAlign(nv, ALIGN_SIZE);
    int64_t aDv = ceilAlign(dv, ALIGN_SIZE);
    int64_t aDk = ceilAlign(dk, ALIGN_SIZE);

    int64_t usedUbBytes = MAX_MTP * (4 * aDk + 2 * aDv);  // 4 for qLocal & kLocal, 2 for vLocal
    usedUbBytes += 128;                                   // reserve 128 Bytes
    usedUbBytes += MAX_MTP * (4 * aNv + 2 * aNv);         // 4 for gamaLocal, 2 for betaLocal

    int64_t ubRestBytes = ubSize - usedUbBytes;

    usedUbBytes += MAX_MTP * (8 * aDk + 4 * aDv + 4 * aNv);  // 8 for qk in ub, 4 for v in ub, 4 for beta in ub
    int64_t coeff = (2 + 2) * aDk + 4;                       // 2 for stateLocal, stateOutLocal, 4 for attnOutLocal
    coeff += (4 + 4) * aDk + 4 + 4;                          // 4 for qInUb, kInUb, vInUb, deltaInUb, attnInUb

    int64_t vStep = (ubSize - usedUbBytes) / coeff / 8 * 8;  // 8 * sizeof(float) = 32
    if (vStep < 8) {                                         // vStep must be no less than 8
        TORCH_CHECK(false, "vStep should be bigger than 8, shape is too big");
        TORCH_CHECK(false, "vStep (" + std::to_string(vStep) + ") should be bigger than 8 ");
    }

    int64_t rptime = ceilDiv(dv, static_cast<uint32_t>(vStep));
    vStep = ceilAlign(ceilDiv(dv, static_cast<uint32_t>(rptime)), 8);  // 8 * sizeof(float) = 32
    ubRestBytes -= ((2 + 2) * aDk + 4) * vStep;  // 2 for stateLocal, stateOutLocal, 4 for attnOutLocal

    // ===================== optional inputs =====================

    bool hasIntermediateState = false;
    void *intermediateStatePtr = nullptr;

    at::Tensor intermediate_state_tensor;

    void *recurrentStatePtr = recurrent_state.data_ptr();
    void *initStatePtr = recurrentStatePtr;
    void *stateOutPtr = recurrentStatePtr;
    void *mtpRecurrentStatePtr = nullptr;

    if (intermediate_state_opt.has_value() && intermediate_state_opt.value().defined()) {
        hasIntermediateState = true;
        intermediate_state_tensor = intermediate_state_opt.value().contiguous();
        intermediateStatePtr = intermediate_state_tensor.data_ptr();

        // MTP input and output
        initStatePtr = intermediateStatePtr;
        stateOutPtr = intermediateStatePtr;
        mtpRecurrentStatePtr = recurrentStatePtr;
    }

    void *cacheIndicesPtr = nullptr;
    at::Tensor cache_indices_tensor;

    if (cache_indices_opt.has_value() && cache_indices_opt.value().defined()) {
        cache_indices_tensor = cache_indices_opt.value().to(at::kInt).contiguous();
        cacheIndicesPtr = cache_indices_tensor.data_ptr();
    }

    bool hasAcceptedTokens = false;
    void *numAcceptedTokensPtr = nullptr;
    at::Tensor num_accepted_tokens_int32;

    if (num_accepted_tokens_opt.has_value() && num_accepted_tokens_opt.value().defined()) {
        hasAcceptedTokens = true;
        num_accepted_tokens_int32 = num_accepted_tokens_opt.value().to(at::kInt).contiguous();

        numAcceptedTokensPtr = num_accepted_tokens_int32.data_ptr();
    }

    bool hasGama = false;
    at::Tensor g_tensor;

    if (g_opt.has_value() && g_opt.value().defined()) {
        hasGama = true;
        g_tensor = g_opt.value().to(at::kFloat).contiguous();
    } else {
        g_tensor = torch::ones({t, nv}, torch::TensorOptions().dtype(at::kFloat).device(mix_qkv.device()));
    }

    void *gPtr = g_tensor.data_ptr();

    void *gkPtr = nullptr;
    at::Tensor gk_local;

    if (gk_opt.has_value() && gk_opt.value().defined()) {
        gk_local = gk_opt.value().to(at::kFloat).contiguous();

        gkPtr = gk_local.data_ptr();
    }

    at::Tensor output = torch::empty({b, s, nv, dv}, mix_qkv.options());

    EXEC_KERNEL_CMD(recurrent_gated_delta_rule, coreNum, mix_qkv, beta, initStatePtr, actual_seq_lengths,
                    ssm_state_indices, mtpRecurrentStatePtr, cacheIndicesPtr, gPtr, gkPtr, numAcceptedTokensPtr, output,
                    stateOutPtr, b, s, nk, dk, nv, dv, hasIntermediateState, hasAcceptedTokens, hasGama, vStep,
                    ubRestBytes, scale);

    return output;
}
}  // namespace npu_kernel
}  // namespace sglang
