/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See
 * LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_runtime.h"
#include "aclnn_sgl_causal_conv1d.h"
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>
#include <initializer_list>
#include <limits>
#include <vector>

namespace {
bool HasElements(const c10::optional<at::Tensor> &tensor)
{
    return tensor.has_value() && tensor->defined() && tensor->numel() > 0;
}

// Legacy metadata/bias pointers are read as dense spans after the original dtype
// conversion, regardless of their Tensor strides. Do not add a contiguous copy.
std::shared_ptr<aclTensor> raw_pointer_descriptor(const at::Tensor &tensor, aclDataType dtype)
{
    std::vector<int64_t> strides(tensor.dim());
    int64_t stride = 1;
    for (int64_t dim = tensor.dim(); dim-- > 0;) {
        strides[dim] = stride;
        stride *= tensor.size(dim);
    }
    auto *descriptor = aclCreateTensor(tensor.sizes().data(), tensor.dim(), dtype, strides.data(), 0, ACL_FORMAT_ND,
                                       tensor.sizes().data(), tensor.dim(), tensor.data_ptr());
    TORCH_CHECK(descriptor, "aclCreateTensor failed: ", aclGetRecentErrMsg());
    return {descriptor, [](aclTensor *value) { aclDestroyTensor(value); }};
}

at::Tensor causal_conv1d(const at::Tensor &x, const at::Tensor &weight, const at::Tensor &conv_states,
                         const c10::optional<at::Tensor> &bias, const c10::optional<at::Tensor> &query_start_loc,
                         const c10::optional<at::Tensor> &cache_indices,
                         const c10::optional<at::Tensor> &has_initial_state,
                         const c10::optional<at::Tensor> &num_accepted_tokens, int64_t activation_mode,
                         int64_t pad_slot_id, int64_t run_mode)
{
    // Keep required-input checks and metadata conversion in the Torch bridge.
    TORCH_CHECK(x.defined(), "x tensor must be defined");
    TORCH_CHECK(weight.defined(), "weight tensor must be defined");
    TORCH_CHECK(conv_states.defined(), "conv_states tensor must be defined");

    TORCH_CHECK(x.dim() == 2 || x.dim() == 3, "x must be 2D or 3D tensor");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D tensor");
    TORCH_CHECK(conv_states.dim() == 3, "conv_states must be 3D tensor");

    const at::ScalarType dtype = x.scalar_type();
    TORCH_CHECK(dtype == at::kBFloat16 || dtype == at::kHalf, "Only BF16 and FP16 are supported");
    TORCH_CHECK(weight.scalar_type() == dtype, "weight dtype must match x dtype");
    TORCH_CHECK(conv_states.scalar_type() == dtype, "conv_states dtype must match x dtype");

    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(conv_states.is_contiguous(), "conv_states must be contiguous");

    int64_t dim = (x.dim() == 2) ? x.size(1) : x.size(2);
    int64_t width = weight.size(0);
    TORCH_CHECK(weight.size(1) == dim, "weight.shape[1] must equal dim");
    TORCH_CHECK(conv_states.size(2) == dim, "conv_states.shape[2] must equal dim");
    TORCH_CHECK(width >= 2 && width <= 4, "Only support width in [2,4]");

    int64_t inputMode = (x.dim() == 2) ? 0 : 1;

    if (inputMode == 0) {
        TORCH_CHECK(query_start_loc.has_value() && query_start_loc->defined(),
                    "query_start_loc must be provided for 2D x");
        int64_t qslSize = query_start_loc->size(0);
        TORCH_CHECK(qslSize >= 2, "query_start_loc must have at least 2 elements");
    }

    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "x must be on an NPU");
    TORCH_CHECK(weight.device() == x.device() && conv_states.device() == x.device(),
                "x, weight and conv_states must be on the same NPU");
    for (const auto *optional : {&bias, &query_start_loc, &cache_indices, &has_initial_state, &num_accepted_tokens}) {
        if (HasElements(*optional)) {
            TORCH_CHECK((*optional)->device() == x.device(), "nonempty optional inputs must be on the same NPU as x");
        }
    }
    const c10::DeviceGuard deviceGuard(x.device());
    const bool hasBias = HasElements(bias);
    const bool hasQuery = HasElements(query_start_loc);
    const bool hasCache = HasElements(cache_indices);
    const bool hasInitial = HasElements(has_initial_state);
    const bool hasAccepted = HasElements(num_accepted_tokens);
    auto y = at::empty_like(x);
    at::Tensor biasTensor, queryTensor, cacheTensor, initialTensor, acceptedTensor;
    if (hasBias) biasTensor = *bias;
    if (hasQuery) queryTensor = query_start_loc->to(at::kLong);
    if (hasCache) cacheTensor = cache_indices->to(at::kLong);
    if (hasInitial) initialTensor = has_initial_state->to(at::kLong);
    if (hasAccepted) acceptedTensor = num_accepted_tokens->to(at::kInt);

    // The generated prototype checks Ref merging and the complete argument order.
    static auto queryWorkspace = reinterpret_cast<decltype(&aclnnSglCausalConv1dGetWorkspaceSize)>(
        sgl_cann::op_api_symbol("aclnnSglCausalConv1dGetWorkspaceSize"));
    static auto execute =
        reinterpret_cast<decltype(&aclnnSglCausalConv1d)>(sgl_cann::op_api_symbol("aclnnSglCausalConv1d"));
    const auto aclDtype = dtype == at::kBFloat16 ? ACL_BF16 : ACL_FLOAT16;
    auto xDesc = sgl_cann::tensor_descriptor(x, aclDtype);
    auto weightDesc = sgl_cann::tensor_descriptor(weight, aclDtype);
    auto stateDesc = sgl_cann::tensor_descriptor(conv_states, aclDtype);
    auto biasDesc = hasBias ? raw_pointer_descriptor(biasTensor, aclDtype) : std::shared_ptr<aclTensor>{};
    auto queryDesc = hasQuery ? raw_pointer_descriptor(queryTensor, ACL_INT64) : std::shared_ptr<aclTensor>{};
    auto cacheDesc = hasCache ? raw_pointer_descriptor(cacheTensor, ACL_INT64) : std::shared_ptr<aclTensor>{};
    auto initialDesc = hasInitial ? raw_pointer_descriptor(initialTensor, ACL_INT64) : std::shared_ptr<aclTensor>{};
    auto acceptedDesc = hasAccepted ? raw_pointer_descriptor(acceptedTensor, ACL_INT32) : std::shared_ptr<aclTensor>{};
    auto yDesc = sgl_cann::tensor_descriptor(y, aclDtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    const auto status = queryWorkspace(xDesc.get(), weightDesc.get(), stateDesc.get(), biasDesc.get(), queryDesc.get(),
                                       cacheDesc.get(), initialDesc.get(), acceptedDesc.get(), activation_mode,
                                       pad_slot_id, run_mode, yDesc.get(), &workspaceSize, &executor);
    TORCH_CHECK(status == ACL_SUCCESS, "CausalConv1d workspace query failed: ", aclGetRecentErrMsg());
    TORCH_CHECK(workspaceSize <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                "ACLNN workspace size exceeds the Tensor allocation limit");
    auto workspace = at::empty({static_cast<int64_t>(workspaceSize)}, x.options().dtype(at::kByte));

    // Tensor captures retain host-queued work; record_stream protects device lifetime.
    std::vector<at::Tensor> tensors{x, weight, conv_states, y, workspace};
    for (const auto &tensor : {biasTensor, queryTensor, cacheTensor, initialTensor, acceptedTensor}) {
        if (tensor.defined()) tensors.push_back(tensor);
    }
    std::vector<std::shared_ptr<aclTensor>> descriptors{xDesc,     weightDesc,  stateDesc,    biasDesc, queryDesc,
                                                        cacheDesc, initialDesc, acceptedDesc, yDesc};
    auto stream = c10_npu::getCurrentNPUStream();
    for (const auto &tensor : tensors)
        tensor.record_stream(stream);
    auto aclStream = stream.stream(false);
    auto launch = [tensors, descriptors, workspace, workspaceSize, executor, aclStream]() -> int {
        const auto result = execute(workspaceSize ? workspace.data_ptr() : nullptr, workspaceSize, executor, aclStream);
        TORCH_CHECK(result == ACL_SUCCESS, "CausalConv1d ACLNN execution failed: ", aclGetRecentErrMsg());
        return result;
    };
    at_npu::native::OpCommand::RunOpApi("aclnnSglCausalConv1d", launch);
    return y;
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def(
        "causal_conv1d(Tensor x, Tensor weight, Tensor(a!) conv_states, Tensor? bias=None, "
        "Tensor? query_start_loc=None, Tensor? cache_indices=None, Tensor? has_initial_state=None, "
        "Tensor? num_accepted_tokens=None, int activation_mode=0, int pad_slot_id=-1, int run_mode=0) -> Tensor");
}
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("causal_conv1d", &causal_conv1d);
}
