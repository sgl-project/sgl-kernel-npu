/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "causal_conv1d.h"

#include <algorithm>
#include <cstddef>

#include "acl/acl.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/causal_conv1d_tiling.h"
#include "causal_conv1d_tiling_data.h"
#include "stub/aclrtlaunch_causal_conv1d_bfloat16_t.h"
#include "stub/aclrtlaunch_causal_conv1d_half.h"
#include "torch_helper.h"

namespace sglang {
namespace npu_kernel {
namespace {

constexpr uint32_t PADDING_BYTE = 32U;

struct CausalConv1dShapeInfo {
    int64_t batch = 0;
    int64_t cuSeqlen = 0;
    int64_t seqLen = 0;
    int64_t inputMode = 0;
    int64_t dim = 0;
    int64_t width = 0;
    int64_t stateLen = 0;
    int64_t numCacheLines = 0;
    bool hasBias = false;
};

void CheckSameDevice(const at::Tensor& lhs, const at::Tensor& rhs, const char* lhs_name, const char* rhs_name)
{
    TORCH_CHECK(lhs.device() == rhs.device(), lhs_name, " and ", rhs_name, " must be on the same device");
}

CausalConv1dShapeInfo ValidateInputs(
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& conv_states,
    const at::Tensor& query_start_loc,
    const at::Tensor& cache_indices,
    const at::Tensor& has_initial_state,
    const at::Tensor& bias)
{
    TORCH_CHECK(x.dim() == 2 || x.dim() == 3,
                "x must be 2D [cu_seqlen, dim] or 3D [batch, seq_len, dim], got shape ", x.sizes());
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D [width, dim], got shape ", weight.sizes());
    TORCH_CHECK(conv_states.dim() == 3,
                "conv_states must be 3D [num_cache_lines, state_len, dim], got shape ", conv_states.sizes());
    TORCH_CHECK(query_start_loc.dim() == 1, "query_start_loc must be 1D [batch + 1], got shape ", query_start_loc.sizes());
    TORCH_CHECK(cache_indices.dim() == 1, "cache_indices must be 1D [batch], got shape ", cache_indices.sizes());
    TORCH_CHECK(has_initial_state.dim() == 1,
                "has_initial_state must be 1D [batch], got shape ", has_initial_state.sizes());

    const at::ScalarType dtype = x.scalar_type();
    TORCH_CHECK(dtype == at::kBFloat16 || dtype == at::kHalf, "Only BF16 and FP16 are supported, got ", dtype);
    TORCH_CHECK(weight.scalar_type() == dtype, "weight dtype must match x dtype");
    TORCH_CHECK(conv_states.scalar_type() == dtype, "conv_states dtype must match x dtype");
    TORCH_CHECK(query_start_loc.scalar_type() == at::kInt, "query_start_loc dtype must be int32");
    TORCH_CHECK(cache_indices.scalar_type() == at::kInt, "cache_indices dtype must be int32");
    TORCH_CHECK(has_initial_state.scalar_type() == at::kBool, "has_initial_state dtype must be bool");

    const bool has_bias = bias.numel() > 0;
    if (has_bias) {
        TORCH_CHECK(bias.dim() == 1, "bias must be 1D [dim], got shape ", bias.sizes());
        TORCH_CHECK(bias.scalar_type() == dtype, "bias dtype must match x dtype");
    }

    CheckSameDevice(x, weight, "x", "weight");
    CheckSameDevice(x, conv_states, "x", "conv_states");
    CheckSameDevice(x, query_start_loc, "x", "query_start_loc");
    CheckSameDevice(x, cache_indices, "x", "cache_indices");
    CheckSameDevice(x, has_initial_state, "x", "has_initial_state");
    if (has_bias) {
        CheckSameDevice(x, bias, "x", "bias");
    }

    TORCH_CHECK(x.is_contiguous(), "x must be contiguous before entering the NPU kernel");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous before entering the NPU kernel");
    TORCH_CHECK(conv_states.is_contiguous(), "conv_states must be contiguous before entering the NPU kernel");
    TORCH_CHECK(query_start_loc.is_contiguous(), "query_start_loc must be contiguous before entering the NPU kernel");
    TORCH_CHECK(cache_indices.is_contiguous(), "cache_indices must be contiguous before entering the NPU kernel");
    TORCH_CHECK(has_initial_state.is_contiguous(),
                "has_initial_state must be contiguous before entering the NPU kernel");
    if (has_bias) {
        TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous before entering the NPU kernel");
    }

    CausalConv1dShapeInfo info;
    if (x.dim() == 2) {
        info.inputMode = 0;
        info.cuSeqlen = x.size(0);
        info.dim = x.size(1);
        info.seqLen = 0;
        TORCH_CHECK(info.dim > 0, "x.shape[1] must be > 0");
        TORCH_CHECK(info.cuSeqlen >= 0, "x.shape[0] must be >= 0");
        TORCH_CHECK(query_start_loc.size(0) >= 1, "query_start_loc.size(0) must be >= 1");
        info.batch = query_start_loc.size(0) - 1;
    } else {
        info.inputMode = 1;
        info.batch = x.size(0);
        info.seqLen = x.size(1);
        info.dim = x.size(2);
        info.cuSeqlen = info.batch * info.seqLen;
        TORCH_CHECK(info.batch > 0, "x.shape[0] must be > 0");
        TORCH_CHECK(info.seqLen > 0, "x.shape[1] must be > 0");
        TORCH_CHECK(info.dim > 0, "x.shape[2] must be > 0");
        TORCH_CHECK(query_start_loc.size(0) == info.batch + 1,
                    "query_start_loc.size(0) must equal batch + 1 for 3D input");
    }

    TORCH_CHECK(info.batch > 0, "batch must be > 0");

    info.width = weight.size(0);
    TORCH_CHECK(weight.size(1) == info.dim, "weight.shape[1] must equal dim");
    TORCH_CHECK(info.width == 4, "Only width == 4 is supported, got ", info.width);
    TORCH_CHECK(info.dim == 1024 || info.dim == 2048 || info.dim == 4096 || info.dim == 8192,
                "Only dim in {1024, 2048, 4096, 8192} is supported, got ", info.dim);

    info.numCacheLines = conv_states.size(0);
    info.stateLen = conv_states.size(1);
    TORCH_CHECK(info.numCacheLines > 0, "conv_states.shape[0] must be > 0");
    TORCH_CHECK(conv_states.size(2) == info.dim, "conv_states.shape[2] must equal dim");
    TORCH_CHECK(info.stateLen >= info.width - 1, "conv_states.shape[1] must be >= width - 1");

    TORCH_CHECK(cache_indices.size(0) == info.batch, "cache_indices.size(0) must equal batch");
    TORCH_CHECK(has_initial_state.size(0) == info.batch, "has_initial_state.size(0) must equal batch");

    if (has_bias) {
        TORCH_CHECK(bias.size(0) == info.dim, "bias.size(0) must equal dim");
    }

    info.hasBias = has_bias;
    return info;
}

}  // namespace

HOST_API at::Tensor causal_conv1d_impl(
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& conv_states,
    const at::Tensor& query_start_loc,
    const at::Tensor& cache_indices,
    const at::Tensor& has_initial_state,
    const at::Tensor& bias,
    bool activation_mode,
    int64_t pad_slot_id)
{
    const CausalConv1dShapeInfo info =
        ValidateInputs(x, weight, conv_states, query_start_loc, cache_indices, has_initial_state, bias);

    auto ascendc_platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(ascendc_platform != nullptr, "Failed to acquire AscendC platform manager");

    const int32_t core_num = static_cast<int32_t>(ascendc_platform->GetCoreNumAiv());
    TORCH_CHECK(core_num > 0, "AscendC returned invalid core_num: ", core_num);

    CausalConv1dTilingData tiling_data {};
    SGLang::CausalConv1d::ComputeTilingData(
        info.batch,
        info.cuSeqlen,
        info.seqLen,
        info.inputMode,
        info.dim,
        info.width,
        info.stateLen,
        info.numCacheLines,
        info.hasBias,
        activation_mode,
        pad_slot_id,
        core_num,
        tiling_data);

    TORCH_CHECK(tiling_data.dimTileSize > 0, "Failed to choose a valid dimTileSize for dim=", info.dim);
    TORCH_CHECK(tiling_data.blocksPerSeq > 0, "Failed to choose a valid blocksPerSeq for dim=", info.dim);

    const int64_t grid_size = info.batch * tiling_data.blocksPerSeq;
    TORCH_CHECK(grid_size > 0, "Invalid grid_size computed for causal_conv1d: ", grid_size);

    const int32_t block_dim = static_cast<int32_t>(std::min<int64_t>(grid_size, core_num));
    const int64_t workspace_size = static_cast<int64_t>(ascendc_platform->GetLibApiWorkSpaceSize());

    const size_t tiling_bytes = sizeof(CausalConv1dTilingData);
    const size_t aligned_tiling_bytes =
        (tiling_bytes + static_cast<size_t>(PADDING_BYTE) - 1U) / static_cast<size_t>(PADDING_BYTE) *
        static_cast<size_t>(PADDING_BYTE);

    auto byte_options = x.options().dtype(at::kByte);
    at::Tensor tiling_tensor = at::empty({static_cast<int64_t>(aligned_tiling_bytes)}, byte_options);
    const aclError copy_ret = aclrtMemcpy(
        tiling_tensor.data_ptr<uint8_t>(),
        aligned_tiling_bytes,
        &tiling_data,
        tiling_bytes,
        ACL_MEMCPY_HOST_TO_DEVICE);
    TORCH_CHECK(copy_ret == ACL_SUCCESS, "aclrtMemcpy for tiling data failed with code ", static_cast<int>(copy_ret));

    at::Tensor workspace_tensor = at::empty({workspace_size}, byte_options);
    at::Tensor y = at::empty_like(x);

    if (x.scalar_type() == at::kBFloat16) {
        EXEC_KERNEL_CMD(
            causal_conv1d_bfloat16_t,
            block_dim,
            x,
            weight,
            info.hasBias ? bias : at::empty({0}, x.options()),
            conv_states,
            query_start_loc,
            cache_indices,
            has_initial_state,
            y,
            workspace_tensor,
            tiling_tensor);
    } else {
        EXEC_KERNEL_CMD(
            causal_conv1d_half,
            block_dim,
            x,
            weight,
            info.hasBias ? bias : at::empty({0}, x.options()),
            conv_states,
            query_start_loc,
            cache_indices,
            has_initial_state,
            y,
            workspace_tensor,
            tiling_tensor);
    }

    return y;
}

}  // namespace npu_kernel
}  // namespace sglang
