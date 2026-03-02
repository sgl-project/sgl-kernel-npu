/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdio>
#include <cstring>
#include <unordered_map>
#include <functional>
#include "acl/acl.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/causal_conv1d_update_tiling.h"
#include "defines.h"
#include "torch_helper.h"
#include "common_tiling.h"
#include "common.h"
#include "stub/aclrtlaunch_causal_conv1d_update_bfloat16_t.h"
#include "stub/aclrtlaunch_causal_conv1d_update_half.h"

namespace sglang {
namespace npu_kernel {

constexpr uint32_t PADDING_BYTE = 32U;
constexpr uint32_t MAX_CAPTURE_NUM = 256;

// Graph mode tiling cache
uint32_t actualCaptureNum = 0;
static std::unordered_map<uint64_t, uint32_t> captureMap;

// Helper struct for hashing tiling parameters
struct CausalConv1dUpdateTilingKey {
    int64_t batch;
    int64_t seqLen;
    int64_t dim;
    int64_t width;
    int64_t stateLen;
    int64_t hasIndices;
    int64_t hasBias;
    int64_t hasNumAccept;
    int64_t hasQueryLoc;
    int64_t activationMode;
    int64_t padSlotId;

    bool operator==(const CausalConv1dUpdateTilingKey& other) const {
        return batch == other.batch && seqLen == other.seqLen && dim == other.dim &&
               width == other.width && stateLen == other.stateLen && hasIndices == other.hasIndices &&
               hasBias == other.hasBias && hasNumAccept == other.hasNumAccept &&
               hasQueryLoc == other.hasQueryLoc && activationMode == other.activationMode &&
               padSlotId == other.padSlotId;
    }
};

// Hash function for CausalConv1dUpdateTilingKey
struct CausalConv1dUpdateTilingKeyHash {
    std::size_t operator()(const CausalConv1dUpdateTilingKey& k) const {
        std::size_t h1 = std::hash<int64_t>{}(k.batch);
        std::size_t h2 = std::hash<int64_t>{}(k.seqLen);
        std::size_t h3 = std::hash<int64_t>{}(k.dim);
        std::size_t h4 = std::hash<int64_t>{}(k.width);
        std::size_t h5 = std::hash<int64_t>{}(k.stateLen);
        std::size_t h6 = std::hash<int64_t>{}(k.hasIndices);
        std::size_t h7 = std::hash<int64_t>{}(k.hasBias);
        std::size_t h8 = std::hash<int64_t>{}(k.hasNumAccept);
        std::size_t h9 = std::hash<int64_t>{}(k.hasQueryLoc);
        std::size_t h10 = std::hash<int64_t>{}(k.activationMode);
        std::size_t h11 = std::hash<int64_t>{}(k.padSlotId);
        return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3) ^ (h5 << 4) ^ (h6 << 5) ^ (h7 << 6) ^ (h8 << 7) ^ (h9 << 8) ^ (h10 << 9) ^ (h11 << 10);
    }
};

// Global tiling buffer for graph mode
static at::Tensor globalTilingBuffer;

void populate_tiling_data(CausalConv1dUpdateTilingData* tiling_data, int32_t block_dim,
                          int64_t batch, int64_t seq_len, int64_t dim, int64_t width, int64_t state_len,
                          bool has_indices, bool has_bias, bool has_num_accept, bool has_query_loc,
                          bool activation_mode, int64_t pad_slot_id)
{
    tiling_data->batch = batch;
    tiling_data->seqLen = seq_len;
    tiling_data->dim = dim;
    tiling_data->width = width;
    tiling_data->stateLen = state_len;
    tiling_data->hasIndices = has_indices ? 1 : 0;
    tiling_data->hasBias = has_bias ? 1 : 0;
    tiling_data->hasNumAccept = has_num_accept ? 1 : 0;
    tiling_data->hasQueryLoc = has_query_loc ? 1 : 0;
    tiling_data->activationMode = activation_mode ? 1 : 0;
    tiling_data->padSlotId = pad_slot_id;
    tiling_data->numCore = block_dim;

    // Compute block factor
    tiling_data->blockFactor = (batch + block_dim - 1) / block_dim;
    tiling_data->blockTailFactor = batch - tiling_data->blockFactor * (block_dim - 1);
    if (tiling_data->blockTailFactor <= 0) {
        tiling_data->blockTailFactor = tiling_data->blockFactor;
    }
}

at::Tensor get_tiling(int32_t& block_dim, int32_t& workspace_size,
                      int64_t batch, int64_t seq_len, int64_t dim, int64_t width, int64_t state_len,
                      bool has_indices, bool has_bias, bool has_num_accept, bool has_query_loc,
                      bool activation_mode, int64_t pad_slot_id, bool enable_graph_mode_cache = false,
                      const at::Device& device = at::kCPU)
{
    auto ascendc_platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int32_t max_aiv_core = static_cast<int32_t>(ascendc_platform->GetCoreNumAiv());
    block_dim = std::min(max_aiv_core, static_cast<int32_t>(batch));
    if (block_dim == 0) {
        block_dim = 1;
    }
    workspace_size = static_cast<int32_t>(ascendc_platform->GetLibApiWorkSpaceSize());

    // align to 32 bytes
    int32_t tiling_size = (sizeof(CausalConv1dUpdateTilingData) + PADDING_BYTE - 1) / PADDING_BYTE * PADDING_BYTE;

    at::Tensor tiling_tensor;

    if (enable_graph_mode_cache) {
        // Graph mode: use cached tiling
        CausalConv1dUpdateTilingKey key{
            .batch = batch,
            .seqLen = seq_len,
            .dim = dim,
            .width = width,
            .stateLen = state_len,
            .hasIndices = has_indices ? 1 : 0,
            .hasBias = has_bias ? 1 : 0,
            .hasNumAccept = has_num_accept ? 1 : 0,
            .hasQueryLoc = has_query_loc ? 1 : 0,
            .activationMode = activation_mode ? 1 : 0,
            .padSlotId = pad_slot_id
        };

        CausalConv1dUpdateTilingKeyHash hasher;
        uint64_t hashValue = hasher(key);

        // Initialize global tiling buffer if needed
        if (!globalTilingBuffer.defined() || globalTilingBuffer.numel() < static_cast<int64_t>(tiling_size * MAX_CAPTURE_NUM)) {
            globalTilingBuffer = at::empty({static_cast<int64_t>(tiling_size * MAX_CAPTURE_NUM)},
                                          at::TensorOptions().dtype(at::kByte).device(device));
        }

        if (captureMap.find(hashValue) != captureMap.end()) {
            // Get cached tiling data from globalTilingBuffer
            tiling_tensor = at::from_blob(globalTilingBuffer.data_ptr<uint8_t>() + (tiling_size * captureMap[hashValue]),
                                         tiling_size, at::kByte);
        } else if (actualCaptureNum >= MAX_CAPTURE_NUM) {
            // Exceeded max captures, use temporary buffer
            static at::Tensor tempTilingBuffer = at::empty({tiling_size},
                                                          at::TensorOptions().dtype(at::kByte).device(at::kCPU));
            CausalConv1dUpdateTilingData* tiling_data = reinterpret_cast<CausalConv1dUpdateTilingData*>(tempTilingBuffer.data_ptr());
            populate_tiling_data(tiling_data, block_dim, batch, seq_len, dim, width, state_len,
                               has_indices, has_bias, has_num_accept, has_query_loc,
                               activation_mode, pad_slot_id);

            // Create temporary device buffer
            static at::Tensor tempDeviceTilingBuffer = at::empty({tiling_size},
                                                                at::TensorOptions().dtype(at::kByte).device(device));
            aclrtMemcpy(tempDeviceTilingBuffer.data_ptr<uint8_t>(), tiling_size, tiling_data, tiling_size, ACL_MEMCPY_HOST_TO_DEVICE);
            tiling_tensor = at::from_blob(tempDeviceTilingBuffer.data_ptr<uint8_t>(), tiling_size, at::kByte);
        } else {
            // Cache new tiling data
            captureMap[hashValue] = actualCaptureNum;

            // Create host buffer
            auto tiling_buffer = at::empty({tiling_size}, at::TensorOptions().dtype(at::kByte).device(at::kCPU));
            CausalConv1dUpdateTilingData* tiling_data = reinterpret_cast<CausalConv1dUpdateTilingData*>(tiling_buffer.data_ptr());
            populate_tiling_data(tiling_data, block_dim, batch, seq_len, dim, width, state_len,
                               has_indices, has_bias, has_num_accept, has_query_loc,
                               activation_mode, pad_slot_id);

            // Copy to global buffer (host to device)
            aclrtMemcpy(globalTilingBuffer.data_ptr<uint8_t>() + actualCaptureNum * tiling_size, tiling_size,
                       tiling_data, tiling_size, ACL_MEMCPY_HOST_TO_DEVICE);
            actualCaptureNum++;

            tiling_tensor = at::from_blob(globalTilingBuffer.data_ptr<uint8_t>() + (tiling_size * captureMap[hashValue]),
                                         tiling_size, at::kByte);
        }
    } else {
        // Eager mode: create tiling tensor normally
        auto tiling_buffer = at::empty({tiling_size}, at::TensorOptions().dtype(at::kByte).device(at::kCPU));
        CausalConv1dUpdateTilingData* tiling_data = reinterpret_cast<CausalConv1dUpdateTilingData*>(tiling_buffer.data_ptr());
        populate_tiling_data(tiling_data, block_dim, batch, seq_len, dim, width, state_len,
                           has_indices, has_bias, has_num_accept, has_query_loc,
                           activation_mode, pad_slot_id);
        tiling_tensor = TorchNpuHelper::CopyTensorHostToDevice(tiling_buffer);
    }

    return tiling_tensor;
}

HOST_API at::Tensor causal_conv1d_update_impl(
    const at::Tensor& x, const at::Tensor& weight, const at::Tensor& conv_state,
    const at::Tensor& conv_state_indices, const at::Tensor& bias,
    const at::Tensor& num_accepted_tokens, const at::Tensor& query_start_loc,
    bool activation_mode, int64_t pad_slot_id)
{
    // Input validation
    TORCH_CHECK(x.dim() == 3, "x must be 3D tensor [batch, seq_len, dim], got shape ", x.sizes());
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D tensor [width, dim], got shape ", weight.sizes());
    TORCH_CHECK(conv_state.dim() == 3, "conv_state must be 3D tensor [cache_len, width-1, dim], got shape ", conv_state.sizes());

    const at::ScalarType dtype = x.scalar_type();
    TORCH_CHECK(dtype == at::kBFloat16 || dtype == at::kHalf, "Only BF16 and FP16 are supported, got ", dtype);
    TORCH_CHECK(weight.scalar_type() == dtype, "weight dtype must match x dtype");
    TORCH_CHECK(conv_state.scalar_type() == dtype, "conv_state dtype must match x dtype");

    const int64_t batch = x.size(0);
    const int64_t seq_len = x.size(1);
    const int64_t dim = x.size(2);
    const int64_t width = weight.size(0);
    const int64_t state_len = conv_state.size(1);

    // Check optional tensors
    const bool has_indices = conv_state_indices.numel() > 0;
    const bool has_bias = bias.numel() > 0;
    const bool has_num_accept = num_accepted_tokens.numel() > 0;
    const bool has_query_loc = query_start_loc.numel() > 0;

    if (has_indices) {
        TORCH_CHECK(conv_state_indices.dim() == 1, "conv_state_indices must be 1D tensor");
        TORCH_CHECK(conv_state_indices.size(0) >= batch, "conv_state_indices size must be >= batch");
    }

    if (has_bias) {
        TORCH_CHECK(bias.dim() == 1, "bias must be 1D tensor");
        TORCH_CHECK(bias.size(0) == dim, "bias size must match dim");
        TORCH_CHECK(bias.scalar_type() == dtype, "bias dtype must match input dtype");
    }

    if (has_num_accept) {
        TORCH_CHECK(num_accepted_tokens.dim() == 1, "num_accepted_tokens must be 1D tensor");
        TORCH_CHECK(num_accepted_tokens.size(0) >= batch, "num_accepted_tokens size must be >= batch");
    }

    if (has_query_loc) {
        TORCH_CHECK(query_start_loc.dim() == 1, "query_start_loc must be 1D tensor");
        TORCH_CHECK(query_start_loc.size(0) >= batch + 1, "query_start_loc size must be >= batch+1");
    }

    // Create output tensor
    at::Tensor y = at::empty_like(x);

    // Detect graph mode by checking if we're using NPU and graph compilation is enabled
    bool enable_graph_mode_cache = false;
    if (x.is_npu()) {
        // Check for graph mode environment variable or context
        const char* graph_mode_env = std::getenv("TORCH_NPU_COMPILE_ENABLE");
        enable_graph_mode_cache = (graph_mode_env != nullptr && std::string(graph_mode_env) != "0");
    }

    // Get tiling data
    int32_t block_dim;
    int32_t workspace_size;
    at::Tensor tiling_tensor = get_tiling(block_dim, workspace_size,
                                           batch, seq_len, dim, width, state_len,
                                           has_indices, has_bias, has_num_accept, has_query_loc,
                                           activation_mode, pad_slot_id, enable_graph_mode_cache,
                                           x.options().device());

    // Create workspace tensor
    auto workspace_tensor = at::empty({workspace_size},
                                      at::TensorOptions().dtype(at::kByte).device(x.options().device()));

    // Prepare tensors for kernel launch
    at::Tensor x_contiguous = x.contiguous();
    at::Tensor weight_contiguous = weight.contiguous();
    at::Tensor conv_state_contiguous = conv_state.contiguous();
    at::Tensor conv_state_indices_contiguous = has_indices ? conv_state_indices.contiguous() : at::empty(0, x.options());
    at::Tensor bias_contiguous = has_bias ? bias.contiguous() : at::empty(0, x.options());
    at::Tensor num_accepted_tokens_contiguous = has_num_accept ? num_accepted_tokens.contiguous() : at::empty(0, x.options());
    at::Tensor query_start_loc_contiguous = has_query_loc ? query_start_loc.contiguous() : at::empty(0, x.options());

    // Launch kernel based on dtype
    if (dtype == at::kBFloat16) {
        EXEC_KERNEL_CMD(causal_conv1d_update_bfloat16_t, block_dim, x_contiguous, weight_contiguous, conv_state_contiguous,
                        conv_state_indices_contiguous, bias_contiguous, num_accepted_tokens_contiguous,
                        query_start_loc_contiguous, y, workspace_tensor, tiling_tensor);
    } else {
        EXEC_KERNEL_CMD(causal_conv1d_update_half, block_dim, x_contiguous, weight_contiguous, conv_state_contiguous,
                        conv_state_indices_contiguous, bias_contiguous, num_accepted_tokens_contiguous,
                        query_start_loc_contiguous, y, workspace_tensor, tiling_tensor);
    }

    // Return y (conv_state is updated in-place)
    return y;
}

}  // namespace npu_kernel
}  // namespace sglang
