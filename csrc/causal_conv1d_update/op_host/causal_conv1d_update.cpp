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
constexpr uint32_t MAX_CAPTURE_NUM = 1024; // 对齐 lightning_indexer

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

    // 关键拦截：防止在图回放中出现步长错乱的转置 Tensor！
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous before entering the NPU kernel. Fix this in Python.");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous. Transposed weights are NOT allowed. Fix this in Python.");
    TORCH_CHECK(conv_state.is_contiguous(), "conv_state must be contiguous. Fix this in Python.");

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

    // Create output tensor
    at::Tensor y = at::empty_like(x);

    auto ascendc_platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int32_t max_aiv_core = static_cast<int32_t>(ascendc_platform->GetCoreNumAiv());
    int32_t block_dim = std::min(max_aiv_core, static_cast<int32_t>(batch));
    if (block_dim == 0) {
        block_dim = 1;
    }
    int32_t workspace_size = static_cast<int32_t>(ascendc_platform->GetLibApiWorkSpaceSize());

    // 1. Prepare Tiling Data Struct (使用你头文件中的辅助函数)
    CausalConv1dUpdateTilingData tiling_data;
    SGLang::CausalConv1dUpdate::ComputeTilingData(
        batch, seq_len, dim, width, state_len,
        has_indices, has_bias, has_num_accept, has_query_loc,
        activation_mode, pad_slot_id, block_dim,
        tiling_data
    );

    int32_t tilingSize = (sizeof(CausalConv1dUpdateTilingData) + PADDING_BYTE - 1) / PADDING_BYTE * PADDING_BYTE;
    at::Tensor tilingTensor;

    // 2. Hash computation
    CausalConv1dUpdateTilingKey key{
        .batch = batch, .seqLen = seq_len, .dim = dim, .width = width, .stateLen = state_len,
        .hasIndices = has_indices ? 1 : 0, .hasBias = has_bias ? 1 : 0, 
        .hasNumAccept = has_num_accept ? 1 : 0, .hasQueryLoc = has_query_loc ? 1 : 0,
        .activationMode = activation_mode ? 1 : 0, .padSlotId = pad_slot_id
    };
    uint64_t hashValue = CausalConv1dUpdateTilingKeyHash{}(key);

    // 3. 全局缓存管理 (对标 lightning_indexer 逻辑)
    static auto globalTilingBuffer = at::empty({tilingSize * MAX_CAPTURE_NUM},
                                               at::TensorOptions().dtype(at::kByte).device(x.options().device()));

    if (captureMap.find(hashValue) != captureMap.end()) {
        tilingTensor = at::from_blob(globalTilingBuffer.data_ptr<uint8_t>() + (tilingSize * captureMap[hashValue]),
                                     tilingSize, at::kByte);
    } else if (actualCaptureNum >= MAX_CAPTURE_NUM) {
        static auto tilingBuffer =
            at::empty({tilingSize}, at::TensorOptions().dtype(at::kByte).device(x.options().device()));
        aclrtMemcpy(tilingBuffer.data_ptr<uint8_t>(), sizeof(CausalConv1dUpdateTilingData), &tiling_data, sizeof(CausalConv1dUpdateTilingData), ACL_MEMCPY_HOST_TO_DEVICE);
        tilingTensor = at::from_blob(tilingBuffer.data_ptr<uint8_t>(), tilingSize, at::kByte);
    } else {
        captureMap[hashValue] = actualCaptureNum;
        aclrtMemcpy(globalTilingBuffer.data_ptr<uint8_t>() + actualCaptureNum * tilingSize, sizeof(CausalConv1dUpdateTilingData), &tiling_data,
                    sizeof(CausalConv1dUpdateTilingData), ACL_MEMCPY_HOST_TO_DEVICE);
        actualCaptureNum++;
        tilingTensor = at::from_blob(globalTilingBuffer.data_ptr<uint8_t>() + (tilingSize * captureMap[hashValue]),
                                     tilingSize, at::kByte);
    }

    // 4. Create workspace
    auto workspace_tensor = at::empty({workspace_size}, at::TensorOptions().dtype(at::kByte).device(x.options().device()));

    // 5. Launch kernel (注意！直接传入原 Tensor，不要做任何 .contiguous 操作！)
    if (dtype == at::kBFloat16) {
        EXEC_KERNEL_CMD(causal_conv1d_update_bfloat16_t, block_dim, x, weight, conv_state,
                        has_indices ? conv_state_indices : at::empty(0, x.options()),
                        has_bias ? bias : at::empty(0, x.options()),
                        has_num_accept ? num_accepted_tokens : at::empty(0, x.options()),
                        has_query_loc ? query_start_loc : at::empty(0, x.options()),
                        y, workspace_tensor, tilingTensor);
    } else {
        EXEC_KERNEL_CMD(causal_conv1d_update_half, block_dim, x, weight, conv_state,
                        has_indices ? conv_state_indices : at::empty(0, x.options()),
                        has_bias ? bias : at::empty(0, x.options()),
                        has_num_accept ? num_accepted_tokens : at::empty(0, x.options()),
                        has_query_loc ? query_start_loc : at::empty(0, x.options()),
                        y, workspace_tensor, tilingTensor);
    }

    return y;
}

}  // namespace npu_kernel
}  // namespace sglang
