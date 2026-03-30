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
#include <functional>
#include <mutex>
#include <unordered_map>
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

namespace {

constexpr uint32_t PADDING_BYTE = 32U;
// Keep the cache capacity aligned with lightning_indexer so graph-mode memory stays predictable.
constexpr uint32_t MAX_CAPTURE_NUM = 1024;

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

class CausalConv1dUpdateTilingCache {
public:
    at::Tensor GetOrCreate(
        const at::Tensor& reference_tensor,
        int64_t tiling_size,
        uint64_t hash_value,
        const CausalConv1dUpdateTilingData& tiling_data)
    {
        const auto options = at::TensorOptions().dtype(at::kByte).device(reference_tensor.device());
        std::lock_guard<std::mutex> lock(mutex_);
        ResetIfNeeded(reference_tensor.device(), tiling_size, options);

        auto it = capture_map_.find(hash_value);
        if (it != capture_map_.end()) {
            return GetSlice(it->second, tiling_size);
        }

        if (actual_capture_num_ >= MAX_CAPTURE_NUM) {
            at::Tensor tiling_tensor = at::empty({tiling_size}, options);
            CopyTilingData(tiling_tensor, tiling_data, tiling_size);
            return tiling_tensor;
        }

        const uint32_t slot = actual_capture_num_++;
        capture_map_.emplace(hash_value, slot);
        at::Tensor tiling_tensor = GetSlice(slot, tiling_size);
        CopyTilingData(tiling_tensor, tiling_data, tiling_size);
        return tiling_tensor;
    }

private:
    static void CopyTilingData(
        const at::Tensor& tiling_tensor,
        const CausalConv1dUpdateTilingData& tiling_data,
        int64_t tiling_size)
    {
        const aclError copy_status = aclrtMemcpy(
            tiling_tensor.data_ptr<uint8_t>(),
            static_cast<size_t>(tiling_size),
            &tiling_data,
            sizeof(CausalConv1dUpdateTilingData),
            ACL_MEMCPY_HOST_TO_DEVICE);
        TORCH_CHECK(
            copy_status == ACL_SUCCESS,
            "aclrtMemcpy failed for causal_conv1d_update tiling data, error code: ",
            static_cast<int>(copy_status));
    }

    at::Tensor GetSlice(uint32_t slot, int64_t tiling_size) const
    {
        return global_tiling_buffer_.narrow(0, static_cast<int64_t>(slot) * tiling_size, tiling_size);
    }

    void ResetIfNeeded(const c10::Device& device, int64_t tiling_size, const at::TensorOptions& options)
    {
        if (global_tiling_buffer_.defined() && global_tiling_buffer_.device() == device && tiling_size_ == tiling_size) {
            return;
        }

        global_tiling_buffer_ = at::empty({tiling_size * MAX_CAPTURE_NUM}, options);
        capture_map_.clear();
        actual_capture_num_ = 0;
        tiling_size_ = tiling_size;
    }

    std::mutex mutex_;
    at::Tensor global_tiling_buffer_;
    std::unordered_map<uint64_t, uint32_t> capture_map_;
    uint32_t actual_capture_num_ = 0;
    int64_t tiling_size_ = 0;
};

CausalConv1dUpdateTilingCache& GetCausalConv1dUpdateTilingCache()
{
    static CausalConv1dUpdateTilingCache cache;
    return cache;
}

}  // namespace

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

    // 1. Prepare Tiling Data Struct
    CausalConv1dUpdateTilingData tiling_data;
    SGLang::CausalConv1dUpdate::ComputeTilingData(
        batch, seq_len, dim, width, state_len,
        has_indices, has_bias, has_num_accept, has_query_loc,
        activation_mode, pad_slot_id, block_dim,
        tiling_data
    );

    const int64_t tiling_size =
        (sizeof(CausalConv1dUpdateTilingData) + PADDING_BYTE - 1) / PADDING_BYTE * PADDING_BYTE;

    // 2. Hash computation
    CausalConv1dUpdateTilingKey key{
        .batch = batch, .seqLen = seq_len, .dim = dim, .width = width, .stateLen = state_len,
        .hasIndices = has_indices ? 1 : 0, .hasBias = has_bias ? 1 : 0, 
        .hasNumAccept = has_num_accept ? 1 : 0, .hasQueryLoc = has_query_loc ? 1 : 0,
        .activationMode = activation_mode ? 1 : 0, .padSlotId = pad_slot_id
    };
    uint64_t hashValue = CausalConv1dUpdateTilingKeyHash{}(key);

    // Reuse graph-safe tiling storage when possible, but never grow past the fixed cache budget.
    at::Tensor tilingTensor =
        GetCausalConv1dUpdateTilingCache().GetOrCreate(x, tiling_size, hashValue, tiling_data);

    // 4. Create workspace
    auto workspace_tensor = at::empty({workspace_size}, at::TensorOptions().dtype(at::kByte).device(x.options().device()));

    // 5. Launch kernel
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
