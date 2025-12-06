/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdio>
#include <string>
#include "acl/acl.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/sparse_flash_attention_tiling.h"
#include "defines.h"
#include "torch_helper.h"
#include "ge_helper.h"
#include "common_tiling.h"
#include "sparse_flash_attention_def.h"
#include "common.h"
#include "aclrtlaunch_sparse_flash_attention.h"

using namespace sglang::SFAHost;
using namespace ge_helper;
constexpr uint32_t MAX_CAPTURE_NUM = 1024;
constexpr uint32_t MAX_DECODE_BS = 512;
// npu tensor max size
constexpr int SIZE = 8;
constexpr int DIM_0 = 0;
constexpr int DIM_1 = 1;
constexpr int DIM_2 = 2;
constexpr int DIM_3 = 3;

// namespace scope global parameters
uint32_t actualCaptureNum = 0;
static std::unordered_map<uint64_t, uint32_t> captureMap;
// at::Tensor workspace; 

namespace sglang {
namespace npu_kernel {

inline at::Tensor ConstructSparseFlashAttentionOutputTensor(const at::Tensor &query, const at::Tensor &key,
                                                            const at::Tensor &value, const at::Tensor &sparse_indices,
                                                            const c10::optional<at::Tensor> &actual_seq_lengths_query,
                                                            std::string query_layout_str,
                                                            std::string kv_layout_str)
{
    at::SmallVector<int64_t, SIZE> outputSize;
    for (size_t i = 0; i < query.sizes().size(); i++) {
        TORCH_CHECK(query.size(i) > 0,
                    "All values within query's shape should be greater "
                    "than 0, but shape[",
                    i, "] is ", query.size(i));
    }
    
    // Output shape should match query shape for attention output
    if (query_layout_str == "BSND") {
        outputSize = {query.size(DIM_0), query.size(DIM_1), query.size(DIM_2)};
    } else {
        // TND layout
        outputSize = {query.size(DIM_0), query.size(DIM_1)};
    }
    
    at::Tensor output = at::empty(outputSize, query.options());
    return output;
}

HOST_API ::Tensor sparse_flash_attention(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const at::Tensor &sparse_indices,
    const c10::optional<at::Tensor> &block_table,
    const c10::optional<at::Tensor> &actual_seq_lengths_query,
    const c10::optional<at::Tensor> &actual_seq_lengths_kv,
    const c10::optional<at::Tensor> &query_rope,
    const c10::optional<at::Tensor> &key_rope,
    at::Tensor &attention_out,
    double scale_value,
    int64_t sparse_block_size,
    c10::optional<c10::string_view> layout_query,
    c10::optional<c10::string_view> layout_kv,
    c10::optional<int64_t> sparse_mode)
    
{
    using namespace SFAHost;
    SparseFlashAttention sfa("sparse_flash_attention");
    auto context = std::make_shared<TilingContext>("sparse_flash_attention");
    TORCH_CHECK(context != nullptr, "TilingContext is null");

    std::string layoutQuery(sfa.GetAttr(ATTR_LAYOUT_QUERY_INDEX).GetString());
    std::string layoutKV(sfa.GetAttr(ATTR_LAYOUT_KV_INDEX).GetString());
    float scaleValue = std::any_cast<float>(sfa.GetAttr(ATTR_SCALE_VALUE_INDEX).GetValue());
    int64_t sparseBlockSize = std::any_cast<int32_t>(sfa.GetAttr(ATTR_SPARSE_BLOCK_SIZE_INDEX).GetValue());
    int64_t sparseMode = std::any_cast<int32_t>(sfa.GetAttr(ATTR_SPARSE_MODE_INDEX).GetValue());

    if (layout_query.has_value()) {
        layoutQuery = std::string(layout_query.value());
        sfa.SetAttrStr("layout_query", layoutQuery);
    }
    if (layout_kv.has_value()) {
        layoutKV = std::string(layout_kv.value());
        sfa.SetAttrStr("layout_kv", layoutKV);
    }
    if (sparse_mode.has_value()) {
        sparseMode = sparse_mode.value();
        sfa.SetAttrAny("sparse_mode", static_cast<int32_t>(sparseMode));
    }
    
    // Set scale_value and sparse_block_size from function arguments
    scaleValue = static_cast<float>(scale_value);
    sparseBlockSize = sparse_block_size;
    sfa.SetAttrAny("scale_value", scaleValue);
    sfa.SetAttrAny("sparse_block_size", static_cast<int32_t>(sparseBlockSize));

    // Note: attention_out is passed as input-output parameter, we don't need to construct it
    // but we should validate its shape matches expected output shape
    at::Tensor expected_attention_out = ConstructSparseFlashAttentionOutputTensor(
        query, key, value, sparse_indices, actual_seq_lengths_query, layoutQuery, layoutKV);
    
    TORCH_CHECK(attention_out.sizes() == expected_attention_out.sizes(),
                "attention_out shape mismatch. Expected: ", expected_attention_out.sizes(),
                ", got: ", attention_out.sizes());

    auto qScalarType = query.scalar_type();

    at::Tensor actualSeqLengthsQuery =
        actual_seq_lengths_query.has_value()
            ? actual_seq_lengths_query.value()
            : at::empty({1}, at::TensorOptions().dtype(at::kInt).device(query.options().device()));

    at::Tensor actualSeqLengthsKV =
        actual_seq_lengths_kv.has_value()
            ? actual_seq_lengths_kv.value()
            : at::empty({1}, at::TensorOptions().dtype(at::kInt).device(query.options().device()));

    at::Tensor blockTable =
        block_table.has_value()
            ? block_table.value()
            : at::empty({1}, at::TensorOptions().dtype(at::kInt).device(query.options().device()));

    at::Tensor queryRope =
        query_rope.has_value()
            ? query_rope.value()
            : at::empty({1}, at::TensorOptions().dtype(qScalarType).device(query.options().device()));

    at::Tensor keyRope =
        key_rope.has_value()
            ? key_rope.value()
            : at::empty({1}, at::TensorOptions().dtype(qScalarType).device(query.options().device()));

    sfa.SetToContext(context, qScalarType);
    context->RegisterTensor(query, true);
    context->RegisterTensor(key, true);
    context->RegisterTensor(value, true);
    context->RegisterTensor(sparse_indices, true);
    context->RegisterTensor(block_table, true);
    context->RegisterTensor(actual_seq_lengths_query, true);
    context->RegisterTensor(actual_seq_lengths_kv, true);
    context->RegisterTensor(query_rope, true);
    context->RegisterTensor(key_rope, true);
    context->RegisterTensor(attention_out, false);

    SFATilingInfo sfaInfo;
    SFAInfoParser sfaInfoParser(context.get());
    TORCH_CHECK(sfaInfoParser.Parse(sfaInfo) == ge::GRAPH_SUCCESS, "sparse_flash_attention Parse failed")

    SFAMlaTiling sfaTiling(context.get());
    TORCH_CHECK(sfaTiling.DoOpTiling(&sfaInfo) == ge::GRAPH_SUCCESS, "sparse_flash_attention DoTiling failed")
    const auto &tilingData = sfaTiling.GetTilingData();

    uint32_t tilingSize = sizeof(SparseFlashAttentionTilingDataMla);
    auto blockDim = tilingData.singleCoreParams.usedCoreNum;
    at::Tensor tilingTensor;

    // Create tuple for hashing similar to lightning_indexer
    auto tup = std::make_tuple(tilingData.baseParams.batchSize,
                               tilingData.baseParams.seqSize,
                               tilingData.baseParams.qSeqSize,
                               tilingData.baseParams.blockSize,
                               tilingData.baseParams.maxBlockNumPerBatch,
                               tilingData.baseParams.sparseBlockSize,
                               tilingData.baseParams.sparseMode);
    auto hashValue = host_utils::TupleHasher::Hash(tup);

    static auto globalTilingBuffer = at::empty({tilingSize * MAX_CAPTURE_NUM},
                                               at::TensorOptions().dtype(at::kByte).device(query.options().device()));

    if (captureMap.find(hashValue) != captureMap.end()) {
        // For decode replay phase and part of prefill phase, get cached tiling data from globalTilingBuffer
        tilingTensor = at::from_blob(globalTilingBuffer.data_ptr<uint8_t>() + (tilingSize * captureMap[hashValue]),
                                     tilingSize, at::kByte);
    } else if (actualCaptureNum >= MAX_CAPTURE_NUM) {
        // For tiling hash that not exist in capture map and exceeds MAX_CAPTURE_NUM, reload its' tiling data to NPU
        static auto tilingBuffer =
            at::empty({tilingSize}, at::TensorOptions().dtype(at::kByte).device(query.options().device()));
        aclrtMemcpy(tilingBuffer.data_ptr<uint8_t>(), tilingSize, &tilingData, tilingSize, ACL_MEMCPY_HOST_TO_DEVICE);
        tilingTensor = at::from_blob(tilingBuffer.data_ptr<uint8_t>(), tilingSize, at::kByte);
    } else {
        // Captured tiling cached here
        captureMap[hashValue] = actualCaptureNum;
        aclrtMemcpy(globalTilingBuffer.data_ptr<uint8_t>() + actualCaptureNum * tilingSize, tilingSize, &tilingData,
                    tilingSize, ACL_MEMCPY_HOST_TO_DEVICE);
        actualCaptureNum++;
        tilingTensor = at::from_blob(globalTilingBuffer.data_ptr<uint8_t>() + (tilingSize * captureMap[hashValue]),
                                     tilingSize, at::kByte);
    }

    size_t workspaceSize = context->GetWorkspaceSize();
    auto workspace = at::empty({workspaceSize}, at::TensorOptions().dtype(at::kByte).device(query.options().device()));
    EXEC_KERNEL_CMD(sparse_flash_attention, blockDim, query, key, value, sparse_indices, blockTable,
                    actualSeqLengthsQuery, actualSeqLengthsKV, queryRope, keyRope, attention_out, workspace, tilingTensor);
    return attention_out;
    }
}
}