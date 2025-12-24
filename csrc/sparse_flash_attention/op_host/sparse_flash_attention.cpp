/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
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
using namespace ge_helper;
inline at::Tensor ConstructSparseFlashAttentionOutputTensor(const at::Tensor &query)
{
    for (size_t i = 0; i < query.sizes().size(); i++) {
        TORCH_CHECK(query.sizes()[i] > 0, "sparse_flash_attention: query tensor shape invalid");
    };
    at::Tensor output = at::empty(query.sizes(), query.options());
    return output;
}

// Helper function to set attributes for SparseFlashAttention
inline void SetSparseFlashAttentionAttributes(SFAHost::SparseFlashAttention &sfa, double scale_value,
                                              int64_t sparse_block_size, c10::optional<c10::string_view> layout_query,
                                              c10::optional<c10::string_view> layout_kv,
                                              c10::optional<int64_t> sparse_mode)
{
    std::string layoutQuery = sfa.GetAttr(LAYOUT_QUERY_ATTR_INDEX).GetString();
    std::string layoutKV = sfa.GetAttr(LAYOUT_KV_ATTR_INDEX).GetString();

    sfa.SetAttrAny("scale_value", static_cast<float>(scale_value));
    sfa.SetAttrAny("sparse_block_size", static_cast<int64_t>(sparse_block_size));

    if (layout_query.has_value()) {
        layoutQuery = std::string(layout_query.value());
        sfa.SetAttrStr("layout_query", layoutQuery);
    }
    if (layout_kv.has_value()) {
        layoutKV = std::string(layout_kv.value());
        sfa.SetAttrStr("layout_kv", layoutKV);
    }

    sfa.SetAttrAny("sparse_mode", static_cast<int64_t>(sparse_mode.value_or(3)));
}

// Helper function to get optional tensor or empty tensor
inline at::Tensor GetOptionalTensor(const c10::optional<at::Tensor> &tensor, const at::Device &device,
                                    c10::ScalarType dtype = at::kInt)
{
    return tensor.has_value() ? tensor.value() : at::empty({0}, at::TensorOptions().dtype(dtype).device(device));
}

// Helper function to register all tensors to context
inline void RegisterSparseFlashAttentionTensors(std::shared_ptr<TilingContext> &context, const at::Tensor &query,
                                                const at::Tensor &key, const at::Tensor &value,
                                                const at::Tensor &sparse_indices,
                                                const c10::optional<at::Tensor> &blockTable,
                                                const at::Tensor &actuaSeqQ, const at::Tensor &actuaSeqKV,
                                                const at::Tensor &queryRope, const at::Tensor &keyRope,
                                                const at::Tensor &attention_out)
{
    context->RegisterTensor(query, true);
    context->RegisterTensor(key, true);
    context->RegisterTensor(value, true);
    context->RegisterTensor(sparse_indices, true);
    context->RegisterTensor(blockTable, true);
    context->RegisterTensor(actuaSeqQ, true);
    context->RegisterTensor(actuaSeqKV, true);
    context->RegisterTensor(queryRope, true);
    context->RegisterTensor(keyRope, true);
    context->RegisterTensor(attention_out, false);
}

inline at::Tensor GetWorkspaceTensor(const std::shared_ptr<TilingContext> &context, const at::Device &device)
{
    size_t workspaceSize = context->GetWorkspaceSize();
    if (workspaceSize > 0) {
        return at::empty({static_cast<int64_t>(workspaceSize)}, at::TensorOptions().dtype(at::kByte).device(device));
    } else {
        return at::empty({0}, at::TensorOptions().dtype(at::kByte).device(device));
    }
}

inline at::Tensor GetSFATilingTensor(const SparseFlashAttentionTilingDataMla &tilingData, const at::Device &device)
{
    uint32_t tilingSize = sizeof(SparseFlashAttentionTilingDataMla);
    auto tup = std::make_tuple(
        tilingData.baseParams.batchSize, tilingData.baseParams.blockSize, tilingData.baseParams.maxBlockNumPerBatch,
        tilingData.baseParams.scaleValue, tilingData.baseParams.nNumOfQInOneGroup, tilingData.baseParams.outputLayout,
        tilingData.baseParams.sparseMode, tilingData.baseParams.sparseBlockSize, tilingData.baseParams.sparseBlockCount,
        tilingData.splitKVParams.s2, tilingData.tilingKey);
    auto hashValue = host_utils::TupleHasher::Hash(tup);
    static auto globalTilingBuffer =
        at::empty({tilingSize * MAX_CAPTURE_NUM}, at::TensorOptions().dtype(at::kByte).device(device));
    at::Tensor tilingTensor;
    if (captureMap.find(hashValue) != captureMap.end()) {
        // For decode replay phase and part of prefill phase, get cached tiling data from globalTilingBuffer
        tilingTensor = at::from_blob(globalTilingBuffer.data_ptr<uint8_t>() + (tilingSize * captureMap[hashValue]),
                                     tilingSize, at::kByte);
    } else if (actualCaptureNum >= MAX_CAPTURE_NUM) {
        // For tiling hash that not exist in capture map and exceeds MAX_CAPTURE_NUM, reload its' tiling data to NPU
        static auto tilingBuffer = at::empty({tilingSize}, at::TensorOptions().dtype(at::kByte).device(device));
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
    return tilingTensor;
}

// Helper function to perform tiling operations
inline void PerformSparseFlashAttentionTiling(std::shared_ptr<TilingContext> &context, SFATilingInfo &sfaInfo,
                                              SFAMlaTiling &sfaTiling)
{
    SFAInfoParser sfaInfoParser(context.get());
    auto parseRet = sfaInfoParser.Parse(sfaInfo);
    TORCH_CHECK(parseRet == ge::GRAPH_SUCCESS, "sparse_flash_attention Parse failed");

    SFATilingCheck tilingchecker(sfaInfo);
    TORCH_CHECK(tilingchecker.Process() == ge::GRAPH_SUCCESS, "sparse_flash_attention TilingCheck failed");

    auto tilingRet = sfaTiling.DoOpTiling(&sfaInfo);
    TORCH_CHECK(tilingRet == ge::GRAPH_SUCCESS, "sparse_flash_attention DoTiling failed");
}

HOST_API at::Tensor sparse_flash_attention(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value, const at::Tensor &sparse_indices,
    double scale_value, int64_t sparse_block_size, const c10::optional<at::Tensor> &block_table,
    const c10::optional<at::Tensor> &actual_seq_lengths_query, const c10::optional<at::Tensor> &actual_seq_lengths_kv,
    const c10::optional<at::Tensor> &query_rope, const c10::optional<at::Tensor> &key_rope,
    c10::optional<c10::string_view> layout_query, c10::optional<c10::string_view> layout_kv,
    c10::optional<int64_t> sparse_mode)
{
    using namespace SFAHost;

    SparseFlashAttention sfa("sparse_flash_attention");

    SetSparseFlashAttentionAttributes(sfa, scale_value, sparse_block_size, layout_query, layout_kv, sparse_mode);

    auto context = std::make_shared<TilingContext>("sparse_flash_attention");
    TORCH_CHECK(context != nullptr, "TilingContext is null");

    auto qScalarType = query.scalar_type();
    sfa.SetToContext(context, qScalarType);

    at::Tensor attention_out = ConstructSparseFlashAttentionOutputTensor(query);
    auto device_opt = query.options().device();

    at::Tensor actuaSeqQ = GetOptionalTensor(actual_seq_lengths_query, device_opt);
    at::Tensor actuaSeqKV = GetOptionalTensor(actual_seq_lengths_kv, device_opt);
    at::Tensor queryRope = GetOptionalTensor(query_rope, device_opt);
    at::Tensor keyRope = GetOptionalTensor(key_rope, device_opt);
    at::Tensor blockTable = GetOptionalTensor(block_table, device_opt);

    RegisterSparseFlashAttentionTensors(context, query, key, value, sparse_indices, blockTable, actuaSeqQ, actuaSeqKV,
                                        queryRope, keyRope, attention_out);

    SFATilingInfo sfaInfo;
    SFAMlaTiling sfaTiling(context.get());
    PerformSparseFlashAttentionTiling(context, sfaInfo, sfaTiling);

    const auto &tilingData = sfaTiling.GetTilingData();

    at::Tensor workspace = GetWorkspaceTensor(context, device_opt);
    at::Tensor tilingTensor = GetSFATilingTensor(tilingData, device_opt);

    auto blockdim = tilingData.singleCoreParams.usedCoreNum;
    EXEC_KERNEL_CMD(sparse_flash_attention, blockdim, query, key, value, sparse_indices, blockTable, actuaSeqQ,
                    actuaSeqKV, queryRope, keyRope, attention_out, workspace, tilingTensor);

    return attention_out;
}
}  // namespace npu_kernel
}  // namespace sglang
