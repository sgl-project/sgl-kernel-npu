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

/*!
 * \file lightning_indexer_v2.cpp
 * \brief torch.ops.npu.lightning_indexer_v2 的 host 侧入口。
 */

#include <string>
#include <tuple>

#include "acl/acl.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tiling/platform/platform_ascendc.h"

#include "common.h"
#include "defines.h"
#include "ge_helper.h"
#include "torch_helper.h"

#include "lightning_indexer_v2_def.h"
#include "tiling/lightning_indexer_v2_tiling.h"
#include "../op_kernel/lightning_indexer_v2_tiling_key.h"

#include "aclrtlaunch_lightning_indexer_v2.h"

namespace sglang::LIV2Host {

using namespace ge_helper;

constexpr int SIZE = 8;
constexpr int DIM_0 = 0;
constexpr int DIM_1 = 1;
constexpr int DIM_2 = 2;

// tilingKey 的 dtype 段必须与 lightning_indexer_v2_tiling_key.h 中的常量一致。
static_assert(GE_DATATYPE_TO_KEY(ge::DT_FLOAT16) == SGL_LI_V2_DT_FP16, "SGL_LI_V2_DT_FP16 mismatch");
static_assert(GE_DATATYPE_TO_KEY(ge::DT_BF16) == SGL_LI_V2_DT_BF16, "SGL_LI_V2_DT_BF16 mismatch");
static_assert(GE_DATATYPE_TO_KEY(ge::DT_INT32) == SGL_LI_V2_DT_INT32, "SGL_LI_V2_DT_INT32 mismatch");
// layout 段必须与 LICommon::LI_LAYOUT / DataLayout 一致。
static_assert(static_cast<uint32_t>(DataLayout::BSND) == SGL_LI_V2_LAYOUT_BSND, "BSND layout code mismatch");
static_assert(static_cast<uint32_t>(DataLayout::TND) == SGL_LI_V2_LAYOUT_TND, "TND layout code mismatch");
static_assert(static_cast<uint32_t>(DataLayout::BnBsND) == SGL_LI_V2_LAYOUT_PA_BSND, "PA_BSND layout code mismatch");

// 输出 shape 规则与上游 lightning_indexer_torch_adpt.h 保持一致。
inline std::tuple<at::Tensor, at::Tensor> ConstructOutputTensors(const at::Tensor &query, const at::Tensor &key,
                                                                 int64_t sparseCount, const std::string &queryLayout,
                                                                 const std::string &keyLayout, bool returnValues)
{
    for (size_t i = 0; i < query.sizes().size(); i++) {
        TORCH_CHECK(query.size(i) > 0, "All values within query shape should be greater than 0, but shape[", i,
                    "] is ", query.size(i));
    }
    for (size_t i = 0; i < key.sizes().size(); i++) {
        TORCH_CHECK(key.size(i) > 0, "All values within key shape should be greater than 0, but shape[", i, "] is ",
                    key.size(i));
    }
    TORCH_CHECK(sparseCount > 0, "sparse count should be greater than 0, but now is ", sparseCount);

    at::SmallVector<int64_t, SIZE> outputSize;
    if (queryLayout == "BSND") {
        outputSize = {query.size(DIM_0), query.size(DIM_1), key.size(DIM_2), sparseCount};
    } else {
        int nDimIndex = (keyLayout == "TND") ? DIM_1 : DIM_2;
        outputSize = {query.size(DIM_0), key.size(nDimIndex), sparseCount};
    }

    at::Tensor sparseIndices = at::empty(outputSize, query.options().dtype(at::kInt));
    // returnValues 为 false 时 kernel 仍需要一个合法地址，这里给一个空 tensor。
    at::Tensor sparseValues = returnValues ? at::empty(outputSize, query.options()) : at::empty({0}, query.options());
    return {sparseIndices, sparseValues};
}

inline at::Tensor OptionalOrEmpty(const c10::optional<at::Tensor> &tensor, const at::Tensor &like)
{
    return tensor.has_value()
               ? tensor.value()
               : at::empty({1}, at::TensorOptions().dtype(at::kInt).device(like.options().device()));
}

}  // namespace sglang::LIV2Host

namespace sglang {
namespace npu_kernel {

HOST_API std::tuple<at::Tensor, at::Tensor> lightning_indexer_v2(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &weights,
    const c10::optional<at::Tensor> &actual_seq_lengths_query,
    const c10::optional<at::Tensor> &actual_seq_lengths_key, const c10::optional<at::Tensor> &block_table,
    c10::optional<c10::string_view> layout_query, c10::optional<c10::string_view> layout_key,
    c10::optional<int64_t> sparse_count, c10::optional<int64_t> sparse_mode, c10::optional<int64_t> pre_tokens,
    c10::optional<int64_t> next_tokens, c10::optional<bool> return_values)
{
    using namespace LIV2Host;

    TORCH_CHECK(query.numel() > 0, "Tensor query is empty.");
    TORCH_CHECK(key.numel() > 0, "Tensor key is empty.");
    TORCH_CHECK(weights.numel() > 0, "Tensor weights is empty.");

    LightningIndexerV2 indexer("lightning_indexer_v2");
    auto context = std::make_shared<TilingContext>("lightning_indexer_v2");
    TORCH_CHECK(context != nullptr, "TilingContext is null");

    std::string layoutQuery(indexer.GetAttr(ATTR_QUERY_LAYOUT_INDEX).GetString());
    std::string layoutKey(indexer.GetAttr(ATTR_KEY_LAYOUT_INDEX).GetString());
    int64_t sparseCount = std::any_cast<int32_t>(indexer.GetAttr(ATTR_SPARSE_COUNT_INDEX).GetValue());
    bool returnValues = std::any_cast<bool>(indexer.GetAttr(ATTR_RETURN_VALUE_INDEX).GetValue());

    if (layout_query.has_value()) {
        layoutQuery = std::string(layout_query.value());
        indexer.SetAttrStr("layout_query", layoutQuery);
    }
    if (layout_key.has_value()) {
        layoutKey = std::string(layout_key.value());
        indexer.SetAttrStr("layout_key", layoutKey);
    }
    if (sparse_count.has_value()) {
        sparseCount = sparse_count.value();
        indexer.SetAttrAny("sparse_count", static_cast<int32_t>(sparseCount));
    }
    if (sparse_mode.has_value()) {
        indexer.SetAttrAny("sparse_mode", static_cast<int32_t>(sparse_mode.value()));
    }
    if (pre_tokens.has_value()) {
        indexer.SetAttrAny("pre_tokens", static_cast<int64_t>(pre_tokens.value()));
    }
    if (next_tokens.has_value()) {
        indexer.SetAttrAny("next_tokens", static_cast<int64_t>(next_tokens.value()));
    }
    if (return_values.has_value()) {
        returnValues = return_values.value();
        indexer.SetAttrAny("return_values", returnValues);
    }

    auto outputs = ConstructOutputTensors(query, key, sparseCount, layoutQuery, layoutKey, returnValues);
    at::Tensor sparseIndices = std::get<0>(outputs);
    at::Tensor sparseValues = std::get<1>(outputs);

    auto qScalarType = query.scalar_type();
    indexer.SetToContext(context, qScalarType);
    // weights 允许为 fp32 而 query/key 为 fp16/bf16（上游的 DT_W_FLAG 档位），
    // 该组合无法由 SetToContext 的单一下标表达，这里按实际 tensor 修正。
    context->OverrideInputDataType(WEIGTHS_INDEX, SCALAR_TYPE_TO_GE_DATATYPE(weights.scalar_type()));

    context->RegisterTensor(query, true);
    context->RegisterTensor(key, true);
    context->RegisterTensor(weights, true);
    context->RegisterTensor(actual_seq_lengths_query, true);
    context->RegisterTensor(actual_seq_lengths_key, true);
    context->RegisterTensor(block_table, true);
    context->RegisterTensor(sparseIndices, false);
    context->RegisterTensor(sparseValues, false);

    LITilingInfo liInfo;
    LIInfoParser parser(context.get());
    TORCH_CHECK(parser.ParseAndCheck(liInfo) == ge::GRAPH_SUCCESS,
                "lightning_indexer_v2 ParseAndCheck failed: ", LastOpError());

    LightningIndexerTiling liTiling(context.get());
    TORCH_CHECK(liTiling.DoTiling(&liInfo) == ge::GRAPH_SUCCESS,
                "lightning_indexer_v2 DoTiling failed: ", LastOpError());
    const LITilingData &tilingData = liTiling.GetTilingData();

    at::Tensor tilingTensor = context->GetTilingTensor(tilingData);

    at::Tensor actualSeqLengthsQuery = OptionalOrEmpty(actual_seq_lengths_query, query);
    at::Tensor actualSeqLengthsKey = OptionalOrEmpty(actual_seq_lengths_key, query);
    at::Tensor blockTable = OptionalOrEmpty(block_table, query);

    size_t workspaceSize = context->GetWorkspaceSize();
    auto workspace = at::empty({static_cast<int64_t>(workspaceSize)},
                               at::TensorOptions().dtype(at::kByte).device(query.options().device()));

    // EXEC_KERNEL_CMD 会把 blockdim 实参直接展开到 lambda 的捕获列表里，
    // 因此只能传一个普通变量名，不能传 tilingData.usedCoreNum 这样的表达式。
    auto blockDim = tilingData.usedCoreNum;
    EXEC_KERNEL_CMD(lightning_indexer_v2, blockDim, query, key, weights, actualSeqLengthsQuery, actualSeqLengthsKey,
                    blockTable, sparseIndices, sparseValues, workspace, tilingTensor);

    return {sparseIndices, sparseValues};
}

}  // namespace npu_kernel
}  // namespace sglang
