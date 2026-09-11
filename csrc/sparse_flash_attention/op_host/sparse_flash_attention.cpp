/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_flash_attention.cpp
 * \brief Host entry for torch.ops.npu.npu_sparse_flash_attention_lse.
 *
 * Ported from vllm-ascend csrc/attention/sparse_flash_attention. Upstream's entry point
 * (sparse_flash_attention_torch_adpt.h) allocates the outputs and then dispatches through ACLNN with
 * EXEC_NPU_CMD(aclnnSparseFlashAttention, ...), which needs the op registered in a CANN op package.
 * This repo launches AscendC kernels directly, so the ACLNN hop is replaced by the ge_helper route
 * used by csrc/sparse_attn_sharedkv: build a tiling context out of the torch tensors, run the
 * vendored tiling on it, then EXEC_KERNEL_CMD. The output-shape logic below is upstream's, unchanged.
 *
 * WHY THIS EXISTS AT ALL -- do not "simplify" it back onto torch_npu:
 *
 * CANN ships its own build of this operator as torch_npu.npu_sparse_flash_attention, and SGLang's
 * Ascend MLA decode calls it today. That build refuses to return the log-sum-exp under a paged KV
 * layout:
 *
 *   OP_CHECK_IF(*opParamInfo_.returnSoftmaxLse && kvLayout_ == SFALayout::PA_BSND,
 *               ... "When return_softmax_lse is true, layout_kv does not support PA_BSND"),
 *               return ge::GRAPH_FAILED);
 *       -- ops-transformer/.../sparse_flash_attention_tiling.cpp:1994
 *
 * and PA_BSND is its only paged layout, so paged and LSE are mutually exclusive there. Decode Context
 * Parallelism needs a per-rank LSE to weight the cross-rank merge, so that refusal is what blocks it.
 * The vendored build this port comes from has no such clause -- it validates layouts on dimension and
 * pairing only -- and vllm-ascend runs PA_BSND together with return_softmax_lse=True in production
 * (vllm_ascend/device/device_op.py:429-445, feeding the merge at
 * vllm_ascend/attention/context_parallel/sfa_cp.py:1250).
 *
 * Registered under a distinct name so the operator CANN provides is left completely alone: the
 * non-DCP serving path keeps calling torch_npu.npu_sparse_flash_attention and is unaffected by
 * anything here.
 *
 * On the returned LSE: softmax_max and softmax_sum are the two halves of a natural-log LSE, combined
 * by the caller as lse = softmax_max + log(softmax_sum), because softmax_sum is sum(exp(qk - max)).
 */

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <utility>

#include "acl/acl.h"
#include "aclrtlaunch_sparse_flash_attention.h"
#include "ge_helper.h"
#include "torch_helper.h"

#include "sparse_flash_attention_def.h"
#include "sparse_flash_attention_tiling.h"

namespace sglang::npu_kernel {
namespace {

using ge_helper::TilingContext;
using optiling::SFAInfoParser;
using optiling::SFALayout;
using optiling::SFAMlaTiling;
using optiling::SFATilingCheck;
using optiling::SFATilingInfo;
using optiling::SparseFlashAttentionTilingDataMla;

constexpr int64_t SIZE = 8;
constexpr int64_t DIM_0 = 0;
constexpr int64_t DIM_1 = 1;
constexpr int64_t DIM_2 = 2;
constexpr int64_t DIM_3 = 3;

// The dispatch key packs the layout and dtype codes that op_kernel/sparse_flash_attention.cpp
// switches on. These pin the codes to the enums the tiling actually uses, so a future edit to either
// side fails to build instead of silently dispatching to the wrong kernel instantiation.
static_assert(static_cast<uint32_t>(SFALayout::BSND) == SGL_SFA_LAYOUT_BSND, "BSND layout code mismatch");
static_assert(static_cast<uint32_t>(SFALayout::TND) == SGL_SFA_LAYOUT_TND, "TND layout code mismatch");
static_assert(static_cast<uint32_t>(SFALayout::PA_BSND) == SGL_SFA_LAYOUT_PA_BSND, "PA_BSND layout code mismatch");

/*
 * Upstream construct_sparse_flash_attention_output_tensor(), unchanged in behaviour.
 *
 * The softmax_max / softmax_sum shapes are not free: the tiling checks them against the layout it
 * derives from layout_query (SFAInfoParser::GetSoftmaxMaxAndSumLayout -- BSND query gives BNSG,
 * TND query gives NTG), and rejects the call if they disagree. So
 *   TND  query (T, N1, D)     -> softmax (N2, T, G)
 *   BSND query (B, S, N1, D)  -> softmax (B, N2, S, G)
 * with G = N1 / N2. When return_softmax_lse is false the tiling wants genuinely empty tensors.
 */
std::tuple<at::Tensor, at::Tensor, at::Tensor> ConstructOutputTensors(const at::Tensor &query, const at::Tensor &key,
                                                                      const std::string &layoutQuery,
                                                                      const std::string &layoutKv,
                                                                      bool returnSoftmaxLse)
{
    TORCH_CHECK(layoutQuery == "BSND" || layoutQuery == "TND",
                "The layout of query only support BSND and TND, but got ", layoutQuery);
    for (size_t i = 0; i < query.sizes().size(); i++) {
        TORCH_CHECK(query.size(i) > 0, "All values within query's shape should be greater than 0, but shape[", i,
                    "] is ", query.size(i));
    }

    at::SmallVector<int64_t, SIZE> outputSize;
    if (layoutQuery == "TND") {
        TORCH_CHECK(query.dim() == DIM_3, "When the layout of query is TND, the query dimension must be 3, but got ",
                    query.dim());
        outputSize = {query.size(DIM_0), query.size(DIM_1), query.size(DIM_2)};
    } else {
        TORCH_CHECK(query.dim() == DIM_3 + 1,
                    "When the layout of query is BSND, the query dimension must be 4, but got ", query.dim());
        outputSize = {query.size(DIM_0), query.size(DIM_1), query.size(DIM_2), query.size(DIM_3)};
    }

    at::Tensor attentionOut = at::empty(outputSize, query.options().dtype(query.dtype()));

    at::SmallVector<int64_t, SIZE> softmaxSize;
    if (returnSoftmaxLse) {
        if (query.dim() == DIM_3) {
            const auto kvHeadNum = (layoutKv == "PA_BSND") ? key.size(DIM_2) : key.size(DIM_1);
            TORCH_CHECK(kvHeadNum > 0 && query.size(DIM_1) % kvHeadNum == 0, "query head num (", query.size(DIM_1),
                        ") must be a positive multiple of the kv head num (", kvHeadNum, ")");
            softmaxSize = {kvHeadNum, query.size(DIM_0), query.size(DIM_1) / kvHeadNum};
        } else {
            const auto kvHeadNum = key.size(DIM_2);
            TORCH_CHECK(kvHeadNum > 0 && query.size(DIM_2) % kvHeadNum == 0, "query head num (", query.size(DIM_2),
                        ") must be a positive multiple of the kv head num (", kvHeadNum, ")");
            softmaxSize = {query.size(DIM_0), kvHeadNum, query.size(DIM_1), query.size(DIM_2) / kvHeadNum};
        }
    } else {
        softmaxSize = {0};
    }

    at::Tensor softmaxMax = at::empty(softmaxSize, query.options().dtype(at::kFloat));
    at::Tensor softmaxSum = at::empty(softmaxSize, query.options().dtype(at::kFloat));
    return {attentionOut, softmaxMax, softmaxSum};
}

void CheckTensor(const at::Tensor &tensor, const at::Tensor &query, const char *name)
{
    TORCH_CHECK(tensor.device().type() == query.device().type() && tensor.device().index() == query.device().index(),
                name, " must be on the same device as query");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void CheckOptionalTensor(const c10::optional<at::Tensor> &tensor, const at::Tensor &query, const char *name)
{
    if (tensor.has_value()) {
        CheckTensor(*tensor, query, name);
    }
}

// The kernel takes a GM address for every parameter including the absent optionals, so an absent one
// still needs a real allocation to point at. The tiling has already been told it is absent (a
// value-less c10::optional registers as a null gert::Tensor), so nothing reads these.
at::Tensor Placeholder(const at::Tensor &query, at::ScalarType dtype)
{
    return at::empty({1}, query.options().dtype(dtype));
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_sparse_flash_attention_lse(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value, const at::Tensor &sparse_indices,
    double scale_value, const c10::optional<at::Tensor> &block_table,
    const c10::optional<at::Tensor> &actual_seq_lengths_query, const c10::optional<at::Tensor> &actual_seq_lengths_kv,
    const c10::optional<at::Tensor> &query_rope, const c10::optional<at::Tensor> &key_rope, int64_t sparse_block_size,
    c10::string_view layout_query, c10::string_view layout_kv, int64_t sparse_mode, int64_t pre_tokens,
    int64_t next_tokens, int64_t attention_mode, bool return_softmax_lse)
{
    TORCH_CHECK(query.scalar_type() == at::kHalf || query.scalar_type() == at::kBFloat16,
                "npu_sparse_flash_attention_lse: query must be float16 or bfloat16, got ", query.scalar_type());
    TORCH_CHECK(query.numel() > 0, "Tensor query is empty.");
    TORCH_CHECK(key.numel() > 0, "Tensor key is empty.");
    TORCH_CHECK(value.numel() > 0, "Tensor value is empty.");
    TORCH_CHECK(sparse_indices.numel() > 0, "Tensor sparse_indices is empty.");

    CheckTensor(key, query, "key");
    CheckTensor(value, query, "value");
    CheckTensor(sparse_indices, query, "sparse_indices");
    CheckOptionalTensor(block_table, query, "block_table");
    CheckOptionalTensor(actual_seq_lengths_query, query, "actual_seq_lengths_query");
    CheckOptionalTensor(actual_seq_lengths_kv, query, "actual_seq_lengths_kv");
    CheckOptionalTensor(query_rope, query, "query_rope");
    CheckOptionalTensor(key_rope, query, "key_rope");

    const std::string layoutQuery(layout_query);
    const std::string layoutKv(layout_kv);

    auto outputs = ConstructOutputTensors(query, key, layoutQuery, layoutKv, return_softmax_lse);
    at::Tensor attentionOut = std::get<0>(outputs);
    at::Tensor softmaxMax = std::get<1>(outputs);
    at::Tensor softmaxSum = std::get<2>(outputs);

    SFAHost::SparseFlashAttention op("sparse_flash_attention");
    // Attrs have to be set before SetToContext(): that is what snapshots them into the RuntimeAttrs
    // the tiling reads. The int64_t casts are load-bearing -- RuntimeAttrs::GetAttrPointer<T>
    // type-checks with typeid, and the tiling reads all five of these as int64_t.
    op.SetAttrAny("scale_value", static_cast<float>(scale_value));
    op.SetAttrAny("sparse_block_size", static_cast<int64_t>(sparse_block_size));
    op.SetAttrStr("layout_query", layoutQuery);
    op.SetAttrStr("layout_kv", layoutKv);
    op.SetAttrAny("sparse_mode", static_cast<int64_t>(sparse_mode));
    op.SetAttrAny("pre_tokens", static_cast<int64_t>(pre_tokens));
    op.SetAttrAny("next_tokens", static_cast<int64_t>(next_tokens));
    op.SetAttrAny("attention_mode", static_cast<int64_t>(attention_mode));
    op.SetAttrAny("return_softmax_lse", return_softmax_lse);

    auto context = std::make_shared<TilingContext>("sparse_flash_attention");
    auto scalarType = query.scalar_type();
    op.SetToContext(context, scalarType);

    // Registration order must match the input/output order in sparse_flash_attention_def.h: the
    // tiling addresses them by the *_INPUT_INDEX / *_INDEX constants in the tiling header.
    context->RegisterTensor(query, true);
    context->RegisterTensor(key, true);
    context->RegisterTensor(value, true);
    context->RegisterTensor(sparse_indices, true);
    context->RegisterTensor(block_table, true);
    context->RegisterTensor(actual_seq_lengths_query, true);
    context->RegisterTensor(actual_seq_lengths_kv, true);
    context->RegisterTensor(query_rope, true);
    context->RegisterTensor(key_rope, true);
    context->RegisterTensor(attentionOut, false);
    context->RegisterTensor(softmaxMax, false);
    context->RegisterTensor(softmaxSum, false);

    SFATilingInfo info;
    SFAInfoParser parser(context.get());
    TORCH_CHECK(parser.Parse(info) == ge::GRAPH_SUCCESS,
                "npu_sparse_flash_attention_lse: parsing tiling inputs failed: ", optiling::SFALastError());

    SFATilingCheck checker(info);
    TORCH_CHECK(checker.Process() == ge::GRAPH_SUCCESS,
                "npu_sparse_flash_attention_lse: tiling validation failed: ", optiling::SFALastError());

    SFAMlaTiling tiling(context.get());
    TORCH_CHECK(tiling.DoOpTiling(&info) == ge::GRAPH_SUCCESS,
                "npu_sparse_flash_attention_lse: tiling failed: ", optiling::SFALastError());

    const SparseFlashAttentionTilingDataMla &tilingData = tiling.GetTilingData();
    auto tilingTensor = context->GetTilingTensor(tilingData);
    auto workspace =
        at::empty({static_cast<int64_t>(tiling.GetWorkspaceSizeBytes())}, query.options().dtype(at::kByte));

    auto queryPlaceholder = Placeholder(query, query.scalar_type());
    auto intPlaceholder = Placeholder(query, at::kInt);
    auto blockTableLaunch = block_table.value_or(intPlaceholder);
    auto actSeqQLaunch = actual_seq_lengths_query.value_or(intPlaceholder);
    auto actSeqKvLaunch = actual_seq_lengths_kv.value_or(intPlaceholder);
    auto queryRopeLaunch = query_rope.value_or(queryPlaceholder);
    auto keyRopeLaunch = key_rope.value_or(queryPlaceholder);

    // Argument order follows the kernel signature in op_kernel/sparse_flash_attention.cpp, where
    // block_table precedes the two actual_seq_lengths -- the same order as the op definition.
    const uint32_t blockDim = tiling.GetBlockDim();
    EXEC_KERNEL_CMD(sparse_flash_attention, blockDim, query, key, value, sparse_indices, blockTableLaunch,
                    actSeqQLaunch, actSeqKvLaunch, queryRopeLaunch, keyRopeLaunch, attentionOut, softmaxMax, softmaxSum,
                    workspace, tilingTensor);

    return {attentionOut, softmaxMax, softmaxSum};
}

}  // namespace sglang::npu_kernel
