/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#include <cstdio>
#include <string>
#include "acl/acl.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/minimax_indexer_tiling.h"
#include "defines.h"
#include "torch_helper.h"
#include "ge_helper.h"
#include "common_tiling.h"
#include "minimax_indexer_def.h"
#include "common.h"
#include "aclrtlaunch_minimax_indexer.h"

namespace sglang::MIHost {

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
static uint32_t actualCaptureNum = 0;  // static: avoids linkage conflict with lightning_indexer.cpp
static std::unordered_map<uint64_t, uint32_t> captureMap;
// at::Tensor workspace;

inline at::Tensor ConstructMinimaxIndexerOutputTensor(const at::Tensor &query, const at::Tensor &key,
                                                      const c10::optional<at::Tensor> &actual_seq_lengths_query,
                                                      int64_t sparse_count, int64_t append_local,
                                                      std::string query_layout_str, std::string key_layout_str)
{
    at::SmallVector<int64_t, SIZE> outputSize;
    for (size_t i = 0; i < query.sizes().size(); i++) {
        TORCH_CHECK(query.size(i) > 0,
                    "All values within query's shape should be greater "
                    "than 0, but shape[",
                    i, "] is ", query.size(i));
    }
    TORCH_CHECK(sparse_count > 0, "sparse count should be greater than 0, but now is ", sparse_count);

    if (query_layout_str == "BSND") {
        // MiniMax indexer: per query-head topk -> [B, S1, N1, topk(+append_local)].
        // Output is [B, S1, N1, topk] (lightning_indexer produced [B, S1, N2, sparse_count];
        // the GQA group; MiniMax keeps every query head independent.) With the
        // fused append the width is topk+1 and the kernel emits the [QH, B, ..]
        // memory layout the GQA sparse-attention kernels consume directly.
        outputSize = {query.size(DIM_0), query.size(DIM_1), query.size(DIM_2), sparse_count + append_local};
    } else {
        int n_dim_index = 0;
        n_dim_index = (key_layout_str == "TND") ? DIM_1 : DIM_2;
        outputSize = {query.size(DIM_0), key.size(n_dim_index), sparse_count + append_local};
    }
    at::Tensor output = at::zeros(outputSize, query.options().dtype(at::kInt));

    return output;
}
}  // namespace sglang::MIHost

namespace sglang {
namespace npu_kernel {
HOST_API at::Tensor minimax_indexer(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &weights,
    const c10::optional<at::Tensor> &actual_seq_lengths_query, const c10::optional<at::Tensor> &actual_seq_lengths_key,
    const c10::optional<at::Tensor> &block_table, c10::optional<c10::string_view> layout_query,
    c10::optional<c10::string_view> layout_key, c10::optional<int64_t> sparse_count, c10::optional<int64_t> sparse_mode,
    c10::optional<int64_t> init_blocks, c10::optional<int64_t> local_blocks, c10::optional<double> sm_scale,
    const c10::optional<at::Tensor> &req_to_token, const c10::optional<at::Tensor> &req_pool_indices,
    c10::optional<int64_t> append_local, c10::optional<int64_t> packed_mode)
{
    using namespace MIHost;
    MinimaxIndexer indexer("minimax_indexer");
    auto context = std::make_shared<TilingContext>("minimax_indexer");
    TORCH_CHECK(context != nullptr, "TilingContext is null");

    std::string layoutQuery(indexer.GetAttr(ATTR_QUERY_LAYOUT_INDEX).GetString());
    std::string layoutKey(indexer.GetAttr(ATTR_KEY_LAYOUT_INDEX).GetString());
    int64_t sparseCount = std::any_cast<int32_t>(indexer.GetAttr(ATTR_SPARSE_COUNT_INDEX).GetValue());

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

    int64_t appendLocal = append_local.value_or(0);
    TORCH_CHECK(appendLocal >= 0 && appendLocal <= 1, "append_local must be 0 or 1, got ", appendLocal);
    int64_t packedMode = packed_mode.value_or(0);
    TORCH_CHECK(packedMode >= 0 && packedMode <= 1, "packed_mode must be 0 or 1, got ", packedMode);
    bool directMode = req_to_token.has_value() || req_pool_indices.has_value();
    TORCH_CHECK(!directMode || (req_to_token.has_value() && req_pool_indices.has_value()),
                "req_to_token and req_pool_indices must be provided together for direct mode");

    at::Tensor sparse_indices = ConstructMinimaxIndexerOutputTensor(query, key, actual_seq_lengths_query, sparseCount,
                                                                    appendLocal, layoutQuery, layoutKey);

    auto qScalarType = query.scalar_type();

    at::Tensor actualSeqLengthsQuery =
        actual_seq_lengths_query.has_value()
            ? actual_seq_lengths_query.value()
            : at::empty({1}, at::TensorOptions().dtype(qScalarType).device(query.options().device()));

    at::Tensor actualSeqLengthsKey =
        actual_seq_lengths_key.has_value()
            ? actual_seq_lengths_key.value()
            : at::empty({1}, at::TensorOptions().dtype(qScalarType).device(query.options().device()));

    at::Tensor blockTable;
    if (block_table.has_value()) {
        blockTable = block_table.value();
    } else if (directMode) {
        // Direct mode: the kernel gathers block ids from req_to_token in-kernel, but
        // the parser derives maxBlockNumPerBatch from block_table.shape[1], so hand
        // it a shape-valid dummy ([B, max_blocks] zeros; never read by the kernel).
        int64_t dummyMaxBlocks = req_to_token.value().size(1) / key.size(1);
        blockTable = at::zeros({query.size(0), dummyMaxBlocks},
                               at::TensorOptions().dtype(at::kInt).device(query.options().device()));
    } else {
        blockTable = at::empty({1}, at::TensorOptions().dtype(qScalarType).device(query.options().device()));
    }
    at::Tensor reqToToken = req_to_token.has_value()
                                ? req_to_token.value()
                                : at::empty({1}, at::TensorOptions().dtype(at::kInt).device(query.options().device()));
    at::Tensor reqPoolIdx = req_pool_indices.has_value()
                                ? req_pool_indices.value()
                                : at::empty({1}, at::TensorOptions().dtype(at::kInt).device(query.options().device()));

    indexer.SetToContext(context, qScalarType);
    context->RegisterTensor(query, true);
    context->RegisterTensor(key, true);
    context->RegisterTensor(weights, true);
    context->RegisterTensor(actual_seq_lengths_query, true);
    context->RegisterTensor(actual_seq_lengths_key, true);
    context->RegisterTensor(blockTable, true);
    context->RegisterTensor(reqToToken, true);
    context->RegisterTensor(reqPoolIdx, true);
    context->RegisterTensor(sparse_indices, false);

    MITilingInfo miInfo;
    MIInfoParser MIInfoParser(context.get());
    TORCH_CHECK(MIInfoParser.ParseAndCheck(miInfo) == ge::GRAPH_SUCCESS, "minimax_indexer ParseAndCheck failed");

    MinimaxIndexerTiling liTiling(context.get());
    liTiling.DoTiling(&miInfo);
    const auto &tilingDataBase = liTiling.GetTilingData();

    // MiniMax extras are injected post-tiling (not via the OpDef attr parser).
    // sm_scale is folded with log2(e) once on host; the kernel applies it to the
    // max-reduced block score in fp32 (max commutes with a positive scale).
    MITilingData tilingData = tilingDataBase;
    tilingData.initBlocks = static_cast<uint32_t>(init_blocks.value_or(0));
    tilingData.localBlocks = static_cast<uint32_t>(local_blocks.value_or(0));
    double smScaleVal = sm_scale.value_or(1.0);
    tilingData.smScaleLog2e = static_cast<float>(smScaleVal * 1.4426950408889634);
    // Fused-interface extras: in-kernel direct page lookup + causal-local append.
    tilingData.directMode = directMode ? 1U : 0U;
    tilingData.maxTokenSlots = directMode ? static_cast<uint32_t>(req_to_token.value().size(1)) : 0U;
    tilingData.appendLocal = static_cast<uint32_t>(appendLocal);
    // packed_mode: actual_seq_lengths_key carries the full [B*gqa] per-row causal
    // lengths (EAGLE3 verify packed draft queries).
    tilingData.packedMode = static_cast<uint32_t>(packedMode);

    uint32_t tilingSize = sizeof(MITilingData);
    auto blockDim = tilingData.usedCoreNum;
    auto bs = tilingData.bSize;
    if (std::getenv("SGLANG_MINIMAX_NPU_IDX_TILING_DUMP")) {
        // DIAG: dump the tiling actually built for THIS call so we can cross-check
        // against the installed .so / isolate the all-zero service bug.
        std::fprintf(stderr,
                     "[IDXTILE] bs=%u g=%u s1=%u s2=%u sC=%u core=%u blkBlk=%u maxBnb=%u spMode=%u "
                     "tkey=%08x init=%u loc=%u smScaleLog2e=%f direct=%u mts=%u ap=%u pack=%u\n",
                     tilingData.bSize, tilingData.gSize, tilingData.s1Size, tilingData.s2Size, tilingData.sparseCount,
                     tilingData.usedCoreNum, tilingData.blockSize, tilingData.maxBlockNumPerBatch,
                     tilingData.sparseMode, tilingData.tilingKey, tilingData.initBlocks, tilingData.localBlocks,
                     tilingData.smScaleLog2e, tilingData.directMode, tilingData.maxTokenSlots, tilingData.appendLocal,
                     tilingData.packedMode);
    }
    at::Tensor tilingTensor;

    auto tup =
        std::make_tuple(tilingData.bSize, tilingData.n2Size, tilingData.gSize, tilingData.s1Size, tilingData.s2Size,
                        tilingData.blockSize, tilingData.maxBlockNumPerBatch, tilingData.tilingKey,
                        tilingData.sparseCount, tilingData.initBlocks, tilingData.localBlocks, tilingData.directMode,
                        tilingData.maxTokenSlots, tilingData.appendLocal);
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
    // WORKSPACE MUST BE ZEROED EVERY CALL: the kernel's LD merge reads EVERY core's
    // per-head topk16 partial region (including rows a given core never computes) and
    // Zero-init: deterministic clean backstop.
    // GM (a next launch's AIV can read the previous launch's stale WS rows). at::empty
    // hands back uninitialized caching-allocator bytes; at::zeros gives every row a
    // deterministic clean init (-1 blocks) so an (over-)early LD merge reads a benign
    // "all -1" instead of garbage ~1e9 block ids (was observed as sporadic empty rows
    // Zero-init: deterministic clean backstop.
    auto workspace = at::zeros({workspaceSize}, at::TensorOptions().dtype(at::kByte).device(query.options().device()));
    EXEC_KERNEL_CMD(minimax_indexer, blockDim, query, key, weights, actualSeqLengthsQuery, actualSeqLengthsKey,
                    blockTable, reqToToken, reqPoolIdx, sparse_indices, workspace, tilingTensor);
    if (std::getenv("SGLANG_MINIMAX_NPU_WS_DUMP2")) {
        // DIAG: sync + inspect native indexer output/WS right after launch to see if
        // the kernel actually wrote output (all-zero output => AIV never ran).
        aclrtSynchronizeStream(c10_npu::getCurrentNPUStream().stream(false));
        auto out_cpu = sparse_indices.cpu().contiguous();
        const int32_t *op = out_cpu.data_ptr<int32_t>();
        int64_t n = out_cpu.numel(), nz = 0;
        for (int64_t i = 0; i < n; i++)
            nz += (op[i] != 0) ? 1 : 0;
        auto ws_cpu = workspace.cpu().contiguous();
        const uint8_t *wp = ws_cpu.data_ptr<uint8_t>();
        const int32_t *wpi = reinterpret_cast<const int32_t *>(wp);
        int64_t wn = ws_cpu.numel() / 4, wnz = 0;
        for (int64_t i = 0; i < wn; i++)
            wnz += (wpi[i] != 0) ? 1 : 0;
        std::fprintf(stderr, "[WSDUMP2] bs=%u core=%u(outSize=%lld) output_nz=%lld/%lld WS_nz=%lld/%lld first_out=%d\n",
                     tilingData.bSize, tilingData.usedCoreNum, (long long)out_cpu.numel(), (long long)nz, (long long)n,
                     (long long)wnz, (long long)wn, op[0]);
    }
    return sparse_indices;
}
}  // namespace npu_kernel
}  // namespace sglang
