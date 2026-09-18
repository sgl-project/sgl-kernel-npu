/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software; you can redistribute it and/or modify it under the terms of
 * CANN Open Software License Agreement Version 2.0 ("the License").
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR
 * FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the
 * full text of the License.
 */

#include "sparse_attention_score.h"

#include <cmath>
#include <tuple>
#include <unordered_map>

#include "acl/acl.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/tilingdata_base.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "torch_helper.h"  // EXEC_KERNEL_CMD
#include "ge_helper.h"     // SCALAR_TYPE_TO_GE_DATATYPE
#include "common.h"        // host_utils::TupleHasher
#include "sparse_attention_score_tiling.h"
#include "aclrtlaunch_sparse_attention_score.h"  // build-generated: ACLRT_LAUNCH_KERNEL + aclrtlaunch_*

namespace sglang {
namespace npu_kernel {

using namespace SAHost;
constexpr uint32_t MAX_CAPTURE_NUM = 1024;

// cuda-graph replay cache: one entry per shape/dtype/soc-derived tiling combo.
// static to avoid linkage clashes with other host ops in the same .so.
static uint32_t actualCaptureNum = 0;
static std::unordered_map<uint64_t, uint32_t> captureMap;
// Stable per-capture tiling storage. The captured H2D copy re-reads this address
// on replay, so it must outlive the call frame (a stack slot would not).
static SATilingData hostTilingStore[MAX_CAPTURE_NUM];

HOST_API at::Tensor sparse_attention_score(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value, const at::Tensor &select_idx,
    const at::Tensor &block_table, const c10::optional<at::Tensor> &select_num_idx,
    const c10::optional<at::Tensor> &q_dequant_scale, const c10::optional<at::Tensor> &k_dequant_scale,
    const c10::optional<at::Tensor> &v_dequant_scale, const c10::optional<at::Tensor> &actual_seq_lengths,
    const c10::optional<at::Tensor> &actual_seq_lengths_kv, int64_t num_key_value_heads, double scale_value,
    int64_t block_size, int64_t top_k, int64_t inner_precise)
{
    // ---- outputs ----
    // attentionOut = query shape; fp8 input -> fp16 output, else same dtype as query.
    at::ScalarType outDtype = (query.scalar_type() == at::kFloat8_e4m3fn) ? at::kHalf : query.scalar_type();
    at::Tensor attentionOut = at::zeros(query.sizes(), query.options().dtype(outDtype));
    // softmaxLse [T,N,1] fp32: the kernel writes it (L0 AllocTensor did this in PTA);
    // not returned -- sglang consumes only attentionOut.
    int64_t totalQ = query.size(0);
    int64_t numHeads = query.size(1);
    at::Tensor softmaxLse =
        at::zeros({totalQ, numHeads, 1}, at::TensorOptions().dtype(at::kFloat).device(query.options().device()));

    // ---- optional inputs: default-empty (caller sanitizes OOB select_idx slots, so
    // unused optionals are never dereferenced by the kernel) ----
    auto devOpts = at::TensorOptions().device(query.options().device());
    at::Tensor selectNumIdx =
        select_num_idx.has_value() ? select_num_idx.value() : at::empty({1}, devOpts.dtype(at::kInt));
    at::Tensor actualSeqLengths =
        actual_seq_lengths.has_value() ? actual_seq_lengths.value() : at::empty({1}, devOpts.dtype(at::kInt));
    at::Tensor actualSeqLengthsKv =
        actual_seq_lengths_kv.has_value() ? actual_seq_lengths_kv.value() : at::empty({1}, devOpts.dtype(at::kInt));
    at::Tensor qDequantScale =
        q_dequant_scale.has_value() ? q_dequant_scale.value() : at::empty({1}, devOpts.dtype(at::kFloat));
    at::Tensor kDequantScale =
        k_dequant_scale.has_value() ? k_dequant_scale.value() : at::empty({1}, devOpts.dtype(at::kFloat));
    at::Tensor vDequantScale =
        v_dequant_scale.has_value() ? v_dequant_scale.value() : at::empty({1}, devOpts.dtype(at::kFloat));

    // ---- fill SAInfo from shapes + attrs + platform (no gert::TilingContext) ----
    SAInfo info;
    // query [T,N,D]
    info.totalQTokens = static_cast<uint32_t>(query.size(0));
    info.numHeads = static_cast<uint32_t>(query.size(1));
    info.embeddingSize = static_cast<uint32_t>(query.size(2));
    // attr kvHeads (may be overridden by select_idx/key shape below, mirroring SASATiling)
    info.kvHeads = static_cast<uint32_t>(num_key_value_heads);
    if (info.kvHeads == 0) {
        info.kvHeads = static_cast<uint32_t>(key.size(2));  // key [blockNum, blockSize, kvHead, D]
    }
    info.blockSize = static_cast<uint32_t>(block_size);
    info.topK = static_cast<uint32_t>(top_k);
    // block_table [batch, maxBlocksPerBatch]
    info.batch = static_cast<uint32_t>(block_table.size(0));
    info.maxBlocksPerBatch = static_cast<uint32_t>(block_table.size(1));
    // select_idx [kvHead, maxQSeqlen, topK] overrides kvHeads/maxQSeqlen/topK (SASATiling behavior)
    info.kvHeads = static_cast<uint32_t>(select_idx.size(0));
    info.maxQSeqlen = static_cast<uint32_t>(select_idx.size(1));
    info.topK = static_cast<uint32_t>(select_idx.size(2));
    // scale
    info.scaleValue = static_cast<float>(scale_value);
    if (info.scaleValue < 1e-9f && info.scaleValue > -1e-9f && info.embeddingSize > 0) {
        info.scaleValue = 1.0f / std::sqrt(static_cast<float>(info.embeddingSize));
    }
    info.innerPrecise = static_cast<uint32_t>(inner_precise);
    info.dataType = SCALAR_TYPE_TO_GE_DATATYPE(query.scalar_type());

    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    info.aicNum = static_cast<uint32_t>(platform->GetCoreNumAic());
    info.libapiSize = static_cast<uint64_t>(platform->GetLibApiWorkSpaceSize());
    info.socVer = static_cast<uint32_t>(platform->GetSocVersion());

    // ---- tiling ----
    SATiling tiling;
    SATilingData tilingData{};
    size_t workspaceSize = 0;
    uint32_t blockDim = 0;
    tiling.DoTiling(info, tilingData, workspaceSize, blockDim);

    // ---- captureMap: cache tiling blob by hash of all tiling-relevant dims ----
    auto tup =
        std::make_tuple(tilingData.batch, tilingData.numHeads, tilingData.kvHeads, tilingData.embeddingSize,
                        tilingData.blockSize, tilingData.topK, tilingData.maxBlocksPerBatch, tilingData.totalQTokens,
                        tilingData.maxQSeqlen, tilingData.tilingKey, tilingData.scaleValue, tilingData.innerPrecise);
    auto hashValue = host_utils::TupleHasher::Hash(tup);

    uint32_t tilingSize = sizeof(SATilingData);
    at::Tensor tilingTensor;
    static auto globalTilingBuffer = at::empty({tilingSize * MAX_CAPTURE_NUM},
                                               at::TensorOptions().dtype(at::kByte).device(query.options().device()));

    if (captureMap.find(hashValue) != captureMap.end()) {
        // replay: cached tiling blob
        tilingTensor = at::from_blob(globalTilingBuffer.data_ptr<uint8_t>() + (tilingSize * captureMap[hashValue]),
                                     tilingSize, at::kByte);
    } else if (actualCaptureNum >= MAX_CAPTURE_NUM) {
        // overflow: per-call H2D copy from a stable static slot (replay-safe).
        static SATilingData overflowStore;
        overflowStore = tilingData;
        static auto tilingBuffer =
            at::empty({tilingSize}, at::TensorOptions().dtype(at::kByte).device(query.options().device()));
        aclrtMemcpy(tilingBuffer.data_ptr<uint8_t>(), tilingSize, &overflowStore, tilingSize,
                    ACL_MEMCPY_HOST_TO_DEVICE);
        tilingTensor = at::from_blob(tilingBuffer.data_ptr<uint8_t>(), tilingSize, at::kByte);
    } else {
        // first sight: persist tiling in a stable per-index slot, then H2D copy.
        // The captured copy re-reads this slot on replay (not a stale stack frame).
        captureMap[hashValue] = actualCaptureNum;
        hostTilingStore[actualCaptureNum] = tilingData;
        aclrtMemcpy(globalTilingBuffer.data_ptr<uint8_t>() + actualCaptureNum * tilingSize, tilingSize,
                    &hostTilingStore[actualCaptureNum], tilingSize, ACL_MEMCPY_HOST_TO_DEVICE);
        actualCaptureNum++;
        tilingTensor = at::from_blob(globalTilingBuffer.data_ptr<uint8_t>() + (tilingSize * captureMap[hashValue]),
                                     tilingSize, at::kByte);
    }

    // launch: zero-init workspace + softmaxLse (kernel may read-before-write them);
    // at::empty would feed stale recycled-memory garbage.
    auto workspace = at::zeros({workspaceSize}, at::TensorOptions().dtype(at::kByte).device(query.options().device()));
    // 15 GM args, kernel entry order: q,k,v,select_idx,block_table,selectNumIdx,
    // actualSeqLens[+kv], q/k/vDequantScale, out,softmaxLse,workspace,tiling.
    EXEC_KERNEL_CMD(sparse_attention_score, blockDim, query, key, value, select_idx, block_table, selectNumIdx,
                    actualSeqLengths, actualSeqLengthsKv, qDequantScale, kDequantScale, vDequantScale, attentionOut,
                    softmaxLse, workspace, tilingTensor);
    return attentionOut;
}

}  // namespace npu_kernel
}  // namespace sglang
