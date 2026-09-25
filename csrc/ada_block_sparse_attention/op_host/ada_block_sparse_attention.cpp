/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 *
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#include "ada_block_sparse_attention_tiling.h"
#include "sparse_attention_host.h"
#include "aclrtlaunch_ada_block_sparse_attention.h"
#include "defines.h"

namespace sglang::npu_kernel {

HOST_API at::Tensor ada_block_sparse_attention(const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
                                               const at::Tensor &sparse_mask, const at::Tensor &sparse_count_table,
                                               std::string input_layout, int64_t sparse_size, int64_t num_heads,
                                               int64_t num_key_value_heads, double scale_value, bool causal,
                                               int64_t inner_precise, int64_t pre_tokens, int64_t next_tokens,
                                               c10::OptionalIntArrayRef actual_seq_lengths,
                                               c10::OptionalIntArrayRef actual_seq_lengths_kv)
{
    namespace sa = sparse_attention;
    auto platform = sa::platform();
    auto qs = sa::check_input(query, input_layout, num_heads);
    auto ks = sa::check_input(key, input_layout, num_key_value_heads);
    auto vs = sa::check_input(value, input_layout, num_key_value_heads);
    sa::check_pair(query, key, qs, ks);
    sa::check_pair(query, value, qs, vs);
    TORCH_CHECK(key.sizes() == value.sizes(), "key and value shapes must match");
    TORCH_CHECK(std::isfinite(scale_value), "scale_value must be finite");
    int32_t sparseSize = sa::checked_int(sparse_size, "sparse_size");
    TORCH_CHECK(sparseSize >= 128 && sparseSize <= 512 && sparseSize % 128 == 0,
                "sparse_size must be a multiple of 128 in [128, 512]");
    TORCH_CHECK(inner_precise >= 0 && inner_precise <= 3, "inner_precise must be in [0, 3]");
    TORCH_CHECK(pre_tokens >= 0 && pre_tokens <= INT32_MAX && next_tokens >= 0 && next_tokens <= INT32_MAX,
                "pre_tokens and next_tokens must fit in a nonnegative int32");
    const int64_t rows = (qs.sequence + sparse_size - 1) / sparse_size;
    const int64_t columns = ((ks.sequence + sparse_size - 1) / sparse_size + 31) / 32 * 32;
    TORCH_CHECK(sparse_mask.device() == query.device() && sparse_mask.scalar_type() == at::kChar &&
                    sparse_mask.sizes() == at::IntArrayRef({qs.batch, qs.heads, rows, columns}),
                "sparse_mask must be int8 [B, Nq, ceil(Sq/sparse_size), align32(ceil(Skv/sparse_size))]");
    TORCH_CHECK(sparse_count_table.device() == query.device() && sparse_count_table.scalar_type() == at::kInt &&
                    sparse_count_table.sizes() == at::IntArrayRef({qs.batch, qs.heads, rows}),
                "sparse_count_table must be int32 [B, Nq, ceil(Sq/sparse_size)]");
    auto qLengths = sa::check_lengths(actual_seq_lengths, qs);
    auto kvLengths = sa::check_lengths(actual_seq_lengths_kv, ks);
    auto qLengthTensor = sa::length_tensor(qLengths);
    auto kvLengthTensor = sa::length_tensor(kvLengths);
    auto q = sa::normalize(query, input_layout);
    auto k = sa::normalize(key, input_layout);
    auto v = sa::normalize(value, input_layout);
    auto mask = sparse_mask.contiguous();
    auto count = sparse_count_table.contiguous();
    auto output = at::empty_like(q);
    auto qShape = sa::storage_shape(q.sizes());
    auto kShape = sa::storage_shape(k.sizes());
    auto vShape = sa::storage_shape(v.sizes());
    auto maskShape = sa::storage_shape(mask.sizes());
    if (input_layout == "BSND") input_layout = "BSH";

    optiling::AdaBlockSparseAttentionCompileInfo info{};
    info.aicNum = platform->GetCoreNumAic();
    info.aivNum = platform->GetCoreNumAiv();
    info.socShortName = platform->GetSocVersion();
    info.defaultSysWorkspaceSize = platform->GetLibApiWorkSpaceSize();
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, info.ubSize);
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::L1, info.l1Size);
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, info.l0ASize);
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, info.l0BSize);
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, info.l0CSize);
    int32_t heads = sa::checked_int(num_heads, "num_heads");
    int32_t kvHeads = sa::checked_int(num_key_value_heads, "num_key_value_heads");
    int32_t sparseMode = 0;
    uint8_t causalValue = causal;
    float scale = static_cast<float>(scale_value);
    size_t workspaceSize = 0;
    optiling::ContextParamsForBSATiling params{};
    params.opName = "AdaBlockSparseAttention";
    params.queryInputShape = params.outputShape = &qShape;
    params.keyInputShape = &kShape;
    params.valueInputShape = &vShape;
    params.sparseMaskShape = &maskShape;
    params.inputDataType = query.scalar_type() == at::kHalf ? ge::DT_FLOAT16 : ge::DT_BF16;
    params.kDataType = params.vDataType = params.outputDataType = params.inputDataType;
    params.pseShiftDataType = params.maskDataType = params.inputDataType;
    params.deqScaleType = params.deqScale2Type = params.inputDataType;
    params.quantScale2Type = params.quantOffset2Type = ge::DT_FLOAT;
    params.actualSeqenceLengthQ = qLengths.empty() ? nullptr : &qLengthTensor;
    params.actualSeqenceLengthKV = kvLengths.empty() ? nullptr : &kvLengthTensor;
    params.headsNumber = &heads;
    params.numKeyValueHeads = &kvHeads;
    params.sparseSize = &sparseSize;
    params.causal = &causalValue;
    params.sparseMode = &sparseMode;
    params.innerPrecisePtr = &inner_precise;
    params.preToken = &pre_tokens;
    params.nextToken = &next_tokens;
    params.scaleValue = &scale;
    params.layout = input_layout.c_str();
    params.workspaceSize = &workspaceSize;
    params.compileInfoPtr = &info;
    optiling::AdaBlockSparseAttentionTilingData tiling;
    tiling.promptAttentionBaseParams.set_causal(causalValue);
    tiling.promptAttentionBaseParams.set_sparseSize(sparseSize);
    tiling.promptAttentionBaseParams.set_sparseMaskS1(rows);
    tiling.promptAttentionBaseParams.set_sparseMaskS2(columns);
    optiling::AdaBlockSparseAttentionTiling tiler(nullptr);
    uint64_t tilingKey = 7;
    uint32_t blockDim = 0;
    TORCH_CHECK(tiler.RunBigKernelTilingWithParams(params, tilingKey, blockDim, tiling) == ge::GRAPH_SUCCESS,
                "AdaBlockSparseAttention tiling failed");
    tilingKey += 1000000000000000000ULL;
    switch (tilingKey) {
        case 1000000000000101012ULL:
        case 1000000000002101012ULL:
        case 1000000000000001012ULL:
        case 1000000000002001012ULL:
        case 1000000000000111112ULL:
        case 1000000000002111112ULL:
        case 1000000000000011112ULL:
        case 1000000000002011112ULL:
            break;
        default:
            TORCH_CHECK(false, "unsupported AdaBlockSparseAttention tiling key: ", tilingKey);
    }
    TORCH_CHECK(blockDim > 0 && workspaceSize <= static_cast<size_t>(INT64_MAX), "invalid launch configuration");
    auto tilingTensor = sa::device_tiling(tiling);
    auto workspace = at::empty({static_cast<int64_t>(workspaceSize)}, q.options().dtype(at::kByte));
    auto qLengthsDevice = sa::device_lengths(qLengths);
    auto kvLengthsDevice = sa::device_lengths(kvLengths);
    void *qLengthsPtr = sa::data_or_null(qLengthsDevice);
    void *kvLengthsPtr = sa::data_or_null(kvLengthsDevice);
    EXEC_KERNEL_CMD(ada_block_sparse_attention, blockDim, q, k, v, qLengthsPtr, kvLengthsPtr, mask, count, output,
                    workspace, tilingTensor, tilingKey);
    return output.view(query.sizes());
}

}  // namespace sglang::npu_kernel
