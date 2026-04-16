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

#include <torch/library.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/core/npu/NPUFormat.h"
#include "pytorch_npu_helper.h"
#include "sparse_block_estimate.h"

using namespace at;
namespace {
constexpr int DIM_INDEX_0 = 0;
constexpr int DIM_INDEX_1 = 1;
constexpr int DIM_INDEX_2 = 2;
constexpr int DIM_INDEX_3 = 3;
constexpr std::string_view SPARSEBLOCKESTIMATE_NAME = "aclnnSparseBlockEstimate";
}
std::tuple<at::Tensor, at::Tensor> sparse_block_estimate(
    const at::Tensor &query, const at::Tensor &key,
    c10::OptionalIntArrayRef actual_seq_lengths,
    c10::OptionalIntArrayRef actual_seq_lengths_kv,
    std::string input_layout, int64_t stride, int64_t sparse_size,
    int64_t num_heads, int64_t num_key_value_heads, double scale_value,
    double threshold, bool causal, bool keep_sink, bool keep_recent, double row_sparse)
{
    TORCH_CHECK(num_heads != 0, "num_heads must be nonzero.");
    TORCH_CHECK(sparse_size != 0, "sparse_size must be nonzero.");

    auto actSeqLen = actual_seq_lengths.value_or(at::IntArrayRef{});
    auto actSeqLenKv = actual_seq_lengths_kv.value_or(at::IntArrayRef{});
    const char* inputLayoutPtr = input_layout.c_str();

    int64_t b;
    int64_t nq;
    int64_t s;
    int64_t d;

    if (input_layout == "BNSD") {
        b = query.size(DIM_INDEX_0);
        nq = query.size(DIM_INDEX_1);
        s = query.size(DIM_INDEX_2);
        d = query.size(DIM_INDEX_3);
    } else if (input_layout == "BSND") {
        b = query.size(DIM_INDEX_0);
        nq = query.size(DIM_INDEX_2);
        s = query.size(DIM_INDEX_1);
        d = query.size(DIM_INDEX_3);
    } else if (input_layout == "BSH") {
        b = query.size(DIM_INDEX_0);
        s = query.size(DIM_INDEX_1);
        d = query.size(DIM_INDEX_2) / num_heads;
        nq = num_heads;
    } else {
        std::cerr << "Error: input_layout only support BNSD, BSND, BSH!!!" << std::endl;
    }
    int64_t seqlenSparse = (s + sparse_size - 1) / sparse_size;
    int64_t seqlenSparseAlign32 = (seqlenSparse + 31) / 32 * 32;

    at::Tensor sparse_mask =
        at_npu::native::empty_with_format({b, nq, seqlenSparse, seqlenSparseAlign32},
        query.options().dtype(c10::ScalarType::Char), at_npu::native::get_npu_format(query));

    at::Tensor sparse_count_table =
        at_npu::native::empty_with_format({b, nq, seqlenSparse}, query.options().dtype(c10::ScalarType::Int),
        at_npu::native::get_npu_format(query));

    EXEC_NPU_CMD<SPARSEBLOCKESTIMATE_NAME>(query, key, actSeqLen,
        actSeqLenKv, inputLayoutPtr, stride, sparse_size,
        num_heads, num_key_value_heads, scale_value,
        threshold, causal, keep_sink, keep_recent, row_sparse, sparse_mask, sparse_count_table);

    return std::tuple<at::Tensor, at::Tensor>(sparse_mask, sparse_count_table);
}