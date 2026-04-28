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

#include <string_view>
#include <torch/library.h>
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/core/npu/NPUFormat.h"
#include "pytorch_npu_helper.h"

#include "rainfusionattention.h"

using namespace at;
using npu_preparation = at_npu::native::OpPreparation;
namespace {
constexpr int EXPECTED_TENSOR_DIMENSION = 3;
constexpr std::string_view RAINFUSIONATTENTIONOP_NAME = "aclnnRainFusionAttention";
}
std::tuple<at::Tensor, at::Tensor> rainfusionattention(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &value,
    const at::Tensor &select_idx,
    const at::Tensor &select_num_idx,
    at::IntArrayRef blockshape,
    const c10::optional<at::Tensor> &attn_mask,
    c10::OptionalIntArrayRef actual_seq_qlen,
    c10::OptionalIntArrayRef actual_seq_kvlen,
    const c10::optional<at::Tensor> &block_table,
    std::string q_input_layout,
    std::string kv_input_layout,
    int64_t head_num, int64_t mask_type, double scale,
    int64_t inner_precise, int64_t block_size)
{
    TORCH_CHECK(q_input_layout == "TND" || q_input_layout == "BNSD",
        "q_input_layout only supports 'TND' and 'BNSD' now.");
    TORCH_CHECK(kv_input_layout == "TND" || kv_input_layout == "BNSD",
        "kv_input_layout only supports 'TND' and 'BNSD' now.");
    const at::Tensor& attenMask = c10::value_or_else(attn_mask, [] {return at::Tensor();});
    auto actualSeqLengths = actual_seq_qlen.value_or(at::IntArrayRef{});
    auto actualSeqLengthsKv = actual_seq_kvlen.value_or(at::IntArrayRef{});
    const at::Tensor& blockTable = c10::value_or_else(block_table, [] {return at::Tensor();});

    const char* qlayoutPtr = q_input_layout.data();
    const char* kvlayoutPtr = kv_input_layout.data();

    at::Tensor attentionOut = at_npu::native::empty_with_format(query.sizes(), query.options(),
        at_npu::native::get_npu_format(query));
    at::Tensor softmaxLse = at_npu::native::empty_with_format({query.sizes()[0], query.sizes()[1], query.sizes()[2]},
        query.options(), at_npu::native::get_npu_format(query));
    
    EXEC_NPU_CMD<RAINFUSIONATTENTIONOP_NAME>(query, key, value, select_idx, select_num_idx, blockshape,
        attenMask, actualSeqLengths, actualSeqLengthsKv, blockTable, qlayoutPtr, kvlayoutPtr,
        head_num, mask_type, scale, inner_precise, block_size, attentionOut, softmaxLse);
    return std::tuple<at::Tensor, at::Tensor>(attentionOut, softmaxLse);
}