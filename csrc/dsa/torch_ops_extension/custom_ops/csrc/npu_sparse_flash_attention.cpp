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

#include <iostream>
#include <torch/library.h>
#include "ops_common.h"

namespace custom {
using namespace at_npu::native;

// 工具函数，推导输出shape
at::Tensor construct_sparse_infer_output_tensor(const at::Tensor &query, const at::Tensor &value, std::string layout)
{
    at::SmallVector<int64_t, 8> output_size;
    if (layout == "TND") {
        output_size = {query.size(DIM_0), query.size(DIM_1), query.size(DIM_2)};
    } else {
        output_size = {query.size(DIM_0), query.size(DIM_1), query.size(DIM_2), query.size(DIM_3)};
    }
    at::Tensor output = at::empty(output_size, query.options().dtype(query.dtype()));

    return output;
}

// step2, 为NPU设备实现前向接口
at::Tensor npu_sparse_flash_attention_npu(const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
                                          const at::Tensor &sparse_indices, double scale_value,
                                          int64_t sparse_block_size, const c10::optional<at::Tensor> &block_table,
                                          const c10::optional<at::Tensor> &actual_seq_lengths_query,
                                          const c10::optional<at::Tensor> &actual_seq_lengths_kv,
                                          const c10::optional<at::Tensor> &query_rope,
                                          const c10::optional<at::Tensor> &key_rope, c10::string_view layout_query,
                                          c10::string_view layout_kv, int64_t sparse_mode)
{
    std::string layout_query_str = std::string(layout_query);
    std::string layout_kv_str = std::string(layout_kv);

    // construct the output tensor
    at::Tensor output = construct_sparse_infer_output_tensor(query, value, layout_query_str);
    // convert str
    char *layout_query_ptr = const_cast<char *>(layout_query_str.c_str());
    char *layout_kv_ptr = const_cast<char *>(layout_kv_str.c_str());

    EXEC_NPU_CMD_V1(aclnnSparseFlashAttention, query, key, value, sparse_indices, block_table, actual_seq_lengths_query,
                    actual_seq_lengths_kv, query_rope, key_rope, scale_value, sparse_block_size, layout_query_ptr,
                    layout_kv_ptr, sparse_mode, output);
    return output;
}

// step3, 为META设备实现前向接口
at::Tensor npu_sparse_flash_attention_meta(const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
                                           const at::Tensor &sparse_indices, double scale_value,
                                           int64_t sparse_block_size, const c10::optional<at::Tensor> &block_table,
                                           const c10::optional<at::Tensor> &actual_seq_lengths_query,
                                           const c10::optional<at::Tensor> &actual_seq_lengths_kv,
                                           const c10::optional<at::Tensor> &query_rope,
                                           const c10::optional<at::Tensor> &key_rope, c10::string_view layout_query,
                                           c10::string_view layout_kv, int64_t sparse_mode)
{
    std::string layout_query_str = std::string(layout_query);
    at::Tensor output = construct_sparse_infer_output_tensor(query, value, layout_query_str);

    return output;
}
}  // namespace custom

// step4, 为NPU设备注册前向实现
TORCH_LIBRARY_IMPL(custom, PrivateUse1, m)
{
    m.impl("npu_sparse_flash_attention", &custom::npu_sparse_flash_attention_npu);
}

// step5, 为META设备注册前向实现
TORCH_LIBRARY_IMPL(custom, Meta, m)
{
    m.impl("npu_sparse_flash_attention", &custom::npu_sparse_flash_attention_meta);
}
