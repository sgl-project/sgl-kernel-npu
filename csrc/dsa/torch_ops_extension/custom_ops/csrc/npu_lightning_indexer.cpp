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

// npu tensor max size
const int SIZE = 8;
const int DIM_0 = 0;
const int DIM_1 = 1;
const int DIM_2 = 2;
const int DIM_3 = 3;

// 工具函数，推导输出shape
at::Tensor construct_lightning_indexer_output_tensor(const at::Tensor &query, const at::Tensor &key,
                                                     const c10::optional<at::Tensor> &actual_seq_lengths_query,
                                                     int64_t sparse_count, std::string query_layout_str)
{
    at::SmallVector<int64_t, SIZE> output_size;
    if (query_layout_str == "BSND") {
        output_size = {query.size(DIM_0), query.size(DIM_1), key.size(DIM_2), sparse_count};
    } else {
        output_size = {query.size(DIM_0), key.size(DIM_2), sparse_count};
    }
    at::Tensor output = at::empty(output_size, query.options().dtype(at::kInt));

    return output;
}

// step2, 为NPU设备实现前向接口
at::Tensor npu_lightning_indexer_npu(const at::Tensor &query, const at::Tensor &key, const at::Tensor &weights,
                                     const c10::optional<at::Tensor> &actual_seq_lengths_query,
                                     const c10::optional<at::Tensor> &actual_seq_lengths_key,
                                     const c10::optional<at::Tensor> &block_table, c10::string_view layout_query,
                                     c10::string_view layout_key, int64_t sparse_count, int64_t sparse_mode)
{
    std::string query_layout_str = std::string(layout_query);
    std::string key_layout_str = std::string(layout_key);

    // construct the output tensor
    at::Tensor lightning_indexer_output =
        construct_lightning_indexer_output_tensor(query, key, actual_seq_lengths_query, sparse_count, query_layout_str);
    // convert str
    char *query_layout_ptr = const_cast<char *>(query_layout_str.c_str());
    char *key_layout_ptr = const_cast<char *>(key_layout_str.c_str());

    EXEC_NPU_CMD_V1(aclnnLightningIndexer, query, key, weights, actual_seq_lengths_query, actual_seq_lengths_key,
                    block_table, query_layout_ptr, key_layout_ptr, sparse_count, sparse_mode, lightning_indexer_output);

    return lightning_indexer_output;
}

// step3, 为META设备实现前向接口
at::Tensor npu_lightning_indexer_meta(const at::Tensor &query, const at::Tensor &key, const at::Tensor &weights,
                                      const c10::optional<at::Tensor> &actual_seq_lengths_query,
                                      const c10::optional<at::Tensor> &actual_seq_lengths_key,
                                      const c10::optional<at::Tensor> &block_table, c10::string_view layout_query,
                                      c10::string_view layout_key, int64_t sparse_count, int64_t sparse_mode)
{
    std::string query_layout_str = std::string(layout_query);
    // construct the output tensor
    at::Tensor lightning_indexer_output =
        construct_lightning_indexer_output_tensor(query, key, actual_seq_lengths_query, sparse_count, query_layout_str);

    return lightning_indexer_output;
}
}  // namespace custom

// step4, 为NPU设备注册前向实现
TORCH_LIBRARY_IMPL(custom, PrivateUse1, m)
{
    m.impl("npu_lightning_indexer", &custom::npu_lightning_indexer_npu);
}

// step5, 为META设备注册前向实现
TORCH_LIBRARY_IMPL(custom, Meta, m)
{
    m.impl("npu_lightning_indexer", &custom::npu_lightning_indexer_meta);
}
