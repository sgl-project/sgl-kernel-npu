/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#ifndef RAINFUSIONATTENTION_IMPL_H
#define RAINFUSIONATTENTION_IMPL_H

#include <ATen/Tensor.h>
#include <c10/util/Optional.h>
#include <string>
#include <tuple>

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
    int64_t inner_precise, int64_t block_size);

#endif // RAINFUSIONATTENTION_IMPL_H