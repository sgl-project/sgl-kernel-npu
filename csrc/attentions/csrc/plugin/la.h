/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * sgl-kernel-npu/svoloch2940194 is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#ifndef LA_IMPL_H
#define LA_IMPL_H

#include <ATen/Tensor.h>
#include <c10/util/Optional.h>
#include <string>
#include <tuple>

std::tuple<at::Tensor, at::Tensor> la(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const c10::optional<at::Tensor> &atten_mask_opt,
    const c10::optional<at::Tensor> &alibi_mask_opt,
    const c10::optional<at::Tensor> &drop_mask_opt,
    double scale_value, int64_t head_num, std::string input_layout,
    double keep_prob, int64_t pre_tokens, int64_t next_tokens, bool is_highPrecision);

#endif // LA_IMPL_H