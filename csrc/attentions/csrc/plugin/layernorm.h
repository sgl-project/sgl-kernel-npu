/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 *
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#ifndef LAYER_NORM_H
#define LAYER_NORM_H

#include <ATen/Tensor.h>
#include <c10/util/Optional.h>


std::tuple<at::Tensor, at::Tensor, at::Tensor> layernorm_npu(
    const at::Tensor &input,
    at::IntArrayRef normalized_shape,
    const c10::optional<at::Tensor> &weight,
    const c10::optional<at::Tensor> &bias, double eps,
    int64_t impl_mode
);
#endif // LAYER_NORM_H