/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file causal_conv1d.h
 * \brief causal_conv1d host-side function declaration
 */

#ifndef CAUSAL_CONV1D_HOST_H_
#define CAUSAL_CONV1D_HOST_H_

#include <ATen/ATen.h>

#include "defines.h"

namespace sglang {
namespace npu_kernel {

HOST_API at::Tensor causal_conv1d_impl(
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& conv_states,
    const at::Tensor& query_start_loc,
    const at::Tensor& cache_indices,
    const at::Tensor& has_initial_state,
    const at::Tensor& bias,
    bool activation_mode,
    int64_t pad_slot_id);

}  // namespace npu_kernel
}  // namespace sglang

#endif  // CAUSAL_CONV1D_HOST_H_
