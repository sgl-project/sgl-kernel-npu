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
#include "layernorm.h"

using namespace at;

constexpr std::string_view LAYER_NORM_OP_NAME = "aclnnLayerNormWithImplMode";
std::tuple<at::Tensor, at::Tensor, at::Tensor> layernorm_npu(const at::Tensor &input, at::IntArrayRef normalized_shape,
                                                             const c10::optional<at::Tensor> &weight,
                                                             const c10::optional<at::Tensor> &bias, double eps,
                                                             int64_t impl_mode)
{
    const at::Tensor &weight_op = c10::value_or_else(weight, [] { return at::Tensor(); });
    const at::Tensor &bias_op = c10::value_or_else(bias, [] { return at::Tensor(); });

    at::Tensor output =
        at_npu::native::empty_with_format(input.sizes(), input.options(), at_npu::native::get_npu_format(input));
    at::Tensor mean_out;
    at::Tensor rstd_out;

    const size_t normNdim = normalized_shape.size();
    const auto inputNdim = input.dim();
    const size_t beginAxis = inputNdim - normNdim;

    const auto input_shape = input.sizes();

    const int64_t m =
        std::accumulate(input_shape.cbegin(), input_shape.cbegin() + beginAxis, 1LL, std::multiplies<int64_t>());
    // shape and dtype of mean and rstd depend on m value and input dtype
    if (m <= 0) {
        mean_out = at_npu::native::empty_with_format({m}, input.options(), at_npu::native::get_npu_format(input));
        rstd_out = at_npu::native::empty_with_format({m}, input.options(), at_npu::native::get_npu_format(input));
    } else {
        at::SmallVector<int64_t, 8> mean_shape;  // 维度最大支持到8
        for (size_t index = 0; index < beginAxis; index++) {
            mean_shape.emplace_back(input.size(index));
        }
        for (size_t index = beginAxis; index < inputNdim; index++) {
            mean_shape.emplace_back(1);
        }
        mean_out =
            at_npu::native::empty_with_format(mean_shape, input.options(), at_npu::native::get_npu_format(input));
        rstd_out =
            at_npu::native::empty_with_format(mean_shape, input.options(), at_npu::native::get_npu_format(input));
    }

    EXEC_NPU_CMD<LAYER_NORM_OP_NAME>(input, normalized_shape, weight_op, bias_op, eps, output, mean_out, rstd_out,
                                     impl_mode);

    return std::tie(output, mean_out, rstd_out);
}
