// SPDX-License-Identifier: Apache-2.0
#include <optional>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include "aclnn_common.h"

namespace op_api {
std::tuple<at::Tensor, at::Tensor>
grouped_matmul_swiglu_quant(const at::Tensor &x, const at::Tensor &weight,
                            const at::Tensor &weight_scale,
                            const at::Tensor &x_scale,
                            const at::Tensor &group_list, double swiglu_limit,
                            const std::optional<at::Tensor> &weight_assist_matrix) {
  TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1,
              "GMM inputs must be on NPU");
  for (const auto &input : {weight, weight_scale, x_scale, group_list}) {
    TORCH_CHECK(input.device() == x.device(), "GMM inputs must share a device");
  }
  const at::Tensor unused;
  const auto &bias = weight_assist_matrix ? *weight_assist_matrix : unused;
  TORCH_CHECK(!bias.defined() || bias.device() == x.device(),
              "GMM inputs must share a device");
  // Packed INT32 stores eight output channels per element.
  const int64_t n = weight.size(2) * (weight.scalar_type() == at::kInt ? 8 : 1);
  auto output = at::empty({x.size(0), n / 2}, x.options());
  auto scale = at::empty({x.size(0)}, x.options().dtype(at::kFloat));
  ACLNN_CMD(aclnnSglGroupedMatmulSwigluQuantWeightNZ, x, weight, bias, unused,
            weight_scale, x_scale, group_list, swiglu_limit, output, scale,
            unused);
  return {output, scale};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("grouped_matmul_swiglu_quant", &grouped_matmul_swiglu_quant);
}
} // namespace op_api
