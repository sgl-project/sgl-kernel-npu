// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>

#include "aclnn_common.h"

namespace op_api {
std::tuple<at::Tensor, at::Tensor>
grouped_matmul_swiglu_quant(const at::Tensor &x, const at::Tensor &weight,
                            const at::Tensor &weight_scale,
                            const at::Tensor &x_scale,
                            const at::Tensor &group_list, double swiglu_limit) {
  TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1,
              "GMM inputs must be on NPU");
  for (const auto &input : {weight, weight_scale, x_scale, group_list}) {
    TORCH_CHECK(input.device() == x.device(), "GMM inputs must share a device");
  }
  auto output = at::empty({x.size(0), weight.size(2) / 2}, x.options());
  auto scale = at::empty({x.size(0)}, x.options().dtype(at::kFloat));
  const at::Tensor unused;
  ACLNN_CMD(aclnnSglGroupedMatmulSwigluQuantWeightNZ, x, weight, unused, unused,
            weight_scale, x_scale, group_list, swiglu_limit, output, scale,
            unused);
  return {output, scale};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("grouped_matmul_swiglu_quant", &grouped_matmul_swiglu_quant);
}
} // namespace op_api
