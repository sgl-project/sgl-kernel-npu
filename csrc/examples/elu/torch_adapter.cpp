// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "aclnn_runtime.h"
#include "aclnn_sgl_elu.h"

#include <c10/core/DeviceGuard.h>
#include <cmath>
#include <limits>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>

namespace {
at::Tensor elu(const at::Tensor &x, double alpha)
{
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "x must be on an NPU");
    TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kFloat, "x must be float16 or float32, got ",
                x.scalar_type());
    TORCH_CHECK(x.dim() >= 1, "x must be at least 1D, got ", x.dim(), "D");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(std::isfinite(alpha), "alpha must be finite, got ", alpha);
    TORCH_CHECK(at_npu::native::get_npu_format(x) == ACL_FORMAT_ND, "x must use ND storage format");
    // The OpDef requires a positive element count; an empty tensor is a legal
    // public input, so answer it before entering the CANN launch path.
    if (x.numel() == 0) {
        return at::empty_like(x);
    }
    const c10::DeviceGuard guard(x.device());
    auto y = at_npu::native::empty_with_format(x.sizes(), x.options(), ACL_FORMAT_ND, true);

    static auto query = reinterpret_cast<decltype(&aclnnSglEluGetWorkspaceSize)>(
        sgl_cann::op_api_symbol("aclnnSglEluGetWorkspaceSize"));
    static auto execute = reinterpret_cast<decltype(&aclnnSglElu)>(sgl_cann::op_api_symbol("aclnnSglElu"));
    const auto dtype = x.scalar_type() == at::kFloat ? ACL_FLOAT : ACL_FLOAT16;
    auto xDesc = sgl_cann::tensor_descriptor(x, dtype);
    auto yDesc = sgl_cann::tensor_descriptor(y, dtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    // Argument order is inputs, attributes, outputs.
    auto status = query(xDesc.get(), static_cast<float>(alpha), yDesc.get(), &workspaceSize, &executor);
    TORCH_CHECK(status == ACL_SUCCESS, "ELU workspace query failed: ", aclGetRecentErrMsg());
    TORCH_CHECK(workspaceSize <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                "ACLNN workspace size exceeds the Tensor allocation limit");
    auto workspace = at::empty({static_cast<int64_t>(workspaceSize)}, x.options().dtype(at::kByte));
    auto stream = c10_npu::getCurrentNPUStream();
    x.record_stream(stream);
    y.record_stream(stream);
    workspace.record_stream(stream);
    auto aclStream = stream.stream(false);
    // Captures protect the host queue; record_stream protects asynchronous device use.
    auto launch = [x, y, workspace, xDesc, yDesc, workspaceSize, executor, aclStream]() -> int {
        auto result = execute(workspaceSize ? workspace.data_ptr() : nullptr, workspaceSize, executor, aclStream);
        TORCH_CHECK(result == ACL_SUCCESS, "ELU ACLNN execution failed: ", aclGetRecentErrMsg());
        return result;
    };
    at_npu::native::OpCommand::RunOpApi("aclnnSglElu", launch);
    return y;
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("elu(Tensor x, float alpha=1.0) -> Tensor");
}
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("elu", &elu);
}
