// BSD 3-Clause License
// Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

#include "aclnn_runtime.h"
#include "aclnn_sgl_add_example.h"
#include <c10/core/DeviceGuard.h>
#include <limits>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>

namespace {
at::Tensor add_example(const at::Tensor &x1, const at::Tensor &x2)
{
    TORCH_CHECK(x1.device().type() == c10::DeviceType::PrivateUse1, "x1 must be on an NPU");
    TORCH_CHECK(x2.device() == x1.device(), "inputs must be on the same NPU");
    TORCH_CHECK(x1.sizes() == x2.sizes(), "inputs must have identical shapes; broadcasting is not supported");
    TORCH_CHECK(x1.scalar_type() == x2.scalar_type(), "inputs must have the same dtype");
    TORCH_CHECK(x1.scalar_type() == at::kFloat || x1.scalar_type() == at::kInt, "inputs must be float32 or int32");
    TORCH_CHECK(x1.is_contiguous() && x2.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(
        at_npu::native::get_npu_format(x1) == ACL_FORMAT_ND && at_npu::native::get_npu_format(x2) == ACL_FORMAT_ND,
        "inputs must use ND storage format");
    TORCH_CHECK(x1.numel() > 0, "empty inputs are not supported");
    const c10::DeviceGuard guard(x1.device());
    auto y = at_npu::native::empty_with_format(x1.sizes(), x1.options(), ACL_FORMAT_ND, true);

    static auto query = reinterpret_cast<decltype(&aclnnSglAddExampleGetWorkspaceSize)>(
        sgl_cann::op_api_symbol("aclnnSglAddExampleGetWorkspaceSize"));
    static auto execute =
        reinterpret_cast<decltype(&aclnnSglAddExample)>(sgl_cann::op_api_symbol("aclnnSglAddExample"));
    const auto dtype = x1.scalar_type() == at::kFloat ? ACL_FLOAT : ACL_INT32;
    auto x1Desc = sgl_cann::tensor_descriptor(x1, dtype);
    auto x2Desc = sgl_cann::tensor_descriptor(x2, dtype);
    auto yDesc = sgl_cann::tensor_descriptor(y, dtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto status = query(x1Desc.get(), x2Desc.get(), yDesc.get(), &workspaceSize, &executor);
    TORCH_CHECK(status == ACL_SUCCESS, "AddExample workspace query failed: ", aclGetRecentErrMsg());
    TORCH_CHECK(workspaceSize <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                "ACLNN workspace exceeds the Tensor allocation limit");
    auto workspace = at::empty({static_cast<int64_t>(workspaceSize)}, x1.options().dtype(at::kByte));
    auto stream = c10_npu::getCurrentNPUStream();
    x1.record_stream(stream);
    x2.record_stream(stream);
    y.record_stream(stream);
    workspace.record_stream(stream);
    auto aclStream = stream.stream(false);
    // Captures protect the host queue; record_stream protects asynchronous device use.
    auto launch = [x1, x2, y, workspace, x1Desc, x2Desc, yDesc, workspaceSize, executor, aclStream]() -> int {
        auto result = execute(workspaceSize ? workspace.data_ptr() : nullptr, workspaceSize, executor, aclStream);
        TORCH_CHECK(result == ACL_SUCCESS, "AddExample ACLNN execution failed: ", aclGetRecentErrMsg());
        return result;
    };
    at_npu::native::OpCommand::RunOpApi("aclnnSglAddExample", launch);
    return y;
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("sgl_add_example(Tensor x1, Tensor x2) -> Tensor");
}
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("sgl_add_example", &add_example);
}
