/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */

#include "aclnn_runtime.h"
#include "aclnn_sgl_situ_mxfp8_quant.h"

#include <c10/core/DeviceGuard.h>
#include <limits>
#include <torch/library.h>
#include <tuple>

namespace {

std::tuple<at::Tensor, at::Tensor> situ_mxfp8_quant(const at::Tensor &x, const at::Tensor &group_list,
                                                    int64_t group_list_type, double beta, double linear_beta)
{
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "x must be on an NPU");
    TORCH_CHECK(x.dim() == 2, "x must be 2D [capacity, 2 * hidden], got dim=", x.dim());
    TORCH_CHECK(x.scalar_type() == at::kBFloat16, "x must be BF16, got ", x.scalar_type());
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.size(0) > 0 && x.size(1) == 6144, "the A5 kernel supports x shape [capacity, 6144], got [", x.size(0),
                ", ", x.size(1), "]");
    TORCH_CHECK(group_list.dim() == 1 && group_list.numel() > 0, "group_list must be a non-empty 1D tensor");
    TORCH_CHECK(group_list.scalar_type() == at::kInt || group_list.scalar_type() == at::kLong,
                "group_list must use int32 or int64, got ", group_list.scalar_type());
    TORCH_CHECK(group_list.is_contiguous(), "group_list must be contiguous");
    TORCH_CHECK(group_list.device() == x.device(), "x and group_list must be on the same NPU");
    TORCH_CHECK(group_list_type == 0 || group_list_type == 1,
                "group_list_type must be 0 (cumulative) or 1 (count), got ", group_list_type);
    TORCH_CHECK(beta > 0.0 && linear_beta > 0.0, "beta and linear_beta must be positive, got beta=", beta,
                ", linear_beta=", linear_beta);

    const c10::DeviceGuard deviceGuard(x.device());
    const int64_t rows = x.size(0);
    constexpr int64_t kOutputCols = 3072;
    auto payload = at::empty({rows, kOutputCols}, x.options().dtype(at::kFloat8_e4m3fn));
    auto scales = at::empty({rows, kOutputCols / 64, 2}, x.options().dtype(at::kFloat8_e8m0fnu));

    static auto query = reinterpret_cast<decltype(&aclnnSglSituMxfp8QuantGetWorkspaceSize)>(
        sgl_cann::op_api_symbol("aclnnSglSituMxfp8QuantGetWorkspaceSize"));
    static auto execute =
        reinterpret_cast<decltype(&aclnnSglSituMxfp8Quant)>(sgl_cann::op_api_symbol("aclnnSglSituMxfp8Quant"));

    auto xDesc = sgl_cann::tensor_descriptor(x, ACL_BF16);
    auto groupDesc =
        sgl_cann::tensor_descriptor(group_list, group_list.scalar_type() == at::kInt ? ACL_INT32 : ACL_INT64);
    auto payloadDesc = sgl_cann::tensor_descriptor(payload, ACL_FLOAT8_E4M3FN);
    auto scalesDesc = sgl_cann::tensor_descriptor(scales, ACL_FLOAT8_E8M0);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    // Order: inputs, attributes, outputs. No output aliases an input.
    auto status =
        query(xDesc.get(), groupDesc.get(), group_list_type, static_cast<float>(beta), static_cast<float>(linear_beta),
              payloadDesc.get(), scalesDesc.get(), &workspaceSize, &executor);
    TORCH_CHECK(status == ACL_SUCCESS, "situ_mxfp8_quant workspace query failed: ", aclGetRecentErrMsg());
    TORCH_CHECK(workspaceSize <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                "ACLNN workspace size exceeds the Tensor allocation limit");
    auto workspace = at::empty({static_cast<int64_t>(workspaceSize)}, x.options().dtype(at::kByte));

    auto stream = c10_npu::getCurrentNPUStream();
    x.record_stream(stream);
    group_list.record_stream(stream);
    payload.record_stream(stream);
    scales.record_stream(stream);
    workspace.record_stream(stream);
    auto aclStream = stream.stream(false);

    // Captures retain queued host resources; record_stream retains asynchronous device storage.
    auto launch = [x, group_list, payload, scales, workspace, xDesc, groupDesc, payloadDesc, scalesDesc, workspaceSize,
                   executor, aclStream]() -> int {
        auto result = execute(workspaceSize ? workspace.data_ptr() : nullptr, workspaceSize, executor, aclStream);
        TORCH_CHECK(result == ACL_SUCCESS, "situ_mxfp8_quant execution failed: ", aclGetRecentErrMsg());
        return result;
    };
    at_npu::native::OpCommand::RunOpApi("aclnnSglSituMxfp8Quant", launch);
    return std::make_tuple(payload, scales);
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def(
        "situ_mxfp8_quant(Tensor x, Tensor group_list, int group_list_type=1, float beta=4.0, "
        "float linear_beta=25.0) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("situ_mxfp8_quant", &situ_mxfp8_quant);
}
