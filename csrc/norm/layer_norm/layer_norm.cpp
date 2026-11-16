// BSD 3-Clause License
// Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

// HOST bridge for the CANN built-in LayerNormWithImplMode operator. The Torch
// adapter prepares the descriptors; this file resolves the ACLNN entry points
// and queues the launch.

#include "layer_norm.h"

#include <ATen/ATen.h>
#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <aclnn/aclnn_base.h>
#include <aclnnop/aclnn_layer_norm.h>
#include <torch_npu/csrc/core/npu/DeviceUtils.h>
#include <torch_npu/csrc/core/npu/NPUFunctions.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <torch_npu/csrc/framework/utils/CalcuOpUtil.h>

#include "aclnn_runtime.h"

namespace sglang::norm {
namespace {

constexpr const char *kOpName = "aclnnLayerNormWithImplMode";

using InitHugeMemFn = int (*)(void *, bool);
using UnInitHugeMemFn = void (*)(void *, bool);
using ReleaseHugeMemFn = void (*)(void *, bool);

}  // namespace

void launch_layer_norm(const LayerNormArgs &args)
{
    static auto queryWorkspace = reinterpret_cast<decltype(&aclnnLayerNormWithImplModeGetWorkspaceSize)>(
        sgl_cann::op_api_symbol_with_fallback("aclnnLayerNormWithImplModeGetWorkspaceSize"));
    static auto execute =
        reinterpret_cast<decltype(&aclnnLayerNormWithImplMode)>(sgl_cann::op_api_symbol_with_fallback(kOpName));
    // Best-effort callbacks, skipped when the loaded libraries do not export them.
    static auto initHugeMem = reinterpret_cast<InitHugeMemFn>(sgl_cann::try_op_api_symbol("InitHugeMemThreadLocal"));
    static auto unInitHugeMem =
        reinterpret_cast<UnInitHugeMemFn>(sgl_cann::try_op_api_symbol("UnInitHugeMemThreadLocal"));
    static auto releaseHugeMem = reinterpret_cast<ReleaseHugeMemFn>(sgl_cann::try_op_api_symbol("ReleaseHugeMem"));

    if (initHugeMem != nullptr) {
        initHugeMem(nullptr, false);
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    const auto status = queryWorkspace(args.input.get(), args.normalized_shape.get(), args.weight.get(),
                                       args.bias.get(), args.eps, args.out.get(), args.mean.get(), args.rstd.get(),
                                       static_cast<int32_t>(args.impl_mode), &workspaceSize, &executor);
    TORCH_CHECK(status == ACL_SUCCESS, "call ", kOpName, "GetWorkspaceSize failed, detail:", aclGetRecentErrMsg());

    // The legacy workspace is allocated for the current NPU device type.
    at::Tensor workspace;
    void *workspaceAddr = nullptr;
    if (workspaceSize != 0) {
        workspace = at::empty({static_cast<int64_t>(workspaceSize)},
                              at::TensorOptions(torch_npu::utils::get_npu_device_type()).dtype(c10::kByte));
        workspaceAddr = const_cast<void *>(workspace.storage().data());
    }

    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);
    auto launch = [args, workspace, workspaceAddr, workspaceSize, executor, aclStream]() -> int {
        // Keep the descriptors and the workspace storage alive through the
        // queued launch.
        (void)args;
        (void)workspace;
        const auto result = execute(workspaceAddr, workspaceSize, executor, aclStream);
        TORCH_CHECK(result == ACL_SUCCESS, "call ", kOpName, " failed, detail:", aclGetRecentErrMsg());
        if (releaseHugeMem != nullptr) {
            releaseHugeMem(nullptr, false);
        }
        return result;
    };
    at_npu::native::OpCommand::RunOpApi(kOpName, launch);

    if (unInitHugeMem != nullptr) {
        unInitHugeMem(nullptr, false);
    }
}

}  // namespace sglang::norm
