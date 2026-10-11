#pragma once
#include "torch_helper.h"

// Keep tensor storage alive while the host task queue holds the native launch.
// Pointer conversion alone does not own temporary workspace or metadata tensors.
#define QSA_EXEC_KERNEL_CMD(kernel_name, blockdim, ...)                                                                \
    do {                                                                                                               \
        auto stream = c10_npu::getCurrentNPUStream().stream(false);                                                    \
        auto owned_args = std::make_tuple(__VA_ARGS__);                                                                \
        auto launch = [stream, blockdim, owned_args]() mutable -> int {                                                \
            return std::apply(                                                                                         \
                [&](auto &...args) -> int {                                                                            \
                    return ACLRT_LAUNCH_KERNEL(kernel_name)(blockdim, stream,                                          \
                                                            sglang::npu_kernel::TorchNpuHelper::ConvertType(args)...); \
                },                                                                                                     \
                owned_args);                                                                                           \
        };                                                                                                             \
        at_npu::native::OpCommand::RunOpApi(#kernel_name, launch);                                                     \
    } while (false)
