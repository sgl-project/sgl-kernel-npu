// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Apache-2.0. Ordered submission adapted from vllm-ascend PR #18383.
#pragma once
#include <dlfcn.h>
#include <vector>
#include "torch_helper.h"

namespace sglang::npu_kernel {
inline void UseOrderedStreamResources(aclrtStream stream, bool enqueue)
{
    const bool needed = enqueue ? c10_npu::check_enqueue_need_use(stream) : c10_npu::check_dequeue_need_use(stream);
    if (!needed) return;
    using Function = aclError (*)(aclrtStream);
    static void *library = dlopen("libascendcl.so", RTLD_NOW | RTLD_LOCAL);
    static auto use =
        library ? reinterpret_cast<Function>(dlsym(library, "aclrtUseStreamResInCurrentThread")) : nullptr;
    TORCH_CHECK(use != nullptr, "aclrtUseStreamResInCurrentThread is unavailable");
    TORCH_CHECK(use(stream) == ACL_SUCCESS, "could not acquire ordered stream resources");
}
}  // namespace sglang::npu_kernel

// Pointer conversion alone does not retain Tensor/workspace owners while
// torch_npu queues the launch. Capture owners in the submitted closure as well.
#define EXEC_OWNED_KERNEL_CMD(kernel_name, blockdim, owners, ...)                                                \
    do {                                                                                                         \
        auto owned_tensors = (owners);                                                                           \
        auto npu_stream = c10_npu::getCurrentNPUStream();                                                        \
        for (const auto &tensor : owned_tensors) {                                                               \
            if (tensor.device().is_privateuseone()) tensor.record_stream(npu_stream);                            \
        }                                                                                                        \
        auto acl_stream = npu_stream.stream(false);                                                              \
        sglang::npu_kernel::UseOrderedStreamResources(acl_stream, true);                                         \
        auto params = sglang::npu_kernel::TorchNpuHelper::ConvertTypes(__VA_ARGS__);                             \
        auto submit = [acl_stream, blockdim, params, owned_tensors]() -> int {                                   \
            (void)owned_tensors;                                                                                 \
            sglang::npu_kernel::UseOrderedStreamResources(acl_stream, false);                                    \
            auto status = std::apply(                                                                            \
                [&](auto &&...args) { return ACLRT_LAUNCH_KERNEL(kernel_name)(blockdim, acl_stream, args...); }, \
                params);                                                                                         \
            TORCH_CHECK(status == 0, #kernel_name " launch failed: ", status);                                   \
            return status;                                                                                       \
        };                                                                                                       \
        at_npu::native::OpCommand::RunOpApiV2(#kernel_name, submit);                                             \
    } while (false)
