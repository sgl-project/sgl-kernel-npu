// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Apache-2.0. Explicit batch scheduling for native kernels with cross-core sync.
#pragma once
#include <cstdint>
#include "acl/acl_rt.h"

namespace sglang::npu_kernel {
inline uint32_t LaunchKernelWithBatchScheduling(void *binary, uint64_t entry, uint32_t blocks, void **args,
                                                uint32_t args_size, const void *stream)
{
    aclrtFuncHandle function = nullptr;
    auto status = aclrtBinaryGetFunctionByEntry(binary, entry, &function);
    if (status != ACL_SUCCESS) return status;
    aclrtLaunchKernelAttr attribute{};
    attribute.id = ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE;
    attribute.value.schemMode = 1;
    aclrtLaunchKernelCfg config{&attribute, 1};
    return aclrtLaunchKernelWithHostArgs(function, blocks, const_cast<void *>(stream), &config, args, args_size,
                                         nullptr, 0);
}
}  // namespace sglang::npu_kernel
