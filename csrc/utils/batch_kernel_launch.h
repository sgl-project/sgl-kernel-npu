// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Apache-2.0. Explicit batch scheduling for native kernels with cross-core sync.
#pragma once
#include <cstdint>
#include "acl/acl_rt.h"

namespace sglang::npu_kernel {
inline uint32_t LaunchKernelWithBatchScheduling(void *binary, uint64_t entry, uint32_t blocks, void **args,
                                                uint32_t args_size, const void *stream, uint32_t local_memory_size = 0)
{
    aclrtFuncHandle function = nullptr;
    auto status = aclrtBinaryGetFunctionByEntry(binary, entry, &function);
    if (status != ACL_SUCCESS) return status;
    aclrtLaunchKernelAttr attributes[2]{};
    attributes[0].id = ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE;
    attributes[0].value.schemMode = 1;
    uint32_t count = 1;
    if (local_memory_size != 0) {
        attributes[1].id = ACL_RT_LAUNCH_KERNEL_ATTR_DYN_UBUF_SIZE;
        attributes[1].value.localMemorySize = local_memory_size;
        ++count;
    }
    aclrtLaunchKernelCfg config{attributes, count};
    return aclrtLaunchKernelWithHostArgs(function, blocks, const_cast<void *>(stream), &config, args, args_size,
                                         nullptr, 0);
}
}  // namespace sglang::npu_kernel
