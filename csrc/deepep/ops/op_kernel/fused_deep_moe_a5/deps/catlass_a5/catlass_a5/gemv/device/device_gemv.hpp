/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMV_DEVICE_DEVICE_GEMV_HPP
#define CATLASS_GEMV_DEVICE_DEVICE_GEMV_HPP

#include <acl/acl.h>
#include "catlass_a5/catlass.hpp"
#include "catlass_a5/status.hpp"
#include "catlass_a5/detail/kernel_adapter.hpp"

namespace Catlass::Gemv::Device {

template <class GemvKernel>
class DeviceGemv
{
public:
    /// Argument structure: User API
    using Arguments = typename GemvKernel::Arguments;
    /// Argument structure: Kernel API
    using Params = typename GemvKernel::Params;

private:
    /// kernel API parameters object
    Params params_;

public:
    DeviceGemv() {}
    ~DeviceGemv() {}

    /// Access the Params structure
    Params const &params() const
    {
        return params_;
    }

    /// Determines whether the GEMM can execute the given problem.
    static Status CanImplement(Arguments const &args)
    {
        if (GemvKernel::CanImplement(args)) {
            return Status::kSuccess;
        } else {
            return Status::kInvalid;
        }
    }

    /// Gets the workspace size
    static size_t GetWorkspaceSize(Arguments const &args)
    {
        size_t workspace_bytes = 0;
        workspace_bytes += GemvKernel::GetWorkspaceSize(args);
        return workspace_bytes;
    }

    /// Initializes GEMV state from arguments
    Status Initialize(Arguments const &args, uint8_t *workspace = nullptr, aclrtStream stream = nullptr)
    {
        // Initialize the Params structure
        params_ = GemvKernel::ToUnderlyingArguments(args, workspace);
        return Status::kSuccess;
    }

    /// Primary run() entry point API that is static allowing users to create and manage their own params.
    /// Supplied params struct must be construct by calling matmul Kernel::to_underlying arguments
    inline Status Run(aclrtStream stream, uint32_t blockDim, uint64_t hardwareSyncAddr)
    {
        if (hardwareSyncAddr == 0) {
            Catlass::KernelAdapter<GemvKernel><<<blockDim, nullptr, stream>>>(params_);
        } else {
            Catlass::KernelAdapter<GemvKernel><<<blockDim, nullptr, stream>>>(params_, hardwareSyncAddr);
        }
        return Status::kSuccess;
    }

    /// Runs the kernel using initialized state
    inline Status operator()(aclrtStream stream, uint32_t blockDim)
    {
        return Run(stream, blockDim, 0);
    }

    inline Status operator()(aclrtStream stream, uint32_t blockDim, uint64_t hardwareSyncAddr)
    {
        return Run(stream, blockDim, hardwareSyncAddr);
    }
};
///////////////////////////////////////////////////////////////////////////////////

}  // namespace Catlass::Gemv::Device
#endif
