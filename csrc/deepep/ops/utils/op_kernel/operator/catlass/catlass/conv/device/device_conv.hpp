/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_CONV_DEVICE_DEVICE_CONV_HPP
#define CATLASS_CONV_DEVICE_DEVICE_CONV_HPP

#include <acl/acl.h>
#include "catlass/catlass.hpp"
#include "catlass/status.hpp"
#include "catlass/conv/device/kernel_adapter.hpp"

#if defined(ENABLE_ASCENDC_DUMP)
#include "catlass/debug.hpp"
#endif

namespace Catlass::Conv::Device {

template <class ConvKernel>
class DeviceConv {
public:
    /// Argument structure: User API
    using Arguments = typename ConvKernel::Arguments;
    /// Argument structure: Kernel API
    using Params = typename ConvKernel::Params;
private:
    /// kernel API parameters object
    Params params_;
public:
    DeviceConv() {}
    ~DeviceConv() {}

    /// Access the Params structure
    Params const &params() const
    {
        return params_;
    }

    /// Determines whether the Conv can execute the given problem.
    static Status CanImplement(Arguments const &args)
    {
        if (ConvKernel::CanImplement(args)) {
            return Status::kSuccess;
        } else {
            return Status::kInvalid;
        }
    }

    /// Gets the workspace size
    static size_t GetWorkspaceSize(Arguments const &args)
    {
        size_t workspace_bytes = 0;
        workspace_bytes += ConvKernel::GetWorkspaceSize(args);
        return workspace_bytes;
    }

    /// Initializes Conv state from arguments
    Status Initialize(Arguments const &args, uint8_t *workspace = nullptr, aclrtStream stream = nullptr)
    {
        // Initialize the Params structure
        params_ = ConvKernel::ToUnderlyingArguments(args, workspace);
        return Status::kSuccess;
    }

    /// Primary run() entry point API that is static allowing users to create and manage their own params.
    /// Supplied params struct must be construct by calling matmul Kernel::to_underling arguments
    inline Status Run(aclrtStream stream, uint32_t blockDim, uint64_t fftsAddr)
    {
#if defined(ENABLE_ASCENDC_DUMP)
        uint8_t *ptrDump{nullptr};
        aclCheck(aclrtMalloc(reinterpret_cast<void **>(&ptrDump), ALL_DUMPSIZE, ACL_MEM_MALLOC_HUGE_FIRST));
        if (fftsAddr == 0) {
            Catlass::KernelAdapter<ConvKernel><<<blockDim, nullptr, stream>>>(params_, ptrDump);
        } else {
            Catlass::KernelAdapter<ConvKernel><<<blockDim, nullptr, stream>>>(params_, fftsAddr, ptrDump);
        }
        aclCheck(aclrtSynchronizeStream(stream));
        Adx::AdumpPrintWorkSpace(ptrDump, ALL_DUMPSIZE, stream, "device_gemm");
        aclCheck(aclrtFree(ptrDump));
#else
        if (fftsAddr == 0) {
            Catlass::KernelAdapter<ConvKernel><<<blockDim, nullptr, stream>>>(params_);
        } else {
            Catlass::KernelAdapter<ConvKernel><<<blockDim, nullptr, stream>>>(params_, fftsAddr);
        }
#endif
        return Status::kSuccess;
    }

    /// Runs the kernel using initialized state
    inline Status operator()(aclrtStream stream, uint32_t blockDim)
    {
        return Run(stream, blockDim, 0);
    }

    inline Status operator()(aclrtStream stream, uint32_t blockDim, uint64_t fftsAddr)
    {
        return Run(stream, blockDim, fftsAddr);
    }
};
///////////////////////////////////////////////////////////////////////////////////

} // namespace Catlass::Gemm::Device
#endif
