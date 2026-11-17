/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE-CANN in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_ADD_EXAMPLE_CPU_KERNELS_H_
#define AICPU_ADD_EXAMPLE_CPU_KERNELS_H_

#include "cpu_kernel.h"

namespace aicpu {
class AddExampleCpuKernel : public CpuKernel
{
public:
    ~AddExampleCpuKernel() = default;
    uint32_t Compute(CpuKernelContext &ctx) override;
    template <typename T>
    uint32_t AddCompute(CpuKernelContext &ctx);
};
}  // namespace aicpu
#endif
