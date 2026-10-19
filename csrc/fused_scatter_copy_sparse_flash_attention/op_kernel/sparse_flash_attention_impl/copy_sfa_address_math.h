// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under the BSD 3-Clause License.

#ifndef SGLANG_NPU_COPY_SFA_ADDRESS_MATH_H
#define SGLANG_NPU_COPY_SFA_ADDRESS_MATH_H

#include <cstdint>

namespace sglang::npu_kernel::copy_sfa {

#if defined(__CCE_AICORE__)
#define COPY_SFA_AICORE __aicore__
#else
#define COPY_SFA_AICORE
#endif

// DataCopyExtParams strides are the byte gap after blockLen, not the physical
// row distance. This also reduces to the legacy split-layout expression when
// physicalRowStrideElements == logicalRowWidthElements.
COPY_SFA_AICORE constexpr int64_t SourceGapBytes(int64_t tokenDelta, int64_t blockTokens,
                                                 int64_t physicalRowStrideElements, int64_t logicalRowWidthElements,
                                                 int64_t elementBytes)
{
    return tokenDelta * physicalRowStrideElements * elementBytes - blockTokens * logicalRowWidthElements * elementBytes;
}

#undef COPY_SFA_AICORE

}  // namespace sglang::npu_kernel::copy_sfa

#endif  // SGLANG_NPU_COPY_SFA_ADDRESS_MATH_H
