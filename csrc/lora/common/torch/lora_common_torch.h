// BSD 3-Clause License
// Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

#ifndef SGL_KERNEL_NPU_LORA_COMMON_TORCH_H
#define SGL_KERNEL_NPU_LORA_COMMON_TORCH_H

#include <torch/extension.h>

namespace sglang {
namespace npu_kernel {
namespace lora {

// Shared dtype contract for every LoRA operator: activations enter the device
// as float32, LoRA weights and the in-place output are half or bfloat16.
inline bool IsLoraWeightDtype(at::ScalarType dtype)
{
    return dtype == at::kHalf || dtype == at::kBFloat16;
}

inline bool IsLoraActivationDtype(at::ScalarType dtype)
{
    return dtype == at::kFloat;
}

}  // namespace lora
}  // namespace npu_kernel
}  // namespace sglang

#endif  // SGL_KERNEL_NPU_LORA_COMMON_TORCH_H
