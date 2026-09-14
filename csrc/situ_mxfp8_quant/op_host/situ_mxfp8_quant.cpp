/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */

#include <algorithm>
#include <tuple>

#include "tiling/platform/platform_ascendc.h"
#include "defines.h"
#include "torch_helper.h"
#include "aclrtlaunch_situ_mxfp8_quant.h"

namespace sglang {
namespace npu_kernel {

HOST_API std::tuple<at::Tensor, at::Tensor> situ_mxfp8_quant(
    const at::Tensor &x, const at::Tensor &group_list,
    int64_t group_list_type, double beta, double linear_beta)
{
    TORCH_CHECK(x.dim() == 2, "x must be 2D [capacity, 2 * hidden], got dim=", x.dim());
    TORCH_CHECK(x.scalar_type() == at::kBFloat16, "x must be BF16, got ", x.scalar_type());
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.size(0) > 0 && x.size(1) == 6144,
                "the first A5 kernel supports x shape [capacity, 6144], got [",
                x.size(0), ", ", x.size(1), "]");
    TORCH_CHECK(group_list.dim() == 1 && group_list.numel() > 0,
                "group_list must be a non-empty 1D tensor");
    TORCH_CHECK(group_list.scalar_type() == at::kInt || group_list.scalar_type() == at::kLong,
                "group_list must use int32 or int64, got ", group_list.scalar_type());
    TORCH_CHECK(group_list.is_contiguous(), "group_list must be contiguous");
    TORCH_CHECK(group_list.device() == x.device(), "x and group_list must be on the same NPU");
    TORCH_CHECK(group_list_type == 0 || group_list_type == 1,
                "group_list_type must be 0 (cumulative) or 1 (count), got ", group_list_type);
    TORCH_CHECK(beta > 0.0 && linear_beta > 0.0,
                "beta and linear_beta must be positive, got beta=", beta,
                ", linear_beta=", linear_beta);

    constexpr int64_t kOutputCols = 3072;
    const int64_t rows = x.size(0);

    auto payload = at::empty({rows, kOutputCols}, x.options().dtype(at::kFloat8_e4m3fn));
    auto scales = at::empty({rows, kOutputCols / 64, 2}, x.options().dtype(at::kFloat8_e8m0fnu));

    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t block_dim = static_cast<uint32_t>(platform->GetCoreNumAiv());
    if (block_dim == 0) {
        block_dim = 1;
    }
    block_dim = std::min<uint32_t>(block_dim, static_cast<uint32_t>(rows));

    uint32_t capacity_rows = static_cast<uint32_t>(rows);
    uint32_t num_experts = static_cast<uint32_t>(group_list.numel());
    uint32_t group_list_type_u32 = static_cast<uint32_t>(group_list_type);
    uint32_t group_dtype = group_list.scalar_type() == at::kLong ? 1U : 0U;
    float beta_f32 = static_cast<float>(beta);
    float linear_beta_f32 = static_cast<float>(linear_beta);
    EXEC_KERNEL_CMD(situ_mxfp8_quant, block_dim, x, group_list, payload, scales,
                    capacity_rows, num_experts, group_list_type_u32, group_dtype,
                    beta_f32, linear_beta_f32);
    return std::make_tuple(payload, scales);
}

}  // namespace npu_kernel
}  // namespace sglang
