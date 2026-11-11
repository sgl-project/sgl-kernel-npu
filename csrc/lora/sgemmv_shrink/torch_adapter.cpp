// BSD 3-Clause License
// Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
//
// Framework bridge for the migrated sgemmv_shrink operator. The validation
// order mirrors the legacy Host contract; the device computation stays in the
// separately copied Kernel.

#include "aclnn_runtime.h"
#include "aclnn_sgl_sgemmv_shrink.h"
#include "../common/torch/lora_common_torch.h"

#include <c10/core/DeviceGuard.h>
#include <torch/library.h>

#include <limits>

namespace {

void sgemmv_shrink(at::Tensor x, at::Tensor weight, at::Tensor lora_indices, at::Tensor seq_len, at::Tensor lora_ranks,
                   at::Tensor lora_scales, at::Tensor y, int64_t slice_count)
{
    TORCH_CHECK(y.device().type() == c10::DeviceType::PrivateUse1, "y must be on an NPU");
    TORCH_CHECK(x.device() == y.device(), "x and y must be on the same NPU");
    TORCH_CHECK(weight.device() == y.device(), "weight and y must be on the same NPU");
    TORCH_CHECK(lora_indices.device() == y.device(), "lora_indices and y must be on the same NPU");
    TORCH_CHECK(seq_len.device() == y.device(), "seq_len and y must be on the same NPU");
    TORCH_CHECK(lora_ranks.device() == y.device(), "lora_ranks and y must be on the same NPU");
    TORCH_CHECK(lora_scales.device() == y.device(), "lora_scales and y must be on the same NPU");

    const c10::DeviceGuard deviceGuard(y.device());

    // Mirror the legacy HOST_API checks and their order where possible.
    TORCH_CHECK(sglang::npu_kernel::lora::IsLoraWeightDtype(x.scalar_type()), "only support half and bf16");
    TORCH_CHECK(x.dim() == 2, "x should be [batch_size, hidden_in]");
    TORCH_CHECK(weight.dim() == 3 || weight.dim() == 4,
                "weight should be [num_loras, hidden_out, hidden_in] or [num_loras, 1, hidden_out, hidden_in]");
    TORCH_CHECK(y.dim() == 2, "y should be [batch_size, hidden_out]");
    TORCH_CHECK(x.size(1) > y.size(1), "hidden in should be greater than hidden out");
    TORCH_CHECK(x.size(0) > 0, "num_tokens_per_core should not be 0");

    // Contract checks required by the Kernel, kept explicit for direct aclnn calls.
    TORCH_CHECK(sglang::npu_kernel::lora::IsLoraActivationDtype(y.scalar_type()), "y must be float32");
    TORCH_CHECK(weight.scalar_type() == x.scalar_type(), "weight and x must have the same dtype");
    TORCH_CHECK(lora_scales.scalar_type() == at::kHalf, "lora_scales must be float16");
    TORCH_CHECK(lora_indices.scalar_type() == at::kInt && seq_len.scalar_type() == at::kInt &&
                    lora_ranks.scalar_type() == at::kInt,
                "lora_indices, seq_len and lora_ranks must be int32");
    TORCH_CHECK(lora_indices.dim() == 1 && seq_len.dim() == 1 && lora_ranks.dim() == 1 && lora_scales.dim() == 1,
                "lora_indices, seq_len, lora_ranks and lora_scales must be 1D");
    TORCH_CHECK(x.is_contiguous() && weight.is_contiguous() && y.is_contiguous() && lora_indices.is_contiguous() &&
                    seq_len.is_contiguous() && lora_ranks.is_contiguous() && lora_scales.is_contiguous(),
                "all inputs must be contiguous");
    TORCH_CHECK(x.size(1) % 64 == 0, "hidden_in must be a multiple of 64");

    const auto inputDtype = x.scalar_type() == at::kHalf ? ACL_FLOAT16 : ACL_BF16;
    auto xDesc = sgl_cann::tensor_descriptor(x, inputDtype);
    auto weightDesc = sgl_cann::tensor_descriptor(weight, inputDtype);
    auto loraIndicesDesc = sgl_cann::tensor_descriptor(lora_indices, ACL_INT32);
    auto seqLenDesc = sgl_cann::tensor_descriptor(seq_len, ACL_INT32);
    auto loraRanksDesc = sgl_cann::tensor_descriptor(lora_ranks, ACL_INT32);
    auto loraScalesDesc = sgl_cann::tensor_descriptor(lora_scales, ACL_FLOAT16);
    // y is a Ref input/output, so the generated API takes it once.
    auto yDesc = sgl_cann::tensor_descriptor(y, ACL_FLOAT);

    static auto query = reinterpret_cast<decltype(&aclnnSglSgemmvShrinkGetWorkspaceSize)>(
        sgl_cann::op_api_symbol("aclnnSglSgemmvShrinkGetWorkspaceSize"));
    static auto execute =
        reinterpret_cast<decltype(&aclnnSglSgemmvShrink)>(sgl_cann::op_api_symbol("aclnnSglSgemmvShrink"));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    // Generated ACLNN order is: all tensor inputs (y is a Ref input), then attributes, then outputs.
    auto status = query(xDesc.get(), weightDesc.get(), loraIndicesDesc.get(), seqLenDesc.get(), loraRanksDesc.get(),
                        loraScalesDesc.get(), yDesc.get(), slice_count, &workspaceSize, &executor);
    TORCH_CHECK(status == ACL_SUCCESS, "sgemmv_shrink workspace query failed: ", aclGetRecentErrMsg());
    TORCH_CHECK(workspaceSize <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                "ACLNN workspace size exceeds the Tensor allocation limit");

    auto workspace = at::empty({static_cast<int64_t>(workspaceSize)}, y.options().dtype(at::kByte));
    auto stream = c10_npu::getCurrentNPUStream();
    x.record_stream(stream);
    weight.record_stream(stream);
    lora_indices.record_stream(stream);
    seq_len.record_stream(stream);
    lora_ranks.record_stream(stream);
    lora_scales.record_stream(stream);
    y.record_stream(stream);
    workspace.record_stream(stream);
    auto aclStream = stream.stream(false);

    // Tensor captures cover the host queue; record_stream covers asynchronous device use.
    auto launch = [x, weight, lora_indices, seq_len, lora_ranks, lora_scales, y, workspace, xDesc, weightDesc,
                   loraIndicesDesc, seqLenDesc, loraRanksDesc, loraScalesDesc, yDesc, workspaceSize, executor,
                   aclStream]() -> int {
        auto result = execute(workspaceSize ? workspace.data_ptr() : nullptr, workspaceSize, executor, aclStream);
        TORCH_CHECK(result == ACL_SUCCESS, "sgemmv_shrink ACLNN execution failed: ", aclGetRecentErrMsg());
        return result;
    };
    at_npu::native::OpCommand::RunOpApi("aclnnSglSgemmvShrink", launch);
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def(
        "sgemmv_shrink(Tensor x, Tensor weight, Tensor lora_indices, Tensor seq_len, Tensor lora_ranks,"
        " Tensor lora_scales, Tensor(a!) y, int slice_count=1) -> ()");
}
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("sgemmv_shrink", &sgemmv_shrink);
}
