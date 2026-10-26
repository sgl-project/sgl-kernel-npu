// BSD 3-Clause License
// Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.

#include "aclnn_runtime.h"
#include "aclnn_sgl_apply_token_bit_mask.h"
#include <torch/library.h>
#include <c10/core/DeviceGuard.h>
#include <limits>

namespace {
at::Tensor apply_token_bitmask(at::Tensor logits, at::Tensor bitmask, c10::optional<at::Tensor> indices)
{
    TORCH_CHECK(logits.device().type() == c10::DeviceType::PrivateUse1, "logits must be on an NPU");
    TORCH_CHECK(bitmask.device() == logits.device(), "logits and bitmask must be on the same NPU");
    if (indices.has_value() && indices->defined() && indices->numel() > 0) {
        TORCH_CHECK(indices->device().is_cpu() || indices->device() == logits.device(),
                    "indices must be on CPU or the same NPU as logits");
    }
    const c10::DeviceGuard deviceGuard(logits.device());
    // Input validation
    TORCH_CHECK(logits.dim() == 2, "logits must be 2D, got ", logits.dim(), "D tensor");
    TORCH_CHECK(bitmask.dim() == 2, "bitmask must be 2D, got ", bitmask.dim(), "D tensor");
    TORCH_CHECK(logits.size(0) == bitmask.size(0), "logits and bitmask batch size mismatch: ", logits.size(0), " vs ",
                bitmask.size(0));
    TORCH_CHECK(logits.is_contiguous(), "logits must be contiguous");
    TORCH_CHECK(bitmask.is_contiguous(), "bitmask must be contiguous");
    TORCH_CHECK(bitmask.scalar_type() == at::kInt, "bitmask must be int32, got ", bitmask.scalar_type());

    at::ScalarType dtype = logits.scalar_type();
    TORCH_CHECK(dtype == at::kFloat || dtype == at::kHalf || dtype == at::kBFloat16,
                "logits must be float32, float16 or bfloat16, got ", dtype);

    int64_t batch = logits.size(0);
    int64_t vocabSize = logits.size(1);
    int64_t bitmaskWidth = bitmask.size(1);

    TORCH_CHECK(vocabSize > 0, "vocab_size must be > 0");

    // Handle indices
    bool hasIndices = indices.has_value() && indices->defined() && indices->numel() > 0;
    at::Tensor rowIndices;
    at::Tensor selectedBitmask;

    if (hasIndices) {
        rowIndices = indices.value().to(at::kLong).contiguous();
        TORCH_CHECK(rowIndices.dim() == 1, "indices must be 1D");
        selectedBitmask = bitmask.index({rowIndices}).contiguous();
    } else {
        selectedBitmask = bitmask;
    }

    int64_t numIndices = hasIndices ? rowIndices.size(0) : batch;
    if (numIndices == 0) return logits;

    // --- Alignment ---
    // Use 256 elements as alignment unit to satisfy both:
    //   logits DataCopy: 256 * sizeof(T) is a multiple of 32 bytes
    //   bitmask DataCopy: 256/32 = 8 int32 words = 32 bytes (DataCopy alignment for int32)
    // Most LLM vocab sizes (32000, 128256) are already multiples of 256.
    constexpr int64_t ALIGN_UNIT = 256;
    int64_t paddedVocabSize = ((vocabSize + ALIGN_UNIT - 1) / ALIGN_UNIT) * ALIGN_UNIT;
    bool needsPadding = (paddedVocabSize != vocabSize);

    // Prepare working logits [numIndices, paddedVocabSize]
    at::Tensor workingLogits;
    at::Tensor selectedLogits;
    if (hasIndices) {
        selectedLogits = logits.index({rowIndices});
        if (!selectedLogits.is_contiguous()) selectedLogits = selectedLogits.contiguous();
    } else {
        selectedLogits = logits;
    }

    if (needsPadding) {
        workingLogits = at::zeros({numIndices, paddedVocabSize}, selectedLogits.options());
        workingLogits.narrow(1, 0, vocabSize).copy_(selectedLogits);
    } else {
        workingLogits = selectedLogits;
    }

    // Prepare working bitmask [numIndices, paddedBitmaskWidth]
    int64_t paddedBitmaskWidth = (paddedVocabSize + 31) / 32;
    at::Tensor workingBitmask;
    if (static_cast<int64_t>(selectedBitmask.size(1)) >= paddedBitmaskWidth) {
        workingBitmask = selectedBitmask;
    } else {
        workingBitmask = at::zeros({numIndices, paddedBitmaskWidth}, selectedBitmask.options());
        workingBitmask.narrow(1, 0, selectedBitmask.size(1)).copy_(selectedBitmask);
    }

    // Ensure working tensors are contiguous
    if (!workingLogits.is_contiguous()) workingLogits = workingLogits.contiguous();
    if (!workingBitmask.is_contiguous()) workingBitmask = workingBitmask.contiguous();

    static auto query = reinterpret_cast<decltype(&aclnnSglApplyTokenBitMaskGetWorkspaceSize)>(
        sgl_cann::op_api_symbol("aclnnSglApplyTokenBitMaskGetWorkspaceSize"));
    static auto execute =
        reinterpret_cast<decltype(&aclnnSglApplyTokenBitMask)>(sgl_cann::op_api_symbol("aclnnSglApplyTokenBitMask"));
    auto aclDtype = dtype == at::kFloat ? ACL_FLOAT : (dtype == at::kHalf ? ACL_FLOAT16 : ACL_BF16);
    auto logitsDesc = sgl_cann::tensor_descriptor(workingLogits, aclDtype);
    auto maskDesc = sgl_cann::tensor_descriptor(workingBitmask, ACL_INT32);
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto status = query(logitsDesc.get(), maskDesc.get(), &workspaceSize, &executor);
    TORCH_CHECK(status == ACL_SUCCESS, "Bitmask workspace query failed: ", aclGetRecentErrMsg());
    TORCH_CHECK(workspaceSize <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                "ACLNN workspace size exceeds the Tensor allocation limit");
    auto workspace = at::empty({static_cast<int64_t>(workspaceSize)}, workingLogits.options().dtype(at::kByte));
    auto stream = c10_npu::getCurrentNPUStream();
    workingLogits.record_stream(stream);
    workingBitmask.record_stream(stream);
    workspace.record_stream(stream);
    auto aclStream = stream.stream(false);
    // Tensor captures cover the host queue; record_stream covers asynchronous device use.
    auto launch = [workingLogits, workingBitmask, workspace, logitsDesc, maskDesc, workspaceSize, executor,
                   aclStream]() -> int {
        auto result = execute(workspaceSize ? workspace.data_ptr() : nullptr, workspaceSize, executor, aclStream);
        TORCH_CHECK(result == ACL_SUCCESS, "Bitmask ACLNN execution failed: ", aclGetRecentErrMsg());
        return result;
    };
    at_npu::native::OpCommand::RunOpApi("aclnnSglApplyTokenBitMask", launch);

    // Copy results back
    auto result = needsPadding ? workingLogits.narrow(1, 0, vocabSize) : workingLogits;
    if (hasIndices) {
        logits.index_put_({rowIndices}, result);
    } else if (needsPadding) {
        logits.copy_(result);
    }

    return logits;
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("apply_token_bitmask(Tensor(a!) logits, Tensor bitmask, Tensor? indices=None) -> Tensor(a!)");
}
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("apply_token_bitmask", &apply_token_bitmask);
}
