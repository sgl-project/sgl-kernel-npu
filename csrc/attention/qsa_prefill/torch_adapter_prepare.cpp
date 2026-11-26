// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under the CANN Open Software License Agreement Version 2.0.
#include "aclnn_runtime.h"
#include "aclnn_sgl_qsa_prefill_prepare.h"
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>

#include <limits>
#include <vector>

namespace {

std::shared_ptr<aclTensor> flat_int32_descriptor(const at::Tensor &tensor, int64_t span)
{
    int64_t stride = 1;
    auto *descriptor = aclCreateTensor(&span, 1, ACL_INT32, &stride, 0, ACL_FORMAT_ND, &span, 1, tensor.data_ptr());
    TORCH_CHECK(descriptor, "aclCreateTensor failed: ", aclGetRecentErrMsg());
    return {descriptor, [](aclTensor *value) { aclDestroyTensor(value); }};
}

std::vector<at::Tensor> qsa_prefill_prepare(const at::Tensor &b, const at::Tensor &table, const at::Tensor &req,
                                            const at::Tensor &flags, int64_t length, int64_t base, int64_t capacity)
{
    TORCH_CHECK(b.device().type() == c10::DeviceType::PrivateUse1, "NPU tensors required");
    const c10::DeviceGuard deviceGuard(b.device());
    TORCH_CHECK(b.dim() == 2 && b.size(1) == 512 && b.scalar_type() == at::kInt && b.is_contiguous(),
                "contiguous int32 [R,512] blocks required");
    TORCH_CHECK(table.dim() == 2 && table.scalar_type() == at::kInt && table.size(0) > 0 && table.stride(1) == 1 &&
                    table.stride(0) >= 0 && table.stride(0) <= std::numeric_limits<int32_t>::max() &&
                    table.device() == b.device(),
                "int32 unit-stride request table required");
    TORCH_CHECK(req.dim() == 1 && req.numel() == 1 && req.scalar_type() == at::kLong && req.is_contiguous() &&
                    req.device() == b.device(),
                "one int64 request row required");
    TORCH_CHECK(length >= 0 && length <= table.size(1) && length <= std::numeric_limits<int32_t>::max() &&
                    base >= std::numeric_limits<int32_t>::min() &&
                    base + b.size(0) < std::numeric_limits<int32_t>::max() && capacity > 0 &&
                    capacity <= std::numeric_limits<int32_t>::max(),
                "invalid sequence/capacity range");
    TORCH_CHECK(flags.dim() == 1 && flags.numel() == b.size(0) * 16 && flags.scalar_type() == at::kInt &&
                    flags.is_contiguous() && flags.device() == b.device(),
                "contiguous int32 stride16 exact-sharing flags required");
    TORCH_CHECK(b.size(0) <= std::numeric_limits<int32_t>::max() / 16, "too many query rows");

    const int64_t rows = b.size(0);
    constexpr int64_t kOutWidth = 2064;
    auto slots = at::empty({rows, kOutWidth}, b.options());
    auto runs = at::empty_like(slots);
    auto counts = at::empty({rows * 16}, b.options());
    if (rows == 0) {
        return {slots, runs, counts};
    }

    // Describe request_table as a flat contiguous span so ACLNN preserves its
    // padded row stride without inserting a ViewCopy.
    const int64_t tableStride = table.stride(0);
    const int64_t tableSpan = (table.size(0) - 1) * tableStride + table.size(1);
    const at::Tensor tableBuf = tableSpan > 0 ? table : at::zeros({1}, table.options());
    const int64_t descSpan = tableSpan > 0 ? tableSpan : 1;

    static auto queryWorkspace = reinterpret_cast<decltype(&aclnnSglQsaPrefillPrepareGetWorkspaceSize)>(
        sgl_cann::op_api_symbol("aclnnSglQsaPrefillPrepareGetWorkspaceSize"));
    static auto execute =
        reinterpret_cast<decltype(&aclnnSglQsaPrefillPrepare)>(sgl_cann::op_api_symbol("aclnnSglQsaPrefillPrepare"));

    auto bDesc = sgl_cann::tensor_descriptor(b, ACL_INT32);
    auto tableDesc = flat_int32_descriptor(tableBuf, descSpan);
    auto reqDesc = sgl_cann::tensor_descriptor(req, ACL_INT64);
    auto flagsDesc = sgl_cann::tensor_descriptor(flags, ACL_INT32);
    auto slotsDesc = sgl_cann::tensor_descriptor(slots, ACL_INT32);
    auto runsDesc = sgl_cann::tensor_descriptor(runs, ACL_INT32);
    auto countsDesc = sgl_cann::tensor_descriptor(counts, ACL_INT32);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    const auto status =
        queryWorkspace(bDesc.get(), tableDesc.get(), reqDesc.get(), flagsDesc.get(), tableStride, length, base,
                       capacity, slotsDesc.get(), runsDesc.get(), countsDesc.get(), &workspaceSize, &executor);
    TORCH_CHECK(status == ACL_SUCCESS, "qsa_prefill_prepare workspace query failed: ", aclGetRecentErrMsg());
    TORCH_CHECK(workspaceSize <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                "ACLNN workspace size exceeds the Tensor allocation limit");
    auto workspace = at::empty({static_cast<int64_t>(workspaceSize)}, b.options().dtype(at::kByte));

    auto stream = c10_npu::getCurrentNPUStream();
    std::vector<at::Tensor> tensors{b, tableBuf, req, flags, slots, runs, counts, workspace};
    for (auto &tensor : tensors) {
        if (tensor.numel() > 0) {
            tensor.record_stream(stream);
        }
    }
    std::vector<std::shared_ptr<aclTensor>> descriptors{bDesc,     tableDesc, reqDesc,   flagsDesc,
                                                        slotsDesc, runsDesc,  countsDesc};
    auto aclStream = stream.stream(false);
    auto launch = [tensors, descriptors, workspace, workspaceSize, executor, aclStream]() -> int {
        const auto result = execute(workspaceSize ? workspace.data_ptr() : nullptr, workspaceSize, executor, aclStream);
        TORCH_CHECK(result == ACL_SUCCESS, "qsa_prefill_prepare ACLNN execution failed: ", aclGetRecentErrMsg());
        return result;
    };
    at_npu::native::OpCommand::RunOpApi("aclnnSglQsaPrefillPrepare", launch);
    return {slots, runs, counts};
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def(
        "qsa_prefill_prepare(Tensor blocks, Tensor table, Tensor request_row, Tensor flags, int length, int base, int "
        "capacity) -> Tensor[]");
}
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("qsa_prefill_prepare", &qsa_prefill_prepare);
}
