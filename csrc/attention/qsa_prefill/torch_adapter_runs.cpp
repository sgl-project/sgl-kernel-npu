// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under the CANN Open Software License Agreement Version 2.0.
#include "aclnn_runtime.h"
#include "aclnn_sgl_qsa_prefill_runs.h"
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>

#include <cmath>
#include <limits>
#include <tuple>
#include <vector>

namespace {

void check_tensor(const at::Tensor &tensor, const at::Tensor &ref, at::ScalarType dtype)
{
    TORCH_CHECK(tensor.device() == ref.device() && tensor.scalar_type() == dtype && tensor.is_contiguous(),
                "QSA tensors must have the expected dtype, be contiguous, and share a device");
}

std::tuple<at::Tensor, at::Tensor> qsa_prefill_runs(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
                                                    const at::Tensor &slots, const at::Tensor &runs,
                                                    const at::Tensor &counts, double scale)
{
    TORCH_CHECK(q.device().type() == c10::DeviceType::PrivateUse1, "NPU tensors required");
    const c10::DeviceGuard deviceGuard(q.device());
    check_tensor(q, q, at::kBFloat16);
    check_tensor(k, q, at::kBFloat16);
    check_tensor(v, q, at::kBFloat16);
    check_tensor(slots, q, at::kInt);
    check_tensor(runs, q, at::kInt);
    check_tensor(counts, q, at::kInt);

    TORCH_CHECK(q.dim() == 3 && q.size(2) == 256, "packed Q must be [G,16*H,256]");
    TORCH_CHECK(k.dim() == 3 && k.size(0) > 0 && k.size(2) == 256 && v.sizes() == k.sizes(),
                "K,V must be [capacity,KV,256]");
    // Packed Q is [G,16*H,256], so a sharing group needs 16*(H/KV) of the 128
    // rows in the per-stage workspace block. The whitelist of validated layouts
    // lives in the Python operator, which is the only supported entry point.
    TORCH_CHECK((k.size(1) == 1 || k.size(1) == 2) && q.size(1) % (16 * k.size(1)) == 0 && q.size(1) / k.size(1) <= 128,
                "unsupported packed Q/KV heads");
    TORCH_CHECK(slots.dim() == 2 && slots.size(1) == 2064 && (slots.size(0) + 15) / 16 == q.size(0),
                "prepared slots must be [R,2064]");
    TORCH_CHECK(runs.sizes() == slots.sizes() && counts.dim() == 1 && counts.numel() == slots.size(0) * 16,
                "prepared metadata shape mismatch");
    const float scaleFloat = static_cast<float>(scale);
    TORCH_CHECK(std::isfinite(scale) && scale > 0.0 && std::isfinite(scaleFloat) && scaleFloat > 0.0f,
                "scale must be positive and finite");
    TORCH_CHECK(q.size(0) <= std::numeric_limits<int32_t>::max() / 2 &&
                    slots.size(0) <= std::numeric_limits<int32_t>::max() / 16 &&
                    k.size(0) <= std::numeric_limits<int32_t>::max(),
                "tensor extent exceeds int32 indexing");

    auto out = at::zeros(q.sizes(), q.options().dtype(at::kFloat));
    auto lse = at::full({q.size(0), q.size(1)}, -INFINITY, q.options().dtype(at::kFloat));
    if (q.size(0) == 0) {
        return {out, lse};
    }

    static auto queryWorkspace = reinterpret_cast<decltype(&aclnnSglQsaPrefillRunsGetWorkspaceSize)>(
        sgl_cann::op_api_symbol("aclnnSglQsaPrefillRunsGetWorkspaceSize"));
    static auto execute =
        reinterpret_cast<decltype(&aclnnSglQsaPrefillRuns)>(sgl_cann::op_api_symbol("aclnnSglQsaPrefillRuns"));

    auto qDesc = sgl_cann::tensor_descriptor(q, ACL_BF16);
    auto kDesc = sgl_cann::tensor_descriptor(k, ACL_BF16);
    auto vDesc = sgl_cann::tensor_descriptor(v, ACL_BF16);
    auto slotsDesc = sgl_cann::tensor_descriptor(slots, ACL_INT32);
    auto runsDesc = sgl_cann::tensor_descriptor(runs, ACL_INT32);
    auto countsDesc = sgl_cann::tensor_descriptor(counts, ACL_INT32);
    auto outDesc = sgl_cann::tensor_descriptor(out, ACL_FLOAT);
    auto lseDesc = sgl_cann::tensor_descriptor(lse, ACL_FLOAT);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    const auto status =
        queryWorkspace(qDesc.get(), kDesc.get(), vDesc.get(), slotsDesc.get(), runsDesc.get(), countsDesc.get(),
                       scaleFloat, outDesc.get(), lseDesc.get(), &workspaceSize, &executor);
    TORCH_CHECK(status == ACL_SUCCESS, "qsa_prefill_runs workspace query failed: ", aclGetRecentErrMsg());
    TORCH_CHECK(workspaceSize <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                "ACLNN workspace size exceeds the Tensor allocation limit");
    auto workspace = at::zeros({static_cast<int64_t>(workspaceSize)}, q.options().dtype(at::kByte));

    auto stream = c10_npu::getCurrentNPUStream();
    std::vector<at::Tensor> tensors{q, k, v, slots, runs, counts, out, lse, workspace};
    std::vector<std::shared_ptr<aclTensor>> descriptors{qDesc,    kDesc,      vDesc,   slotsDesc,
                                                        runsDesc, countsDesc, outDesc, lseDesc};
    auto aclStream = stream.stream(false);
    auto launch = [tensors, descriptors, workspace, workspaceSize, executor, aclStream]() -> int {
        const auto result = execute(workspaceSize ? workspace.data_ptr() : nullptr, workspaceSize, executor, aclStream);
        TORCH_CHECK(result == ACL_SUCCESS, "qsa_prefill_runs ACLNN execution failed: ", aclGetRecentErrMsg());
        return result;
    };
    at_npu::native::OpCommand::RunOpApi("aclnnSglQsaPrefillRuns", launch);
    return {out, lse};
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def(
        "qsa_prefill_runs(Tensor q, Tensor k, Tensor v, Tensor slots, Tensor runs, Tensor counts, float scale) -> "
        "(Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("qsa_prefill_runs", &qsa_prefill_runs);
}
