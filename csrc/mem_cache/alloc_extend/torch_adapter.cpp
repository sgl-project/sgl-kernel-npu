#include "aclnn_runtime.h"
#include "aclnn_sgl_alloc_extend.h"
#include <array>
#include <c10/core/DeviceGuard.h>
#include <limits>
#include <stdexcept>
#include <torch/library.h>

namespace {
void alloc_extend(at::Tensor pre_lens, at::Tensor seq_lens, at::Tensor last_loc, at::Tensor free_pages,
                  int64_t page_size, at::Tensor out_indices, at::Tensor values)
{
    if (pre_lens.options().dtype() != at::kLong || seq_lens.options().dtype() != at::kLong ||
        last_loc.options().dtype() != at::kLong || free_pages.options().dtype() != at::kLong ||
        out_indices.options().dtype() != at::kLong) {
        throw std::invalid_argument("Only support int64 input dtype");
    }
    const c10::DeviceGuard deviceGuard(pre_lens.device());
    static auto query = reinterpret_cast<decltype(&aclnnSglAllocExtendGetWorkspaceSize)>(
        sgl_cann::op_api_symbol("aclnnSglAllocExtendGetWorkspaceSize"));
    static auto execute =
        reinterpret_cast<decltype(&aclnnSglAllocExtend)>(sgl_cann::op_api_symbol("aclnnSglAllocExtend"));
    std::array<at::Tensor, 6> tensors = {pre_lens, seq_lens, last_loc, free_pages, out_indices, values};
    std::array<std::shared_ptr<aclTensor>, 6> descriptors;
    for (size_t i = 0; i < tensors.size(); ++i) {
        descriptors[i] = sgl_cann::tensor_descriptor(tensors[i], ACL_INT64);
    }
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto status = query(descriptors[0].get(), descriptors[1].get(), descriptors[2].get(), descriptors[3].get(),
                        descriptors[4].get(), descriptors[5].get(), page_size, &workspaceSize, &executor);
    TORCH_CHECK(status == ACL_SUCCESS, "AllocExtend workspace query failed: ", aclGetRecentErrMsg());
    TORCH_CHECK(workspaceSize <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                "ACLNN workspace size exceeds the Tensor allocation limit");
    auto workspace = at::empty({static_cast<int64_t>(workspaceSize)}, pre_lens.options().dtype(at::kByte));
    auto stream = c10_npu::getCurrentNPUStream();
    for (const auto &tensor : tensors)
        tensor.record_stream(stream);
    workspace.record_stream(stream);
    auto aclStream = stream.stream(false);
    auto launch = [tensors, descriptors, workspace, workspaceSize, executor, aclStream]() -> int {
        auto result = execute(workspaceSize ? workspace.data_ptr() : nullptr, workspaceSize, executor, aclStream);
        TORCH_CHECK(result == ACL_SUCCESS, "AllocExtend execution failed: ", aclGetRecentErrMsg());
        return result;
    };
    at_npu::native::OpCommand::RunOpApi("aclnnSglAllocExtend", launch);
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def(
        "alloc_extend(Tensor pre_lens, Tensor seq_lens, Tensor last_loc, Tensor free_pages, int page_size, "
        "Tensor(a!) out_indices, Tensor(b!) values) -> ()");
}
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("alloc_extend", &alloc_extend);
}
