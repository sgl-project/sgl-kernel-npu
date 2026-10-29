#pragma once

#include "aclnn_runtime.h"
#include <array>
#include <c10/core/DeviceGuard.h>
#include <limits>

namespace sgl_cache_loc {

// Public operators validate dtype and location contiguity; direct Torch callers must satisfy that contract.
template <typename Query, typename Execute>
inline at::Tensor Launch(at::Tensor req_indices, at::Tensor token_pool, at::Tensor start_offset, at::Tensor end_offset,
                         at::Tensor out_cache_loc, bool is_update, const char *name, Query query, Execute execute)
{
    auto target = is_update ? out_cache_loc : token_pool;
    const c10::DeviceGuard deviceGuard(target.device());
    std::array<at::Tensor, 5> tensors = {req_indices, token_pool, start_offset, end_offset, out_cache_loc};
    auto reqDtype = req_indices.scalar_type() == at::kInt ? ACL_INT32 : ACL_INT64;
    std::array<std::shared_ptr<aclTensor>, 5> descriptors = {
        sgl_cann::tensor_descriptor(req_indices, reqDtype),    sgl_cann::tensor_descriptor(token_pool, ACL_INT32),
        sgl_cann::tensor_descriptor(start_offset, ACL_INT64),  sgl_cann::tensor_descriptor(end_offset, ACL_INT64),
        sgl_cann::tensor_descriptor(out_cache_loc, ACL_INT32),
    };
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto status = query(descriptors[0].get(), descriptors[1].get(), descriptors[2].get(), descriptors[3].get(),
                        descriptors[4].get(), &workspaceSize, &executor);
    TORCH_CHECK(status == ACL_SUCCESS, name, " workspace query failed: ", aclGetRecentErrMsg());
    TORCH_CHECK(workspaceSize <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                "ACLNN workspace size exceeds the Tensor allocation limit");
    auto workspace = at::empty({static_cast<int64_t>(workspaceSize)}, target.options().dtype(at::kByte));
    auto stream = c10_npu::getCurrentNPUStream();
    for (const auto &tensor : tensors)
        tensor.record_stream(stream);
    workspace.record_stream(stream);
    auto aclStream = stream.stream(false);
    // Captures retain queued host resources; record_stream retains asynchronous device storage.
    auto launch = [tensors, descriptors, workspace, workspaceSize, executor, aclStream, execute, name]() -> int {
        auto result = execute(workspaceSize ? workspace.data_ptr() : nullptr, workspaceSize, executor, aclStream);
        TORCH_CHECK(result == ACL_SUCCESS, name, " execution failed: ", aclGetRecentErrMsg());
        return result;
    };
    at_npu::native::OpCommand::RunOpApi(name, launch);
    return target;
}

}  // namespace sgl_cache_loc
