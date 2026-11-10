#include "aclnn_runtime.h"
#include "aclnn_sgl_assign_cache.h"
#include <acl/acl_rt.h>
#include <array>
#include <c10/core/DeviceGuard.h>
#include <limits>
#include <torch/library.h>

namespace {
// The kernel copies bit patterns using integer carriers selected by destination byte width.
aclDataType PayloadType(size_t bytes)
{
    switch (bytes) {
        case 1:
            return ACL_INT8;
        case 2:
            return ACL_INT16;
        case 4:
            return ACL_INT32;
        case 8:
            return ACL_INT64;
        default:
            return ACL_DT_UNDEFINED;
    }
}
std::shared_ptr<aclTensor> SourceBytes(const at::Tensor &source)
{
    const int64_t size = source.numel() * source.element_size();
    const int64_t stride = 1;
    auto *descriptor = aclCreateTensor(&size, 1, ACL_UINT8, &stride, 0, ACL_FORMAT_ND, &size, 1, source.data_ptr());
    TORCH_CHECK(descriptor, "aclCreateTensor failed: ", aclGetRecentErrMsg());
    return {descriptor, [](aclTensor *value) { aclDestroyTensor(value); }};
}
bool assign_cache_op(at::Tensor dst, at::Tensor src, at::Tensor dst_start, at::Tensor dst_end, at::Tensor src_start,
                     at::Tensor src_end)
{
    const auto payloadType = PayloadType(dst.element_size());
    // No kernel branch performs a copy for other element widths.
    if (payloadType == ACL_DT_UNDEFINED) return true;
    const c10::DeviceGuard deviceGuard(dst.device());
    static auto query = reinterpret_cast<decltype(&aclnnSglAssignCacheGetWorkspaceSize)>(
        sgl_cann::op_api_symbol("aclnnSglAssignCacheGetWorkspaceSize"));
    static auto execute =
        reinterpret_cast<decltype(&aclnnSglAssignCache)>(sgl_cann::op_api_symbol("aclnnSglAssignCache"));
    const auto deviceId = static_cast<uint32_t>(dst.get_device());
    int64_t cores = 0;
    auto coreStatus = aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &cores);
    if (coreStatus != ACL_SUCCESS || cores <= 0) {
        coreStatus = aclGetDeviceCapability(deviceId, ACL_DEVICE_INFO_VECTOR_CORE_NUM, &cores);
    }
    TORCH_CHECK(coreStatus == ACL_SUCCESS && cores > 0, "AssignCache VECTOR_CORE_NUM query failed: device=", deviceId,
                ", status=", coreStatus, ", cores=", cores);
    auto sync = at::zeros({static_cast<int64_t>(cores * 64 + 32)}, dst.options().dtype(at::kByte));
    std::array<at::Tensor, 7> tensors = {dst, src, dst_start, dst_end, src_start, src_end, sync};
    std::array<std::shared_ptr<aclTensor>, 7> descriptors;
    for (size_t i = 0; i < tensors.size(); ++i) {
        if (i == 1) {
            descriptors[i] = SourceBytes(src);
        } else {
            auto dtype = i == 0 ? payloadType : (i == 6 ? ACL_UINT8 : ACL_INT64);
            descriptors[i] = sgl_cann::tensor_descriptor(tensors[i], dtype);
        }
    }
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto status = query(descriptors[0].get(), descriptors[1].get(), descriptors[2].get(), descriptors[3].get(),
                        descriptors[4].get(), descriptors[5].get(), descriptors[6].get(), &workspaceSize, &executor);
    TORCH_CHECK(status == ACL_SUCCESS, "AssignCache workspace query failed: ", aclGetRecentErrMsg());
    TORCH_CHECK(workspaceSize <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                "ACLNN workspace size exceeds the Tensor allocation limit");
    auto workspace = at::empty({static_cast<int64_t>(workspaceSize)}, dst.options().dtype(at::kByte));
    auto stream = c10_npu::getCurrentNPUStream();
    for (const auto &tensor : tensors)
        tensor.record_stream(stream);
    workspace.record_stream(stream);
    auto aclStream = stream.stream(false);
    auto launch = [tensors, descriptors, workspace, workspaceSize, executor, aclStream]() -> int {
        auto result = execute(workspaceSize ? workspace.data_ptr() : nullptr, workspaceSize, executor, aclStream);
        TORCH_CHECK(result == ACL_SUCCESS, "AssignCache execution failed: ", aclGetRecentErrMsg());
        return result;
    };
    at_npu::native::OpCommand::RunOpApi("aclnnSglAssignCache", launch);
    return true;
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def(
        "assign_cache_op(Tensor(a!) out, Tensor src, Tensor dst_start_idx, Tensor dst_end_idx, "
        "Tensor src_start_idx, Tensor src_end_idx) -> bool");
}
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("assign_cache_op", &assign_cache_op);
}
