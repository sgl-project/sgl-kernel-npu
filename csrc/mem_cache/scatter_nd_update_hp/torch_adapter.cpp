// SPDX-License-Identifier: Apache-2.0
#include "aclnn_runtime.h"
#include "aclnn_sgl_scatter_nd_update_hp.h"

#include <c10/core/DeviceGuard.h>
#include <limits>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>

namespace {

using TensorDescriptor = std::unique_ptr<aclTensor, decltype(&aclDestroyTensor)>;
using IntArrayDescriptor = std::unique_ptr<aclIntArray, decltype(&aclDestroyIntArray)>;

aclDataType ToAclDataType(at::ScalarType dtype)
{
    switch (dtype) {
        case at::kChar:
            return ACL_INT8;
        case at::kShort:
            return ACL_INT16;
        case at::kInt:
            return ACL_INT32;
        case at::kLong:
            return ACL_INT64;
        case at::kHalf:
            return ACL_FLOAT16;
        case at::kFloat:
            return ACL_FLOAT;
        case at::kBool:
            return ACL_BOOL;
        case at::kBFloat16:
            return ACL_BF16;
        case at::kFloat8_e4m3fn:
            return ACL_FLOAT8_E4M3FN;
        case at::kFloat8_e5m2:
            return ACL_FLOAT8_E5M2;
        default:
            TORCH_CHECK(false, "Unsupported cache dtype: ", dtype);
    }
}

aclDataType CheckDtypes(const at::Tensor &cache, const at::Tensor &indices, const at::Tensor &updates)
{
    const auto dtype = cache.scalar_type();
    const auto acl_dtype = ToAclDataType(dtype);
    TORCH_CHECK(indices.scalar_type() == at::kLong || indices.scalar_type() == at::kInt,
                "indices must be int32 or int64, got ", indices.scalar_type());
    TORCH_CHECK(updates.scalar_type() == dtype, "updates dtype must match cache dtype");
    return acl_dtype;
}

TensorDescriptor Describe(const at::Tensor &tensor, aclDataType dtype)
{
    auto format = static_cast<aclFormat>(at_npu::native::get_npu_format(tensor));
    c10::SmallVector<int64_t, 5> storage_dims;
    // Describe views into larger allocations using their storage metadata.
    switch (format) {
        case ACL_FORMAT_ND:
        case ACL_FORMAT_NCHW:
        case ACL_FORMAT_NHWC:
        case ACL_FORMAT_NCDHW:
            switch (tensor.dim()) {
                case 3:
                    format = ACL_FORMAT_NCL;
                    break;
                case 4:
                    format = ACL_FORMAT_NCHW;
                    break;
                case 5:
                    format = ACL_FORMAT_NCDHW;
                    break;
                default:
                    format = ACL_FORMAT_ND;
                    break;
            }
            storage_dims.push_back(tensor.storage().nbytes() / tensor.element_size());
            break;
        default: {
            const auto sizes = at_npu::native::get_npu_storage_sizes(tensor);
            storage_dims.assign(sizes.begin(), sizes.end());
            break;
        }
    }
    auto *descriptor =
        aclCreateTensor(tensor.sizes().data(), tensor.dim(), dtype, tensor.strides().data(), tensor.storage_offset(),
                        format, storage_dims.data(), storage_dims.size(), const_cast<void *>(tensor.storage().data()));
    TORCH_CHECK(descriptor, "aclCreateTensor failed: ", aclGetRecentErrMsg());
    return TensorDescriptor(descriptor, &aclDestroyTensor);
}

// Prepared descriptors are owned together until the queued launch completes.
struct PreparedCall {
    TensorDescriptor self{nullptr, &aclDestroyTensor};
    TensorDescriptor index{nullptr, &aclDestroyTensor};
    TensorDescriptor update{nullptr, &aclDestroyTensor};
    IntArrayDescriptor strides{nullptr, &aclDestroyIntArray};
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
};

void Scatter(at::Tensor &cache, const at::Tensor &indices, const at::Tensor &updates)
{
    TORCH_CHECK(cache.device().type() == c10::DeviceType::PrivateUse1, "cache must be on NPU");
    TORCH_CHECK(indices.device() == cache.device() && updates.device() == cache.device(),
                "All tensors must be on the same NPU");
    const c10::DeviceGuard guard(cache.device());
    const auto cache_dtype = CheckDtypes(cache, indices, updates);
    if (cache.numel() == 0 || indices.numel() == 0 || updates.numel() == 0) {
        return;
    }
    auto contiguous_indices = indices.contiguous();
    auto contiguous_updates = updates.contiguous();
    auto stream = c10_npu::getCurrentNPUStream();
    auto acl_stream = stream.stream(false);
    static auto query = reinterpret_cast<decltype(&aclnnSglScatterNdUpdateHpGetWorkspaceSize)>(
        sgl_cann::op_api_symbol("aclnnSglScatterNdUpdateHpGetWorkspaceSize"));
    static auto execute =
        reinterpret_cast<decltype(&aclnnSglScatterNdUpdateHp)>(sgl_cann::op_api_symbol("aclnnSglScatterNdUpdateHp"));
    auto prepared = std::make_shared<PreparedCall>();
    prepared->self = Describe(cache, cache_dtype);
    prepared->index = Describe(contiguous_indices, indices.scalar_type() == at::kLong ? ACL_INT64 : ACL_INT32);
    prepared->update = Describe(contiguous_updates, cache_dtype);
    prepared->strides.reset(aclCreateIntArray(cache.strides().data(), cache.dim()));
    TORCH_CHECK(prepared->strides, "aclCreateIntArray failed: ", aclGetRecentErrMsg());
    auto status = query(prepared->self.get(), prepared->index.get(), prepared->update.get(), prepared->strides.get(),
                        false, &prepared->workspace_size, &prepared->executor);
    TORCH_CHECK(status == ACL_SUCCESS, "Scatter HP prepare failed: ", aclGetRecentErrMsg());
    TORCH_CHECK(prepared->workspace_size <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                "ACLNN workspace size exceeds the Tensor allocation limit");
    // Allocate on the caller: allocator retries may wait for the host queue,
    // which would deadlock if allocation ran inside its current handler.
    void *workspace_addr = nullptr;
    at::Tensor workspace;
    if (prepared->workspace_size != 0) {
        workspace = at::empty({static_cast<int64_t>(prepared->workspace_size)}, cache.options().dtype(at::kByte));
        workspace_addr = workspace.data_ptr();
    }
    // Capture converted descriptors and raw launch arguments, without retaining
    // the original tensors or workspace tensor.
    auto launch = [prepared, workspace_addr, acl_stream]() -> int {
        auto status = execute(workspace_addr, prepared->workspace_size, prepared->executor, acl_stream);
        TORCH_CHECK(status == ACL_SUCCESS, "Scatter HP launch failed: ", aclGetRecentErrMsg());
        return status;
    };
    at_npu::native::OpCommand::RunOpApi("aclnnSglScatterNdUpdateHp", launch);
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("scatter_nd_update_hp(Tensor(a!) cache, Tensor indices, Tensor updates) -> ()");
}
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("scatter_nd_update_hp", &Scatter);
}
