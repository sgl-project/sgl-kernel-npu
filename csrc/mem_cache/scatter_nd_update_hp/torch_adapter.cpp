// SPDX-License-Identifier: Apache-2.0
#include "aclnn_runtime.h"
#include "aclnn_scatter_nd_update_sk.h"

#include <c10/core/DeviceGuard.h>
#include <c10/util/ScopeExit.h>
#include <limits>
#include <torch/library.h>
#include <torch_npu/csrc/core/NPUStorageImpl.h>

namespace {

class HostMemoryPool
{
    using InitFn = int (*)(void *, bool);
    using VoidFn = void (*)(void *, bool);

public:
    HostMemoryPool()
    {
        // Resolve cleanup callbacks before activating the thread-local pool.
        (void)Release();
        (void)UnInit();
        if (auto init = Init()) {
            init(nullptr, false);
        }
    }
    ~HostMemoryPool()
    {
        if (auto uninit = UnInit()) {
            uninit(nullptr, false);
        }
    }
    HostMemoryPool(const HostMemoryPool &) = delete;
    HostMemoryPool &operator=(const HostMemoryPool &) = delete;

    static VoidFn Release()
    {
        static auto fn = reinterpret_cast<VoidFn>(sgl_cann::try_op_api_symbol("ReleaseHugeMem"));
        return fn;
    }

private:
    static InitFn Init()
    {
        static auto fn = reinterpret_cast<InitFn>(sgl_cann::try_op_api_symbol("InitHugeMemThreadLocal"));
        return fn;
    }
    static VoidFn UnInit()
    {
        static auto fn = reinterpret_cast<VoidFn>(sgl_cann::try_op_api_symbol("UnInitHugeMemThreadLocal"));
        return fn;
    }
};

// Prepared descriptors are owned together until the queued launch completes.
auto Describe(const at::Tensor &tensor)
{
    aclDataType dtype;
    switch (tensor.scalar_type()) {
        case at::kByte:
            dtype = ACL_UINT8;
            break;
        case at::kChar:
            dtype = ACL_INT8;
            break;
        case at::kShort:
            dtype = ACL_INT16;
            break;
        case at::kInt:
            dtype = ACL_INT32;
            break;
        case at::kLong:
            dtype = ACL_INT64;
            break;
        case at::kHalf:
            dtype = ACL_FLOAT16;
            break;
        case at::kFloat:
            dtype = ACL_FLOAT;
            break;
        case at::kBool:
            dtype = ACL_BOOL;
            break;
        case at::kBFloat16:
            dtype = ACL_BF16;
            break;
        case at::kFloat8_e4m3fn:
            dtype = ACL_FLOAT8_E4M3FN;
            break;
        case at::kFloat8_e5m2:
            dtype = ACL_FLOAT8_E5M2;
            break;
        default:
            TORCH_CHECK(false, "Unsupported dtype: ", tensor.scalar_type());
    }
    // torch_npu does not export NPUBridge::GetNpuStorageImplDesc on all builds.
    // Use its storage type directly.
    const auto &storage =
        static_cast<const torch_npu::NPUStorageImpl *>(tensor.storage().unsafeGetStorageImpl())->npu_desc_;
    auto format = storage.npu_format_;
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
        default:
            storage_dims.assign(storage.storage_sizes_.begin(), storage.storage_sizes_.end());
            break;
    }
    auto *descriptor =
        aclCreateTensor(tensor.sizes().data(), tensor.dim(), dtype, tensor.strides().data(), tensor.storage_offset(),
                        format, storage_dims.data(), storage_dims.size(), const_cast<void *>(tensor.storage().data()));
    TORCH_CHECK(descriptor, "aclCreateTensor failed: ", aclGetRecentErrMsg());
    return std::unique_ptr<aclTensor, decltype(&aclDestroyTensor)>(descriptor, &aclDestroyTensor);
}

struct PreparedCall {
    std::unique_ptr<aclTensor, decltype(&aclDestroyTensor)> self{nullptr, &aclDestroyTensor};
    std::unique_ptr<aclTensor, decltype(&aclDestroyTensor)> index{nullptr, &aclDestroyTensor};
    std::unique_ptr<aclTensor, decltype(&aclDestroyTensor)> update{nullptr, &aclDestroyTensor};
    std::unique_ptr<aclIntArray, decltype(&aclDestroyIntArray)> strides{nullptr, &aclDestroyIntArray};
    decltype(HostMemoryPool::Release()) release_memory = HostMemoryPool::Release();
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    void Release()
    {
        // Destroy descriptors before returning their pooled memory.
        // Initialize/uninitialize on the caller, release on the launch thread.
        strides.reset();
        update.reset();
        index.reset();
        self.reset();
        if (release_memory != nullptr) {
            release_memory(nullptr, false);
            release_memory = nullptr;
        }
    }

    ~PreparedCall()
    {
        Release();
    }
};

void Scatter(at::Tensor &cache, const at::Tensor &indices, const at::Tensor &updates)
{
    TORCH_CHECK(cache.device().type() == c10::DeviceType::PrivateUse1, "cache must be on NPU");
    TORCH_CHECK(indices.device() == cache.device() && updates.device() == cache.device(),
                "All tensors must be on the same NPU");
    const c10::DeviceGuard guard(cache.device());
    auto stream = c10_npu::getCurrentNPUStream();
    auto acl_stream = stream.stream(false);
    static auto query = reinterpret_cast<decltype(&aclnnSglScatterNdUpdateHpGetWorkspaceSize)>(
        sgl_cann::op_api_symbol("aclnnSglScatterNdUpdateHpGetWorkspaceSize"));
    static auto execute =
        reinterpret_cast<decltype(&aclnnSglScatterNdUpdateHp)>(sgl_cann::op_api_symbol("aclnnSglScatterNdUpdateHp"));
    HostMemoryPool pool;
    auto prepared = std::make_shared<PreparedCall>();
    prepared->self = Describe(cache);
    prepared->index = Describe(indices);
    prepared->update = Describe(updates);
    prepared->strides.reset(aclCreateIntArray(cache.strides().data(), cache.dim()));
    TORCH_CHECK(prepared->strides, "aclCreateIntArray failed: ", aclGetRecentErrMsg());
    auto status = query(prepared->self.get(), prepared->index.get(), prepared->update.get(), prepared->strides.get(),
                        &prepared->workspace_size, &prepared->executor);
    TORCH_CHECK(status == ACL_SUCCESS, "Scatter HP prepare failed: ", aclGetRecentErrMsg());
    TORCH_CHECK(prepared->workspace_size <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                "ACLNN workspace size exceeds the Tensor allocation limit");
    // Allocate on the caller: allocator retries may wait for the host queue,
    // which would deadlock if allocation ran inside its current handler.
    void *workspace_addr = nullptr;
    if (prepared->workspace_size != 0) {
        auto workspace = at::empty({static_cast<int64_t>(prepared->workspace_size)}, cache.options().dtype(at::kByte));
        workspace_addr = const_cast<void *>(workspace.storage().data());
    }
    // Capture converted descriptors and raw launch arguments, without retaining
    // the original tensors or workspace tensor.
    auto launch = [prepared, workspace_addr, acl_stream]() -> int {
        auto cleanup = c10::make_scope_exit([&] { prepared->Release(); });
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
