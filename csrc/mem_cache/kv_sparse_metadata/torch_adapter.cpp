#include "metadata_args.h"
#include "aclnn_runtime.h"
#include "aclnn_build_kv_sparse_metadata.h"
#include <ATen/record_function.h>
#include <c10/core/DeviceGuard.h>
#include <pybind11/stl.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/core/npu/NPUEvent.h>
#include <array>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace sglang::kv_metadata {
namespace {
at::Tensor PrepareIndices(const at::Tensor &tensor, const at::Device &device, const c10_npu::NPUStream &stream)
{
    TORCH_CHECK(tensor.device() == device && tensor.scalar_type() == at::kLong &&
                tensor.dim() == 1, "AICPU metadata indices must be 1-D int64 tensors on the KV NPU");
    tensor.record_stream(stream);
    auto indices = tensor.contiguous();
    TORCH_CHECK(at_npu::native::get_npu_format(indices) == ACL_FORMAT_ND, "AICPU metadata indices must use ND format");
    return indices;
}

uint64_t PageBytes(const at::Tensor &tensor, int64_t page_size)
{
    TORCH_CHECK(tensor.dim() == 5 && tensor.size(2) == page_size, "Expected 5-D KV buffers with matching page_size");
    uint64_t bytes = tensor.element_size();
    int64_t stride = 1;
    for (int i = 4; i >= 2; --i) {
        TORCH_CHECK(tensor.size(i) > 0 && tensor.stride(i) == stride, "KV page payload must be nonempty and packed");
        TORCH_CHECK(bytes <= static_cast<uint64_t>(INT32_MAX) / tensor.size(i),
                    "MF sparse_copy page bytes must fit positive int32");
        bytes *= tensor.size(i);
        stride *= tensor.size(i);
    }
    TORCH_CHECK(tensor.stride(0) >= 0 && tensor.stride(1) >= 0, "Negative KV strides are unsupported");
    return bytes;
}

void CheckAddressRange(const Component &m, bool host)
{
    const uint64_t base = host ? m.host_base : m.device_base;
    const uint64_t pages = host ? m.host_pages : m.device_pages;
    const uint64_t ls = host ? m.host_layer_stride : m.device_layer_stride;
    const uint64_t ps = host ? m.host_page_stride : m.device_page_stride;
    const __uint128_t last = static_cast<__uint128_t>(base) +
        static_cast<__uint128_t>(m.layers - 1) * ls +
        static_cast<__uint128_t>(pages - 1) * ps + m.page_bytes;
    TORCH_CHECK(base && pages && last <= std::numeric_limits<uint64_t>::max(),
                "KV address range overflows uint64");
}
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> BuildMetadata(
    const at::Tensor &device_indices, const at::Tensor &host_indices, at::TensorList devices,
    at::TensorList hosts, int64_t page_size, int64_t direction)
{
    TORCH_CHECK(page_size > 0 && (direction == 1 || direction == 2), "Invalid KV page size or direction");
    TORCH_CHECK(!devices.empty() && devices.size() <= kMaxComponents && devices.size() == hosts.size(),
                "Expected one to four paired KV components");
    const auto device = devices[0].device();
    TORCH_CHECK(device.type() == c10::DeviceType::PrivateUse1, "KV buffers must be on NPU");
    c10::OptionalDeviceGuard guard(device);
    TORCH_CHECK(device_indices.numel() == host_indices.numel() && device_indices.numel() % page_size == 0,
                "Indices must describe equal numbers of complete pages");
    const auto stream = c10_npu::getCurrentNPUStream();
    const auto di = PrepareIndices(device_indices, device, stream);
    const auto hi = PrepareIndices(host_indices, device, stream);
    Args args{};
    args.device_indices = reinterpret_cast<uintptr_t>(di.data_ptr());
    args.host_indices = reinterpret_cast<uintptr_t>(hi.data_ptr());
    args.pages = di.numel() / page_size;
    args.page_size = page_size;
    args.direction = direction;
    args.components = devices.size();
    uint64_t total = 0;
    bool split_pages = false;
    bool aligned_pages = true;
    for (size_t c = 0; c < devices.size(); ++c) {
        const auto &dev = devices[c];
        const auto &host = hosts[c];
        TORCH_CHECK(dev.device() == device && host.device().is_cpu() &&
                    dev.scalar_type() == host.scalar_type(), "KV component device/dtype mismatch");
        const auto bytes = PageBytes(dev, page_size);
        TORCH_CHECK(PageBytes(host, page_size) == bytes && dev.size(3) == host.size(3) &&
                    dev.size(4) == host.size(4), "KV page shapes must match");
        TORCH_CHECK(dev.size(0) > 0 && dev.size(1) > 0 && host.size(0) > 0 &&
                    dev.size(0) == host.size(1), "KV layer/page ranges must be nonempty and paired");
        const uint64_t layers = dev.size(0);
        const uint64_t chunks = ChunksPerPage(bytes);
        TORCH_CHECK(args.pages <= (kMaxRecords - total) / layers / chunks,
                    "MF sparse_copy supports at most 65536 records per batch");
        total += args.pages * layers * chunks;
        split_pages |= chunks > 1;
        aligned_pages &= bytes % kAivCopyAlignment == 0;
        const uint64_t item = dev.element_size();
        args.component[c] = {
            reinterpret_cast<uintptr_t>(dev.data_ptr()), reinterpret_cast<uintptr_t>(host.data_ptr()),
            static_cast<uint64_t>(dev.size(1)), static_cast<uint64_t>(host.size(0)), layers,
            static_cast<uint64_t>(dev.stride(0)) * item, static_cast<uint64_t>(dev.stride(1)) * item,
            static_cast<uint64_t>(host.stride(1)) * item, static_cast<uint64_t>(host.stride(0)) * item, bytes};
        CheckAddressRange(args.component[c], false);
        CheckAddressRange(args.component[c], true);
    }
    // MF places the second record at local[first_length], which must be
    // 32-byte aligned. Full chunks and their page tails must preserve this.
    TORCH_CHECK(!split_pages || total < kMinAivPairedRecords || aligned_pages,
                "Split MF AIV batches require page byte counts aligned to 32 bytes");
    total = CopyRecordCount(total, split_pages);
    auto src = at_npu::native::empty_with_format(
        {static_cast<int64_t>(total)}, di.options(), ACL_FORMAT_ND, true);
    auto dst = at_npu::native::empty_with_format(
        {static_cast<int64_t>(total)}, di.options(), ACL_FORMAT_ND, true);
    auto lengths = at_npu::native::empty_with_format(
        {static_cast<int64_t>(total)}, di.options().dtype(at::kInt), ACL_FORMAT_ND, true);
    auto count = at_npu::native::empty_with_format(
        {1}, di.options().dtype(at::kInt), ACL_FORMAT_ND, true);
    std::vector<int64_t> info(args.components * kComponentFields);
    std::memcpy(info.data(), args.component, info.size() * sizeof(int64_t));
    std::shared_ptr<aclIntArray> infoDesc(
        aclCreateIntArray(info.data(), info.size()), [](aclIntArray *v) { aclDestroyIntArray(v); });
    TORCH_CHECK(infoDesc, "Cannot create KV component attribute: ", aclGetRecentErrMsg());
    std::array<std::shared_ptr<aclTensor>, 6> descriptors{
        sgl_cann::tensor_descriptor(di, ACL_INT64), sgl_cann::tensor_descriptor(hi, ACL_INT64),
        sgl_cann::tensor_descriptor(src, ACL_INT64), sgl_cann::tensor_descriptor(dst, ACL_INT64),
        sgl_cann::tensor_descriptor(lengths, ACL_INT32), sgl_cann::tensor_descriptor(count, ACL_INT32)};
    static auto query = reinterpret_cast<decltype(&aclnnBuildKvSparseMetadataGetWorkspaceSize)>(
        sgl_cann::op_api_symbol("aclnnBuildKvSparseMetadataGetWorkspaceSize"));
    static auto execute = reinterpret_cast<decltype(&aclnnBuildKvSparseMetadata)>(
        sgl_cann::op_api_symbol("aclnnBuildKvSparseMetadata"));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto status = query(
        descriptors[0].get(), descriptors[1].get(), infoDesc.get(), page_size, direction,
        descriptors[2].get(), descriptors[3].get(), descriptors[4].get(), descriptors[5].get(),
        &workspaceSize, &executor);
    TORCH_CHECK(status == ACL_SUCCESS, "KV metadata workspace query failed: ", aclGetRecentErrMsg());
    TORCH_CHECK(workspaceSize <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                "ACLNN workspace exceeds Tensor allocation limit");
    at::Tensor workspace;
    if (workspaceSize > 0) {
        workspace = at::empty({static_cast<int64_t>(workspaceSize)}, di.options().dtype(at::kByte));
    }
    for (const auto &tensor : devices) {
        tensor.record_stream(stream);
    }
    const auto aclStream = stream.stream(false);
    // Metadata and workspace are allocated and consumed on this stream.
    // Keep their owners alive through the queued ACLNN launch.
    std::array<at::Tensor, 7> keepalive{di, hi, src, dst, lengths, count, workspace};
    auto launch = [keepalive, descriptors, infoDesc, workspaceSize, executor, aclStream]() -> int {
        const auto result = execute(
            workspaceSize ? keepalive.back().data_ptr() : nullptr, workspaceSize, executor, aclStream);
        TORCH_CHECK(result == ACL_SUCCESS, "KV metadata ACLNN execution failed: ", aclGetRecentErrMsg());
        return result;
    };
    at_npu::native::OpCommand::RunOpApi("aclnnBuildKvSparseMetadata", launch);
    return {src, dst, lengths, count};
}

namespace {
struct SparseSubmission {
    explicit SparseSubmission(c10_npu::NPUStream stream) : stream(stream) {}
    c10_npu::NPUStream stream;
    c10_npu::NPUEvent done;
};

struct SparseSubmitState {
    std::mutex mutex;
    std::unordered_map<c10::DeviceIndex, std::unique_ptr<SparseSubmission>> previous;

    void Wait(const c10_npu::NPUStream &stream)
    {
        auto it = previous.find(stream.device_index());
        if (it != previous.end() && it->second->stream != stream) {
            it->second->done.block(stream);
        }
    }

    void Record(const c10_npu::NPUStream &stream)
    {
        auto submission = std::make_unique<SparseSubmission>(stream);
        submission->done.record(stream);
        previous[stream.device_index()] = std::move(submission);
    }
};

void Submit(SparseSubmitState &state, const pybind11::function &sparse_copy,
            const at::Tensor &device_indices, const at::Tensor &host_indices,
            const std::vector<at::Tensor> &devices, const std::vector<at::Tensor> &hosts,
            int64_t page_size, int64_t direction, const pybind11::object &mf_device)
{
    // Do not hold the GIL while waiting for a submitter whose MF callback needs it.
    std::unique_lock<std::mutex> lock(state.mutex, std::defer_lock);
    {
        pybind11::gil_scoped_release release;
        lock.lock();
    }
    const auto device = device_indices.device();
    TORCH_CHECK(device.type() == c10::DeviceType::PrivateUse1, "KV indices must be on NPU");
    c10::OptionalDeviceGuard guard(device);
    const auto stream = c10_npu::getCurrentNPUStream();
    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> metadata;
    {
        pybind11::gil_scoped_release release;
        {
            // Retain the builder-only profiling range when bypassing Python's
            // Torch dispatcher. MF copy and event submission stay outside it.
            RECORD_FUNCTION("npu::kv_sparse_metadata_aicpu", c10::ArrayRef<c10::IValue>{});
            metadata = BuildMetadata(device_indices, host_indices, devices, hosts, page_size, direction);
        }
        state.Wait(stream);
    }

    int status;
    try {
        // MF exposes this pool through Python, while its public C API uses a
        // different singleton. Invoke the existing pool on the caller thread,
        // never from a RunOpApi worker. The tables remain on the NPU.
        status = sparse_copy(std::get<0>(metadata), std::get<1>(metadata),
                             std::get<2>(metadata), std::get<3>(metadata), mf_device).cast<int>();
    } catch (...) {
        pybind11::gil_scoped_release release;
        state.Record(stream);
        throw;
    }
    {
        pybind11::gil_scoped_release release;
        state.Record(stream);
    }
    TORCH_CHECK(status == 0, "MF sparse_copy failed: code=", status);
}
}  // namespace
}

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("kv_sparse_metadata_aicpu(Tensor device_indices, Tensor host_indices, "
          "Tensor[] device_components, Tensor[] host_components, int page_size, int direction) "
          "-> (Tensor, Tensor, Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("kv_sparse_metadata_aicpu", TORCH_FN(sglang::kv_metadata::BuildMetadata));
}

// Export a private Python entry from the same Torch library. This needs no
// separate extension or change to the shared operator build framework.
PYBIND11_MODULE(_kv_sparse_metadata, m)
{
    auto state = std::make_shared<sglang::kv_metadata::SparseSubmitState>();
    m.def("submit", [state](const pybind11::function &sparse_copy,
                           const at::Tensor &device_indices, const at::Tensor &host_indices,
                           const std::vector<at::Tensor> &devices, const std::vector<at::Tensor> &hosts,
                           int64_t page_size, int64_t direction, const pybind11::object &device) {
        sglang::kv_metadata::Submit(*state, sparse_copy, device_indices, host_indices,
                                   devices, hosts, page_size, direction, device);
    });
}
