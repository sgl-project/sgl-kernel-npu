#include "defines.h"
#include "torch_helper.h"
#include "aclrtlaunch_restore_short_kv_history.h"
#include <cstdint>
#include <limits>

namespace sglang {
namespace npu_kernel {
namespace {
void CheckIndex(const at::Tensor &t, const at::Tensor &ref, const char *name)
{
    TORCH_CHECK(t.device() == ref.device(), name, " must be on the cache device");
    TORCH_CHECK(t.dim() == 1 && (t.scalar_type() == at::kInt || t.scalar_type() == at::kLong), name,
                " must be a 1-D int32 or int64 tensor");
    TORCH_CHECK(t.stride(0) > 0 && t.stride(0) <= UINT32_MAX, name, " stride is invalid");
}
void CheckExtent(const at::Tensor &t, const at::Tensor &ref, const char *name)
{
    TORCH_CHECK(t.device() == ref.device() && t.dim() == 1 && t.is_contiguous() && t.scalar_type() == at::kInt &&
                    t.numel() == ref.size(0),
                name, " must be contiguous int32 [rows]");
}
bool Overlap(const at::Tensor &a, const at::Tensor &b)
{
    uintptr_t x = reinterpret_cast<uintptr_t>(a.data_ptr()), y = reinterpret_cast<uintptr_t>(b.data_ptr());
    uint64_t xn = static_cast<uint64_t>(a.numel()) * a.element_size();
    uint64_t yn = static_cast<uint64_t>(b.numel()) * b.element_size();
    return x <= y ? y - x < xn : x - y < yn;
}
}  // namespace
HOST_API void restore_short_kv_history(const at::Tensor &host, at::Tensor &cache, const at::Tensor &table,
                                       const at::Tensor &requests, const at::Tensor &lengths, const at::Tensor &first,
                                       const at::Tensor &oldBegin, const at::Tensor &oldEnd, at::Tensor &nextBegin,
                                       at::Tensor &nextEnd, at::Tensor &nextFirst, int64_t capacity, int64_t contextLen,
                                       int64_t srcPtr, int64_t blockDim)
{
    TORCH_CHECK(cache.device().type() == at::DeviceType::PrivateUse1 && cache.dim() == 4 && cache.is_contiguous(),
                "cache must be contiguous NPU [rows, stride, head, dim]");
    TORCH_CHECK(host.device().is_cpu() && host.dim() == 4 && host.is_contiguous() &&
                    host.scalar_type() == cache.scalar_type() && host.size(2) == cache.size(2) &&
                    host.size(3) == cache.size(3),
                "host must be contiguous CPU [layers, physical_tokens, head, dim] with cache dtype");
    TORCH_CHECK(table.device() == cache.device() && table.dim() == 2 && table.is_contiguous() &&
                    (table.scalar_type() == at::kInt || table.scalar_type() == at::kLong),
                "req_to_token must be contiguous NPU int32 or int64 [rows, context]");
    CheckIndex(requests, cache, "requests");
    CheckIndex(lengths, cache, "lengths");
    CheckIndex(first, cache, "first_positions");
    TORCH_CHECK(requests.numel() == lengths.numel() && requests.numel() == first.numel(), "batch lengths must match");
    CheckExtent(oldBegin, cache, "old_begin");
    CheckExtent(oldEnd, cache, "old_end");
    CheckExtent(nextBegin, cache, "next_begin");
    CheckExtent(nextEnd, cache, "next_end");
    CheckExtent(nextFirst, cache, "next_first");
    const at::Tensor extents[] = {oldBegin, oldEnd, nextBegin, nextEnd, nextFirst};
    for (int i = 0; i < 5; ++i)
        for (int j = i + 1; j < 5; ++j)
            TORCH_CHECK(!Overlap(extents[i], extents[j]), "extent buffers must not overlap");
    TORCH_CHECK(capacity > 0 && capacity <= INT32_MAX && capacity <= cache.size(1) && contextLen > 0 &&
                    contextLen <= table.size(1),
                "invalid capacity or context_len");
    TORCH_CHECK(srcPtr > 0 && blockDim > 0 && blockDim <= UINT32_MAX, "invalid src_ptr or block_dim");
    const uint64_t bytes = static_cast<uint64_t>(cache.size(2)) * cache.size(3) * cache.element_size();
    TORCH_CHECK(bytes > 0 && bytes <= 32768, "row bytes must be within 1..32768");
    const uint64_t hostBytes = static_cast<uint64_t>(host.size(1)) * bytes;
    TORCH_CHECK(hostBytes > 0 && hostBytes % bytes == 0 && hostBytes <= UINT32_MAX, "invalid host storage size");
    TORCH_CHECK(static_cast<uint64_t>(cache.numel()) * cache.element_size() <= UINT32_MAX,
                "cache exceeds uint32 address range");
    TORCH_CHECK(cache.size(0) <= UINT32_MAX && cache.size(1) <= UINT32_MAX && table.size(0) <= UINT32_MAX &&
                    contextLen <= UINT32_MAX && table.stride(0) <= UINT32_MAX && requests.numel() <= UINT32_MAX,
                "shape exceeds uint32 range");
    TORCH_CHECK(static_cast<uint64_t>(requests.numel()) * requests.stride(0) <= UINT32_MAX &&
                    static_cast<uint64_t>(lengths.numel()) * lengths.stride(0) <= UINT32_MAX &&
                    static_cast<uint64_t>(first.numel()) * first.stride(0) <= UINT32_MAX,
                "metadata stride exceeds uint32 range");
    if (requests.numel() == 0) return;
    auto stream = c10_npu::getCurrentNPUStream();
    cache.record_stream(stream);
    table.record_stream(stream);
    requests.record_stream(stream);
    lengths.record_stream(stream);
    first.record_stream(stream);
    oldBegin.record_stream(stream);
    oldEnd.record_stream(stream);
    nextBegin.record_stream(stream);
    nextEnd.record_stream(stream);
    nextFirst.record_stream(stream);
    uint32_t types = (requests.scalar_type() == at::kLong ? 1U : 0U) | (lengths.scalar_type() == at::kLong ? 2U : 0U) |
                     (first.scalar_type() == at::kLong ? 4U : 0U) | (table.scalar_type() == at::kLong ? 8U : 0U);
    void *srcAddr = reinterpret_cast<void *>(static_cast<uintptr_t>(srcPtr));
    uint32_t batchU32 = static_cast<uint32_t>(requests.numel());
    uint32_t rowsU32 = static_cast<uint32_t>(cache.size(0));
    uint32_t tableRowsU32 = static_cast<uint32_t>(table.size(0));
    uint32_t contextU32 = static_cast<uint32_t>(contextLen);
    uint32_t tableStrideU32 = static_cast<uint32_t>(table.stride(0));
    uint32_t strideU32 = static_cast<uint32_t>(cache.size(1));
    uint32_t srcRowsU32 = static_cast<uint32_t>(hostBytes / bytes);
    uint32_t bytesU32 = static_cast<uint32_t>(bytes);
    uint32_t capacityU32 = static_cast<uint32_t>(capacity);
    uint32_t reqStrideU32 = static_cast<uint32_t>(requests.stride(0));
    uint32_t lenStrideU32 = static_cast<uint32_t>(lengths.stride(0));
    uint32_t firstStrideU32 = static_cast<uint32_t>(first.stride(0));
    uint32_t blockDimU32 = static_cast<uint32_t>(blockDim);
    EXEC_KERNEL_CMD(restore_short_kv_history, blockDimU32, srcAddr, cache, table, requests, lengths, first, oldBegin,
                    oldEnd, nextBegin, nextEnd, nextFirst, batchU32, rowsU32, tableRowsU32, contextU32, tableStrideU32,
                    strideU32, srcRowsU32, bytesU32, capacityU32, reqStrideU32, lenStrideU32, firstStrideU32, types);
}
}  // namespace npu_kernel
}  // namespace sglang
