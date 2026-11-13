#include <algorithm>
#include <cstdint>
#include <limits>

#include <ATen/MemoryOverlap.h>
#include <c10/core/DeviceGuard.h>

#ifndef SGL_KERNEL_ENABLE_A3_ONLY_OPS
#include "aclnn_runtime.h"
#include "aclnn_sgl_dcp_verify_page_tables.h"
#else
#include "aclrtlaunch_sgl_dcp_verify_page_tables.h"
#include "torch_helper.h"
#endif

#include <torch/library.h>

namespace {
constexpr int64_t kTile = 128;
constexpr int64_t kMaxBlocks = 32;

void dcp_verify_page_tables(const at::Tensor &slots, const at::Tensor &requests, const at::Tensor &ordinary,
                            const at::Tensor &local, int64_t active_cols, int64_t query_len, int64_t page_size,
                            int64_t dcp_size)
{
    const auto device = slots.device();
    TORCH_CHECK(device.type() == c10::DeviceType::PrivateUse1 && requests.device() == device &&
                    ordinary.device() == device && local.device() == device,
                "DSA DCP page-table tensors must be on the same NPU");
    TORCH_CHECK(slots.dim() == 2 && requests.dim() == 1 && ordinary.dim() == 2 && local.dim() == 2,
                "Expected two-dimensional page tables and one-dimensional request indices");
    TORCH_CHECK(slots.is_contiguous() && requests.is_contiguous() && ordinary.is_contiguous() && local.is_contiguous(),
                "DSA DCP page-table tensors must be contiguous");
    TORCH_CHECK(slots.scalar_type() == at::kInt && ordinary.scalar_type() == at::kInt &&
                    local.scalar_type() == at::kInt &&
                    (requests.scalar_type() == at::kInt || requests.scalar_type() == at::kLong),
                "DSA DCP page tables require int32 slots/tables and int32 or int64 request indices");
    TORCH_CHECK((dcp_size == 2 || dcp_size == 4) && query_len > 0 && page_size > 0 && page_size <= 65536 &&
                    (page_size & (page_size - 1)) == 0,
                "Expected DCP2 or DCP4, positive query length and a power-of-two page size");
    constexpr int64_t kMaxIndex = std::numeric_limits<uint32_t>::max();
    TORCH_CHECK(slots.size(0) <= kMaxIndex && slots.size(1) <= kMaxIndex && ordinary.numel() <= kMaxIndex &&
                    local.numel() <= kMaxIndex && query_len <= kMaxIndex && requests.numel() <= kMaxIndex,
                "DSA DCP page-table sizes exceed the kernel index range");

    const int64_t rows = requests.numel();
    TORCH_CHECK(ordinary.size(0) == rows && local.size(0) % query_len == 0 && local.size(0) / query_len == rows,
                "Expected one ordinary row per request and one DCP row per query");
    TORCH_CHECK(active_cols >= 0 && active_cols <= (slots.size(1) + page_size - 1) / page_size &&
                    active_cols <= ordinary.size(1) && (active_cols + dcp_size - 1) / dcp_size <= local.size(1),
                "Active pages exceed the input or output table capacity");
    if (rows == 0) {
        return;
    }
    const int64_t cols = std::max(ordinary.size(1), local.size(1) * dcp_size);
    const int64_t tiles = (cols + kTile - 1) / kTile;
    TORCH_CHECK(cols <= kMaxIndex - kTile && (tiles == 0 || rows <= (kMaxIndex - kMaxBlocks) / tiles),
                "DSA DCP page-table grid exceeds the kernel index range");
    if (tiles == 0) {
        return;
    }
    at::assert_no_overlap(ordinary, local);
    at::assert_no_overlap(ordinary, slots);
    at::assert_no_overlap(local, slots);
    at::assert_no_overlap(ordinary, requests);
    at::assert_no_overlap(local, requests);
    const c10::DeviceGuard guard(device);
#ifndef SGL_KERNEL_ENABLE_A3_ONLY_OPS
    // An empty tensor can make ACLNN skip the launch. Clear nonempty tables
    // explicitly when there cannot be any valid output pages.
    if (slots.numel() == 0 || ordinary.numel() == 0 || local.numel() == 0) {
        ordinary.zero_();
        local.zero_();
        return;
    }
    static const auto query = reinterpret_cast<decltype(&aclnnSglDcpVerifyPageTablesGetWorkspaceSize)>(
        sgl_cann::op_api_symbol("aclnnSglDcpVerifyPageTablesGetWorkspaceSize"));
    static const auto execute = reinterpret_cast<decltype(&aclnnSglDcpVerifyPageTables)>(
        sgl_cann::op_api_symbol("aclnnSglDcpVerifyPageTables"));
    auto slots_desc = sgl_cann::tensor_descriptor(slots, ACL_INT32);
    const auto index_type = requests.scalar_type() == at::kLong ? ACL_INT64 : ACL_INT32;
    auto requests_desc = sgl_cann::tensor_descriptor(requests, index_type);
    auto ordinary_desc = sgl_cann::tensor_descriptor(ordinary, ACL_INT32);
    auto local_desc = sgl_cann::tensor_descriptor(local, ACL_INT32);
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    const auto status = query(slots_desc.get(), requests_desc.get(), ordinary_desc.get(), local_desc.get(), active_cols,
                              query_len, page_size, dcp_size, &workspace_size, &executor);
    TORCH_CHECK(status == ACL_SUCCESS, "aclnnSglDcpVerifyPageTables workspace query failed: ", aclGetRecentErrMsg());
    TORCH_CHECK(workspace_size <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                "ACLNN workspace size exceeds the Tensor allocation limit");
    auto workspace = at::empty({static_cast<int64_t>(workspace_size)}, slots.options().dtype(at::kByte));
    auto stream = c10_npu::getCurrentNPUStream();
    slots.record_stream(stream);
    requests.record_stream(stream);
    ordinary.record_stream(stream);
    local.record_stream(stream);
    workspace.record_stream(stream);
    auto acl_stream = stream.stream(false);
    // Retain descriptors and tensors until the queued host launch has run.
    auto launch = [slots, requests, ordinary, local, slots_desc, requests_desc, ordinary_desc, local_desc, workspace,
                   workspace_size, executor, acl_stream]() -> int {
        const auto result =
            execute(workspace_size ? workspace.data_ptr() : nullptr, workspace_size, executor, acl_stream);
        TORCH_CHECK(result == ACL_SUCCESS, "aclnnSglDcpVerifyPageTables execution failed: ", aclGetRecentErrMsg());
        return result;
    };
    at_npu::native::OpCommand::RunOpApi("aclnnSglDcpVerifyPageTables", launch);
#else
    const auto blocks = static_cast<uint32_t>(std::min(kMaxBlocks, rows * tiles));
    const auto batch = static_cast<uint32_t>(rows);
    const auto slot_rows = static_cast<uint32_t>(slots.size(0));
    const auto slot_cols = static_cast<uint32_t>(slots.size(1));
    const auto ordinary_cols = static_cast<uint32_t>(ordinary.size(1));
    const auto local_cols = static_cast<uint32_t>(local.size(1));
    const auto active = static_cast<uint32_t>(active_cols);
    const auto width = static_cast<uint32_t>(query_len);
    const auto page = static_cast<uint32_t>(page_size);
    const auto dcp = static_cast<uint32_t>(dcp_size);
    const auto index_bytes = static_cast<uint32_t>(requests.element_size());
    EXEC_KERNEL_CMD(sgl_dcp_verify_page_tables, blocks, slots, requests, ordinary, local, batch, slot_rows, slot_cols,
                    ordinary_cols, local_cols, active, width, page, dcp, index_bytes);
#endif
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def(
        "sgl_dcp_verify_page_tables(Tensor slots, Tensor requests, Tensor(a!) ordinary, Tensor(b!) local, "
        "int active_cols, int query_len, int page_size, int dcp_size) -> ()");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("sgl_dcp_verify_page_tables", dcp_verify_page_tables);
}
