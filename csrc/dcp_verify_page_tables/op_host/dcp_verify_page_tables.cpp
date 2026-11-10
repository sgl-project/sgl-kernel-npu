#include <algorithm>
#include <cstdint>
#include <limits>

#include <c10/core/DeviceGuard.h>

#include "aclrtlaunch_sgl_dcp_verify_page_tables.h"
#include "torch_helper.h"

namespace
{
constexpr int64_t kTile = 128;
constexpr int64_t kMaxBlocks = 32;

void dcp_verify_page_tables(const at::Tensor &slots, const at::Tensor &requests, const at::Tensor &ordinary,
                            const at::Tensor &local, int64_t active_cols, int64_t query_len, int64_t page_size,
                            int64_t dcp_size)
{
    const auto device = slots.device();
    TORCH_CHECK(device.type() == DEVICE_TYPE && requests.device() == device && ordinary.device() == device &&
                    local.device() == device,
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
    TORCH_CHECK(cols <= kMaxIndex - kTile && rows * tiles <= kMaxIndex - kMaxBlocks,
                "DSA DCP page-table grid exceeds the kernel index range");
    if (tiles == 0) {
        return;
    }
    const c10::DeviceGuard guard(device);
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
}
} // namespace

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("sgl_dcp_verify_page_tables(Tensor slots, Tensor requests, Tensor(a!) ordinary, Tensor(b!) local, "
          "int active_cols, int query_len, int page_size, int dcp_size) -> ()");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m) { m.impl("sgl_dcp_verify_page_tables", dcp_verify_page_tables); }
