#include <algorithm>
#include <cstdint>
#include <limits>

#include <c10/core/DeviceGuard.h>

#include "aclrtlaunch_sgl_dcp_sparse_index_remap.h"
#include "torch_helper.h"

namespace
{
at::Tensor dcp_sparse_index_remap(const at::Tensor &input, int64_t dcp_size, int64_t rank, int64_t page_size)
{
    TORCH_CHECK(input.device().type() == DEVICE_TYPE && input.scalar_type() == at::kInt,
                "DSA DCP remap requires int32 NPU indices");
    TORCH_CHECK(input.dim() == 2 && input.size(1) == 2048 && input.is_contiguous(),
                "DSA DCP remap requires contiguous [rows, 2048] indices");
    TORCH_CHECK((dcp_size == 2 || dcp_size == 4) && rank >= 0 && rank < dcp_size,
                "DSA DCP remap requires a valid rank in a DCP2 or DCP4 group");
    TORCH_CHECK(page_size > 0 && page_size <= 65536 && !(page_size & (page_size - 1)),
                "DSA DCP remap requires a power-of-two page size");
    TORCH_CHECK(input.numel() <= std::numeric_limits<uint32_t>::max(),
                "DSA DCP remap input exceeds the kernel index range");
    const c10::DeviceGuard guard(input.device());
    auto output = at::empty_like(input);
    const uint32_t rows = input.size(0);
    if (rows == 0) {
        return output;
    }
    const uint32_t blocks = std::min(rows, 32U);
    const int32_t page_shift = __builtin_ctzll(static_cast<uint64_t>(page_size));
    const int32_t dcp_shift = dcp_size == 2 ? 1 : 2;
    const int32_t dcp_rank = rank;
    EXEC_KERNEL_CMD(sgl_dcp_sparse_index_remap, blocks, input, output, rows, page_shift, dcp_shift, dcp_rank);
    return output;
}
} // namespace

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("sgl_dcp_sparse_index_remap(Tensor indices, int dcp_size, int dcp_rank, int page_size) -> Tensor");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m) { m.impl("sgl_dcp_sparse_index_remap", dcp_sparse_index_remap); }
