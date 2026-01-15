// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "defines.h"
#include "alloc_extend_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "aclrtlaunch_alloc_extend.h"
#include "torch_helper.h"

namespace sglang {
namespace npu_kernel {
constexpr uint32_t PADDING_BYTE = 32U;

at::Tensor get_tiling(int32_t &block_dim, int32_t &workspace_size, const int64_t &page_size, int32_t &batch_size,
                      int64_t &total_extend_tokens)
{
    auto ascendc_platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int32_t max_aiv_core = static_cast<int32_t>(ascendc_platform->GetCoreNumAiv());
    block_dim = std::min(max_aiv_core, batch_size);
    workspace_size = static_cast<int32_t>(ascendc_platform->GetLibApiWorkSpaceSize());

    int32_t tiling_size = (sizeof(AllocExtendTilingData) + PADDING_BYTE - 1) / PADDING_BYTE * PADDING_BYTE;
    auto tiling_buffer = at::empty({tiling_size}, at::TensorOptions().dtype(at::kByte).device(at::kCPU));

    AllocExtendTilingData *tiling_data = reinterpret_cast<AllocExtendTilingData *>(tiling_buffer.data_ptr());
    tiling_data->batch_size = batch_size;
    tiling_data->page_size = static_cast<int32_t>(page_size);
    tiling_data->used_core_num = block_dim;
    tiling_data->total_extend_tokens = total_extend_tokens;

    auto tiling_tensor = TorchNpuHelper::CopyTensorHostToDevice(tiling_buffer);
    return tiling_tensor;
}

HOST_API void alloc_extend(const at::Tensor &pre_lens, const at::Tensor &seq_lens, const at::Tensor &last_loc,
                           const at::Tensor &free_pages, int64_t pages_size, at::Tensor &out_indices,
                           at::Tensor &values)
{
    if (pre_lens.options().dtype() != at::kLong || seq_lens.options().dtype() != at::kLong ||
        last_loc.options().dtype() != at::kLong || free_pages.options().dtype() != at::kLong ||
        out_indices.options().dtype() != at::kLong) {
        throw std::invalid_argument("Only support int64 input dtype");
    }
    int32_t block_dim;
    int32_t workspace_size;
    int32_t batch_size = pre_lens.sizes()[0];
    int64_t total_extend_tokens = out_indices.sizes()[0];  // 64k

    at::Tensor tiling_tensor = get_tiling(block_dim, workspace_size, pages_size, batch_size, total_extend_tokens);

    auto workspace_tensor =
        at::empty({workspace_size}, at::TensorOptions().dtype(at::kByte).device(pre_lens.options().device()));
    /* launch the kernel function via torch */
    EXEC_KERNEL_CMD(alloc_extend, block_dim, pre_lens, seq_lens, last_loc, free_pages, out_indices, values,
                    workspace_tensor, tiling_tensor);
}

}  // namespace npu_kernel
}  // namespace sglang
