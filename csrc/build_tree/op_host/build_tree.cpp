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
#include "build_tree_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "aclrtlaunch_build_tree_efficient.h"
#include "torch_helper.h"

namespace sglang {
namespace npu_kernel {
constexpr uint32_t PADDING_BYTE = 32U;

at::Tensor get_tiling(int32_t &block_dim, int32_t &workspace_size, int32_t batch_size, int32_t mask_size, int64_t topk,
                      int64_t depth, int64_t draft_token_num, int64_t tree_mask_mode)
{
    auto ascendc_platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int32_t max_aiv_core = static_cast<int32_t>(ascendc_platform->GetCoreNumAiv());
    block_dim = std::min(max_aiv_core, batch_size);
    workspace_size = static_cast<int32_t>(ascendc_platform->GetLibApiWorkSpaceSize());

    // align to 32 bytes
    int32_t tiling_size = (sizeof(BuildTreeTilingData) + PADDING_BYTE - 1) / PADDING_BYTE * PADDING_BYTE;
    auto tiling_buffer = at::empty({tiling_size}, at::TensorOptions().dtype(at::kByte).device(at::kCPU));

    BuildTreeTilingData *tiling_data = reinterpret_cast<BuildTreeTilingData *>(tiling_buffer.data_ptr());
    tiling_data->batch_size = batch_size;
    tiling_data->mask_size = mask_size;
    tiling_data->topk = topk;
    tiling_data->depth = depth;
    tiling_data->draft_token_num = draft_token_num;
    tiling_data->tree_mask_mode = tree_mask_mode;

    auto num_big_core = batch_size % block_dim;
    tiling_data->big_core_num = num_big_core == 0 ? block_dim : num_big_core;
    tiling_data->big_core_tile_num = (batch_size + block_dim - 1) / block_dim;
    tiling_data->small_core_tile_num = batch_size / block_dim;

    auto tiling_tensor = TorchNpuHelper::CopyTensorHostToDevice(tiling_buffer);
    return tiling_tensor;
}

HOST_API void build_tree_efficient(const at::Tensor &parent_list, const at::Tensor &selected_index,
                                   const at::Tensor &verified_seq_len, const at::Tensor &tree_mask,
                                   const at::Tensor &positions, const at::Tensor &retrive_index,
                                   const at::Tensor &retrive_next_token, const at::Tensor &retrive_next_sibling,
                                   int64_t topk, int64_t depth, int64_t draft_token_num, int64_t tree_mask_mode)
{
    if (QLEN_ONLY_BITPACKING == tree_mask_mode) {
        throw std::runtime_error("Not implemented");
    }

    if (parent_list.options().dtype() != at::kLong || selected_index.options().dtype() != at::kLong ||
        verified_seq_len.options().dtype() != at::kLong || tree_mask.options().dtype() != at::kBool ||
        positions.options().dtype() != at::kLong || retrive_index.options().dtype() != at::kLong ||
        retrive_next_token.options().dtype() != at::kLong || retrive_next_sibling.options().dtype() != at::kLong) {
        throw std::invalid_argument(
            "Invalid input datetype. "
            "Support combo: int64, int64, int64, bool, int64, int64, int64, int64");
    }
    int32_t block_dim;
    int32_t workspace_size;
    int32_t batch_size = parent_list.sizes()[0];
    int32_t mask_size = tree_mask.size(0);

    at::Tensor tiling_tensor =
        get_tiling(block_dim, workspace_size, batch_size, mask_size, topk, depth, draft_token_num, tree_mask_mode);

    auto workspace_tensor =
        at::empty({workspace_size}, at::TensorOptions().dtype(at::kByte).device(parent_list.options().device()));
    /* launch the kernel function via torch */
    EXEC_KERNEL_CMD(build_tree_efficient, block_dim, parent_list, selected_index, verified_seq_len, tree_mask,
                    positions, retrive_index, retrive_next_token, retrive_next_sibling, workspace_tensor,
                    tiling_tensor);
}

}  // namespace npu_kernel
}  // namespace sglang
