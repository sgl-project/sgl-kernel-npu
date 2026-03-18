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
#include "torch_helper.h"

#include "tiling_tri_inv_cube.h"
#include "aclrtlaunch_tri_inv_cube_col_sweep_fp16.h"
#include "tiling/platform/platform_ascendc.h"

namespace sglang {

namespace npu_kernel {

at::Tensor calc_tiling(const TriInvColumnSweepCubeTiling &tiling)
{
    constexpr uint32_t PADDING_BYTE = 32U;

    // align to 32 bytes
    int32_t tiling_size = (sizeof(TriInvColumnSweepCubeTiling) + PADDING_BYTE - 1) / PADDING_BYTE * PADDING_BYTE;
    auto tiling_buffer = at::empty({tiling_size}, at::TensorOptions().dtype(at::kByte).device(at::kCPU));

    TriInvColumnSweepCubeTiling *tiling_data =
        reinterpret_cast<TriInvColumnSweepCubeTiling *>(tiling_buffer.data_ptr());
    tiling_data->num_blocks = tiling.num_blocks;
    tiling_data->num_elems = tiling.num_elems;
    tiling_data->matrix_size = tiling.matrix_size;
    tiling_data->ws_circular_buffer_len = tiling.ws_circular_buffer_len;

    auto tiling_tensor = TorchNpuHelper::CopyTensorHostToDevice(tiling_buffer);
    return tiling_tensor;
}

HOST_API at::Tensor tri_inv_cube_col_sweep(const at::Tensor &tensor)
{
    platform_ascendc::PlatformAscendC *platformAscendC = platform_ascendc::PlatformAscendCManager::GetInstance();
    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);

    const auto dtype = tensor.options().dtype();
    if (tensor.dim() < 2) {
        throw std::runtime_error("Input tensor must have at least 2 dimensions.\n");
    }

    const uint32_t matrix_size = static_cast<uint32_t>(tensor.size(-1));
    if (matrix_size != tensor.size(-2)) {
        throw std::runtime_error("Only square matrices are supported.\n");
    }

    const uint32_t num_elems = static_cast<uint32_t>(tensor.numel());
    const uint32_t block_dim = static_cast<uint32_t>(num_elems / (matrix_size * matrix_size));

    auto tensor_out = at::empty_like(tensor, at::kFloat);

    const uint32_t WS_CIRCULAR_BUFFER_LEN = 4;
    const TriInvColumnSweepCubeTiling tiling{block_dim, num_elems, matrix_size, WS_CIRCULAR_BUFFER_LEN};
    const at::Tensor tiling_device = calc_tiling(tiling);

    // workspace
    const uint64_t system_workspace_size = static_cast<uint64_t>(platformAscendC->GetLibApiWorkSpaceSize());
    const uint64_t workspace_size = system_workspace_size + num_elems * WS_CIRCULAR_BUFFER_LEN * tensor.itemsize();
    const auto options = at::TensorOptions().dtype(at::kByte).device(tensor.options().device());
    auto workspace = at::empty({static_cast<int64_t>(workspace_size)}, options);

    if (dtype == at::kHalf) {
        EXEC_KERNEL_CMD(tri_inv_cube_col_sweep_fp16, block_dim, tensor, tensor_out, workspace, tiling_device);
    } else {
        throw std::runtime_error("Unsupported data type for tri_inv_cube_col_sweep. fp16 is currently supported.");
    }
    aclrtSynchronizeStream(acl_stream);
    return tensor_out;
}

}  // namespace npu_kernel
}  // namespace sglang
