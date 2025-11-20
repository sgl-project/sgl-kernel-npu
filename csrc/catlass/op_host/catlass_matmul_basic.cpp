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
#include "tiling/platform/platform_ascendc.h"
#include "torch_helper.h"
#include "catlass_matmul_tiling.h"
#include "aclrtlaunch_catlass_matmul_basic.h"

namespace sglang {
namespace npu_kernel {
constexpr uint32_t PADDING_BYTE = 32U;

at::Tensor get_tiling(int32_t &m, int32_t &n, int32_t k, int64_t weight_format_mode,
                      int64_t data_format_mode, uint32_t &blockDim)
{

    auto ascendc_platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    blockDim = static_cast<uint32_t>(ascendc_platform->GetCoreNumAiv());
    // workspace_size = static_cast<int32_t>(ascendc_platform->GetLibApiWorkSpaceSize());

    // align to 32 bytes
    int32_t tiling_size = (sizeof(KernelCatlassMatmulTilingData) + PADDING_BYTE - 1) / PADDING_BYTE * PADDING_BYTE;
    auto tiling_buffer = at::empty({tiling_size}, at::TensorOptions().dtype(at::kByte).device(at::kCPU));

    KernelCatlassMatmulTilingData *tiling_data = reinterpret_cast<KernelCatlassMatmulTilingData *>(tiling_buffer.data_ptr());
    tiling_data->m = m;
    tiling_data->n = n;
    tiling_data->k = k;
    tiling_data->weight_format_mode = weight_format_mode;
    tiling_data->data_format_mode = data_format_mode;

    auto tiling_tensor = TorchNpuHepler::CopyTensorHostToDevice(tiling_buffer);
    return tiling_tensor;
}

HOST_API void catlass_matmul_basic(const at::Tensor &input_a, const at::Tensor &input_b,
                                   at::Tensor &output_c)
{
    std::map<c10::ScalarType, DataFormatMode> dTypeMap = {{at::ScalarType::Half, DataFormatMode::FP16}, \
        {at::ScalarType::BFloat16, DataFormatMode::BF16}};

    at::ScalarType aType = input_a.scalar_type();
    at::ScalarType bType = input_b.scalar_type();
    at::ScalarType cType = output_c.scalar_type();
    TORCH_CHECK(aType == bType && bType == cType, "tensor type is not the same");
    TORCH_CHECK((aType == at::ScalarType::BFloat16) || (aType == at::ScalarType::Half),
                "tensor type only support half or bf16");

    int32_t m = input_a.size(0);
    int32_t k = input_a.size(1);
    int32_t n = input_b.size(1);

    TORCH_CHECK(input_b.size(0) == k, "input shape mismatch");

    uint32_t blockDim;
    auto tiling_tensor = get_tiling(m, n, k, NO_NZ, dTypeMap[aType], blockDim);

    auto workspace_tensor =
        at::empty({1}, at::TensorOptions().dtype(at::kByte).device(input_a.options().device()));
    /* launch the kernel function via torch */
    EXEC_KERNEL_CMD(catlass_matmul_basic, blockDim, input_a, input_b, output_c, workspace_tensor, tiling_tensor);
}

}  // namespace npu_kernel
}  // namespace sglang
