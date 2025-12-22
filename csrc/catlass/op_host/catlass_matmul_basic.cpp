// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <map>

#include "defines.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_helper.h"
#include "catlass_matmul_tiling.h"
#include "aclrtlaunch_catlass_matmul_basic.h"

namespace sglang {
namespace npu_kernel {

constexpr uint32_t PADDING_BYTE = 32U;

std::map<c10::ScalarType, DataFormatMode> dTypeMap = {{at::ScalarType::Half, DataFormatMode::FP16},
                                                      {at::ScalarType::BFloat16, DataFormatMode::BF16},
                                                      {at::ScalarType::Float, DataFormatMode::FP32}};

std::unordered_map<c10::string_view, uint16_t> weightFormatMap = {{"ND", WeightFormatMode::WEIGHT_ND},
                                                                  {"NZ", WeightFormatMode::WEIGHT_NZ}};

template <typename MapType>
inline int GetModeVal(const MapType &mode_map, c10::optional<c10::string_view> mode_opt, c10::string_view default_mode,
                      const char *mode_name)
{
    std::string modeStr(mode_name);
    c10::string_view mode_str = mode_opt.value_or(default_mode);
    auto it = mode_map.find(mode_str);
    // if input mode is unsupported, use default value
    TORCH_CHECK(it != mode_map.end(), modeStr, c10::str(": Unsupported mode value ", mode_str));
    return it->second;
}

at::Tensor get_tiling(int32_t &m, int32_t &n, int32_t k, int64_t weight_format_mode, int64_t data_format_mode,
                      uint32_t &blockDim)
{
    auto ascendc_platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    blockDim = static_cast<uint32_t>(ascendc_platform->GetCoreNumAiv());

    // align to 32 bytes
    int32_t tiling_size = (sizeof(KernelCatlassMatmulTilingData) + PADDING_BYTE - 1) / PADDING_BYTE * PADDING_BYTE;
    auto tiling_buffer = at::empty({tiling_size}, at::TensorOptions().dtype(at::kByte).device(at::kCPU));

    KernelCatlassMatmulTilingData *tiling_data =
        reinterpret_cast<KernelCatlassMatmulTilingData *>(tiling_buffer.data_ptr());
    tiling_data->m = m;
    tiling_data->n = n;
    tiling_data->k = k;
    tiling_data->weight_format_mode = weight_format_mode;
    tiling_data->data_format_mode = data_format_mode;

    auto tiling_tensor = TorchNpuHelper::CopyTensorHostToDevice(tiling_buffer);
    return tiling_tensor;
}

HOST_API void catlass_matmul_basic(const at::Tensor &input_a, const at::Tensor &input_b, at::Tensor &output_c,
                                   c10::optional<c10::string_view> format_mode)
{
    // ops valid check
    at::ScalarType aType = input_a.scalar_type();
    at::ScalarType bType = input_b.scalar_type();
    at::ScalarType cType = output_c.scalar_type();
    TORCH_CHECK(aType == bType && bType == cType, "tensor type is not the same");
    TORCH_CHECK(
        (aType == at::ScalarType::BFloat16) || (aType == at::ScalarType::Half) || (aType == at::ScalarType::Float),
        "tensor type only support half / bf16 / fp32");

    auto formatMode = static_cast<WeightFormatMode>(GetModeVal(weightFormatMap, format_mode, "ND", "format_mode"));
    TORCH_CHECK(formatMode == WeightFormatMode::WEIGHT_ND, "current ops only support weightFormat ND");

    int32_t m = input_a.size(0);
    int32_t k = input_a.size(1);
    int32_t n = input_b.size(1);
    TORCH_CHECK(input_b.size(0) == k, "input k dim shape mismatch");

    uint32_t blockDim;
    auto tiling_tensor = get_tiling(m, n, k, formatMode, dTypeMap[aType], blockDim);

    // launch the kernel function via torch
    auto workspace_tensor = at::empty({1}, at::TensorOptions().dtype(at::kByte).device(input_a.options().device()));
    EXEC_KERNEL_CMD(catlass_matmul_basic, blockDim, input_a, input_b, output_c, workspace_tensor, tiling_tensor);
}

}  // namespace npu_kernel
}  // namespace sglang
