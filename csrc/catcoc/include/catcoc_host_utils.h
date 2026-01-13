// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef KERNEL_CATCOC_HOST_UTILS_H
#define KERNEL_CATCOC_HOST_UTILS_H

#include <cstdint>
#include "catcoc_host_tiling.h"

namespace sglang {
namespace npu_kernel {

constexpr uint32_t PADDING_BYTE = 32U;

inline std::map<c10::ScalarType, DataFormatMode> dTypeMap = {{at::ScalarType::Half, DataFormatMode::FP16},
                                                             {at::ScalarType::BFloat16, DataFormatMode::BF16}};

inline std::unordered_map<c10::string_view, uint16_t> weightFormatMap = {{"ND", WeightFormatMode::WEIGHT_ND},
                                                                         {"NZ", WeightFormatMode::WEIGHT_NZ}};

// batch size -> memory index
constexpr uint32_t MAX_CAPTURE_NUM = 2048;

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

inline at::Tensor get_tiling_tensor(uint32_t &m, uint32_t &n, uint32_t &k, int64_t weight_format_mode,
                                    int64_t data_format_mode, uint32_t &blockDim)
{
    auto ascendc_platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    blockDim = static_cast<uint32_t>(ascendc_platform->GetCoreNumAiv());

    // align to 32 bytes
    int32_t tiling_size = (sizeof(KernelCATCOCHostTilingData) + PADDING_BYTE - 1) / PADDING_BYTE * PADDING_BYTE;
    auto tiling_buffer = at::empty({tiling_size}, at::TensorOptions().dtype(at::kByte).device(at::kCPU));

    KernelCATCOCHostTilingData *tiling_data = reinterpret_cast<KernelCATCOCHostTilingData *>(tiling_buffer.data_ptr());
    tiling_data->m = m;
    tiling_data->n = n;
    tiling_data->k = k;
    tiling_data->weight_format_mode = weight_format_mode;
    tiling_data->data_format_mode = data_format_mode;

    return tiling_buffer;
}

}  // namespace npu_kernel
}  // namespace sglang

#endif  // KERNEL_CATCOC_HOST_UTILS_H
