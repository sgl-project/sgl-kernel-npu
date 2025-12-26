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
#include "acl/acl.h"

#include "defines.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_helper.h"
#include "catcoc_host_tiling.h"
#include "aclrtlaunch_catcoc_allgather_matmul_kernel.h"

// shmem_device
#include "shmem_api.h"

namespace sglang {
namespace npu_kernel {

constexpr uint32_t PADDING_BYTE = 32U;

std::map<c10::ScalarType, DataFormatMode> dTypeMap = {{at::ScalarType::BFloat16, DataFormatMode::BF16}};

std::unordered_map<c10::string_view, uint16_t> weightFormatMap = {{"ND", WeightFormatMode::WEIGHT_ND},
                                                                  {"NZ", WeightFormatMode::WEIGHT_NZ}};

// batch size -> memory index
constexpr uint32_t MAX_CAPTURE_NUM = 128;

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

at::Tensor get_tiling_tensor(uint32_t &m, uint32_t &n, uint32_t &k, int64_t weight_format_mode,
                             int64_t data_format_mode, uint32_t &blockDim)
{
    auto ascendc_platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    blockDim = static_cast<uint32_t>(ascendc_platform->GetCoreNumAiv());

    // align to 32 bytes
    int32_t tiling_size = (sizeof(KernelCATCOCHostTilingData) + PADDING_BYTE - 1) / PADDING_BYTE * PADDING_BYTE;
    auto tiling_buffer = at::empty({tiling_size}, at::TensorOptions().dtype(at::kByte).device(at::kCPU));

    KernelCATCOCHostTilingData *tiling_data =
        reinterpret_cast<KernelCATCOCHostTilingData *>(tiling_buffer.data_ptr());
    tiling_data->m = m;
    tiling_data->n = n;
    tiling_data->k = k;
    tiling_data->weight_format_mode = weight_format_mode;
    tiling_data->data_format_mode = data_format_mode;

    // auto tiling_tensor = TorchNpuHelper::CopyTensorHostToDevice(tiling_buffer);
    return tiling_buffer;
}

HOST_API void catcoc_allgather_matmul(const at::Tensor &input_a, const at::Tensor &input_b, at::Tensor &output_c,
                                      int64_t symmAddr, int64_t teamId = 0,
                                      c10::optional<c10::string_view> format_mode = c10::nullopt)
{
    // ops valid check
    at::ScalarType aType = input_a.scalar_type();
    at::ScalarType bType = input_b.scalar_type();
    at::ScalarType cType = output_c.scalar_type();
    TORCH_CHECK(aType == bType && bType == cType, "tensor type is not the same");
    TORCH_CHECK((aType == at::ScalarType::Half),
        "tensor type only support half");

    auto formatMode = static_cast<WeightFormatMode>(GetModeVal(weightFormatMap, format_mode, "ND", "format_mode"));
    TORCH_CHECK(formatMode == WeightFormatMode::WEIGHT_ND, "current ops only support weightFormat ND");

    uint32_t m = input_a.size(0);
    uint32_t k = input_a.size(1);
    uint32_t n = input_b.size(1);
    TORCH_CHECK(input_b.size(0) == k, "input k dim shape mismatch");

    uint32_t blockDim;
    auto cpu_tiling_tensor = get_tiling_tensor(m, n, k, formatMode, dTypeMap[aType], blockDim);

    auto tiling_data_cpu = reinterpret_cast<KernelCATCOCHostTilingData *>(cpu_tiling_tensor.data_ptr<uint8_t>());
    printf("m is: %d ;", tiling_data_cpu->m);
    printf("n is: %d ;", tiling_data_cpu->n);
    printf("k is: %d ;\n", tiling_data_cpu->k);

    int32_t batchIdx = m - 1;
    uint32_t tilingSize = sizeof(KernelCATCOCHostTilingData);
    static auto global_tiling_data = at::empty(
            {tilingSize * MAX_CAPTURE_NUM}, at::TensorOptions().dtype(at::kByte).device(input_a.options().device())).contiguous();
    if (batchIdx >= 0 && batchIdx < MAX_CAPTURE_NUM) {
      aclrtMemcpy(global_tiling_data.data_ptr<uint8_t>() + (tilingSize * batchIdx), tilingSize,
                  cpu_tiling_tensor.data_ptr<uint8_t>(), tilingSize, ACL_MEMCPY_HOST_TO_DEVICE);
    } else {
      // Handle the case where batchIdx is out of range
      TORCH_CHECK(false, "caching tiling batchIdx is out of range: ", batchIdx);
    }
    // c10_npu::getCurrentNPUStream().synchronize();
    at::Tensor tiling_tensor =
            at::from_blob(global_tiling_data.data_ptr<uint8_t>() + (tilingSize * batchIdx), tilingSize,
                          at::TensorOptions().dtype(at::kByte).device(input_a.options().device()));

    // launch the kernel function via torch opcmd
    void *a_ptr = input_a.data_ptr();
    void *b_ptr = input_b.data_ptr();
    void *c_ptr = output_c.data_ptr();
    void *symm_ptr = reinterpret_cast<void*>(symmAddr);
    void *tiling_ptr = tiling_tensor.data_ptr();
    auto fftsAddr = shmemx_get_ffts_config();
    // void *tiling_ptr = reinterpret_cast<void*>(static_cast<uint8_t*>(global_tiling_data.data_ptr<uint8_t>()) + (tilingSize * batchIdx));
    printf("tiling_ptr on host is %ld\n", tiling_ptr);

    // uint32_t block_dim = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    // EXEC_KERNEL_CMD(catcoc_allgather_matmul, block_dim, fftsAddr, input_a, input_b, output_c, symm_ptr, tiling_ptr);
    /*
    at::Tensor cpu_tensor = tiling_tensor.to(at::kCPU).contiguous();
    uint8_t * data_ptr = cpu_tensor.data_ptr<uint8_t>();
    printf("tiling_ptr on host is %ld\n", tiling_ptr);
    printf("M element (hex): %02x %02x %02x %02x\n", data_ptr[0], data_ptr[1], data_ptr[2], data_ptr[3]);
    printf("N element (hex): %02x %02x %02x %02x\n", data_ptr[4], data_ptr[5], data_ptr[6], data_ptr[7]);
    printf("K element (hex): %02x %02x %02x %02x\n", data_ptr[8], data_ptr[9], data_ptr[10], data_ptr[11]);
    */

    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto teamIdx = (uint64_t)teamId;
    uint32_t aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    auto acl_call = [aicCoreNum, stream, fftsAddr, teamIdx, a_ptr, b_ptr, c_ptr, symm_ptr, tiling_ptr]() -> int {
        printf("tiling_ptr on launch is %ld\n", tiling_ptr);
        ACLRT_LAUNCH_KERNEL(catcoc_allgather_matmul_kernel)
            (aicCoreNum, stream, fftsAddr, teamIdx, a_ptr, b_ptr, c_ptr, symm_ptr, tiling_ptr);
        return 0;
        };
    at_npu::native::OpCommand::RunOpApi("catcoc_allgather_matmul_kernel", acl_call);

}

}  // namespace npu_kernel
}  // namespace sglang
