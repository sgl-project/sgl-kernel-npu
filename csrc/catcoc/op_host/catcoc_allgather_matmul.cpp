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
#include <cstdlib>

#include "defines.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_helper.h"
#include "../include/catcoc_host_tiling.h"
#include "../include/catcoc_host_utils.h"
#include "../include/catcoc_kernel.h"
// #include "aclrtlaunch_catcoc_allgather_matmul_kernel.h"

extern "C" int rtGetC2cCtrlAddr(uint64_t *config, uint32_t *len);

namespace sglang {
namespace npu_kernel {

/*
static uint64_t gNpuReserveSpace = 1024UL * 1024UL * 1024;
static uint64_t gNpuMallocSpace = 128UL * 1024UL * 1024;
static std::string ipPort = "tcp://127.0.0.1:19233";

void shmem_init() {
  auto get_env_int = [](const char* name, int default_val = 0) {
      char* val = std::getenv(name);
      return val ? std::stoi(val) : default_val;
  };
  auto rankId = get_env_int("RANK");
  auto rankSize = get_env_int("WORLD_SIZE");
  printf("rankId is: %d ; rankSize is: %d", rankId, rankSize);
  auto status = shmem_set_conf_store_tls(false, nullptr, 0);
  printf("shmem_set_conf_store_tls is: %d ;", status);
  shmem_init_attr_t *attributes;
  status = shmem_set_attr(rankId, rankSize, gNpuReserveSpace, ipPort.c_str(), &attributes);
  printf("shmem_set_attr is: %d ;", status);
  status = shmem_init_attr(attributes);
  printf("shmem_init_attr is: %d ;", status);
  status = shmem_init_status();
  printf("shmem_init_status is: %d ;", status);
}
*/

HOST_API void catcoc_allgather_matmul(const at::Tensor &input_a, const at::Tensor &input_b, at::Tensor &output_c,
                                      int64_t symmAddr, int64_t teamId = 0,
                                      c10::optional<c10::string_view> format_mode = c10::nullopt)
{
    // init shmem
    // shmem_init();
    assert(shm::g_state.is_shmem_initialized);

    // ops valid check
    at::ScalarType aType = input_a.scalar_type();
    at::ScalarType bType = input_b.scalar_type();
    at::ScalarType cType = output_c.scalar_type();
    TORCH_CHECK(aType == bType && bType == cType, "tensor type is not the same");
    TORCH_CHECK((aType == at::ScalarType::Half) || (aType == at::ScalarType::BFloat16),
                "tensor type only support half and bf16");

    auto formatMode = static_cast<WeightFormatMode>(GetModeVal(weightFormatMap, format_mode, "ND", "format_mode"));

    uint32_t m = input_a.size(0);
    uint32_t k = input_a.size(1);
    uint32_t n = input_b.size(1);
    TORCH_CHECK(input_b.size(0) == k, "input k dim shape mismatch");

    uint32_t blockDim;
    auto cpu_tiling_tensor = get_tiling_tensor(m, n, k, formatMode, dTypeMap[aType], blockDim);

    auto tiling_data_cpu = reinterpret_cast<KernelCATCOCHostTilingData *>(cpu_tiling_tensor.data_ptr<uint8_t>());

    int32_t batchIdx = m - 1;
    uint32_t tilingSize = sizeof(KernelCATCOCHostTilingData);
    // for graphed decode(max batch controlled by MAX_CAPTURE_NUM)
    static auto global_batched_tiling =
        at::empty({tilingSize * MAX_CAPTURE_NUM},
                  at::TensorOptions().dtype(at::kByte).device(input_a.options().device()))
            .contiguous();
    // for single prefill each time
    static auto global_single_tiling =
        at::empty({tilingSize}, at::TensorOptions().dtype(at::kByte).device(input_a.options().device())).contiguous();
    at::Tensor tiling_tensor;
    if (batchIdx >= 0 && batchIdx < MAX_CAPTURE_NUM) {
        aclrtMemcpy(global_batched_tiling.data_ptr<uint8_t>() + (tilingSize * batchIdx), tilingSize,
                    cpu_tiling_tensor.data_ptr<uint8_t>(), tilingSize, ACL_MEMCPY_HOST_TO_DEVICE);
        tiling_tensor = at::from_blob(global_batched_tiling.data_ptr<uint8_t>() + (tilingSize * batchIdx), tilingSize,
                                      at::TensorOptions().dtype(at::kByte).device(input_a.options().device()));
    } else {
        // FIXME: if decode processing into this step will be undefined action
        aclrtMemcpy(global_single_tiling.data_ptr<uint8_t>(), tilingSize, cpu_tiling_tensor.data_ptr<uint8_t>(),
                    tilingSize, ACL_MEMCPY_HOST_TO_DEVICE);
        tiling_tensor = global_single_tiling;
    }
    // c10_npu::getCurrentNPUStream().synchronize();

    // gmWorkspace is a dummy input for ascendc compile with tiling, catcoc ops use gmSymmetric as actual workspace
    auto workspace_tensor = at::empty({1}, at::TensorOptions().dtype(at::kByte).device(input_a.options().device()));

    // launch the kernel function via torch opcmd
    auto a_ptr = reinterpret_cast<uint8_t *>(input_a.data_ptr());
    auto b_ptr = reinterpret_cast<uint8_t *>(input_b.data_ptr());
    auto c_ptr = reinterpret_cast<uint8_t *>(output_c.data_ptr());
    // void *symm_ptr = shmem_malloc(gNpuMallocSpace * sizeof(__fp16));
    auto symm_ptr = reinterpret_cast<uint8_t *>(symmAddr);
    auto tiling_ptr = reinterpret_cast<uint8_t *>(tiling_tensor.data_ptr());
    // auto fftsAddr = shmemx_get_ffts_config();
    uint32_t len;
    uint64_t fftsAddr;
    rtGetC2cCtrlAddr(&fftsAddr, &len);
    auto workspace_ptr = reinterpret_cast<uint8_t *>(workspace_tensor.data_ptr());

    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto teamIdx = (uint64_t)teamId;
    uint32_t aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    std::function<int()> acl_call;
    if ((aType == at::ScalarType::Half) && (formatMode == WeightFormatMode::WEIGHT_ND)) {
        acl_call = [aicCoreNum, stream, fftsAddr, teamIdx, a_ptr, b_ptr, c_ptr, symm_ptr, workspace_ptr,
                    tiling_ptr]() -> int {
            // printf("[catcoc_allgather_matmul_fp16_wnd_kernel] tiling_ptr on launch is %ld\n", tiling_ptr);
            catcoc_allgather_matmul_fp16_wnd_kernel(aicCoreNum, stream, fftsAddr, teamIdx, a_ptr, b_ptr, c_ptr,
                                                    symm_ptr, workspace_ptr, tiling_ptr);
            return 0;
        };
    } else if ((aType == at::ScalarType::Half) && (formatMode == WeightFormatMode::WEIGHT_NZ)) {
        acl_call = [aicCoreNum, stream, fftsAddr, teamIdx, a_ptr, b_ptr, c_ptr, symm_ptr, workspace_ptr,
                    tiling_ptr]() -> int {
            // printf("[catcoc_allgather_matmul_fp16_wnz_kernel] tiling_ptr on launch is %ld\n", tiling_ptr);
            catcoc_allgather_matmul_fp16_wnz_kernel(aicCoreNum, stream, fftsAddr, teamIdx, a_ptr, b_ptr, c_ptr,
                                                    symm_ptr, workspace_ptr, tiling_ptr);
            return 0;
        };
    } else if ((aType == at::ScalarType::BFloat16) && (formatMode == WeightFormatMode::WEIGHT_ND)) {
        acl_call = [aicCoreNum, stream, fftsAddr, teamIdx, a_ptr, b_ptr, c_ptr, symm_ptr, workspace_ptr,
                    tiling_ptr]() -> int {
            // printf("[catcoc_allgather_matmul_bf16_wnd_kernel] tiling_ptr on launch is %ld\n", tiling_ptr);
            catcoc_allgather_matmul_bf16_wnd_kernel(aicCoreNum, stream, fftsAddr, teamIdx, a_ptr, b_ptr, c_ptr,
                                                    symm_ptr, workspace_ptr, tiling_ptr);
            return 0;
        };
    } else if ((aType == at::ScalarType::BFloat16) && (formatMode == WeightFormatMode::WEIGHT_NZ)) {
        acl_call = [aicCoreNum, stream, fftsAddr, teamIdx, a_ptr, b_ptr, c_ptr, symm_ptr, workspace_ptr,
                    tiling_ptr]() -> int {
            // printf("[catcoc_allgather_matmul_bf16_wnz_kernel] tiling_ptr on launch is %ld\n", tiling_ptr);
            catcoc_allgather_matmul_bf16_wnz_kernel(aicCoreNum, stream, fftsAddr, teamIdx, a_ptr, b_ptr, c_ptr,
                                                    symm_ptr, workspace_ptr, tiling_ptr);
            return 0;
        };
    } else {
        AT_ERROR("Unknown tiling cases, ops exec failed!");
    }
    at_npu::native::OpCommand::RunOpApiV2("catcoc_allgather_matmul_kernel", acl_call);
}

}  // namespace npu_kernel
}  // namespace sglang
