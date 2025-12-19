// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <memory>
#include "acl/acl.h"
#include "defines.h"
#include "shmem_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "aclrtlaunch_allgather.h"
#include "allgather_tiling_data.h"
#include "torch_helper"

constexpr int64_t SYNC_FLAG_INTERVAL = 16;
constexpr int64_t GVA_BUFF_MAX_SIZE = 100 * 1024 * 1024;

namespace sglang {
namespace npu_kernel {

std::unique_ptr<AllGatherTilingData> get_tiling(int32_t &block_dim, uint64_t elements)
{
    auto ascendc_platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    block_dim = static_cast<int32_t>(ascendc_platform->GetCoreNumAiv());
    int64_t pe_size = shmem_n_pes();
    auto tiling_data = std::make_unique<AllGatherTilingData>();
    tiling_data->input_num_per_core = elements / block_dim;
    tiling_data->input_last_num_core = elements % block_dim;
    const uint32_t core_per_rank = block_dim / pe_size;
    tiling_data->output_core_per_rank = core_per_rank;
    tiling_data->output_num_per_core = elements / core_per_rank;
    tiling_data->output_last_num_core = elements % core_per_rank;
    return tiling_data;
}

HOST_API int ZcclAllGather(void *input, void *output, uint64_t numel, HcclDataType dataType, int teamId, aclrtStream stream)
{
    int32_t block_dim;

    void *tiling_device_ptr;
    aclrtMalloc(&tiling_device_ptr, sizeof(AllGatherTilingData), ACL_MEM_MALLOC_HUGE_FIRST);
    std::unique_ptr<AllGatherTilingData> tiling_host;
    tiling_host = get_tiling(block_dim, numel);

    aclrtMallocHost(reinterpret_cast<void**>(tiling_host.get()), sizeof(AllGatherTilingData));
    aclrtMemcpy(tiling_device_ptr, sizeof(AllGatherTilingData), tiling_host.get(), sizeof(AllGatherTilingData), ACL_MEMCPY_HOST_TO_DEVICE);
    uint64_t fftsAddr = shmemx_get_ffts_config();

    void *gva = shmem_malloc(block_dim * SYNC_FLAG_INTERVAL * sizeof(int) + GVA_BUFF_MAX_SIZE / sizeof(int));

    ACLRT_LAUNCH_KERNEL(allgather)(block_dim, stream, input, output, gva, numel, teamId, fftsAddr, tiling_device_ptr)
}

}  // namespace npu_kernel
}  // namespace sglang
