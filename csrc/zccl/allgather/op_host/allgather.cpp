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
#include "acl/acl.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_helper.h"
#include "aclrtlaunch_sgmv_expand_half.h"
#include "allgather_tiling_data.h"

namespace sglang {
namespace npu_kernel {

AllGatherTilingData* get_tiling(int32_t &block_dim, uint64_t elements)
{
    auto ascendc_platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int32_t block_dim = static_cast<int32_t>(ascendc_platform->GetCoreNumAiv());
    int64_t pe_size = shmem_n_pes();
    AllGatherTilingData tiling_data;
    tiling_data->input_num_per_core = elements / block_dim;
    tiling_data->input_last_num_core = elements % block_dim;
    const uint32_t core_per_rank = block_dim / pe_size;
    tiling_data->output_num_per_core = elements / core_per_rank;
    tiling_data->output_last_num_core = elements / core_per_rank;
    return &tiling_data;
}

HOST_API int ZcclAllGather(void *input, void *output, uint64_t numel, HcclDataType dataType, int teamId, aclrtStream stream)
{
    int32_t block_dim;

    void *tiling_device_ptr;
    aclrtMalloc(&tiling_device_ptr, sizeof(AllGatherTilingData), ACL_MEM_MALLOC_HUGE_FIRST);
    AllGatherTilingData *tiling_host;
    tiling_host = get_tiling(block_dim, numel);

    aclrtMallocHost(reinterpret_cast<void**>(tiling_host), sizeof(AllGatherTilingData));
    aclrtMemcpy(tiling_device_ptr, sizeof(AllGatherTilingData), tiling_host, sizeof(AllGatherTilingData), ACL_MEMCPY_HOST_TO_DEVICE);
    uint64_t fftsAddr = shmemx_get_ffts_config();

    ACLRT_LAUNCH_KERNEL(allgather)(block_dim, stream, input, output, numel, teamId, fftsAddr, tiling_tensor)
}

}  // namespace npu_kernel
}  // namespace sglang
