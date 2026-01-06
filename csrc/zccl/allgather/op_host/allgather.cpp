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
#include "aclrtlaunch_allgather.h"
#include "aclrtlaunch_allgatherZeroBuff.h"
#include "allgather_tiling_data.h"
#include "torch_helper.h"
#include "../../include/zccl.h"

constexpr int64_t SYNC_FLAG_INTERVAL = 16;
constexpr int64_t GVA_BUFF_MAX_SIZE = 100 * 1024 * 1024;
constexpr int64_t BIG_DATA_SIZE = 40 * 1024 * 1024;

namespace sglang {
namespace zccl {

std::shared_ptr<AllGatherTilingData> get_tiling(int32_t block_dim, uint64_t elements, int team_id)
{
    int64_t pe_size = shmem_team_n_pes(team_id);
    auto tiling_data = std::make_shared<AllGatherTilingData>();
    tiling_data->input_num_per_core = elements / block_dim;
    tiling_data->input_last_num_core = elements - (block_dim - 1) * tiling_data->input_num_per_core;
    const uint32_t core_per_rank = block_dim / pe_size;
    tiling_data->output_core_per_rank = core_per_rank;
    tiling_data->output_num_per_core = elements / core_per_rank;
    tiling_data->output_last_num_core = elements - (core_per_rank - 1) * tiling_data->output_num_per_core;
    return tiling_data;
}

extern "C" HOST_API int ZcclAllGather(void *input, void *output, uint64_t numel, ZCCLDataType data_type, int team_id, aclrtStream stream)
{
    int32_t block_dim = 0;
    if (numel * getSizeFromTypeEnum(data_type) < BIG_DATA_SIZE) {
        block_dim = 8;
    }else {
        block_dim = 16;
    }
    int magic = 1024;

    void *tiling_device_ptr;
    aclrtMalloc(&tiling_device_ptr, sizeof(AllGatherTilingData), ACL_MEM_MALLOC_HUGE_FIRST);
    std::shared_ptr<AllGatherTilingData> tiling_host;
    aclrtMallocHost(reinterpret_cast<void**>(tiling_host.get()), sizeof(AllGatherTilingData));
    tiling_host = get_tiling(block_dim, numel, team_id);

    aclrtMemcpy(tiling_device_ptr, sizeof(AllGatherTilingData), tiling_host.get(), sizeof(AllGatherTilingData), ACL_MEMCPY_HOST_TO_DEVICE);
    uint64_t ffts_addr = shmemx_get_ffts_config();
    size_t gva_size = block_dim * SYNC_FLAG_INTERVAL * sizeof(int) + GVA_BUFF_MAX_SIZE;
    void *gva = shmem_malloc(gva_size);
    aclrtMemset(gva, gva_size, 0, gva_size);
    int data_type_int = static_cast<int>(data_type);

    ACLRT_LAUNCH_KERNEL(allgather)(block_dim, stream, input, output, gva, numel, data_type_int, team_id, ffts_addr, magic, tiling_device_ptr);
    aclrtFreeHost(tiling_host.get());
    aclrtFree(tiling_device_ptr);
    shmem_free(gva);
    return 0;
}

extern "C" HOST_API int ZcclAllGatherZeroBuff(void *input, void *output, uint64_t numel, ZCCLDataType data_type, int team_id, aclrtStream stream)
{
    int32_t block_dim = 0;
    if (numel * sizeof(int) < BIG_DATA_SIZE) {
        block_dim = 8;
    }else {
        block_dim = 16;
    }
    int magic = 1024;

    void *tiling_device_ptr;
    aclrtMalloc(&tiling_device_ptr, sizeof(AllGatherTilingData), ACL_MEM_MALLOC_HUGE_FIRST);
    std::shared_ptr<AllGatherTilingData> tiling_host;
    aclrtMallocHost(reinterpret_cast<void**>(tiling_host.get()), sizeof(AllGatherTilingData));
    tiling_host = get_tiling(block_dim, numel, team_id);

    aclrtMemcpy(tiling_device_ptr, sizeof(AllGatherTilingData), tiling_host.get(), sizeof(AllGatherTilingData), ACL_MEMCPY_HOST_TO_DEVICE);
    uint64_t ffts_addr = shmemx_get_ffts_config();
    size_t gva_size = block_dim * SYNC_FLAG_INTERVAL * sizeof(int) + GVA_BUFF_MAX_SIZE;
    void *gva = shmem_malloc(gva_size);
    aclrtMemset(gva, gva_size, 0, gva_size);
    int data_type_int = static_cast<int>(data_type);

    ACLRT_LAUNCH_KERNEL(allgatherZeroBuff)(block_dim, stream, input, output, gva, numel, data_type_int, team_id, ffts_addr, magic, tiling_device_ptr);
    aclrtFreeHost(tiling_host.get());
    aclrtFree(tiling_device_ptr);
    shmem_free(gva);
    return 0;
}

}  // namespace zccl
}  // namespace sglang
