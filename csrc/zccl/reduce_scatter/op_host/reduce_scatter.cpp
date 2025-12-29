// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "acl/acl.h"
#include "defines.h"
#include "reduce_scatter_tilling.h"
#include "tiling/platform/platform_ascendc.h"
#include "aclrtlaunch_ShmemReduceScatter.h"
#include "aclrtlaunch_ShmemZeroBuffReduceScatter.h"
#include "torch_helper.h"
#include "shmem_api.h"
#include "../../include/zccl.h"

namespace sglang {
namespace zccl {

constexpr int64_t SYNC_FLAG_INTERVAL = 16;
constexpr int64_t GVA_BUFF_MAX_SIZE = 100 * 1024 * 1024;
constexpr uint32_t BIG_DATA_THRESHOLD = 2 * 1024 * 1024;
constexpr uint32_t BLOCK_NUM_SMALL_DATA = 8;
constexpr uint32_t BLOCK_NUM_LARGE_DATA = 16;
constexpr uint32_t BLOCK_NUM_ZERO_BUFF = 48;


extern "C" HOST_API int ZcclReduceScatter(uint8_t *inp, uint8_t *out,
    size_t inpNumel, ZCCLDataType dataType, int teamId, aclrtStream stream, uint32_t reduceOp)
{
    /* define the block dim */
    uint32_t blockDim = 0;

    // get team info
    uint32_t rank = shmem_team_my_pe(teamId);
    uint32_t rankSize = shmem_team_n_pes(teamId);

    size_t typeSize = getSizeFromTypeEnum(dataType);
    if (inpNumel * typeSize < BIG_DATA_THRESHOLD) {
        blockDim = BLOCK_NUM_SMALL_DATA;
    } else {
        blockDim = BLOCK_NUM_LARGE_DATA;
    }
    uint32_t dataTypeNum = static_cast<uint32_t>(dataType);

    // Prepare FFTS address
    uint64_t fftsAddr = shmemx_get_ffts_config();
    // allocate gva buffer
    size_t gvaSize = blockDim * SYNC_FLAG_INTERVAL * sizeof(int32_t) + GVA_BUFF_MAX_SIZE;
    void *ptr = shmem_malloc(gvaSize);
    aclrtMemset(ptr, gvaSize, 0, gvaSize);
    // set output empty
    size_t outputSize = inpNumel / rankSize;
    aclrtMemset(out, outputSize, 0, outputSize);

    /* launch the kernel function via ACLRT_LAUNCH_KERNEL */
    ACLRT_LAUNCH_KERNEL(ShmemReduceScatter)(blockDim, stream, inp, out, (uint8_t *)ptr,
                                            fftsAddr, dataTypeNum, inpNumel, teamId, reduceOp);
    shmem_free(ptr);
    return 0;
}

extern "C" HOST_API int ZcclReduceScatterZeroBuff(uint8_t *inp, uint8_t *out,
    size_t inpNumel, ZCCLDataType dataType, int teamId, aclrtStream stream, uint32_t reduceOp)
{
    /* define the block dim */
    uint32_t blockDim = 0;

    // get team info
    uint32_t rankSize = shmem_team_n_pes(teamId);

    size_t typeSize = getSizeFromTypeEnum(dataType);
    if (inpNumel * typeSize < BIG_DATA_THRESHOLD) {
        blockDim = BLOCK_NUM_SMALL_DATA;
    } else {
        blockDim = BLOCK_NUM_ZERO_BUFF;
    }
    uint32_t dataTypeNum = static_cast<uint32_t>(dataType);

    // Prepare FFTS address
    uint64_t fftsAddr = shmemx_get_ffts_config();
    // allocate gva buffer
    size_t gvaSize = blockDim * SYNC_FLAG_INTERVAL * sizeof(int32_t);
    void *ptr = shmem_malloc(gvaSize);
    aclrtMemset(ptr, gvaSize, 0, gvaSize);
    // set output empty
    size_t outputSize = inpNumel / rankSize;
    aclrtMemset(out, outputSize, 0, outputSize);

    /* launch the kernel function via ACLRT_LAUNCH_KERNEL */
    ACLRT_LAUNCH_KERNEL(ShmemZeroBuffReduceScatter)(blockDim, stream, inp, out, (uint8_t *)ptr,
                                                    fftsAddr, dataTypeNum, inpNumel, teamId, reduceOp);
    return 0;
}

}
}
