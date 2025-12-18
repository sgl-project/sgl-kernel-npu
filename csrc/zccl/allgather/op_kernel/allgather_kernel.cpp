// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef SGL_KERNEL_NPU_ALL_GATHER_KERNEL_H
#define SGL_KERNEL_NPU_ALL_GATHER_KERNEL_H

#include "kernel_operator.h"
#include "acl/acl.h"
#include "shmem_api.h"

constexpr int64_t SYNC_FLAG_INTERVAL = 16;
constexpr int64_t UB_DMA_MAX_SIZE = 190 * 1024;
constexpr int64_t GVA_BUFF_MAX_SIZE = 100 * 1024 * 1024;

class AllGatherKernel
{
public:
    __aicore__ inline AllGatherKernel() {}

    __aicore__ inline void Process(GM_ADDR input, GM_ADDR output, uint64_t elements)
    {
        AllGatherSmallData<int>(input, output, elements);
    }

    template<typename T>
    __aicore__ inline void AllGatherSmallData(__gm__ T *inputGM, __gm__ T *outputGM, uint64_t elements)
    {
        const int64_t aivNum = GetBlockNum();
        const int64_t aivIndex = GetBlockIdx();

        const int64_t data_offset = aivNum * SYNC_FLAG_INTERVAL;
        const int64_t flag_offset = aivIndex * SYNC_FLAG_INTERVAL;

        void *ptr = shmem_malloc(aiv_num * SYNC_FLAG_INTERVAL * sizeof(T) + GVA_BUFF_MAX_SIZE / sizeof(T));
        __gm__ T *gva = (__gm__ T *)ptr;

        int64_t my_rank = shmem_my_pe();
        int64_t pe_size = shmem_n_pes();
        
        AscendC::GlobalTensor<T> inputGT;
        inputGT.SetGlobalBuffer(inputGM, elements);
        AscendC::GlobalTensor<T> outputGT;
        outputGT.SetGlobalBuffer(outputGM, elements);
        AscendC::GlobalTensor<T> dataGT;
        dataGT.SetGlobalBuffer(gva + data_offset, elements);
        
        __gm__ int32_t *gva_sync_gm = (__gm__ int32_t *)gva;

        AscendC::LocalTensor<T> tmp_buff;

        // data move parameters
        const uint32_t ub_size = UB_DMA_MAX_SIZE;
        uint32_t input_offset, output_offset, gva_offset, num_per_core;

        // [AllGather Step 1] local input gm -> symmetric mem.

        input_offset = aivIndex * num_per_core;
        gva_offset = aivIndex * num_per_core;
        if (aivIndex == aivNum - 1) {
            num_per_core = elements - num_per_core * aivIndex;
        }

        shmem_mte_put_mem_nbi(dataGT[gva_offset], inputGT[input_offset], tmp_buff, num_per_core, my_rank, EVENT_ID0);

        const int64_t core_per_rank = aivNum / pe_size;
        const int64_t core_rank_idx = aivIndex % core_per_rank;
        const int64_t x = aivIndex / core_per_rank;

        // Sync Ensure Corresponding Tasks Done.
        shmem_quiet();
        shmemi_barrier_core_soft();

        int magic = 1024;
        shmemx_signal_op(gva_sync_gm + flag_offset, magic, SHMEM_SIGNAL_SET, my_rank);
        shmem_signal_wait_until((__gm__ int32_t *)shmem_ptr(gva_sync_gm, x) + flag_offset, SHMEM_CMP_EQ, magic);

        // [AllGather Step 2] symmetric mem -> local output.
        num_per_core = elements / core_per_rank;
        output_offset = x * elements + core_rank_idx * num_per_core;
        gva_offset = core_rank_idx * num_per_core;
        if (core_rank_idx == core_per_rank - 1) {
            num_per_core = elements - num_per_core * core_rank_idx;
        }

        shmem_mte_get_mem_nbi(outputGT[output_offset], dataGT[gva_offset], tmp_buff, num_per_core, my_rank, EVENT_ID0);
    }

private:
    int64_t aivNum;
    uint32_t tileNum;
    uint32_t tileLength;
};


extern "C" __global__ __aicore__ void allgather(GM_ADDR input, GM_ADDR output, uint32_t numel, uint32_t teamId, 
    uint32_t fftsAddr, tiling_tensor)
{
    AllGatherKernel op;
    op.Process(input, output, numel, teamId, fftsAddr, tiling_tensor);
}

#endif  // SGL_KERNEL_NPU_ALL_GATHER_KERNEL_H
