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
#include "shmem_api.h"
#include "../op_host/allgather_tiling_data.h"

constexpr int64_t SYNC_FLAG_INTERVAL = 16;
constexpr int64_t UB_DMA_MAX_SIZE = 190 * 1024;
constexpr int64_t GVA_BUFF_MAX_SIZE = 100 * 1024 * 1024;

class AllGatherKernel
{
public:
    __aicore__ inline AllGatherKernel() {}

    template<typename T>
    __aicore__ inline void Process(GM_ADDR input, GM_ADDR output, GM_ADDR gva, uint64_t elements, int32_t teamId, uint64_t fftsAddr, GM_ADDR tiling_data_in)
    {
        __gm__ sglang::npu_kernel::AllGatherTilingData *tiling_data = reinterpret_cast<__gm__ sglang::npu_kernel::AllGatherTilingData*>(tiling_data_in);
        this->input_num_per_core = tiling_data->input_num_per_core;
        this->output_num_per_core = tiling_data->output_num_per_core;
        this->output_core_per_rank = tiling_data->output_core_per_rank;
        this->input_last_num_core = tiling_data->input_last_num_core;
        this->output_last_num_core = tiling_data->output_last_num_core;
        shmemx_set_ffts_config(fftsAddr);
        AllGatherSmallData<T>(input, output, elements);
    }

    template<typename T>
    __aicore__ inline void AllGatherSmallData(__gm__ uint8_t *inputGM, __gm__ uint8_t *outputGM, __gm__ uint8_t *gva, uint64_t elements, int32_t teamId)
    {
        const int64_t aivNum = AscendC::GetBlockNum();
        const int64_t aivIndex = AscendC::GetBlockIdx();

        const int64_t data_offset = aivNum * SYNC_FLAG_INTERVAL;
        const int64_t flag_offset = aivIndex * SYNC_FLAG_INTERVAL;

        int64_t my_rank = shmem_my_pe();
        
        AscendC::GlobalTensor<T> inputGT;
        inputGT.SetGlobalBuffer((__gm__ T *)inputGM, elements);
        AscendC::GlobalTensor<T> outputGT;
        outputGT.SetGlobalBuffer((__gm__ T *)outputGM, elements);
        AscendC::GlobalTensor<T> dataGT;
        dataGT.SetGlobalBuffer((__gm__ T *)gva + data_offset, elements);
        
        __gm__ int32_t *gva_sync_gm = (__gm__ int32_t *)gva;

        AscendC::LocalTensor<T> tmp_buff;

        // data move parameters
        const uint32_t ub_size = UB_DMA_MAX_SIZE;
        uint32_t input_offset, output_offset, gva_offset, num_per_core;

        // [AllGather Step 1] local input gm -> symmetric mem.

        num_per_core = this->input_num_per_core;
        input_offset = aivIndex * num_per_core;
        gva_offset = aivIndex * num_per_core;
        if (aivIndex == aivNum - 1) {
            num_per_core = this->input_last_num_core;
        }

        shmem_mte_put_mem_nbi(dataGT[gva_offset], inputGT[input_offset], tmp_buff, num_per_core, my_rank, EVENT_ID0);

        const int64_t core_per_rank = this->output_core_per_rank;
        const int64_t core_rank_idx = aivIndex % core_per_rank;
        const int64_t x = aivIndex / core_per_rank;

        // Sync Ensure Corresponding Tasks Done.
        shmem_quiet();
        shmemi_barrier_core_soft();

        int magic = 1024;
        shmemx_signal_op(gva_sync_gm + flag_offset, magic, SHMEM_SIGNAL_SET, my_rank);
        shmem_signal_wait_until((__gm__ int32_t *)shmem_ptr(gva_sync_gm, x) + flag_offset, SHMEM_CMP_EQ, magic);

        // [AllGather Step 2] symmetric mem -> local output.
        num_per_core = elements / this->output_core_per_rank;
        output_offset = x * elements + core_rank_idx * num_per_core;
        gva_offset = core_rank_idx * num_per_core;
        if (core_rank_idx == core_per_rank - 1) {
            num_per_core = this->output_last_num_core;
        }

        shmem_mte_get_mem_nbi(outputGT[output_offset], dataGT[gva_offset], tmp_buff, num_per_core, my_rank, EVENT_ID0);
    }

private:
    int64_t aivNum;
    uint32_t input_num_per_core;
    uint32_t output_num_per_core;
    uint32_t output_core_per_rank;
    uint32_t input_last_num_core;
    uint32_t output_last_num_core;
};


extern "C" __global__ __aicore__ void allgather(GM_ADDR input, GM_ADDR output, GM_ADDR gva, uint32_t numel, uint32_t teamId, 
    uint64_t fftsAddr, GM_ADDR tiling_tensor)
{
    AllGatherKernel op;
    op.Process<int>(input, output, numel, teamId, fftsAddr, tiling_tensor);
}

#endif  // SGL_KERNEL_NPU_ALL_GATHER_KERNEL_H
