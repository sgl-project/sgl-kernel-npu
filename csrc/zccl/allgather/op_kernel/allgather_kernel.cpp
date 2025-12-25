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
#include "fp16_t.h"
#include "bfloat16.h"
#include "../../../mla_preprocess/op_kernel/kernel/common.h"
#include "../../../mla_preprocess/op_kernel/kernel/hardware.h"
#include "../../../mla_preprocess/op_kernel/kernel/mma.h"
#include "../../../mla_preprocess/op_kernel/kernel/utils.h"
#include "../../../mla_preprocess/op_kernel/kernel/iterator.h"
#include "../../include/zccl.h"
#include "../op_host/allgather_tiling_data.h"

using namespace AscendC;

using fp16_t = op::fp16_t;
using bfloat16 = op::bfloat16;

constexpr int64_t SYNC_FLAG_INTERVAL = 16;
constexpr int64_t UB_DMA_MAX_SIZE = 190 * 1024;
constexpr int64_t GVA_BUFF_MAX_SIZE = 100 * 1024 * 1024;
constexpr int64_t BIG_DATA_SIZE = 2 * 1024 * 1024;

class AllGatherKernel
{
public:
    __aicore__ inline AllGatherKernel() {}

    template<typename T>
    __aicore__ inline void Process(GM_ADDR input, GM_ADDR output, GM_ADDR gva, uint64_t elements, int32_t team_id, 
        uint64_t ffts_addr, int magic, GM_ADDR tiling_data_in)
    {
        auto *tiling_data = reinterpret_cast<__gm__ sglang::zccl::AllGatherTilingData*>(tiling_data_in);
        this->input_num_per_core = tiling_data->input_num_per_core;
        this->output_num_per_core = tiling_data->output_num_per_core;
        this->output_core_per_rank = tiling_data->output_core_per_rank;
        this->input_last_num_core = tiling_data->input_last_num_core;
        this->output_last_num_core = tiling_data->output_last_num_core;
        shmemx_set_ffts_config(ffts_addr);
        if (elements * sizeof(T) < BIG_DATA_SIZE) {
            AllGatherSmallData<T>(input, output, gva, elements, team_id, magic);
        } else {
            AllGatherSmallData<T>(input, output, gva, elements, team_id, magic);
        }
        
    }

    template<typename T>
    __aicore__ inline void AllGatherBigData(GM_ADDR inputGM, GM_ADDR outputGM, GM_ADDR gva, uint64_t elements, int32_t team_id, int magic)
    {
        const int64_t max_gva_num = GVA_BUFF_MAX_SIZE / sizeof(T);
        int times = (elements + max_gva_num - 1) / max_gva_num;
        int total_num = elements;
        AscendC::GlobalTensor<T> inputGT;
        inputGT.SetGlobalBuffer((__gm__ T *)inputGM, elements);
        AscendC::GlobalTensor<T> outputGT;
        outputGT.SetGlobalBuffer((__gm__ T *)outputGM, elements);
        for (int i = 0; i < times; i++) {
            AscendC::PipeBarrier<PIPE_ALL>();
            int32_t len = total_num > max_gva_num ? max_gva_num : total_num;
            shmemx_barrier_all_vec();
            AllGatherOrigin(inputGT[i * max_gva_num], outputGT[i * max_gva_num], gva, max_gva_num, elements, len, team_id,
                (magic + i) * 1024);
            total_num -= max_gva_num;
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

    template<typename T>
    __aicore__ inline void AllGatherOrigin(AscendC::GlobalTensor<T> inputGM, AscendC::GlobalTensor<T> outputGM, GM_ADDR gva, 
        uint64_t max_gva_num, uint64_t elements, int32_t team_id, int magic)
    {
        const int64_t aivNum = AscendC::GetBlockNum();
        const int64_t aivIndex = AscendC::GetBlockIdx();

        const int64_t data_offset = aivNum * SYNC_FLAG_INTERVAL;
        const int64_t flag_offset = aivIndex * SYNC_FLAG_INTERVAL;

        int64_t my_rank = shmem_team_my_pe(team_id);
        int64_t pe_size = shmem_team_n_pes(team_id);

        AscendC::GlobalTensor<T> gvaDataGT;
        gvaDataGT.SetGlobalBuffer((__gm__ T *)gva + data_offset);

        __gm__ int32_t *gva_sync_gm = (__gm__ int32_t *)gva;

        AsdopsBuffer<ArchType::ASCEND_V220> buf;

        // signal_op needed
        __ubuf__ int32_t *flags_ub1[16];
        __ubuf__ int32_t *flags_ub2[16];
        for (int i = 0; i < 16; i++) {
            flags_ub1[i] = (__ubuf__ int32_t *)(32) + i * 16;
            flags_ub2[i] = (__ubuf__ int32_t *)(544) + i * 16;
        }

        // 0-7 copy data to local symmetric mem, 8-15 copy remote data from symmetric mem.
        int core_group_num = aivNum / 2;
        int core_per_rank = core_group_num / pe_size;
        int len_per_core = len / core_group_num;

        int group_per_num = len_per_core;
        if (aivIndex == core_group_num - 1) {  // Remain Handle
            group_per_num = len - group_per_num * aivIndex;
        }

        // GM to SymmPtr
        if (aivIndex < core_group_num) {
            AscendC::LocalTensor<T> tmp_buff = buf.GetBuffer<BufferType:ASCEND_UB, T>(1024 + 32);
            uint32_t copy_ub_size = UB_DMA_MAX_SIZE;
            uint32_t copy_ub_num = copy_ub_size / sizeof(T);
            tmp_buff.SetSize(copy_ub_num);
            uint32_t copy_total_size = group_per_num * sizeof(T);

            int64_t times = 0;
            int64_t flag = 0;
            while (copy_total_size >= copy_ub_size) {
                shmem_mte_put_mem_nbi(gvaDataGT[aivIndex * len_per_core + times * copy_ub_num],
                                    inputGM[aivIndex * len_per_core + times * copy_ub_num], tmp_buff,
                                    copy_ub_num, my_rank, EVENT_ID0);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
                times += 1;
                flag = times + magic;
                shmemx_signal_op(gva_sync_gm + flag_offset, flag, SHMEM_SIGNAL_SET, my_rank);

                AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);

                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);

                copy_total_size -= copy_ub_size;
            }
            if (copy_total_size <= 0) {
                return;
            }
            shmem_mte_put_mem_nbi(gvaDataGT[aivIndex * len_per_core + times * copy_ub_num],
                                inputGM[aivIndex * len_per_core + times * copy_ub_num], tmp_buff,
                                copy_total_size / sizeof(T), my_rank, EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
            times += 1;
            flag = times + magic;
            shmemx_signal_op(gva_sync_gm + flag_offset, flag, SHMEM_SIGNAL_SET, my_rank);
            return;
        }

        // while style
        for (int64_t i = 0; i < core_group_num; i++) {
            *flags_ub1[i] = 0;
            *flags_ub2[i] = 0;
        }

        AscendC::LocalTensor<T> ping_buff = buf.GetBuffer<BufferType:ASCEND_UB, T>(1024 + 32);
        AscendC::LocalTensor<T> pong_buff = buf.GetBuffer<BufferType:ASCEND_UB, T>(96 * 1024 + 32);
        uint32_t copy_ub_size = UB_DMA_MAX_SIZE / 2;
        uint32_t copy_ub_num = copy_ub_size / sizeof(T);
        ping_buff.SetSize(copy_ub_num);
        pong_buff.SetSize(copy_ub_num);
        int x = (aivIndex - core_group_num) / core_per_rank;

        int pingpongId = 0;
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        while (true) {
            for (int group_idx = 0; group_idx < core_group_num; group_idx++) {
                if (*flags_ub1[group_idx] == INT32_MAX) {
                    continue;
                }

                int64_t all_data_size = len_per_core * sizeof(T);
                if (group_idx == core_group_num - 1) {
                    all_data_size = (len - group_idx * len_per_core) * sizeof(T);
                }

                if (*flags_ub1[group_idx] * UB_DMA_MAX_SIZE >= all_data_size) {
                    *flags_ub1[group_idx] = INT32_MAX;
                    continue;
                }

                shmem_get_int32_mem_nbi(flags_ub2[group_idx], gva_sync_gm + group_idx * SYNC_FLAG_INTERVAL, 1, x);
                AscendC::PipeBarrier<PIPE_ALL>();

                int64_t ready_num = *flags_ub2[group_idx] - magic;
                if (ready_num <= 0 || *flags_ub1[group_idx] >= ready_num) {
                    continue;
                }

                int group_recv_offset = x * elements + group_idx * len_per_core;
                int group_send_offset = group_idx * len_per_core;

                int send_offset = *flags_ub1[group_idx] * UB_DMA_MAX_SIZE / sizeof(T);
                int recv_offset = *flags_ub1[group_idx] * UB_DMA_MAX_SIZE / sizeof(T);
                int num_total = (ready_num - *flags_ub1[group_idx]) * UB_DMA_MAX_SIZE / sizeof(T);
                if (ready_num * UB_DMA_MAX_SIZE > all_data_size) {
                    num_total = (all_data_size - *flags_ub1[group_idx] * UB_DMA_MAX_SIZE) / sizeof(T);
                }
                AscendC::PipeBarrier<PIPE_ALL>();
                for (int i = 0; num_total > 0; i++) {
                    AscendC::TEventID EVENT_ID = pingpongId == 0 ? EVENT_ID0 : EVENT_ID1;
                    AscendC::LocalTensor<T> buf = pingpongId == 0 ? ping_buff : pong_buff;

                    uint32_t copy_num = num_total > copy_ub_num ? copy_ub_num : num_total;

                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
                    shmem_mte_get_mem_nbi(outputGM[group_recv_offset + recv_offset],
                                        gvaDataGT[group_send_offset + send_offset], buf, copy_num, x,
                                        EVENT_ID);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);

                    send_offset += copy_num;
                    recv_offset += copy_num;
                    num_total -= copy_num;
                    pingpongId = 1 - pingpongId;
                }
                AscendC::PipeBarrier<PIPE_ALL>();
                *flags_ub1[group_idx] = ready_num;
                AscendC::PipeBarrier<PIPE_ALL>();
            }
            bool finished = true;
            for (int64_t group_idx = 0; group_idx < core_group_num; group_idx++) {
                if (*flags_ub1[group_idx] != INT32_MAX) {
                    finished = false;
                    break;
                }
            }
            if (finished) {
                break;
            }
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    }

    template<typename T>
    __aicore__ inline void AllGatherSmallData(GM_ADDR inputGM, GM_ADDR outputGM, GM_ADDR gva, uint64_t elements, int32_t team_id, int magic)
    {
        const int64_t aivNum = AscendC::GetBlockNum();
        const int64_t aivIndex = AscendC::GetBlockIdx();

        const int64_t data_offset = aivNum * SYNC_FLAG_INTERVAL;
        const int64_t flag_offset = aivIndex * SYNC_FLAG_INTERVAL;

        int64_t my_rank = shmem_team_my_pe(team_id);
        
        AscendC::GlobalTensor<T> inputGT;
        inputGT.SetGlobalBuffer((__gm__ T *)inputGM, elements);
        AscendC::GlobalTensor<T> outputGT;
        outputGT.SetGlobalBuffer((__gm__ T *)outputGM, elements);
        AscendC::GlobalTensor<T> dataGT;
        dataGT.SetGlobalBuffer((__gm__ T *)gva + data_offset, elements);
        
        __gm__ int32_t *gva_sync_gm = (__gm__ int32_t *)gva;

        AsdopsBuffer<ArchType::ASCEND_V220> buf;
        AscendC::LocalTensor<T> tmp_buff = buf.GetBuffer<BufferType:ASCEND_UB, T>(64);

        // data move parameters
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
        num_per_core = this->output_core_per_rank;
        output_offset = x * elements + core_rank_idx * num_per_core;
        gva_offset = core_rank_idx * num_per_core;
        if (core_rank_idx == core_per_rank - 1) {
            num_per_core = this->output_last_num_core;
        }

        shmem_mte_get_mem_nbi(outputGT[output_offset], dataGT[gva_offset], tmp_buff, num_per_core, x, EVENT_ID0);
    }

private:
    int64_t aivNum;
    uint32_t input_num_per_core;
    uint32_t output_num_per_core;
    uint32_t output_core_per_rank;
    uint32_t input_last_num_core;
    uint32_t output_last_num_core;
};


extern "C" __global__ __aicore__ void allgather(GM_ADDR input, GM_ADDR output, GM_ADDR gva, uint32_t numel, int data_type, uint32_t team_id, 
    uint64_t ffts_addr, int magic, GM_ADDR tiling_tensor)
{
    AllGatherKernel op;
    ZCCLDataType zccl_data_type = static_cast<ZCCLDataType>(dataType);
    switch (zccl_data_type){
        case ZCCLDataType::ZCCL_DATA_TYPE_INT8:
            op.Process<int8_t>(input, output, gva, numel, team_id, ffts_addr, magic, tiling_tensor);
            break;
        case ZCCLDataType::ZCCL_DATA_TYPE_INT16:
            op.Process<int16_t>(input, output, gva, numel, team_id, ffts_addr, magic, tiling_tensor);
            break;
        case ZCCLDataType::ZCCL_DATA_TYPE_INT32:
            op.Process<int32_t>(input, output, gva, numel, team_id, ffts_addr, magic, tiling_tensor);
            break;
        case ZCCLDataType::ZCCL_DATA_TYPE_FP16:
            op.Process<float>(input, output, gva, numel, team_id, ffts_addr, magic, tiling_tensor);
            break;
        case ZCCLDataType::ZCCL_DATA_TYPE_FP32:
            op.Process<float>(input, output, gva, numel, team_id, ffts_addr, magic, tiling_tensor);
            break;    
        case ZCCLDataType::ZCCL_DATA_TYPE_INT64:
            op.Process<int64_t>(input, output, gva, numel, team_id, ffts_addr, magic, tiling_tensor);        
            break;
        case ZCCLDataType::ZCCL_DATA_TYPE_BFP16:
            op.Process<bfloat16>(input, output, gva, numel, team_id, ffts_addr, magic, tiling_tensor);
            break;                                            
        default:
            break;
    }
    
}

#endif  // SGL_KERNEL_NPU_ALL_GATHER_KERNEL_H
