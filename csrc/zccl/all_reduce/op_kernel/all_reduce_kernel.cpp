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
#include "bfloat16.h"
#include "../../../mla_preprocess/op_kernel/kernel/common.h"
#include "../../../mla_preprocess/op_kernel/kernel/hardware.h"
#include "../../../mla_preprocess/op_kernel/kernel/mma.h"
#include "../../../mla_preprocess/op_kernel/kernel/utils.h"
#include "../../../mla_preprocess/op_kernel/kernel/iterator.h"
#include "zccl.h"
#include "../op_host/all_reduce_tiling.h"

using namespace AscendC;

using bfloat16 = op::bfloat16;

constexpr int64_t SYNC_FLAG_INTERVAL = 16;
constexpr int64_t UB_DMA_MAX_SIZE = 190 * 1024;
constexpr int64_t GVA_BUFF_MAX_SIZE = 100 * 1024 * 1024;
constexpr int64_t BIG_DATA_SIZE = 2 * 1024 * 1024;

enum class ReduceOp {
    REDUCE_SUM = 0,
    REDUCE_PROD = 1,
    REDUCE_MAX = 2,
    REDUCE_MIN = 3,
    REDUCE_RESERVED = 255
};

template <typename T>
SHMEM_DEVICE void setAtomicOp(ReduceOp reduceOp)
{
    switch (reduceOp) {
        case ReduceOp::REDUCE_SUM:
            AscendC::SetAtomicAdd<T>();
            break;
        case ReduceOp::REDUCE_MAX:
            AscendC::SetAtomicMax<T>();
            break;
        case ReduceOp::REDUCE_MIN:
            AscendC::SetAtomicMin<T>();
            break;
        default:
            AscendC::SetAtomicNone();
            break;
    }
}

// 使用 AscendC 原子操作接口进行规约
template<typename T>
__aicore__ inline void AtomicReduce(AscendC::GlobalTensor<T>& dataGt,
                                   uint32_t destOffset, uint32_t srcOffset,
                                   uint32_t numElements, ReduceOp reduceOp)
{
    // 临时缓冲区用于数据传输
    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    AscendC::LocalTensor<T> tempBuff = buf.GetBuffer<BufferType::ASCEND_UB, T>(numElements);

    // 将源数据加载到本地缓冲区
    AscendC::DataCopy(tempBuff, dataGt[srcOffset], numElements);
}

class AllReduceKernel
{
public:
    __aicore__ inline AllReduceKernel() {}

    template<typename T>
    __aicore__ inline void Process(GM_ADDR input, GM_ADDR output, GM_ADDR gva, uint64_t elements, int32_t teamId, 
        uint64_t fftsAddr, int magic, GM_ADDR tilingConfig, uint32_t reduceOp)
    {
        auto *tilingCfg = reinterpret_cast<__gm__ sglang::zccl::AllReduceTilingData*>(tilingConfig);
        this->inputNumPerCore_ = tilingCfg->inputNumPerCore_;
        this->outputNumPerCore_ = tilingCfg->outputNumPerCore_;
        this->outputCorePerRank_ = tilingCfg->outputCorePerRank_;
        this->inputLastNumCore_ = tilingCfg->inputLastNumCore_;
        this->outputLastNumCore_ = tilingCfg->outputLastNumCore_;
        this->reduceOp_ = static_cast<ReduceOp>(reduceOp);
        shmemx_set_ffts_config(fftsAddr);
        if (elements * sizeof(T) < BIG_DATA_SIZE) {
            AllReduceSmallData<T>(input, output, gva, elements, teamId, magic);
        } else {
            AllReduceSmallData<T>(input, output, gva, elements, teamId, magic);
        }
    }

    template<typename T>
    __aicore__ inline void AllReduceLargeData(GM_ADDR inputGm, GM_ADDR outputGm, GM_ADDR gva, uint64_t elements,
        int32_t teamId, int magic, ReduceOp reduceOp)
    {
        const int64_t aivNum = AscendC::GetBlockNum();
        const int64_t aivIndex = AscendC::GetBlockIdx();
        const int64_t dataOffset = aivNum * SYNC_FLAG_INTERVAL;
        const int64_t flagOffset = aivIndex * SYNC_FLAG_INTERVAL;
        int64_t myRank = shmem_team_my_pe(teamId);

        AscendC::GlobalTensor<T> inputGt, outputGt, gvaGt;
        inputGt.SetGlobalBuffer((__gm__ T *)inputGm, elements);
        outputGt.SetGlobalBuffer((__gm__ T *)outputGm, elements);
        gvaGt.SetGlobalBuffer((__gm__ T *)((__gm__ char *)gva + dataOffset), elements);

        __gm__ int32_t *gvaSyncGm = (__gm__ int32_t *)gva;
        AsdopsBuffer<ArchType::ASCEND_V220> buf;
        AscendC::LocalTensor<T> tmpBuff = buf.GetBuffer<BufferType::ASCEND_UB, T>(64);

        // 使用预计算的 tiling 参数
        const uint32_t coreRankIdx = aivIndex % outputCorePerRank_;
        const uint32_t rank = aivIndex / outputCorePerRank_;  // 当前节点ID
        const uint32_t coreTargetRank = (rank + 1) % aivNum;  // 目标节点ID

        // 计算当前核心处理的数据量
        uint32_t lenPerCore = (coreRankIdx < (elements % aivNum)) ?
                             (elements / aivNum + 1) : (elements / aivNum);

        uint32_t gvaCopyInOffset, gvaCopyOutOffset;

        if (coreRankIdx < (elements % aivNum)) {
            gvaCopyInOffset = coreTargetRank * (elements / aivNum) + coreRankIdx * (elements / aivNum + 1);
            gvaCopyOutOffset = rank * (elements / aivNum) + coreRankIdx * (elements / aivNum + 1);
        } else {
            gvaCopyInOffset = coreTargetRank * (elements / aivNum) + (elements % aivNum) * (elements / aivNum + 1) +
                             (coreRankIdx - (elements % aivNum)) * (elements / aivNum);
            gvaCopyOutOffset = rank * (elements / aivNum) + (elements % aivNum) * (elements / aivNum + 1) +
                              (coreRankIdx - (elements % aivNum)) * (elements / aivNum);
        }

        // [AllReduce Step 1: ReduceScatter]
        shmem_mte_put_mem_nbi(gvaGt[gvaCopyInOffset], inputGt, tmpBuff, lenPerCore, rank, EVENT_ID0);

        shmem_quiet();
        shmemi_barrier_core_soft();

        shmemx_signal_op(gvaSyncGm + flagOffset, magic, SHMEM_SIGNAL_SET, myRank);
        __gm__ int32_t * waitAddr = (__gm__ int32_t *)shmem_ptr(gvaSyncGm, coreTargetRank);
        int32_t waitOffset = (rank * outputCorePerRank_ + coreRankIdx) * SYNC_FLAG_INTERVAL;
        shmem_signal_wait_until(waitAddr + waitOffset, SHMEM_CMP_EQ, magic);

        // [AllReduce Step 2: Reduce with Atomic Operation]
        setAtomicOp<T>(reduceOp);
        AscendC::PipeBarrier<PIPE_ALL>();

        shmem_mte_get_mem_nbi(outputGt, gvaGt[gvaCopyOutOffset], tmpBuff, lenPerCore, coreTargetRank, EVENT_ID0);

        AscendC::SetAtomicNone();
        AscendC::PipeBarrier<PIPE_ALL>();

        // [AllReduce Step 3: AllGather]
        uint32_t outputCopyInOffset, outputCopyOutOffset;

        if (coreRankIdx < (elements % aivNum)) {
            outputCopyInOffset = rank * (elements / aivNum) + coreRankIdx * (elements / aivNum + 1);
            outputCopyOutOffset = coreTargetRank * (elements / aivNum) + coreRankIdx * (elements / aivNum + 1);
        } else {
            outputCopyInOffset = rank * (elements / aivNum) + (elements % aivNum) * (elements / aivNum + 1) +
                                (coreRankIdx - (elements % aivNum)) * (elements / aivNum);
            outputCopyOutOffset = coreTargetRank * (elements / aivNum) + (elements % aivNum) * (elements / aivNum + 1) +
                                 (coreRankIdx - (elements % aivNum)) * (elements / aivNum);
        }

        shmem_mte_put_mem_nbi(gvaGt[outputCopyInOffset], outputGt, tmpBuff, lenPerCore, rank, EVENT_ID0);

        shmem_quiet();
        shmemi_barrier_core_soft();

        shmemx_signal_op(gvaSyncGm + flagOffset, magic + 1, SHMEM_SIGNAL_SET, myRank);
        shmem_signal_wait_until(waitAddr + waitOffset, SHMEM_CMP_EQ, magic + 1);

        shmem_mte_get_mem_nbi(outputGt, gvaGt[outputCopyOutOffset], tmpBuff, lenPerCore, coreTargetRank, EVENT_ID0);
    }


    template<typename T>
    __aicore__ inline void AllReduceSmallData(GM_ADDR inputGm, GM_ADDR outputGm, GM_ADDR gva,
        uint64_t elements, int32_t teamId, int magic)
    {
        #ifdef __DAV_C220_VEC__
            // 小数据场景：直接 AllGather + Reduce
    const int64_t aivNum = AscendC::GetBlockNum();
    const int64_t aivIndex = AscendC::GetBlockIdx();
    const int64_t dataOffset = aivNum * SYNC_FLAG_INTERVAL;
    const int64_t flagOffset = aivIndex * SYNC_FLAG_INTERVAL;
    int64_t myRank = shmem_team_my_pe(teamId);

    AscendC::GlobalTensor<T> inputGt, outputGt, dataGt;
    inputGt.SetGlobalBuffer((__gm__ T *)inputGm, elements);
    outputGt.SetGlobalBuffer((__gm__ T *)outputGm, elements);
    dataGt.SetGlobalBuffer((__gm__ T *)((__gm__ char *)gva + dataOffset), elements);

    __gm__ int32_t *gvaSyncGm = (__gm__ int32_t *)gva;
    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    AscendC::LocalTensor<T> tmpBuff = buf.GetBuffer<BufferType::ASCEND_UB, T>(64);

    // 计算每个核心的数据量
    uint32_t numPerCore = this->inputNumPerCore_;
    uint32_t inputOffset = aivIndex * numPerCore;
    uint32_t gvaOffset = aivIndex * numPerCore;

    if (aivIndex == aivNum - 1) {
        numPerCore = this->inputLastNumCore_;
    }

    // 阶段1：收集所有节点数据到共享内存
    shmem_mte_put_mem_nbi(dataGt[gvaOffset], inputGt[inputOffset], tmpBuff,
                          numPerCore, myRank, EVENT_ID0);

    const int64_t corePerRank = this->outputCorePerRank_;
    const int64_t coreRankIdx = aivIndex % corePerRank;
    const int64_t x = aivIndex / corePerRank;

    // 同步
    shmem_quiet();
    shmemi_barrier_core_soft();
    shmemx_signal_op(gvaSyncGm + flagOffset, magic, SHMEM_SIGNAL_SET, myRank);
    shmem_signal_wait_until((__gm__ int32_t *)shmem_ptr(gvaSyncGm, x) + flagOffset,
                           SHMEM_CMP_EQ, magic);

    // 阶段2：单节点规约（只在节点0执行）
    if (aivIndex == 0) {
        // 设置原子操作模式
        setAtomicOp<T>(reduceOp_);
        for (int64_t srcRank = 1; srcRank < aivNum; srcRank++) {
            uint32_t srcOffset = srcRank * this->inputNumPerCore_;
            uint32_t destOffset = 0;  // 规约到位置0

            if (srcRank == aivNum - 1) {
                numPerCore = this->inputLastNumCore_;
            } else {
                numPerCore = this->inputNumPerCore_;
            }

            AtomicReduce<T>(dataGt, destOffset, srcOffset, numPerCore, reduceOp_);
        }
        // 重置为普通模式
        AscendC::SetAtomicNone();
    }

    // 规约完成同步
    shmem_quiet();
    shmemi_barrier_core_soft();
    if (aivIndex == 0) {
        shmemx_signal_op(gvaSyncGm, magic + 1, SHMEM_SIGNAL_SET, myRank);
    }
    shmem_signal_wait_until(gvaSyncGm, SHMEM_CMP_EQ, magic + 1);

    // 阶段3：广播结果
    numPerCore = this->outputCorePerRank_;
    uint32_t outputOffset = x * elements + coreRankIdx * numPerCore;
    gvaOffset = coreRankIdx * numPerCore;

    if (coreRankIdx == corePerRank - 1) {
        numPerCore = this->outputLastNumCore_;
    }

    shmem_mte_get_mem_nbi(outputGt[outputOffset], dataGt[gvaOffset],
                          tmpBuff, numPerCore, 0, EVENT_ID0);
        #endif
    }

private:
    int64_t aivNum;
    uint32_t inputNumPerCore_;
    uint32_t outputNumPerCore_;
    uint32_t outputCorePerRank_;
    uint32_t inputLastNumCore_;
    uint32_t outputLastNumCore_;
    ReduceOp reduceOp_;
};

// 宏定义：减少重复代码
#define HANDLE_DATA_TYPE(CASE_TYPE, TYPE) \
    case CASE_TYPE: \
        op.Process<TYPE>(input, output, gva, numel, teamId, fftsAddr, magic, tilingConfig, reduceOp); \
        break;


extern "C" __global__ __aicore__ void AllReduce(GM_ADDR input, GM_ADDR output, GM_ADDR gva, uint32_t numel,
    int dataType, uint32_t teamId,
    uint64_t fftsAddr, int magic, GM_ADDR tilingConfig, uint32_t reduceOp)
{
    AllReduceKernel op;
    ZCCLDataType zcclDataType = static_cast<ZCCLDataType>(dataType);
    switch (zcclDataType) {
        HANDLE_DATA_TYPE(ZCCL_DATA_TYPE_INT8, int8_t)
        HANDLE_DATA_TYPE(ZCCL_DATA_TYPE_INT16, int16_t)
        HANDLE_DATA_TYPE(ZCCL_DATA_TYPE_INT32, int32_t)
        HANDLE_DATA_TYPE(ZCCL_DATA_TYPE_FP16, float)
        HANDLE_DATA_TYPE(ZCCL_DATA_TYPE_FP32, float)
        HANDLE_DATA_TYPE(ZCCL_DATA_TYPE_INT64, int64_t)
        HANDLE_DATA_TYPE(ZCCL_DATA_TYPE_BFP16, bfloat16)
        default:
        break;
    }
    
}

#endif  // SGL_KERNEL_NPU_ALL_GATHER_KERNEL_H
