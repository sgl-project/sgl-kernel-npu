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
SHMEM_DEVICE void SetAtomicOp(ReduceOp reduceOp)
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

template<typename T>
class AllReduceKernel
{
public:
    __aicore__ inline AllReduceKernel() {}

	__aicore__ inline void Init(
		GM_ADDR input, GM_ADDR output, GM_ADDR gva, GM_ADDR tilingConfig, uint32_t teamId, uint32_t reduceOp)
	{
    	remotePe_ = shmem_team_my_pe(teamId);
        peSize_ = shmem_team_n_pes(teamId);
		auto *tilingCfg = reinterpret_cast<__gm__ sglang::zccl::AllReduceTilingData*>(tilingConfig);
        formerNum_ = tilingCfg->formerNum_;
        formerLength_ = tilingCfg->formerLength_;
        tailLength_ = tilingCfg->tailLength_;
        tailNum_ = tilingCfg->tailNum_;
		eleNumPerRank_ = tilingCfg->eleNumPerRank_;
        reduceOp_ = static_cast<ReduceOp>(reduceOp);

        const uint32_t aivNum = AscendC::GetBlockNum();
        const uint32_t aivIndex = AscendC::GetBlockIdx();

		// [0, 1, 2, 3 || 4, 5, 6, 7 || 8, 9, 10, 11 || 12, 13, 14, 15]
        uint32_t groupId_ = aivIndex / tilingCfg->coreNumPerRank_;

		// [0, 1, 2, 3 || 0, 1, 2, 3 || 0, 1, 2, 3 || 0, 1, 2, 3]
        uint32_t coreInGroupIdx = aivIndex % tilingCfg->coreNumPerRank_;

		// 计算偏移 通信组基址 + 组内偏移
		if (coreInGroupIdx < formerNum_) {
			lenPerCore_ = formerLength_;
            copyOffset_ = groupId_ * eleNumPerRank_ + coreInGroupIdx * formerLength_;
        } else {
			lenPerCore_ = tailLength_;
			// 计算偏移 通信组基址 + 组内偏移
            copyOffset_ =
				groupId_ * eleNumPerRank_ + formerNum_ * formerLength_ + (coreInGroupIdx - formerNum_) * tailLength_;
        }

        inputGm_.SetGlobalBuffer((__gm__ T *)input + copyOffset_, lenPerCore_);
       	outputGm_.SetGlobalBuffer((__gm__ T *)output + copyOffset_, lenPerCore_);

		// [SYNC_FLAG_INTERVAL, SYNC_FLAG_INTERVAL, SYNC_FLAG_INTERVAL || data, data, data]
		uint32_t gvaSyncLen = aivNum * SYNC_FLAG_INTERVAL;
		gvaSyncOffset_ = aivIndex * SYNC_FLAG_INTERVAL;
        gvaGm_.SetGlobalBuffer((__gm__ T *)((__gm__ int32_t *)gva + gvaSyncLen), GVA_BUFF_MAX_SIZE / sizeof(T));
        gvaSyncGm_.SetGlobalBuffer((__gm__ int32_t *)gva, gvaSyncLen);
	}

    __aicore__ inline void Process(uint64_t elements, uint64_t fftsAddr, int magic)
    {
        shmemx_set_ffts_config(fftsAddr);
        if (elements * sizeof(T) < BIG_DATA_SIZE) {
            AllReduceSmallData(magic);
        } else {
            AllReduceSmallData(magic);
        }
    }

    // 小数据场景：直接 ReduceScatter + AllGather
    __aicore__ inline void AllReduceSmallData(int magic)
    {
        #ifdef __DAV_C220_VEC__

        const __gm__ int32_t* gvaSyncGmStart = gvaSyncGm_.GetPhyAddr();
        __gm__ int32_t* gvaSyncGmAddr = (__gm__ int32_t *)gvaSyncGmStart + gvaSyncOffset_;

        AsdopsBuffer<ArchType::ASCEND_V220> buf;
        AscendC::LocalTensor<T> tmpBuff = buf.GetBuffer<BufferType::ASCEND_UB, T>(64);

        // 阶段1：收集所有节点数据到共享内存, 在发送到system mem 时计算
        shmem_mte_put_mem_nbi(gvaGm_[copyOffset_], inputGm_, tmpBuff, lenPerCore_, remotePe_, EVENT_ID0);

        // 同步
        shmem_quiet();
        shmemi_barrier_core_soft();
        shmemx_signal_op(gvaSyncGmAddr, magic, SHMEM_SIGNAL_SET, remotePe_);

        // 阶段2：从system拷贝到本节点 output gm
        AscendC::PipeBarrier<PIPE_ALL>();
		SetAtomicOp<T>(reduceOp_);
        for (int pe = 0; pe < peSize_; pe++) {
            __gm__ int32_t * waitAddr = (__gm__ int32_t *)shmem_ptr((__gm__ int32_t *)gvaSyncGmAddr, pe);
            shmem_signal_wait_until(waitAddr, SHMEM_CMP_EQ, magic);
            shmem_mte_get_mem_nbi(outputGm_, gvaGm_[copyOffset_], tmpBuff, lenPerCore_, pe, EVENT_ID0);
        }
        // 重置原子操作
        AscendC::SetAtomicNone();
        AscendC::PipeBarrier<PIPE_ALL>();
        #endif
    }

private:
    uint32_t formerNum_;
    uint32_t formerLength_;
    uint64_t tailLength_;
    uint32_t tailNum_;
	uint32_t lenPerCore_;
	uint32_t eleNumPerRank_;
	uint32_t copyOffset_;
	uint32_t gvaSyncOffset_;
	uint32_t gvaCopyInOffset_;
	uint32_t groupId_;
	uint32_t remotePe_;
	uint32_t peSize_;
    ReduceOp reduceOp_;
    AscendC::GlobalTensor<T> inputGm_;
    AscendC::GlobalTensor<T> outputGm_;
    AscendC::GlobalTensor<T> gvaGm_;
    AscendC::GlobalTensor<int32_t> gvaSyncGm_;
};

// 宏定义：减少重复代码
#define HANDLE_DATA_TYPE(CASE_TYPE, TYPE) \
    case CASE_TYPE: {\
        AllReduceKernel<TYPE> op; \
		op.Init(input, output, gva, tilingConfig, teamId, reduceOp); \
        op.Process(numel, fftsAddr, magic); \
        break; \
    } \


extern "C" __global__ __aicore__ void AllReduce(GM_ADDR input, GM_ADDR output, GM_ADDR gva, uint32_t numel,
    int dataType, uint32_t teamId,
    uint64_t fftsAddr, int magic, GM_ADDR tilingConfig, uint32_t reduceOp)
{
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
