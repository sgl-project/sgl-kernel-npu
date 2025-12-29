// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef REDUCE_SCATTER_KERNEL_H
#define REDUCE_SCATTER_KERNEL_H

#include <cstdint>

#include "kernel_operator.h"
#include "shmem_api.h"
#include "zccl.h"

using namespace sglang::zccl;


constexpr int64_t SYNC_FLAG_INTERVAL = 16;
constexpr uint32_t UB_DMA_MAX_SIZE = 190 * 1024;


enum ReduceOp{
    REDUCE_SUM = 0,
    REDUCE_PROD = 1,
    REDUCE_MAX = 2,
    REDUCE_MIN = 3,
    REDUCE_RESERVED = 255
};

__aicore__ inline size_t getSizeFromTypeEnum(ZCCLDataType dtype)
{
    switch (dtype) {
        case ZCCLDataType::ZCCL_DATA_TYPE_INT8:
            return sizeof(int8_t);
        case ZCCLDataType::ZCCL_DATA_TYPE_INT16:
            return sizeof(int16_t);
        case ZCCLDataType::ZCCL_DATA_TYPE_INT32:
            return sizeof(int32_t);
        case ZCCLDataType::ZCCL_DATA_TYPE_INT64:
            return sizeof(int64_t);
        case ZCCLDataType::ZCCL_DATA_TYPE_FP16:
            return sizeof(int16_t);
        case ZCCLDataType::ZCCL_DATA_TYPE_FP32:
            return sizeof(float);
        case ZCCLDataType::ZCCL_DATA_TYPE_BFP16:
            return sizeof(int16_t);
        default:
            break;
    }
}

template <typename T>
SHMEM_DEVICE void SetAtomicOp(uint32_t atomicOp)
{
    switch ((enum ReduceOp)atomicOp) {
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


template <typename T>
inline __aicore__ T CeilDiv(const T dividend, const T divisor)
{
    return (divisor == 0) ? 0 : ((dividend + divisor - 1) / divisor);
}


template <typename T>
class ZeroBuffReduceScatterKernel
{
public:
    __aicore__ inline ZeroBuffReduceScatterKernel() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR gva,
                                uint32_t rank, uint32_t rankSize, uint32_t totalLength,
                                uint32_t magic, uint64_t fftsAddr, uint32_t atomicOp = 0)
    {
        shmemx_set_ffts_config(fftsAddr);
        this->atomicOp = atomicOp;
        this->magic = magic;
        this->rank = rank;

        const uint32_t aivNum = AscendC::GetBlockNum();
        const uint32_t aivIndex = AscendC::GetBlockIdx();

        uint32_t coreGroupNum = aivNum;
        uint32_t lenPerRank = totalLength / rankSize;

        // core_target_rank = [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3] core_rank_idx = [0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]
        this->corePerRank = coreGroupNum / rankSize;
        this->coreRankIdx = aivIndex % corePerRank;
        this->coreTargetRank = aivIndex / corePerRank;

        uint32_t lenPerRankAlignToCore = CeilDiv(lenPerRank, corePerRank) * corePerRank;
        uint32_t formerLength = lenPerRankAlignToCore / corePerRank;
        uint32_t tailLength = lenPerRank / corePerRank;
        uint32_t formerNum = lenPerRank % corePerRank;
        uint32_t tailNum = corePerRank - formerNum;
        uint32_t xOffset;
        uint32_t yOffset;

        if (coreRankIdx < formerNum) {
            this->lenPerCore = formerLength;
            xOffset = rank * lenPerRank + coreRankIdx * formerLength;
            yOffset = coreRankIdx * formerLength;
        } else {
            this->lenPerCore = tailLength;
            xOffset = rank * lenPerRank +
                formerNum * formerLength + (coreRankIdx - formerNum) * tailLength;
            yOffset = formerNum * formerLength + (coreRankIdx - formerNum) * tailLength;
        }

        uint32_t gvaDataOffset = aivNum * SYNC_FLAG_INTERVAL;
        gvaSyncOffset = aivIndex * SYNC_FLAG_INTERVAL;
        xGm.SetGlobalBuffer((__gm__ T *)x + xOffset, this->lenPerCore);
        yGm.SetGlobalBuffer((__gm__ T *)y + yOffset, this->lenPerCore);
        gvaSyncGm.SetGlobalBuffer((__gm__ int32_t *)gva, gvaDataOffset);
    }

    __aicore__ inline void Process()
    {
#ifdef __DAV_C220_VEC__
        const uint32_t ubSize = UB_DMA_MAX_SIZE;
        AscendC::LocalTensor<T> tmpBuff(AscendC::TPosition::VECIN, 64, ubSize);

        const __gm__ int32_t *gvaSyncGmAddr = gvaSyncGm.GetPhyAddr();
        __gm__ int32_t *coreGvaSyncGmAddr = (__gm__ int32_t *)gvaSyncGmAddr + gvaSyncOffset;
        shmemx_signal_op(coreGvaSyncGmAddr, magic, SHMEM_SIGNAL_SET, rank);
        __gm__ int32_t * waitAddr = (__gm__ int32_t *)shmem_ptr((__gm__ int32_t *)gvaSyncGmAddr, coreTargetRank);
        int32_t waitOffset = (rank * corePerRank + coreRankIdx) * SYNC_FLAG_INTERVAL;
        shmem_signal_wait_until(waitAddr + waitOffset, SHMEM_CMP_EQ, magic);

        // [ReduceScatter Step 1] local input gm -> output gm.
        SetAtomicOp<T>(atomicOp);
        AscendC::PipeBarrier<PIPE_ALL>();

        shmem_mte_get_mem_nbi(yGm, xGm, tmpBuff, lenPerCore, coreTargetRank, EVENT_ID0);

        AscendC::SetAtomicNone();
        AscendC::PipeBarrier<PIPE_ALL>();
        // Sync Ensure Corresponding Tasks Done.
        shmem_quiet();
        shmemi_barrier_core_soft();
#endif
    }

private:
    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<T> yGm;
    AscendC::GlobalTensor<int32_t> gvaSyncGm;
    uint32_t rank;
    uint64_t fftsAddr;
    uint32_t atomicOp;
    uint32_t lenPerCore;
    uint32_t coreTargetRank;
    uint32_t coreRankIdx;
    uint32_t corePerRank;
    uint32_t gvaSyncOffset;
    uint32_t magic;
};


extern "C" __global__ __aicore__ void ShmemZeroBuffReduceScatter(
    GM_ADDR input, GM_ADDR output, GM_ADDR gva,
    uint64_t fftsAddr, uint32_t dataType, uint32_t totalLength,
    int teamId, uint32_t reduceOp)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    uint32_t magic = 1;
    uint32_t rank = shmem_team_my_pe(teamId);
    uint32_t rankSize = shmem_team_n_pes(teamId);
    ZCCLDataType zcclDataType = static_cast<ZCCLDataType>(dataType);
    size_t typeSize = getSizeFromTypeEnum(zcclDataType);
    switch (zcclDataType) {
        case ZCCLDataType::ZCCL_DATA_TYPE_INT32: {
            ZeroBuffReduceScatterKernel<int32_t> op;
            op.Init(input, output, gva, rank, rankSize, totalLength, magic, fftsAddr, reduceOp);
            op.Process();
            break;
        }
        case ZCCLDataType::ZCCL_DATA_TYPE_FP32: {
            ZeroBuffReduceScatterKernel<float> op;
            op.Init(input, output, gva, rank, rankSize, totalLength, magic, fftsAddr, reduceOp);
            op.Process();
            break;
        }
        default:
            return;
    }
}

#endif // REDUCE_SCATTER_KERNEL_H