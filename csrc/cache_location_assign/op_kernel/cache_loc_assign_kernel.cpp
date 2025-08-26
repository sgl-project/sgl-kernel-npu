// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef SGL_KERNEL_NPU_KERNEL_CACHE_LOC_ASSIGN_H
#define SGL_KERNEL_NPU_KERNEL_CACHE_LOC_ASSIGN_H

/* include file of ascendc */
#include "kernel_operator.h"
#include "../op_host/tiling/cache_loc_assign.h"

/* tensor num for each queue */
constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class CacheLocAssignKernel {
public:
    __aicore__ inline CacheLocAssignKernel()
    {}

    __aicore__ inline void Init(GM_ADDR reqPoolIndices, GM_ADDR tokenPool, GM_ADDR startOffset, GM_ADDR endOffset,
        GM_ADDR outCacheLoc, __gm__ AssignCacheTillingData *tempTilingGM)
    {
        this->coreId = AscendC::GetBlockIdx();
        this->batchSize = tempTilingGM->batchSize;
        this->vcoreNum = tempTilingGM->vcoreNum;
        this->rowNumNoTail = tempTilingGM->rowNumNoTail;
        this->tailNum = tempTilingGM->tailNum;
        if (this->coreId < this->tailNum) {
            this->rowNum = this->rowNumNoTail + 1;
            this->tailOffset = this->coreId;
        } else {
            this->rowNum = this->rowNumNoTail;
            this->tailOffset = this->tailNum;
        }
        this->rowSize = tempTilingGM->rowSize;
        this->reqInxBufferCount = tempTilingGM->reqInxBufferCount;
        this->tokenCountAlignInt32 = tempTilingGM->tokenCountAlignInt32;
        this->offsetCountAlignInt64 = tempTilingGM->offsetCountAlignInt64;
        this->cacheLocCountAlignInt32 = tempTilingGM->cacheLocCountAlignInt32;

        this->rowOffset = this->rowNumNoTail * this->coreId + this->tailOffset;
        this->reqPoolIndicesGM.SetGlobalBuffer((__gm__ T *)reqPoolIndices, this->batchSize);
        this->tokenPoolGM.SetGlobalBuffer((__gm__ int32_t *)tokenPool, tempTilingGM->poolSize * this->rowSize);
        this->startOffsetGm.SetGlobalBuffer((__gm__ int64_t *)startOffset, this->batchSize);
        this->endOffsetGM.SetGlobalBuffer((__gm__ int64_t *)endOffset, this->batchSize);
        this->cacheLocGM.SetGlobalBuffer((__gm__ int32_t *)outCacheLoc, tempTilingGM->cacheLocSize);

        AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuff1, tmpBuff2, tmpBuff3, tmpBuff4, tmpBuff5, tmpBuff6, tmpBuff7;
        this->pipe.InitBuffer(tmpBuff1, tempTilingGM->reqInxBufferSize);
        this->pipe.InitBuffer(tmpBuff2, tempTilingGM->offsetColAlignInt64);
        this->pipe.InitBuffer(tmpBuff3, tempTilingGM->offsetColAlignInt64);
        uint64_t offsetBufferInt32 = this->batchSize * sizeof(int32_t);
        this->pipe.InitBuffer(tmpBuff4, offsetBufferInt32);
        this->pipe.InitBuffer(tmpBuff5, offsetBufferInt32);
        this->pipe.InitBuffer(tmpBuff6, offsetBufferInt32);
        this->pipe.InitBuffer(tmpBuff7, tempTilingGM->cacheLocAlignInt32);

        this->ubReqPoolIndices = tmpBuff1.Get<T>();
        this->ubStartOffset = tmpBuff2.Get<int64_t>();
        this->ubEndOffset = tmpBuff3.Get<int64_t>();
        this->ubStartOffsetInt32 = tmpBuff4.Get<int32_t>();
        this->ubEndOffsetInt32 = tmpBuff5.Get<int32_t>();

        this->cacheLength = tmpBuff6.Get<int32_t>();
        this->ubCacheLoc = tmpBuff7.Get<int32_t>();

        this->pipe.InitBuffer(this->inQueue1, BUFFER_NUM, tempTilingGM->tokenColAlignInt32);
    }

    __aicore__ inline void Process()
    {
        if (this->rowNum > 0) {
            AscendC::DataCopy(this->ubReqPoolIndices, this->reqPoolIndicesGM, this->reqInxBufferCount);
            AscendC::DataCopy(this->ubStartOffset, this->startOffsetGm, this->offsetCountAlignInt64);
            AscendC::DataCopy(this->ubEndOffset, this->endOffsetGM, this->offsetCountAlignInt64);
            AscendC::DataCopy(this->ubCacheLoc, this->cacheLocGM, this->cacheLocCountAlignInt32);

            int32_t eventIDMTE2TOV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIDMTE2TOV);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIDMTE2TOV);

            AscendC::Cast(this->ubStartOffsetInt32,
                this->ubStartOffset,
                AscendC::RoundMode::CAST_NONE,
                this->offsetCountAlignInt64);
            AscendC::Cast(
                this->ubEndOffsetInt32, this->ubEndOffset, AscendC::RoundMode::CAST_NONE, this->offsetCountAlignInt64);
            this->cacheLength = this->ubEndOffsetInt32 - this->ubStartOffsetInt32;

            for (int32_t i = 0; i < this->rowNum; i++) {
                uint64_t rowIdx = this->rowOffset + i;
                uint64_t reqIdx = this->ubReqPoolIndices.GetValue(rowIdx);
                CopyIn(rowIdx, reqIdx);
                Compute(rowIdx, reqIdx);
            }
        }
    }

private:
    __aicore__ inline void CopyIn(uint64_t rowIdx, uint64_t reqIdx)
    {
        AscendC::LocalTensor<int32_t> tokenPoolLocal = this->inQueue1.template AllocTensor<int32_t>();
        int64_t start = this->ubStartOffset.GetValue(rowIdx);
        AscendC::DataCopy(tokenPoolLocal, tokenPoolGM[reqIdx * this->rowSize + start], this->tokenCountAlignInt32);
        this->inQueue1.EnQue(tokenPoolLocal);
    }

    __aicore__ inline void Compute(uint64_t rowIdx, uint64_t reqIdx)
    {
        AscendC::LocalTensor<int32_t> tokenPoolLocal = this->inQueue1.template DeQue<int32_t>();
        int64_t start = this->ubStartOffset.GetValue(rowIdx);
        int64_t step = this->ubEndOffset.GetValue(rowIdx) - start;

        int64_t idxStart = 0;
        GetCacheIdx(rowIdx, idxStart);
        for (int64_t j = 0; j < step; j++) {
            int32_t cache = this->ubCacheLoc.GetValue(idxStart);
            tokenPoolLocal.SetValue(j, cache);
            idxStart += 1;
        }

        AscendC::DataCopy(tokenPoolGM[reqIdx * this->rowSize + start], tokenPoolLocal, this->tokenCountAlignInt32);
        this->inQueue1.FreeTensor(tokenPoolLocal);
    }

    __aicore__ inline void GetCacheIdx(uint64_t rowIdx, int64_t &result)
    {
        for (int64_t i = 0; i < rowIdx; i++) {
            result += this->cacheLength.GetValue(i);
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::LocalTensor<T> ubReqPoolIndices;
    AscendC::LocalTensor<int64_t> ubStartOffset;
    AscendC::LocalTensor<int64_t> ubEndOffset;
    AscendC::LocalTensor<int32_t> ubStartOffsetInt32;
    AscendC::LocalTensor<int32_t> ubEndOffsetInt32;

    AscendC::LocalTensor<int32_t> cacheLength;
    AscendC::LocalTensor<int32_t> ubCacheLoc;

    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueue1;
    AscendC::GlobalTensor<T> reqPoolIndicesGM;
    AscendC::GlobalTensor<int32_t> tokenPoolGM;
    AscendC::GlobalTensor<int64_t> startOffsetGm;
    AscendC::GlobalTensor<int64_t> endOffsetGM;
    AscendC::GlobalTensor<int32_t> cacheLocGM;

    uint64_t coreId;
    uint32_t batchSize;
    uint64_t vcoreNum;
    uint64_t rowNum;
    uint64_t rowNumNoTail;
    uint64_t tailNum;
    uint64_t tailOffset;
    uint64_t rowOffset;
    uint64_t rowSize;

    uint64_t reqInxBufferCount;
    uint64_t tokenCountAlignInt32;
    uint64_t offsetCountAlignInt64;
    uint64_t cacheLocCountAlignInt32;
};

extern "C" __global__ __aicore__ void cache_loc_assign(GM_ADDR reqPoolIndices, GM_ADDR tokenPool, GM_ADDR startOffset,
    GM_ADDR endOffset, GM_ADDR outCacheLoc, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(AssignCacheTillingData);
    __gm__ AssignCacheTillingData *tempTilingGM = reinterpret_cast<__gm__ AssignCacheTillingData *>(tilingGM);
    if (tempTilingGM->key == 1) {
        CacheLocAssignKernel<int32_t> op;
        op.Init(reqPoolIndices, tokenPool, startOffset, endOffset, outCacheLoc, tempTilingGM);
        if ASCEND_IS_AIV {
            op.Process();
        }
    } else if (tempTilingGM->key == 2) {
        CacheLocAssignKernel<int64_t> op;
        op.Init(reqPoolIndices, tokenPool, startOffset, endOffset, outCacheLoc, tempTilingGM);
        if ASCEND_IS_AIV {
            op.Process();
        }
    }
}

#endif  // SGL_KERNEL_NPU_KERNEL_CACHE_LOC_ASSIGN_H
