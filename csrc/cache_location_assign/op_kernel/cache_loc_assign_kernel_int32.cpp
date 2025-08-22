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
#include "../op_host/tiling/assign_cache_tiling.h"

/* tensor num for each queue */
constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t BLK_SIZE_ALIN_FOR_INT64 = 4;
constexpr int32_t DEFAULT_SYNCALL_NEED_SIZE = 8;

class CacheLocAssignKernel {
public:
    __aicore__ inline CacheLocAssignKernel()
    {}

    __aicore__ inline void Init(GM_ADDR tokenPool, GM_ADDR startOffset, GM_ADDR endOffset, GM_ADDR outCacheLoc,
        GM_ADDR outCacheLocIdx, GM_ADDR workspace, GM_ADDR tilingGM)
    {
        __gm__ AssignCacheTillingData *tempTilingGM = reinterpret_cast<__gm__ AssignCacheTillingData *>(tilingGM);
        this->coreId = AscendC::GetBlockIdx();
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
        this->blockLength = this->rowNum * this->rowSize;

        this->tokenCountAlignInt32 = tempTilingGM->tokenCountAlignInt32;
        this->tokenColAlignInt32 = tempTilingGM->tokenColAlignInt32;

        this->cacheLocCountAlignIn32 = tempTilingGM->cacheLocCountAlignIn32;
        this->cacheLocAlignIn32 = tempTilingGM->cacheLocAlignIn32;

        this->cacheLocIdxCountAlignIn32 = tempTilingGM->cacheLocIdxCountAlignIn32;
        this->cacheLocIdxAlignIn32 = tempTilingGM->cacheLocIdxAlignIn32;

        this->offsetCountAlignInt64 =
            (this->rowNum + BLK_SIZE_ALIN_FOR_INT64 - 1) / BLK_SIZE_ALIN_FOR_INT64 * BLK_SIZE_ALIN_FOR_INT64;
        this->offsetColAlignInt64 = this->offsetCountAlignInt64 * sizeof(int64_t);

        this->rowOffset = this->rowNumNoTail * this->coreId + this->tailOffset;

        this->tokenPoolGM.SetGlobalBuffer(
            (__gm__ int32_t *)tokenPool + this->rowOffset * this->rowSize, this->blockLength);
        this->startOffsetGm.SetGlobalBuffer((__gm__ int64_t *)startOffset + this->rowOffset, this->rowNum);
        this->endOffsetGM.SetGlobalBuffer((__gm__ int64_t *)endOffset + this->rowOffset, this->rowNum);
        this->cacheLocGM.SetGlobalBuffer((__gm__ int32_t *)outCacheLoc, tempTilingGM->cacheLocSize);
        this->cacheLocIdxGM.SetGlobalBuffer((__gm__ int32_t *)outCacheLocIdx, tempTilingGM->cacheLocIdxSize);

        AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuff1, tmpBuff2, tmpBuff3, tmpBuff4;
        this->pipe.InitBuffer(tmpBuff1, this->offsetColAlignInt64);
        this->pipe.InitBuffer(tmpBuff2, this->offsetColAlignInt64);
        this->pipe.InitBuffer(tmpBuff3, this->cacheLocAlignIn32);
        this->pipe.InitBuffer(tmpBuff4, this->cacheLocIdxAlignIn32);
        this->ubStartOffset = tmpBuff1.Get<int64_t>();
        this->ubEndOffset = tmpBuff2.Get<int64_t>();
        this->ubCacheLoc = tmpBuff3.Get<int32_t>();
        this->ubCacheLocIdx = tmpBuff4.Get<int32_t>();

        this->pipe.InitBuffer(this->inQueue1, BUFFER_NUM, this->tokenColAlignInt32);
    }

    __aicore__ inline void Process()
    {
        if (this->rowNum > 0) {
            AscendC::DataCopy(this->ubStartOffset, this->startOffsetGm, this->offsetCountAlignInt64);
            AscendC::DataCopy(this->ubEndOffset, this->endOffsetGM, this->offsetCountAlignInt64);
            AscendC::DataCopy(this->ubCacheLoc, this->cacheLocGM, this->cacheLocCountAlignIn32);
            AscendC::DataCopy(this->ubCacheLocIdx, this->cacheLocIdxGM, this->cacheLocIdxCountAlignIn32);
            for (uint64_t i = 0; i < this->rowNum; i++) {
                CopyIn(i);
                Compute(i);
            }
        }
    }

private:
    __aicore__ inline void CopyIn(uint64_t rowIdx)
    {
        AscendC::LocalTensor<int32_t> tokenPoolLocal = this->inQueue1.AllocTensor<int32_t>();
        int64_t start = this->ubStartOffset.GetValue(rowIdx);
        AscendC::DataCopy(tokenPoolLocal, tokenPoolGM[rowIdx * this->rowSize + start], this->tokenCountAlignInt32);
        this->inQueue1.EnQue(tokenPoolLocal);
    }

    __aicore__ inline void Compute(uint64_t rowIdx)
    {
        AscendC::LocalTensor<int32_t> tokenPoolLocal = this->inQueue1.DeQue<int32_t>();
        int64_t start = this->ubStartOffset.GetValue(rowIdx);
        int64_t step = this->ubEndOffset.GetValue(rowIdx) - start;
        if (step > this->tokenCountAlignInt32) {
            // Clamp step to prevent buffer overflow. This may cause silent data truncation.
            step = this->tokenCountAlignInt32;
        }

        int32_t idxStart = this->ubCacheLocIdx.GetValue(this->rowOffset + rowIdx);
        for (int64_t j = 0; j < step; j++) {
            int32_t cache = this->ubCacheLoc.GetValue(idxStart);
            tokenPoolLocal.SetValue(j, cache);
            idxStart += 1;
        }

        AscendC::DataCopy(tokenPoolGM[rowIdx * this->rowSize + start], tokenPoolLocal, this->tokenCountAlignInt32);
        this->inQueue1.FreeTensor(tokenPoolLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::LocalTensor<int64_t> ubStartOffset;
    AscendC::LocalTensor<int64_t> ubEndOffset;
    AscendC::LocalTensor<int32_t> ubCacheLoc;
    AscendC::LocalTensor<int32_t> ubCacheLocIdx;

    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueue1;
    AscendC::GlobalTensor<int32_t> tokenPoolGM;
    AscendC::GlobalTensor<int64_t> startOffsetGm;
    AscendC::GlobalTensor<int64_t> endOffsetGM;
    AscendC::GlobalTensor<int32_t> cacheLocGM;
    AscendC::GlobalTensor<int32_t> cacheLocIdxGM;
    AscendC::GlobalTensor<int32_t> resultGm;

    uint64_t coreId;
    uint64_t vcoreNum;
    uint64_t rowNum;
    uint64_t rowNumNoTail;
    uint64_t tailNum;
    uint64_t tailOffset;
    uint64_t rowOffset;

    uint64_t blockLength;
    uint64_t rowSize;
    uint64_t tokenColAlignInt32;
    uint64_t offsetColAlignInt64;
    uint64_t cacheLocAlignIn32;
    uint64_t cacheLocIdxAlignIn32;

    uint64_t tokenCountAlignInt32;
    uint64_t offsetCountAlignInt64;
    uint64_t cacheLocCountAlignIn32;
    uint64_t cacheLocIdxCountAlignIn32;
};

extern "C" __global__ __aicore__ void cache_loc_assign(GM_ADDR tokenPool, GM_ADDR startOffset, GM_ADDR endOffset,
    GM_ADDR outCacheLoc, GM_ADDR outCacheLocIdx, GM_ADDR workspace, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(AssignCacheTillingData);
    CacheLocAssignKernel op;
    op.Init(tokenPool, startOffset, endOffset, outCacheLoc, outCacheLocIdx, workspace, tilingGM);
    if ASCEND_IS_AIV {
        op.Process();
    }
}

#endif  // SGL_KERNEL_NPU_KERNEL_CACHE_LOC_ASSIGN_H
