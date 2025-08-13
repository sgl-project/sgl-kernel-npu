// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef SGL_KERNEL_NPU_KERNEL_HELLOWORLD_H
#define SGL_KERNEL_NPU_KERNEL_HELLOWORLD_H

/* include file of ascendc */
#include "kernel_operator.h"
#include "../op_host/tiling/assign_cache_tiling.h"

/* tensor num for each queue */
constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t BLK_SIZE = 32;
constexpr uint32_t REPEAT_TIME_8 = 8;

class CacheLocAssignKernel {
public:
    __aicore__ inline CacheLocAssignKernel()
    {}

    __aicore__ inline void Init(GM_ADDR tokenPool, GM_ADDR startOffset, GM_ADDR endOffset, GM_ADDR outCacheLoc,
        GM_ADDR outCacheLocIdx, GM_ADDR workspace, GM_ADDR tilingGM)
    {
        auto tempTilingGM = reinterpret_cast<__gm__ AssignCacheTillingData *>(tilingGM);

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

        this->tokenColAlignInt32 = tempTilingGM->tokenColAlignInt32;
        this->offsetColAlignInt32 = (this->rowNum * sizeof(int32_t) + BLK_SIZE - 1) / BLK_SIZE * BLK_SIZE;
        this->cacheLocAlignIn32 = tempTilingGM->cacheLocAlignIn32;
        this->cacheLocIdxAlignIn32 = tempTilingGM->cacheLocIdxAlignIn32;

        this->tokenCountAlignInt32 = tempTilingGM->tokenCountAlignInt32;
        this->offsetCountAlignInt32 = (this->rowNum + REPEAT_TIME_8 - 1) / REPEAT_TIME_8 * REPEAT_TIME_8;
        this->cacheLocCountAlignIn32 = tempTilingGM->cacheLocCountAlignIn32;
        this->cacheLocIdxCountAlignIn32 = tempTilingGM->cacheLocIdxCountAlignIn32;
        this->rowOffset = this->rowNumNoTail * this->coreId + this->tailOffset;

        this->tokenPoolGM.SetGlobalBuffer(
            (__gm__ int32_t *)tokenPool + this->rowOffset * this->rowSize, this->blockLength);
        this->startOffsetGm.SetGlobalBuffer((__gm__ int32_t *)startOffset + this->rowOffset, this->rowNum);
        this->endOffsetGM.SetGlobalBuffer((__gm__ int32_t *)endOffset + this->rowOffset, this->rowNum);
        this->cacheLocGM.SetGlobalBuffer((__gm__ int32_t *)outCacheLoc, tempTilingGM->cacheLocSize);
        this->cacheLocIdxGM.SetGlobalBuffer((__gm__ int32_t *)outCacheLocIdx, tempTilingGM->cacheLocIdxSize);

        // 32 blk size
        this->syncGm.SetGlobalBuffer((__gm__ int32_t *)(workspace), tempTilingGM->workspaceSize);
        pipe.InitBuffer(vecIn, 1, this->vcoreNum * sizeof(int32_t));

        AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuff1, tmpBuff2, tmpBuff3, tmpBuff4;
        pipe.InitBuffer(tmpBuff1, this->offsetColAlignInt32);
        pipe.InitBuffer(tmpBuff2, this->offsetColAlignInt32);
        pipe.InitBuffer(tmpBuff3, this->cacheLocAlignIn32);
        pipe.InitBuffer(tmpBuff4, this->cacheLocIdxAlignIn32);
        this->ubStartOffset = tmpBuff1.Get<int32_t>();
        this->ubEndOffset = tmpBuff2.Get<int32_t>();
        this->ubCacheLoc = tmpBuff3.Get<int32_t>();
        this->ubCacheLocIdx = tmpBuff4.Get<int32_t>();

        pipe.InitBuffer(inQueue1, 1, this->tokenColAlignInt32);
    }

    __aicore__ inline void Process()
    {
        AscendC::DataCopy(this->ubStartOffset, this->startOffsetGm, this->offsetCountAlignInt32);
        AscendC::DataCopy(this->ubEndOffset, this->endOffsetGM, this->offsetCountAlignInt32);
        AscendC::DataCopy(this->ubCacheLoc, this->cacheLocGM, this->cacheLocCountAlignIn32);
        AscendC::DataCopy(this->ubCacheLocIdx, this->cacheLocIdxGM, this->cacheLocIdxCountAlignIn32);
        CopyIn();
        Compute();
    }

private:
    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<int32_t> tokenPoolLocal = inQueue1.AllocTensor<int32_t>();
        AscendC::DataCopy(tokenPoolLocal, tokenPoolGM, this->tokenCountAlignInt32);
        inQueue1.EnQue(tokenPoolLocal);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<int32_t> tokenPoolLocal = inQueue1.DeQue<int32_t>();
        for (int32_t i = 0; i < this->rowNum; i++) {
            int32_t start = this->ubStartOffset.GetValue(i);
            int32_t end = this->ubEndOffset.GetValue(i);
            int32_t idxStart = this->ubCacheLocIdx.GetValue(this->rowOffset + i);
            for (int32_t j = start; j < end; j++) {
                int32_t cache = this->ubCacheLoc.GetValue(idxStart);
                tokenPoolLocal.SetValue(j + i * this->rowSize, cache);
                idxStart += 1;
            }
        }

        if (this->coreId == 0) {
            auto syncBuf = this->vecIn.AllocTensor<int32_t>();
            AscendC::DataCopy(tokenPoolGM, tokenPoolLocal, this->tokenCountAlignInt32);
            AscendC::IBSet(this->syncGm, syncBuf, 0, 0);
            this->vecIn.FreeTensor(syncBuf);
        } else {
            auto syncBuf = vecIn.AllocTensor<int32_t>();
            AscendC::IBWait(this->syncGm, syncBuf, this->coreId - 1, 0);
            AscendC::DataCopy(tokenPoolGM, tokenPoolLocal, this->tokenCountAlignInt32);
            if (this->coreId != this->vcoreNum - 1) {
                AscendC::IBSet(this->syncGm, syncBuf, this->coreId, 0);
            }
            this->vecIn.FreeTensor(syncBuf);
        }
        inQueue1.FreeTensor(tokenPoolLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::LocalTensor<int32_t> ubStartOffset;
    AscendC::LocalTensor<int32_t> ubEndOffset;
    AscendC::LocalTensor<int32_t> ubCacheLoc;
    AscendC::LocalTensor<int32_t> ubCacheLocIdx;

    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueue1, vecIn;
    AscendC::GlobalTensor<int32_t> syncGm;
    AscendC::GlobalTensor<int32_t> tokenPoolGM;
    AscendC::GlobalTensor<int32_t> startOffsetGm;
    AscendC::GlobalTensor<int32_t> endOffsetGM;
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
    uint64_t offsetColAlignInt32;
    uint64_t cacheLocAlignIn32;
    uint64_t cacheLocIdxAlignIn32;

    uint64_t tokenCountAlignInt32;
    uint64_t offsetCountAlignInt32;
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

#endif  // SGL_KERNEL_NPU_KERNEL_HELLOWORLD_H
