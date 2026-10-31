// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "kernel_operator.h"
#include "cache_loc_tiling.h"

/* tensor num for each queue */
constexpr int32_t BUFFER_NUM = 2;

constexpr uint32_t ASSIGN_TO_POOL = 0;
constexpr uint32_t RETRIEVE_FROM_POOL = 1;

template <typename T>
class CacheLocAssignKernel
{
public:
    __aicore__ inline CacheLocAssignKernel() {}

    __aicore__ inline void Init(GM_ADDR reqPoolIndices, GM_ADDR tokenPool, GM_ADDR startOffset, GM_ADDR endOffset,
                                GM_ADDR outCacheLoc, const optiling::SglCacheLocTilingData &tiling)
    {
        this->coreId = AscendC::GetBlockIdx();
        this->batchSize = tiling.batchSize;
        this->vcoreNum = tiling.vcoreNum;
        this->rowNumNoTail = tiling.rowNumNoTail;
        this->tailNum = tiling.tailNum;
        if (this->coreId < this->tailNum) {
            this->rowNum = this->rowNumNoTail + 1;
            this->tailOffset = this->coreId;
        } else {
            this->rowNum = this->rowNumNoTail;
            this->tailOffset = this->tailNum;
        }
        this->rowSize = tiling.rowSize;
        this->reqInxBufferCount = tiling.reqInxBufferCount;
        this->tokenCountAlignInt32 = tiling.tokenCountAlignInt32;
        this->offsetCountAlignInt64 = tiling.offsetCountAlignInt64;
        this->cacheLocCountAlignInt32 = tiling.cacheLocCountAlignInt32;
        this->cacheLocSize = tiling.cacheLocSize;

        this->rowOffset = this->rowNumNoTail * this->coreId + this->tailOffset;
        this->reqPoolIndicesGM.SetGlobalBuffer((__gm__ T *)reqPoolIndices, this->batchSize);
        this->tokenPoolGM.SetGlobalBuffer((__gm__ int32_t *)tokenPool, tiling.poolSize * this->rowSize);
        this->startOffsetGm.SetGlobalBuffer((__gm__ int64_t *)startOffset, this->batchSize);
        this->endOffsetGM.SetGlobalBuffer((__gm__ int64_t *)endOffset, this->batchSize);
        this->cacheLocGM.SetGlobalBuffer((__gm__ int32_t *)outCacheLoc, this->cacheLocSize);

        AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuff1, tmpBuff2, tmpBuff3, tmpBuff4, tmpBuff5, tmpBuff6, tmpBuff7;
        this->pipe.InitBuffer(tmpBuff1, tiling.reqInxBufferSize);
        this->pipe.InitBuffer(tmpBuff2, tiling.offsetColAlignInt64);
        this->pipe.InitBuffer(tmpBuff3, tiling.offsetColAlignInt64);
        uint64_t offsetBufferInt32 = this->batchSize * sizeof(int32_t);
        this->pipe.InitBuffer(tmpBuff4, offsetBufferInt32);
        this->pipe.InitBuffer(tmpBuff5, offsetBufferInt32);
        this->pipe.InitBuffer(tmpBuff6, offsetBufferInt32);
        this->pipe.InitBuffer(tmpBuff7, tiling.cacheLocAlignInt32);

        this->ubReqPoolIndices = tmpBuff1.Get<T>();
        this->ubStartOffset = tmpBuff2.Get<int64_t>();
        this->ubEndOffset = tmpBuff3.Get<int64_t>();
        this->ubStartOffsetInt32 = tmpBuff4.Get<int32_t>();
        this->ubEndOffsetInt32 = tmpBuff5.Get<int32_t>();

        this->ubCacheLength = tmpBuff6.Get<int32_t>();
        this->ubCacheLoc = tmpBuff7.Get<int32_t>();

        this->pipe.InitBuffer(this->inQueue1, BUFFER_NUM, tiling.tokenColAlignInt32);
    }

    __aicore__ inline void ProcessForTokenPoolAssign()
    {
        if (this->rowNum > 0) {
            PreProcess();
            for (int32_t i = 0; i < this->rowNum; i++) {
                uint64_t rowIdx = this->rowOffset + i;
                // Empty intervals do not read or write the token pool.
                if (this->ubStartOffset.GetValue(rowIdx) == this->ubEndOffset.GetValue(rowIdx)) {
                    continue;
                }
                uint64_t reqIdx = this->ubReqPoolIndices.GetValue(rowIdx);
                CopyIn(rowIdx, reqIdx);
                ComputeForTokenPoolAssign(rowIdx, reqIdx);
            }
        }
    }

    __aicore__ inline void ProcessForCacheUpdate()
    {
        if (this->rowNum > 0) {
            PreProcess();
            for (int32_t i = 0; i < this->rowNum; i++) {
                uint64_t rowIdx = this->rowOffset + i;
                uint64_t reqIdx = this->ubReqPoolIndices.GetValue(rowIdx);
                CopyIn(rowIdx, reqIdx);
                ComputeForCacheUpdate(rowIdx);
            }

            int32_t eventIDVTOMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIDVTOMTE3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIDVTOMTE3);

            uint32_t cacheBytes = static_cast<uint32_t>(this->cacheLocSize * sizeof(int32_t));
            AscendC::DataCopyExtParams copyParams{1, cacheBytes, 0, 0, 0};
            AscendC::DataCopyPad(this->cacheLocGM, this->ubCacheLoc, copyParams);
        }
    }

private:
    __aicore__ inline void PreProcess()
    {
        AscendC::DataCopy(this->ubReqPoolIndices, this->reqPoolIndicesGM, this->reqInxBufferCount);
        AscendC::DataCopy(this->ubStartOffset, this->startOffsetGm, this->offsetCountAlignInt64);
        AscendC::DataCopy(this->ubEndOffset, this->endOffsetGM, this->offsetCountAlignInt64);
        AscendC::DataCopy(this->ubCacheLoc, this->cacheLocGM, this->cacheLocCountAlignInt32);

        int32_t eventIDMTE2TOV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIDMTE2TOV);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIDMTE2TOV);

        AscendC::Cast(this->ubStartOffsetInt32, this->ubStartOffset, AscendC::RoundMode::CAST_NONE,
                      this->offsetCountAlignInt64);
        AscendC::Cast(this->ubEndOffsetInt32, this->ubEndOffset, AscendC::RoundMode::CAST_NONE,
                      this->offsetCountAlignInt64);
        this->ubCacheLength = this->ubEndOffsetInt32 - this->ubStartOffsetInt32;
    }

    __aicore__ inline void CopyIn(uint64_t rowIdx, uint64_t reqIdx)
    {
        AscendC::LocalTensor<int32_t> tokenPoolLocal = this->inQueue1.template AllocTensor<int32_t>();
        int64_t start = this->ubStartOffset.GetValue(rowIdx);
        AscendC::DataCopy(tokenPoolLocal, tokenPoolGM[reqIdx * this->rowSize + start], this->tokenCountAlignInt32);
        this->inQueue1.EnQue(tokenPoolLocal);
    }

    __aicore__ inline void ComputeForTokenPoolAssign(uint64_t rowIdx, uint64_t reqIdx)
    {
        AscendC::LocalTensor<int32_t> tokenPoolLocal = this->inQueue1.template DeQue<int32_t>();
        int64_t start = this->ubStartOffset.GetValue(rowIdx);
        int64_t step = this->ubEndOffset.GetValue(rowIdx) - start;

        GetCacheIdx(rowIdx, this->lastRowIdx, this->cacheIdxStart);
        for (int64_t j = 0; j < step; j++) {
            int32_t cache = this->ubCacheLoc.GetValue(this->cacheIdxStart + j);
            tokenPoolLocal.SetValue(j, cache);
        }

        uint32_t tokenBytes = static_cast<uint32_t>(optiling::MAX_STEP * sizeof(int32_t));
        AscendC::DataCopyExtParams copyParams{1, tokenBytes, 0, 0, 0};
        AscendC::DataCopyPad(tokenPoolGM[reqIdx * this->rowSize + start], tokenPoolLocal, copyParams);

        this->inQueue1.FreeTensor(tokenPoolLocal);
    }

    __aicore__ inline void ComputeForCacheUpdate(uint64_t rowIdx)
    {
        AscendC::LocalTensor<int32_t> tokenPoolLocal = this->inQueue1.template DeQue<int32_t>();
        int64_t start = this->ubStartOffset.GetValue(rowIdx);
        int64_t step = this->ubEndOffset.GetValue(rowIdx) - start;

        GetCacheIdx(rowIdx, this->lastRowIdx, this->cacheIdxStart);
        for (int64_t j = 0; j < step; j++) {
            int32_t tokenPosition = tokenPoolLocal.GetValue(j);
            this->ubCacheLoc.SetValue(this->cacheIdxStart + j, tokenPosition);
        }

        this->inQueue1.FreeTensor(tokenPoolLocal);
    }

    __aicore__ inline void GetCacheIdx(uint64_t rowIdx, uint64_t &lastRowIdx, int64_t &cacheIdxStart)
    {
        for (int64_t i = lastRowIdx; i < rowIdx; i++) {
            cacheIdxStart += this->ubCacheLength.GetValue(i);
        }
        lastRowIdx = rowIdx;
    }

private:
    AscendC::TPipe pipe;
    AscendC::LocalTensor<T> ubReqPoolIndices;
    AscendC::LocalTensor<int64_t> ubStartOffset;
    AscendC::LocalTensor<int64_t> ubEndOffset;
    AscendC::LocalTensor<int32_t> ubStartOffsetInt32;
    AscendC::LocalTensor<int32_t> ubEndOffsetInt32;

    AscendC::LocalTensor<int32_t> ubCacheLength;
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
    uint64_t cacheLocSize;

    int64_t cacheIdxStart{0};
    uint64_t lastRowIdx{0};

    uint64_t reqInxBufferCount;
    uint64_t tokenCountAlignInt32;
    uint64_t offsetCountAlignInt64;
    uint64_t cacheLocCountAlignInt32;
};

extern "C" __global__ __aicore__ void sgl_cache_loc_assign(GM_ADDR req_indices, GM_ADDR token_pool,
                                                           GM_ADDR start_offset, GM_ADDR end_offset,
                                                           GM_ADDR out_cache_loc, GM_ADDR token_pool_ref,
                                                           GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(optiling::SglCacheLocTilingData);
    GET_TILING_DATA(data, tiling);
    if (data.key == 1) {
        CacheLocAssignKernel<int32_t> op;
        op.Init(req_indices, token_pool_ref, start_offset, end_offset, out_cache_loc, data);
        if ASCEND_IS_AIV {
            op.ProcessForTokenPoolAssign();
        }
    } else if (data.key == 2) {
        CacheLocAssignKernel<int64_t> op;
        op.Init(req_indices, token_pool_ref, start_offset, end_offset, out_cache_loc, data);
        if ASCEND_IS_AIV {
            op.ProcessForTokenPoolAssign();
        }
    }
}


extern "C" __global__ __aicore__ void sgl_cache_loc_update(GM_ADDR req_indices, GM_ADDR token_pool,
                                                           GM_ADDR start_offset, GM_ADDR end_offset,
                                                           GM_ADDR out_cache_loc, GM_ADDR out_cache_loc_ref,
                                                           GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(optiling::SglCacheLocTilingData);
    GET_TILING_DATA(data, tiling);
    if (data.key == 1) {
        CacheLocAssignKernel<int32_t> op;
        op.Init(req_indices, token_pool, start_offset, end_offset, out_cache_loc_ref, data);
        if ASCEND_IS_AIV {
            op.ProcessForCacheUpdate();
        }
    } else if (data.key == 2) {
        CacheLocAssignKernel<int64_t> op;
        op.Init(req_indices, token_pool, start_offset, end_offset, out_cache_loc_ref, data);
        if ASCEND_IS_AIV {
            op.ProcessForCacheUpdate();
        }
    }
}
