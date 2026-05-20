#ifndef SGLANG_UPDATE_OE_TOKEN_TABLE_H
#define SGLANG_UPDATE_OE_TOKEN_TABLE_H

#include "kernel_operator.h"
#include "../op_host/update_oe_token_table_tiling.h"

using namespace AscendC;

class UpdateOeTokenTable {
public:
    __aicore__ inline UpdateOeTokenTable() = default;
    __aicore__ inline UpdateOeTokenTable(
        TPipe *pipe, __gm__ const sglang::npu_kernel::UpdateOeTokenTableTilingData *tiling)
        : pipe_(pipe), tl_(tiling)
    {
    }

    __aicore__ inline void Init(GM_ADDR tokens, GM_ADDR req_lens, GM_ADDR row_indices,
                                GM_ADDR column_starts, GM_ADDR ignore_tokens,
                                GM_ADDR oe_token_table, GM_ADDR workspace)
    {
        (void)workspace;
        blockIdx_ = GetBlockIdx();
        blockFactor_ = tl_->blockFactor;
        batchSize_ = tl_->batchSize;
        maxContextLen_ = tl_->maxContextLen;
        ignoreTokenNum_ = tl_->ignoreTokenNum;

        tokensGm_.SetGlobalBuffer((__gm__ int32_t *)tokens);
        oeTokenTableGm_.SetGlobalBuffer((__gm__ int32_t *)oe_token_table);
        reqLensGm_.SetGlobalBuffer((__gm__ int32_t *)req_lens, batchSize_);
        ignoreTokensGm_.SetGlobalBuffer((__gm__ int32_t *)ignore_tokens, ignoreTokenNum_);
        rowIndicesGm_.SetGlobalBuffer((__gm__ int64_t *)row_indices + blockIdx_ * blockFactor_);
        columnStartsGm_.SetGlobalBuffer((__gm__ int32_t *)column_starts + blockIdx_ * blockFactor_);

        pipe_->InitBuffer(isIgnoreQue_, 1, ignoreTokenNum_ * sizeof(int32_t));
        pipe_->InitBuffer(reqLensQue_, 1, batchSize_ * sizeof(int32_t));
        pipe_->InitBuffer(tokenQue_, DOUBLE_BUFF, tl_->ubFactor * sizeof(int32_t));
        pipe_->InitBuffer(oeTokenTableQue_, DOUBLE_BUFF, tl_->ubFactor * sizeof(int32_t));
        pipe_->InitBuffer(maskQue_, tl_->ubFactor * sizeof(uint8_t));
        pipe_->InitBuffer(minusQue_, tl_->ubFactor * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        int64_t currBlockFactor = blockFactor_;
        if (blockIdx_ == tl_->usedCoreNum - 1) {
            currBlockFactor = tl_->tailBlockFactor;
        }

        int32_t srcOffset = 0;
        LoadReqLens();
        LoadIgnore();
        auto reqLensTensor = reqLensQue_.DeQue<int32_t>();
        auto isIgnoreTensor = isIgnoreQue_.DeQue<int32_t>();
        TEventID eventIdMTE2ToS = GetTPipePtr()->FetchEventID(HardEvent::MTE2_S);
        SetFlag<HardEvent::MTE2_S>(eventIdMTE2ToS);
        WaitFlag<HardEvent::MTE2_S>(eventIdMTE2ToS);

        for (uint32_t i = 0; i < blockIdx_ * blockFactor_; i++) {
            srcOffset += reqLensTensor.GetValue(i);
        }

        for (int64_t idx = 0; idx < currBlockFactor; idx++) {
            auto currReqLen = reqLensGm_.GetValue(blockIdx_ * blockFactor_ + idx);
            auto rowIndice = rowIndicesGm_.GetValue(idx);
            auto colIndice = columnStartsGm_.GetValue(idx);
            auto dstOffset = rowIndice * tl_->maxContextLen + colIndice;
            int64_t ubLoopCnt = CeilDiv(currReqLen, tl_->ubFactor);
            int64_t curCopyLen = tl_->ubFactor;
            if (currReqLen == 0) {
                continue;
            }
            for (int64_t loopIdx = 0; loopIdx < ubLoopCnt; loopIdx++) {
                if (loopIdx == ubLoopCnt - 1 && currReqLen % tl_->ubFactor != 0) {
                    curCopyLen = currReqLen % tl_->ubFactor;
                }
                CopyIn(srcOffset, curCopyLen);
                IsIgnoreToken(isIgnoreTensor, curCopyLen);
                CopyOut(dstOffset, curCopyLen);
                srcOffset += curCopyLen;
                dstOffset += curCopyLen;
            }
        }
        reqLensQue_.FreeTensor(reqLensTensor);
        isIgnoreQue_.FreeTensor(isIgnoreTensor);
    }

private:
    __aicore__ inline void LoadReqLens()
    {
        LocalTensor<int32_t> reqLensTensor = reqLensQue_.AllocTensor<int32_t>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(batchSize_ * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
        DataCopyPad(reqLensTensor, reqLensGm_[0], copyParams, padParams);
        reqLensQue_.EnQue(reqLensTensor);
    }

    __aicore__ inline void LoadIgnore()
    {
        LocalTensor<int32_t> ignoreTensor = isIgnoreQue_.AllocTensor<int32_t>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(ignoreTokenNum_ * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
        DataCopyPad(ignoreTensor, ignoreTokensGm_[0], copyParams, padParams);
        isIgnoreQue_.EnQue(ignoreTensor);
    }

    __aicore__ inline void CopyIn(int64_t srcOffset, uint32_t copyLen)
    {
        LocalTensor<int32_t> tokenTensor = tokenQue_.AllocTensor<int32_t>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(copyLen * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
        DataCopyPad(tokenTensor, tokensGm_[srcOffset], copyParams, padParams);
        tokenQue_.EnQue(tokenTensor);
    }

    __aicore__ inline void IsIgnoreToken(const LocalTensor<int32_t> &isIgnoreTensor, uint32_t count)
    {
        auto tokenTensor = tokenQue_.DeQue<int32_t>();
        auto alignCnt = CeilDiv(count, BLOCK_SIZE) * BLOCK_SIZE;
        auto minusTensor = minusQue_.Get<int32_t>();
        Duplicate(minusTensor, static_cast<int32_t>(-1), count);
        PipeBarrier<PIPE_V>();
        for (int64_t idx = 0; idx < ignoreTokenNum_; idx++) {
            int32_t ignoreVal = isIgnoreTensor.GetValue(idx);
            auto maskTensor = maskQue_.Get<uint8_t>();
            CompareScalar(maskTensor, tokenTensor, ignoreVal, CMPMODE::EQ,
                          CeilAlign(count, ONE_REPEAT_ELE_NUM));
            PipeBarrier<PIPE_V>();
            Select(tokenTensor.ReinterpretCast<float>(), maskTensor,
                   minusTensor.ReinterpretCast<float>(), tokenTensor.ReinterpretCast<float>(),
                   SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
            PipeBarrier<PIPE_V>();
        }
        auto outTokenTensor = oeTokenTableQue_.AllocTensor<int32_t>();
        DataCopy(outTokenTensor, tokenTensor, alignCnt);
        tokenQue_.FreeTensor(tokenTensor);
        oeTokenTableQue_.EnQue(outTokenTensor);
    }

    __aicore__ inline void CopyOut(int64_t dstOffset, uint32_t copyLen)
    {
        LocalTensor<int32_t> outTokenTensor = oeTokenTableQue_.DeQue<int32_t>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(copyLen * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(oeTokenTableGm_[dstOffset], outTokenTensor, copyParams);
        oeTokenTableQue_.FreeTensor(outTokenTensor);
    }

    __aicore__ inline int64_t CeilDiv(int64_t x, int64_t align)
    {
        if (align == 0) {
            return x;
        }
        return (x + align - 1) / align;
    }

    __aicore__ inline int64_t CeilAlign(int64_t x, int64_t align)
    {
        return CeilDiv(x, align) * align;
    }

private:
    constexpr static int64_t BLOCK_SIZE = 32;
    constexpr static int64_t DOUBLE_BUFF = 2;
    constexpr static int64_t ONE_REPEAT_ELE_NUM = 256 / sizeof(int32_t);

    TPipe *pipe_ = nullptr;
    __gm__ const sglang::npu_kernel::UpdateOeTokenTableTilingData *tl_ = nullptr;
    int64_t blockIdx_ = 0;
    int64_t blockFactor_ = 0;
    int64_t batchSize_ = 0;
    int64_t maxContextLen_ = 0;
    int64_t ignoreTokenNum_ = 0;

    AscendC::GlobalTensor<int32_t> tokensGm_;
    AscendC::GlobalTensor<int32_t> oeTokenTableGm_;
    AscendC::GlobalTensor<int32_t> reqLensGm_;
    AscendC::GlobalTensor<int64_t> rowIndicesGm_;
    AscendC::GlobalTensor<int32_t> columnStartsGm_;
    AscendC::GlobalTensor<int32_t> ignoreTokensGm_;
    AscendC::TQue<QuePosition::VECIN, 1> tokenQue_, reqLensQue_, isIgnoreQue_;
    AscendC::TQue<QuePosition::VECOUT, 1> oeTokenTableQue_;
    AscendC::TBuf<QuePosition::VECCALC> maskQue_, minusQue_;
};

#endif  // SGLANG_UPDATE_OE_TOKEN_TABLE_H
