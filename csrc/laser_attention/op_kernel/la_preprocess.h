/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */
#ifndef __SRC_OPS_KERNEL_LA_PREPROCESS_H__
#define __SRC_OPS_KERNEL_LA_PREPROCESS_H__

#include <kernel_operator.h>

struct LaPreprocessTiling {
    uint32_t batchSize;
    uint32_t qSeqLen;
    uint32_t kSeqLen;
    uint32_t vSeqLen;
    uint32_t headDim;
    uint32_t headNum;
    uint32_t alignLen;
    uint32_t ubSize;
};

struct AscendLaserAttentionTilingData {
    int32_t batchSize;       // B
    int32_t headNum;         // N
    int32_t seqSize;         // S
    int32_t headDim;         // D
    int32_t coreNumPerGroup; // Y
    int32_t coreGroupNum;    // F

    int32_t qSeqLength;      // qkv不等长
    int32_t kSeqLength;      // qkv不等长
    int32_t vSeqLength;      // qkv不等长
    int32_t maskSeqLength;   // 预留
    float scale;             // 预留
    float keep_prob;         // 预留
    int32_t pre_tokens;      // 预留
    int32_t next_tokens;     // 预留

    bool isTriangle;        // 是否倒三角
    int32_t attenType;       // 0:MHA/1:GQA
    int32_t sparseMode;      // 0:dense/1:sparse
    int32_t headGroupSize;   // N/G
    int32_t windowLen;       // sparse的滑动窗口
    bool isHighPrecision;    // 高性能
};

namespace mmdit_ops {

namespace kernels {

template <typename T, typename DST, int32_t HEAD_DIM = 128, int32_t QUEUE_DEPTH = 1>
class LaPreprocess {
public:
    __aicore__ inline LaPreprocess()
        : blockIdx_(AscendC::GetBlockIdx()), blockDim_(AscendC::GetBlockNum()) {}

    __aicore__ inline void Init(
        GM_ADDR query, GM_ADDR key, GM_ADDR value,
        GM_ADDR outQuery, GM_ADDR outKey, GM_ADDR outValue,
        __gm__ const LaPreprocessTiling *tiling, AscendC::TPipe *pipe)
    {
        batchSize_ = tiling->batchSize;
        headNum_ = tiling->headNum;

        qSeqLen_ = tiling->qSeqLen;
        kSeqLen_ = tiling->kSeqLen;
        vSeqLen_ = tiling->vSeqLen;

        alignLen_ = tiling->alignLen;
        ubSize_ = tiling->ubSize;

        if constexpr (std::is_same_v<T, bfloat16_t>) {
            blockSeqLen_ =
                (ubSize_ - HEAD_DIM * sizeof(T)) /
                (QUEUE_DEPTH * headNum_ * HEAD_DIM) / (sizeof(float) + sizeof(T));
        } else {
            blockSeqLen_ =
                (ubSize_ - HEAD_DIM * sizeof(T)) /
                (QUEUE_DEPTH * headNum_ * HEAD_DIM) / sizeof(T);
        }

        pipe_ = pipe;

        InitBuffers();
        InitGlobal(query, key, value, outQuery, outKey, outValue);
    }

    __aicore__ inline void Process()
    {
        SplitTask(qSeqLen_);
        CopyData(outQueryGm_, queryGm_);

        SplitTask(kSeqLen_);
        CopyData(outKeyGm_, keyGm_);

        SplitTask(vSeqLen_);
        CopyData(outValueGm_, valueGm_);
    }

private:
    __aicore__ inline void InitBuffers()
    {
        bufLen_ = blockSeqLen_ * headNum_ * HEAD_DIM;
        if constexpr (std::is_same_v<T, bfloat16_t>) {
            pipe_->InitBuffer(inQue_, QUEUE_DEPTH, bufLen_ * sizeof(float));
            pipe_->InitBuffer(outQue_, QUEUE_DEPTH, bufLen_ * sizeof(T));
        } else {
            pipe_->InitBuffer(movQueBind_, QUEUE_DEPTH, bufLen_ * sizeof(T));
        }
        pipe_->InitBuffer(zeroBuf_, HEAD_DIM * sizeof(DST));

        zeroTensor_ = zeroBuf_.Get<DST>();
        AscendC::Duplicate<DST>(zeroTensor_, static_cast<DST>(0.0), HEAD_DIM);
    }

    __aicore__ inline void InitGlobal(
        GM_ADDR query, GM_ADDR key, GM_ADDR value,
        GM_ADDR outQuery, GM_ADDR outKey, GM_ADDR outValue)
    {
        queryGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(query));
        keyGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(key));
        valueGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(value));
        outQueryGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DST *>(outQuery));
        outKeyGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DST *>(outKey));
        outValueGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DST *>(outValue));
    }

    __aicore__ inline void SplitTask(uint32_t seqLen)
    {
        uint32_t tailLen = 0;

        curSeqLen_ = seqLen;

        singleSeqLen_ = curSeqLen_ / blockDim_;
        tailLen = curSeqLen_ % blockDim_;
        curSeq_ = singleSeqLen_ * blockIdx_;
        if (blockIdx_ < tailLen) {
            singleSeqLen_ += 1;
            curSeq_ += blockIdx_;
        } else {
            curSeq_ += tailLen;
        }

        curSeqAlignLen_ = AscendC::AlignUp(curSeqLen_, alignLen_);

        // for pad
        curPadSeqLen_ = curSeqAlignLen_ - curSeqLen_;

        singlePadSeqLen_ = curPadSeqLen_ / blockDim_;
        tailLen = curPadSeqLen_ % blockDim_;
        curPadSeq_ = singlePadSeqLen_ * blockIdx_ + curSeqLen_;
        if (blockIdx_ < tailLen) {
            singlePadSeqLen_ += 1;
            curPadSeq_ += blockIdx_;
        } else {
            curPadSeq_ += tailLen;
        }
    }

    __aicore__ inline void CopyIn(const AscendC::GlobalTensor<T>& src, uint32_t seqLen)
    {
        if constexpr (std::is_same_v<T, bfloat16_t>) {
            AscendC::LocalTensor<T> srcLocal = inQue_.template AllocTensor<T>();

            AscendC::DataCopyExtParams inCopyParams{
                1, seqLen * headNum_ * HEAD_DIM * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<T> inPadParams{false, 0, 0, 0};
            AscendC::DataCopyPad(srcLocal[bufLen_], src, inCopyParams, inPadParams);

            inQue_.EnQue(srcLocal);
            AscendC::LocalTensor<T> castLocal = inQue_.template DeQue<T>();
            AscendC::LocalTensor<DST> dstLocal = outQue_.template AllocTensor<DST>();

            AscendC::Cast<float, T>(
                castLocal.template ReinterpretCast<float>(), castLocal[bufLen_],
                AscendC::RoundMode::CAST_NONE, bufLen_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast<DST, float>(
                dstLocal, castLocal.template ReinterpretCast<float>(),
                AscendC::RoundMode::CAST_RINT, bufLen_);

            outQue_.EnQue(dstLocal);
            inQue_.FreeTensor(castLocal);
        } else {
            AscendC::LocalTensor<T> srcLocal = movQueBind_.template AllocTensor<T>();

            AscendC::DataCopyExtParams inCopyParams{
                1, seqLen * headNum_ * HEAD_DIM * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<T> inPadParams{false, 0, 0, 0};
            AscendC::DataCopyPad(srcLocal, src, inCopyParams, inPadParams);

            movQueBind_.EnQue(srcLocal);
        }
    }

    __aicore__ inline void CopyOut(const AscendC::GlobalTensor<DST>& dst, uint32_t seqLen)
    {
        if constexpr (std::is_same_v<T, bfloat16_t>) {
            AscendC::LocalTensor<DST> dstLocal = outQue_.template DeQue<DST>();

            uint32_t dstStride = (curSeqAlignLen_ - 1) * HEAD_DIM * sizeof(DST);

            for (uint32_t i = 0; i < seqLen; ++i) {
                AscendC::DataCopyExtParams outCopyParams{
                    static_cast<uint16_t>(headNum_), HEAD_DIM * sizeof(T), 0, dstStride, 0};
                AscendC::DataCopyPad(dst[i * HEAD_DIM], dstLocal[i * headNum_ * HEAD_DIM], outCopyParams);
            }

            outQue_.FreeTensor(dstLocal);
        } else {
            AscendC::LocalTensor<DST> dstLocal = movQueBind_.template DeQue<DST>();

            uint32_t dstStride = (curSeqAlignLen_ - 1) * HEAD_DIM * sizeof(DST);

            for (uint32_t i = 0; i < seqLen; ++i) {
                AscendC::DataCopyExtParams outCopyParams{
                    static_cast<uint16_t>(headNum_), HEAD_DIM * sizeof(DST), 0, dstStride, 0};
                AscendC::DataCopyPad(dst[i * HEAD_DIM], dstLocal[i * headNum_ * HEAD_DIM], outCopyParams);
            }

            movQueBind_.FreeTensor(dstLocal);
        }
    }

    __aicore__ inline void PadOut(const AscendC::GlobalTensor<DST>& dst, uint32_t seqLen)
    {
        for (uint32_t i = 0; i < headNum_; ++i) {
            for (uint32_t j = 0; j < seqLen; ++j) {
                AscendC::DataCopyExtParams outCopyParams{
                    1, HEAD_DIM * sizeof(DST), 0, 0, 0};
                AscendC::DataCopyPad(
                    dst[j * HEAD_DIM + i * curSeqAlignLen_ * HEAD_DIM], zeroTensor_, outCopyParams);
            }
        }
    }

    __aicore__ inline void CopyData(const AscendC::GlobalTensor<DST>& dst, const AscendC::GlobalTensor<T>& src)
    {
        for (uint32_t i = 0; i < batchSize_; ++i) {
            uint32_t inBatchOffset = i * curSeqLen_ * headNum_ * HEAD_DIM;
            uint32_t outBatchOffset = i * curSeqAlignLen_ * headNum_ * HEAD_DIM;

            if (singlePadSeqLen_ > 0) {
                PadOut(dst[outBatchOffset + curPadSeq_ * HEAD_DIM], singlePadSeqLen_);
            }

            for (uint32_t j = curSeq_; j < curSeq_ + singleSeqLen_; j += blockSeqLen_) {
                uint32_t seqLen =
                    j + blockSeqLen_ > curSeq_ + singleSeqLen_ ?
                    curSeq_ + singleSeqLen_ - j : blockSeqLen_;

                CopyIn(src[inBatchOffset + j * headNum_ * HEAD_DIM], seqLen);

                CopyOut(dst[outBatchOffset + j * HEAD_DIM], seqLen);
            }
        }
    }

    uint32_t blockIdx_;
    uint32_t blockDim_;

    uint32_t batchSize_;
    uint32_t qSeqLen_;
    uint32_t kSeqLen_;
    uint32_t vSeqLen_;

    uint32_t curSeqAlignLen_;

    uint32_t headNum_;
    uint32_t alignLen_;
    uint32_t ubSize_;
    uint32_t blockSeqLen_;

    uint32_t singleSeqLen_;
    uint32_t curSeq_;
    uint32_t curSeqLen_;

    uint32_t singlePadSeqLen_;
    uint32_t curPadSeq_;
    uint32_t curPadSeqLen_;

    uint32_t bufLen_;

    AscendC::TQue<AscendC::TPosition::VECIN, QUEUE_DEPTH> inQue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, QUEUE_DEPTH> outQue_;
    AscendC::TQueBind<AscendC::TPosition::VECIN, AscendC::TPosition::VECOUT, QUEUE_DEPTH> movQueBind_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> zeroBuf_;

    AscendC::LocalTensor<DST> zeroTensor_;

    AscendC::GlobalTensor<T> queryGm_;
    AscendC::GlobalTensor<T> keyGm_;
    AscendC::GlobalTensor<T> valueGm_;
    AscendC::GlobalTensor<DST> outQueryGm_;
    AscendC::GlobalTensor<DST> outKeyGm_;
    AscendC::GlobalTensor<DST> outValueGm_;

    AscendC::TPipe *pipe_ = nullptr;
};

} // namespace kernels

} // namespace mmdit_ops

#endif // __SRC_OPS_KERNEL_LA_PREPROCESS_H__
