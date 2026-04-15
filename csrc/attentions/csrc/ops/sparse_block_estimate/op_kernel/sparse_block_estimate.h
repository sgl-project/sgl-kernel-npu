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

#ifndef __SPARSEBLOCKESTIMATE_H__
#define __SPARSEBLOCKESTIMATE_H__

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"

#define DUMP_TENSOR(mmResUb, m, n)                                                          \
    do {                                                                                    \
        printf("LINE %d:\n", __LINE__);                                                     \
        uint32_t array_##__LINE__[] = {static_cast<uint32_t>(m), static_cast<uint32_t>(n)}; \
        AscendC::ShapeInfo shapeInfo_##__LINE__(2, array_##__LINE__);                       \
        AscendC::DumpTensor(mmResUb, 2, (m) * (n), shapeInfo_##__LINE__);                   \
    } while (0)

struct SPFAEstParam {
    uint32_t bIdx;
    uint32_t batchNOffset;
    uint32_t sOuterLoopIdx;
    uint32_t sInnerLoopIdx;
    uint32_t singleProcessSOuterSize;
    uint32_t sOuterOffset;
    uint32_t mm1SingleCoreN;
    uint32_t actualSeqLengthPerBatch;
    uint32_t actualSeqLengthKVPerBatch;
    uint32_t sInnerLoopTimes;
    uint32_t singleProcessSInnerBmmTail;
    uint64_t tensorAOffset;
    uint64_t tensorBOffset;
    uint32_t gmPingpong;
};

// for TILING_KEY start
enum class INPUT_LAYOUT {
    BNSD = 0,
    BSH,
    TND,
};

template <INPUT_LAYOUT L, typename T, const bool F = false, typename... Args>
struct INVOKE_TYPE {
    using inputType = T;
    static constexpr INPUT_LAYOUT layout = L;
    static constexpr bool causal = F;
};

using namespace AscendC;

template <typename PFAT>
class SparseBlockEstimate {
public:
    using Q_T = typename PFAT::inputType;
    using KV_T = typename PFAT::inputType;
    using T = half;
    using MM_OUT_T = float;

    static constexpr INPUT_LAYOUT LAYOUT_T = PFAT::layout;
    static constexpr bool CAUSAL = PFAT::causal;
    static constexpr uint32_t MAX_QK_LEN = 128 * 1024;
    static constexpr float BOOL_ATTEN_MASK_SCALAR_VALUE = -1000000000000.0f;
    static constexpr int32_t PFA_PARAMS_QUEUE_CAPBABILITY = 4;

    static constexpr uint32_t BUFFER_SIZE_BYTE_1K = 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_2K = 2048;
    static constexpr uint32_t BUFFER_SIZE_BYTE_4K = 4096;
    static constexpr uint32_t BUFFER_SIZE_BYTE_8K = 8192;
    static constexpr uint32_t BUFFER_SIZE_BYTE_16K = 16384;
    static constexpr uint32_t BUFFER_SIZE_BYTE_32K = 32768;
    static constexpr uint32_t BUFFER_SIZE_BYTE_64K = 2 * 32768;
    static constexpr uint32_t BYTE_BLOCK_PFA = 32;
    static constexpr uint32_t FP32_REPEAT_ELEMENT_NUM = 64;
    static constexpr uint32_t FP32_BLOCK_ELEMENT_NUM = 8;
    static constexpr float EPS = 1e-5f;

    AscendC::TPipe pipe;
    AscendC::GlobalTensor<Q_T> queryGm;
    AscendC::GlobalTensor<KV_T> keyGm;
    AscendC::GlobalTensor<MM_OUT_T> bmm1ResGmDb[2];
    AscendC::GlobalTensor<T> firstReduceGm;
    AscendC::GlobalTensor<int64_t> actualSeqLengthsGm;
    AscendC::GlobalTensor<int64_t> actualSeqLengthsKVGm;

    AscendC::GlobalTensor<Q_T> queryGmTmp[2];
    AscendC::GlobalTensor<int8_t> maskGm;
    AscendC::GlobalTensor<int32_t> sparseCountTableGm;
    AscendC::LocalTensor<T> mmResUb[2];
    AscendC::LocalTensor<MM_OUT_T> softmaxMaxUb;
    AscendC::LocalTensor<MM_OUT_T> softmaxSumUb;
    AscendC::LocalTensor<MM_OUT_T> softmaxExpUb;
    AscendC::LocalTensor<MM_OUT_T> maxLocal;
    AscendC::LocalTensor<int8_t> maskUb;
    AscendC::LocalTensor<half> countUb;
    AscendC::LocalTensor<T> firstReduceUb;

    static constexpr float SOFTMAX_MIN_NUM = -2e38;
    uint32_t batchSize;
    uint32_t head_num_q;
    uint32_t head_num_kv;
    uint32_t seqLenQ;
    uint32_t seqLenK;
    uint32_t softmaxSouterStepLen;
    uint32_t dim;
    uint32_t stride;
    uint32_t blockIdx;
    uint32_t sparseSize;
    uint32_t usedCoreNum;
    uint32_t headNumRatio;
    bool causal;
    float rowSparse;

    uint32_t singleCoreMBase;
    uint32_t singleCoreNBase;
    uint32_t mm1SingleCoreNPrev = 0;
    uint64_t tensorACoreOffset;
    uint64_t tensorBCoreOffset;

    float scaleValue;
    float threshold;
    bool setFirst;
    bool setDiag;

    uint32_t MultiHeadQ;   // qH
    uint32_t MultiHeadKV;  // kvH
    uint32_t singleProcessSOuterSizeTail;
    bool isActualLenDimsNull = true;
    bool isActualLenDimsKVNull = true;
    bool isGlobalFirstCompute = true;

    TBuf<> tmpBuff1;
    TBuf<> mmResBuff;
    TBuf<> firstReduceBuff;
    TBuf<> softmaxMaxBuff;
    TBuf<> softmaxSumBuff;
    TBuf<> softmaxExpBuff;
    TBuf<> maskBuff;
    TBuf<> countBuff;
    TBuf<> QKTailBuff;
    TBuf<> maxLocalBuff;

    event_t bmm1ResCopyInEvent[2];
    using AType = MatmulType<TPosition::GM, CubeFormat::ND, Q_T, false>;
    using b1Type = MatmulType<TPosition::GM, CubeFormat::ND, KV_T, true>;
    using bias1Type = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using c1Type = MatmulType<TPosition::GM, CubeFormat::ND_ALIGN, MM_OUT_T>;
    // static constexpr MatmulConfig MM_CFG = GetMDLConfig(false, false, 2);
    Matmul<AType, b1Type, c1Type, bias1Type> mm;  // ,MM_CFG
    const SparseBlockEstimateTilingData *__restrict tilingData;

    SPFAEstParam pfaParamsQueue[PFA_PARAMS_QUEUE_CAPBABILITY];
    SPFAEstParam *tailParams;
    SPFAEstParam *headParams;
    int32_t headId = 0;
    int32_t tailId = 0;
    int32_t queSize = 0;
    int32_t queSizeLimit = PFA_PARAMS_QUEUE_CAPBABILITY - 2;

    SoftMaxTiling softmaxFlashTilingData;
    SoftMaxTiling softmaxFlashTilingDataNew;

    __aicore__ inline SparseBlockEstimate()
    {}

    template <typename T>
    static __aicore__ inline T Align(T num, T rnd)
    {
        return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd)));
    }

    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR actual_seq_len, GM_ADDR actual_seq_len_kv,
        GM_ADDR sparse_mask, GM_ADDR sparse_count_table, GM_ADDR workspace,
        const SparseBlockEstimateTilingData &tilingData)
    {
        this->blockIdx = AscendC::GetBlockIdx();
        this->batchSize = tilingData.batchSize;
        this->head_num_q = tilingData.headNumQ;
        this->head_num_kv = tilingData.headNumKV;
        this->seqLenQ = tilingData.seqLenQ;
        this->seqLenK = tilingData.seqLenK;
        this->dim = tilingData.dim;
        this->stride = tilingData.stride;
        this->sparseSize = tilingData.sparseSize;
        this->scaleValue = tilingData.scaleFactor;
        this->singleCoreMBase = tilingData.sOuterFactor;
        this->singleCoreNBase = tilingData.sInnerFactor;
        this->headNumRatio = head_num_q / head_num_kv;
        this->usedCoreNum = tilingData.actualCoreNums;
        this->threshold = tilingData.threshold;
        this->setFirst = tilingData.setFirstCol;
        this->setDiag = tilingData.setDiag;
        this->causal = tilingData.causal;
        this->tilingData = &tilingData;
        this->rowSparse = tilingData.rowSparse;  // 强制稀疏率，当设置大于等于 1 时不生效，0~1之间生效。

        tailId = 0;
        headId = 0;
        queSize = 0;
        tailParams = &pfaParamsQueue[tailId];
        headParams = &pfaParamsQueue[headId];
        tailParams->gmPingpong = 0;
        auto sparse_seq_len_q = (seqLenQ + sparseSize - 1) / sparseSize;
        auto sparse_seq_len_k = (seqLenK + sparseSize - 1) / sparseSize;
        sparse_seq_len_k = Align(sparse_seq_len_k, BYTE_BLOCK_PFA);

        uint32_t offset = 0;
        auto totalBlockNum = GetBlockNum() * GetTaskRation();
        queryGm.SetGlobalBuffer((__gm__ Q_T *)query);
        keyGm.SetGlobalBuffer((__gm__ KV_T *)key);
        if (tilingData.actualSeqLengthsSize > 0) {
            actualSeqLengthsGm.SetGlobalBuffer((__gm__ int64_t *)actual_seq_len, batchSize);
            isActualLenDimsNull = false;
        }

        if (tilingData.actualSeqLengthsKVSize > 0) {
            actualSeqLengthsKVGm.SetGlobalBuffer((__gm__ int64_t *)actual_seq_len_kv, batchSize);
            isActualLenDimsKVNull = false;
        }

        maskGm.SetGlobalBuffer(
            (__gm__ int8_t *)sparse_mask, sparse_seq_len_q * sparse_seq_len_k * this->batchSize * this->head_num_q);
        sparseCountTableGm.SetGlobalBuffer((__gm__ int32_t *)sparse_count_table,
            Align<int32_t>(sparse_seq_len_q, BYTE_BLOCK_PFA / sizeof(int32_t)) * this->batchSize * this->head_num_q);

        uint32_t mmResN = singleCoreNBase + BYTE_BLOCK_PFA;
        bmm1ResGmDb[0].SetGlobalBuffer(
            (__gm__ MM_OUT_T *)(workspace + offset +
                                this->blockIdx * mmResN * this->singleCoreMBase * sizeof(MM_OUT_T)));
        offset += totalBlockNum * mmResN * this->singleCoreMBase * sizeof(MM_OUT_T);

        bmm1ResGmDb[1].SetGlobalBuffer(
            (__gm__ MM_OUT_T *)(workspace + offset +
                                this->blockIdx * mmResN * this->singleCoreMBase * sizeof(MM_OUT_T)));
        offset += totalBlockNum * mmResN * this->singleCoreMBase * sizeof(MM_OUT_T);

        queryGmTmp[0].SetGlobalBuffer(
            (__gm__ Q_T *)(workspace + offset +
                           this->blockIdx * this->singleCoreMBase * stride * this->dim * sizeof(Q_T)));  // Q, reordered
        offset += totalBlockNum * this->singleCoreMBase * stride * this->dim * sizeof(Q_T);
        queryGmTmp[1].SetGlobalBuffer(
            (__gm__ Q_T *)(workspace + offset +
                           this->blockIdx * this->singleCoreMBase * stride * this->dim * sizeof(Q_T)));  // Q, reordered
        offset += totalBlockNum * this->singleCoreMBase * stride * this->dim * sizeof(Q_T);

        firstReduceGm.SetGlobalBuffer((
            __gm__ T *)(workspace + offset + this->blockIdx * singleCoreNBase * 2 * this->singleCoreMBase * sizeof(T)));
        offset += totalBlockNum * singleCoreNBase * 2 * this->singleCoreMBase * sizeof(T); // 2个singleCoreNBase大小

        if (GetSysWorkSpacePtr() == nullptr) {
            return;
        }
    }

    __aicore__ inline void InitBuffers()
    {
        constexpr auto sizeBase40k = BUFFER_SIZE_BYTE_4K * 10;
        pipe.InitBuffer(mmResBuff, 2 * sizeBase40k); // 2个sizeBase40k大小
        pipe.InitBuffer(firstReduceBuff, sizeBase40k / 2); // sizeBase40k / 2
        pipe.InitBuffer(tmpBuff1, BUFFER_SIZE_BYTE_16K * 3);  // 3个BUFFER_SIZE_BYTE_16K大小
        // 以上同时作为qtemp （128 + 16）每个task能覆盖

        pipe.InitBuffer(maskBuff, BUFFER_SIZE_BYTE_8K);  // 156
        pipe.InitBuffer(QKTailBuff, BUFFER_SIZE_BYTE_1K);
        pipe.InitBuffer(countBuff, BUFFER_SIZE_BYTE_1K);  // 158
        pipe.InitBuffer(softmaxMaxBuff, BUFFER_SIZE_BYTE_4K);
        pipe.InitBuffer(softmaxExpBuff, BUFFER_SIZE_BYTE_4K);
        pipe.InitBuffer(softmaxSumBuff, BUFFER_SIZE_BYTE_4K);  // 170
        pipe.InitBuffer(maxLocalBuff,
            BUFFER_SIZE_BYTE_8K);  // 128k / singlecoreN / stride * singlecoreM * sizeof(half) = 4k float = 8k

        softmaxMaxUb = softmaxMaxBuff.Get<MM_OUT_T>();
        softmaxSumUb = softmaxSumBuff.Get<MM_OUT_T>();
        softmaxExpUb = softmaxExpBuff.Get<MM_OUT_T>();
        maxLocal = maxLocalBuff.Get<MM_OUT_T>();
        mmResUb[0] = mmResBuff.Get<T>();
        mmResUb[1] = mmResUb[0][sizeBase40k / sizeof(T)];
        maskUb = maskBuff.Get<int8_t>();
        countUb = countBuff.Get<half>();
        firstReduceUb = firstReduceBuff.Get<T>();
    }

    __aicore__ inline void Process()
    {
        if (g_coreType == AIV && blockIdx >= usedCoreNum) {
            return;
        }
        AllocGlobalResources();
        ComputeEachCore(this->blockIdx);
        while (this->queSize > 0) {
            this->queSize--;
            ComputeEachCoreSInnerLoop();
            this->headId = (this->headId + 1) % PFA_PARAMS_QUEUE_CAPBABILITY;
            this->headParams = &this->pfaParamsQueue[this->headId];
        }
        FreeGlobalResources();
    }

    __aicore__ inline void AllocGlobalResources()
    {
        for (int i = 0; i < 2; ++i) { // 循环2次
            this->bmm1ResCopyInEvent[i] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
        }
    }

    __aicore__ inline void FreeGlobalResources()
    {
        for (int i = 0; i < 2; ++i) { // 循环2次
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(this->bmm1ResCopyInEvent[i]);
        }
    }

    __aicore__ inline void SInnerLoopFunc(int curBatch)
    {
        SPFAEstParam *&params = this->tailParams;
        auto basicSInnerSize = singleCoreNBase;  // 1k  64 * 16 除了stride
        if constexpr (CAUSAL) {
            params->actualSeqLengthKVPerBatch =
                params->sOuterOffset + params->singleProcessSOuterSize / stride * stride;  // causal忽略key尾块
        }
        auto computeSize = params->actualSeqLengthKVPerBatch / stride;                // <= 16k
        auto innerLoopTimes = (computeSize + basicSInnerSize - 1) / basicSInnerSize;  // 内循环次数,<=16
        for (auto sInnerLoopIdx = 0; sInnerLoopIdx < innerLoopTimes; sInnerLoopIdx++) {
            params->sInnerLoopTimes = innerLoopTimes;
            params->sInnerLoopIdx = sInnerLoopIdx;
            params->tensorAOffset = tensorACoreOffset;

            if constexpr (LAYOUT_T == INPUT_LAYOUT::TND || LAYOUT_T == INPUT_LAYOUT::BSH) {
                params->tensorBOffset =
                    tensorBCoreOffset + sInnerLoopIdx * basicSInnerSize * stride * head_num_kv * dim;
            } else if constexpr (LAYOUT_T == INPUT_LAYOUT::BNSD) {
                params->tensorBOffset = tensorBCoreOffset + sInnerLoopIdx * basicSInnerSize * stride * dim;
            }
            // 当前sInnerLoop的BOffset，tensorBCoreOffset + InnerIdx * stride * headDim * singleProcessSInnerSiz
            // actualsingleProcessSInnerSize
            auto lastInner = sInnerLoopIdx == innerLoopTimes - 1;
            params->singleProcessSInnerBmmTail =
                lastInner ? computeSize - (innerLoopTimes - 1) * basicSInnerSize : basicSInnerSize;
            // actualsingleProcessSInnerSizeAlign
            params->mm1SingleCoreN = Align(params->singleProcessSInnerBmmTail + (lastInner && (!CAUSAL) ? 1 : 0),
                BYTE_BLOCK_PFA);  //! plus 1 for possible k % stride > 0 when not causal
            if (this->queSize >= this->queSizeLimit) {
                ComputeEachCoreSInnerLoop();
                this->headId = (this->headId + 1) % PFA_PARAMS_QUEUE_CAPBABILITY;
                this->headParams = &this->pfaParamsQueue[this->headId];

                this->tailId = (this->tailId + 1) % PFA_PARAMS_QUEUE_CAPBABILITY;
                SPFAEstParam *nextTailParams = &this->pfaParamsQueue[this->tailId];
                if (sInnerLoopIdx < PFA_PARAMS_QUEUE_CAPBABILITY - 1) {
                    this->CopyParamsAttrOutOfInnerLoop(nextTailParams, this->tailParams);
                }
                nextTailParams->gmPingpong = this->tailParams->gmPingpong ^ 1;
                this->tailParams = nextTailParams;
            } else {
                this->tailId = (this->tailId + 1) % PFA_PARAMS_QUEUE_CAPBABILITY;
                SPFAEstParam *nextTailParams = &this->pfaParamsQueue[this->tailId];
                this->CopyParamsAttrOutOfInnerLoop(nextTailParams, this->tailParams);
                nextTailParams->gmPingpong = this->tailParams->gmPingpong ^ 1;
                this->tailParams = nextTailParams;
                this->queSize++;
            }
        }
    }

    __aicore__ inline void GetSingleCoreParam(int sIdx)
    {
        int64_t actualSeqLengthPerBatch = 0;
        int64_t actualSeqLengthKVPerBatch = 0;
        if (isActualLenDimsNull) {
            actualSeqLengthPerBatch = seqLenQ;
        } else {
            if constexpr (LAYOUT_T == INPUT_LAYOUT::TND) {
                actualSeqLengthPerBatch =
                    sIdx == 0 ? actualSeqLengthsGm.GetValue(0)
                              : actualSeqLengthsGm.GetValue(sIdx) - actualSeqLengthsGm.GetValue(sIdx - 1);
            } else {
                actualSeqLengthPerBatch = (tilingData->actualSeqLengthsSize == 1) ? actualSeqLengthsGm.GetValue(0)
                                                                                  : actualSeqLengthsGm.GetValue(sIdx);
            }
        }
        if (isActualLenDimsKVNull) {
            actualSeqLengthKVPerBatch = seqLenK;
        } else {
            if constexpr (LAYOUT_T == INPUT_LAYOUT::TND) {
                actualSeqLengthKVPerBatch =
                    sIdx == 0 ? actualSeqLengthsKVGm.GetValue(0)
                              : actualSeqLengthsKVGm.GetValue(sIdx) - actualSeqLengthsKVGm.GetValue(sIdx - 1);
            } else {
                actualSeqLengthKVPerBatch = tilingData->actualSeqLengthsKVSize == 1
                                                ? actualSeqLengthsKVGm.GetValue(0)
                                                : actualSeqLengthsKVGm.GetValue(sIdx);
            }
        }
        MultiHeadQ = this->head_num_q * dim;      // qH !原本是headSize，被我改成了dim
        MultiHeadKV = MultiHeadQ / headNumRatio;  // kvH
        uint32_t singleProcessSOuterSizeWhole = singleCoreMBase * stride;
        singleProcessSOuterSizeTail = (actualSeqLengthPerBatch % singleProcessSOuterSizeWhole != 0)
                                          ? actualSeqLengthPerBatch % singleProcessSOuterSizeWhole
                                          : singleProcessSOuterSizeWhole;
        this->tailParams->actualSeqLengthPerBatch = actualSeqLengthPerBatch;
        this->tailParams->actualSeqLengthKVPerBatch = actualSeqLengthKVPerBatch;
    }

    __aicore__ inline void ComputeEachCore(uint32_t coreIdx)
    {
        auto actualCoreNums = usedCoreNum;
        uint32_t sOuterCoreIdx = coreIdx;
        uint32_t sOuterSize = stride * singleCoreMBase;

        uint32_t sIdStart = this->tilingData->sparseBlockEstimateSeqParams.actualCoreNums[sOuterCoreIdx];
        uint32_t sIdEnd = this->tilingData->sparseBlockEstimateSeqParams.singleCoreHeadNumSize[sOuterCoreIdx];
        uint32_t outerLoopStart = this->tilingData->sparseBlockEstimateSeqParams.coreSeqPosStart[sOuterCoreIdx];
        uint32_t outerLoopEnd = this->tilingData->sparseBlockEstimateSeqParams.coreSeqPosEnd[sOuterCoreIdx];
        uint32_t nLoopStart = this->tilingData->sparseBlockEstimateSeqParams.coreHeadNumTail[sOuterCoreIdx];
        uint32_t nLoopEnd = this->tilingData->sparseBlockEstimateSeqParams.actualS1[sOuterCoreIdx];  // actualS1
        uint32_t tmpOuterLoopEnd;
        uint32_t tmpSLoopEnd;
        bool isLast = false;
        int64_t actualSeqLengthbIdx = 0;
        // You must pass the reference assignment params because the head address is updated internally.
        SPFAEstParam *&params = this->tailParams;
        for (uint32_t loopNIdx = nLoopStart; loopNIdx < nLoopEnd; loopNIdx++) {
            params->batchNOffset = loopNIdx;  // q nIdx
            if (loopNIdx != nLoopEnd - 1) {
                tmpSLoopEnd = batchSize;
            } else {
                tmpSLoopEnd = sIdEnd;
                isLast = true;
            }

            for (uint32_t bIdx = sIdStart; bIdx < tmpSLoopEnd; bIdx++) {
                GetSingleCoreParam(bIdx);
                uint32_t sOuterBlockNum = (params->actualSeqLengthPerBatch + sOuterSize - 1) / sOuterSize;
                tmpOuterLoopEnd = (isLast && bIdx == tmpSLoopEnd - 1) ? outerLoopEnd : sOuterBlockNum;
                for (uint32_t sOuterLoopIdx = outerLoopStart; sOuterLoopIdx < tmpOuterLoopEnd; sOuterLoopIdx++) {
                    params->singleProcessSOuterSize = sOuterLoopIdx == sOuterBlockNum - 1
                                                          ? singleProcessSOuterSizeTail
                                                          : sOuterSize;  // with stride but not tail align
                    params->sOuterOffset = sOuterLoopIdx * sOuterSize;   // with stride
                    params->sOuterLoopIdx = sOuterLoopIdx;
                    params->bIdx = bIdx;
                    params->batchNOffset = loopNIdx;

                    if constexpr (LAYOUT_T == INPUT_LAYOUT::BNSD) {
                        tensorACoreOffset =
                            (bIdx * head_num_q + loopNIdx) * dim * seqLenQ + sOuterLoopIdx * sOuterSize * dim;  // bnsd
                        tensorBCoreOffset = (bIdx * head_num_kv + loopNIdx / headNumRatio) * dim * seqLenK;     // bnsd
                    } else if constexpr (LAYOUT_T == INPUT_LAYOUT::BSH) {
                        tensorACoreOffset = bIdx * seqLenQ * head_num_q * dim +
                                            sOuterLoopIdx * sOuterSize * head_num_q * dim + loopNIdx * dim;
                        tensorBCoreOffset = bIdx * seqLenK * head_num_kv * dim + loopNIdx / headNumRatio * dim;
                    } else if constexpr (LAYOUT_T == INPUT_LAYOUT::TND) {
                        tensorACoreOffset = (bIdx == 0 ? 0 : actualSeqLengthsGm.GetValue(bIdx - 1)) * head_num_q * dim +
                                            sOuterLoopIdx * sOuterSize * head_num_q * dim + loopNIdx * dim;
                        tensorBCoreOffset =
                            (bIdx == 0 ? 0 : actualSeqLengthsKVGm.GetValue(bIdx - 1)) * head_num_kv * dim +
                            loopNIdx / headNumRatio * dim;
                    }
                    SInnerLoopFunc(bIdx);
                }
                outerLoopStart = 0;
            }
            sIdStart = 0;
        }
    }

    __aicore__ inline void Bmm1ComputeIterate(SPFAEstParam *params)
    {
        if (this->mm1SingleCoreNPrev != params->mm1SingleCoreN) {
            this->mm.SetOrgShape(this->tilingData->cubeTilingData.M,
                this->tilingData->cubeTilingData.N,
                this->tilingData->cubeTilingData.Ka,
                this->tilingData->cubeTilingData.Kb,
                params->mm1SingleCoreN);
            this->mm1SingleCoreNPrev = params->mm1SingleCoreN;
        }
        this->mm.SetTail((params->singleProcessSOuterSize + stride - 1) / stride, params->singleProcessSInnerBmmTail);
        this->mm.SetTensorA(this->queryGmTmp[params->sOuterLoopIdx & 1]);
        this->mm.SetTensorB(this->keyGm[params->tensorBOffset], true);

        this->mm.template IterateAll<false>(this->bmm1ResGmDb[params->gmPingpong], false, false, true, false);
    }

    __aicore__ inline void Bmm1VecInputCopyIn()
    {
        SPFAEstParam *params = this->headParams;
        uint32_t reduceSize = sparseSize / stride;
        auto n = params->mm1SingleCoreN < singleCoreNBase ? params->mm1SingleCoreN : singleCoreNBase;
        this->softmaxSouterStepLen =
            BUFFER_SIZE_BYTE_32K / sizeof(MM_OUT_T) / n / 8 * 8;  // 16 = blocksize/stride;  16 * 1024 * 4 * db；8最小单位
        DataCopy(this->mmResUb[0].template ReinterpretCast<MM_OUT_T>(),
            this->bmm1ResGmDb[params->gmPingpong],
            this->softmaxSouterStepLen * params->mm1SingleCoreN);
        SetFlag<HardEvent::MTE2_V>(this->bmm1ResCopyInEvent[0]);
    }

    __aicore__ inline void ComputeEachCoreSInnerLoop()
    {
        SPFAEstParam *params = this->headParams;
        SPFAEstParam *nextParams = &(this->pfaParamsQueue[(this->headId + 1) % PFA_PARAMS_QUEUE_CAPBABILITY]);

        // mm1 compute
        if (this->isGlobalFirstCompute) {
            ReorderQKTail(params);  // R1
            this->Bmm1ComputeIterate(params);
            SoftMaxShapeInfo softmaxShapeInfo{16 * sizeof(T) / sizeof(MM_OUT_T),
                uint32_t(singleCoreNBase + BYTE_BLOCK_PFA),
                16 * sizeof(T) / sizeof(MM_OUT_T),
                singleCoreNBase};
            this->softmaxFlashTilingData = SoftMaxFlashV2TilingFunc(
                softmaxShapeInfo, sizeof(MM_OUT_T), sizeof(MM_OUT_T),
                BUFFER_SIZE_BYTE_16K * 3, true, false); // 3个BUFFER_SIZE_BYTE_16K大小
        }
        if (this->queSize > 0 && nextParams->sInnerLoopIdx == 0) {
            ReorderQKTail(nextParams);  // R2
        }
        this->mm.WaitIterateAll();

        if (this->queSize > 0) {
            this->Bmm1ComputeIterate(nextParams);
        }
        Bmm1VecInputCopyIn();
        this->Res1VecCompute(params);  // V1 // V2
        this->isGlobalFirstCompute = false;
    }

    template <typename CT>
    static __aicore__ inline void RowMuls(LocalTensor<CT> dstUb, LocalTensor<CT> src0Ub, LocalTensor<CT> src1Ub,
        uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
    {
        // Muls by row, 每行的元素除以相同的元素
        // dstUb[i, (j * 8) : (j * 8 + 7)] = src0Ub[i, (j * 8) : (j * 8 + 7)] / src1Ub[i, 0 : 7]
        // src0Ub:[dealRowCount, columnCount], src1Ub:[dealRowCount, FP32_BLOCK_ELEMENT_NUM] dstUb:[dealRowCount,
        // columnCount]
        uint32_t dtypeMask = 256 / sizeof(CT);
        uint32_t dLoop = actualColumnCount / dtypeMask;
        uint32_t dRemain = actualColumnCount % dtypeMask;
        constexpr auto FP32_BLOCK_ELEMENT_NUM = BYTE_BLOCK_PFA / sizeof(CT);
        BinaryRepeatParams repeatParamsMul;
        repeatParamsMul.src0BlkStride = 1;
        repeatParamsMul.src1BlkStride = 0;
        repeatParamsMul.dstBlkStride = 1;
        repeatParamsMul.src0RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
        repeatParamsMul.src1RepStride = 1;
        repeatParamsMul.dstRepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
        uint32_t columnRepeatCount = dLoop;
        if (columnRepeatCount <= dealRowCount) {
            uint32_t offset = 0;
            for (uint32_t i = 0; i < dLoop; i++) {
                Mul(dstUb[offset], src0Ub[offset], src1Ub, dtypeMask, dealRowCount, repeatParamsMul);
                offset += dtypeMask;
            }
        } else {
            BinaryRepeatParams columnRepeatParams;
            columnRepeatParams.src0BlkStride = 1;
            columnRepeatParams.src1BlkStride = 0;
            columnRepeatParams.dstBlkStride = 1;
            columnRepeatParams.src0RepStride = 8;  // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
            columnRepeatParams.src1RepStride = 0;
            columnRepeatParams.dstRepStride = 8;  // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
            uint32_t offset = 0;
            for (uint32_t i = 0; i < dealRowCount; i++) {
                Mul(dstUb[offset],
                    src0Ub[offset],
                    src1Ub[i * FP32_BLOCK_ELEMENT_NUM],
                    dtypeMask,
                    columnRepeatCount,
                    columnRepeatParams);
                offset += columnCount;
            }
        }
        if (dRemain > 0) {
            Mul(dstUb[dLoop * dtypeMask], src0Ub[dLoop * dtypeMask], src1Ub, dRemain, dealRowCount, repeatParamsMul);
        }
    }

    __aicore__ inline void AccumulateByRow(const LocalTensor<T> &srcUb, uint32_t rowCount, uint32_t columnCount)
    {
        for (uint32_t i = rowCount; i > 1;) {
            i >>= 1;
            pipe_barrier(PIPE_V);
            Add(srcUb, srcUb, srcUb[i * columnCount], i * columnCount);
        }
        pipe_barrier(PIPE_V);
    }

    __aicore__ inline void ColumnSum(LocalTensor<T> srcUb, uint32_t dealRowCount, uint32_t columnCount)
    {
        // 将srcUb的dealRowCount行累加到第一行,每行columnCount个元素
        for (uint32_t mask = sparseSize / stride; mask > 0; mask = mask / 2) { // mask/2是mask减半（二进制右移1位），缩小二分累加步长
            if (dealRowCount & mask) {
                if (dealRowCount > mask) {
                    pipe_barrier(PIPE_V);
                    Add(srcUb, srcUb, srcUb[mask * columnCount], (dealRowCount - mask) * columnCount);
                }
                AccumulateByRow(srcUb, mask, columnCount);
                break;
            }
        }
    }

    template <typename CT>
    static __aicore__ inline void RowSum(const LocalTensor<CT> dstUb, LocalTensor<CT> srcUb, uint32_t dealRowCount,
        uint32_t columnCount, uint32_t actualColumnCount)
    {
        uint32_t dtypeMask = 256 / sizeof(CT);
        uint32_t blockCount = actualColumnCount / dtypeMask;
        uint32_t remain = actualColumnCount % dtypeMask;

        BinaryRepeatParams repeatParamsMax;
        repeatParamsMax.src0BlkStride = 1;
        repeatParamsMax.src1BlkStride = 1;
        repeatParamsMax.dstBlkStride = 1;
        repeatParamsMax.src0RepStride = columnCount / (BYTE_BLOCK_PFA / sizeof(CT));
        repeatParamsMax.src1RepStride = columnCount / (BYTE_BLOCK_PFA / sizeof(CT));
        repeatParamsMax.dstRepStride = columnCount / (BYTE_BLOCK_PFA / sizeof(CT));
        if (blockCount > 0 && remain > 0) {
            Add(srcUb, srcUb, srcUb[blockCount * dtypeMask], remain, dealRowCount, repeatParamsMax);
            pipe_barrier(PIPE_V);
        }

        for (uint32_t loopCount = blockCount / 2; loopCount > 0; loopCount = blockCount / 2) {
            blockCount = (blockCount + 1) / 2;
            for (uint32_t j = 0; j < loopCount; j++) {
                Add(srcUb[j * dtypeMask],
                    srcUb[j * dtypeMask],
                    srcUb[(j + blockCount) * dtypeMask],
                    dtypeMask,
                    dealRowCount,
                    repeatParamsMax);
            }
            pipe_barrier(PIPE_V);
        }
        WholeReduceSum(dstUb,
            srcUb,
            (actualColumnCount < dtypeMask) ? actualColumnCount : dtypeMask,
            dealRowCount,
            1,
            1,
            columnCount / (BYTE_BLOCK_PFA / sizeof(CT)));
    }

    template <bool nonFirst>
    __aicore__ inline void SoftmaxCompute(const LocalTensor<MM_OUT_T> &mmResUb,
        const LocalTensor<MM_OUT_T> &softmaxMaxUb, const LocalTensor<MM_OUT_T> &softmaxSumUb,
        const LocalTensor<MM_OUT_T> &softmaxExpMaxUb, const LocalTensor<uint8_t> &tmpUb, uint32_t souterSize,
        uint32_t colCount, uint32_t actColCount)
    {
        SoftMaxShapeInfo softmaxShapeInfo{static_cast<uint32_t>(souterSize),
            static_cast<uint32_t>(colCount),
            static_cast<uint32_t>(souterSize),
            static_cast<uint32_t>(actColCount)};
        SoftMaxTiling &softmaxFlashTilingData = this->softmaxSouterStepLen == this->softmaxFlashTilingData.srcM
                                                    ? this->softmaxFlashTilingData
                                                    : this->softmaxFlashTilingDataNew;
        SoftmaxFlashV2<MM_OUT_T, nonFirst, true, false>(mmResUb,
            softmaxSumUb,
            softmaxMaxUb,
            mmResUb,
            softmaxExpMaxUb,
            softmaxSumUb,
            softmaxMaxUb,
            tmpUb,
            softmaxFlashTilingData,
            softmaxShapeInfo);
    }

    __aicore__ inline void MaskCopyOut(uint32_t rowCount, uint32_t blockNumKGlobal, uint32_t sparseCountTableGmOffset)
    {
        auto reduceSize = sparseSize / stride;
        auto tmpUb = tmpBuff1.Get<half>();
        auto ansUb = countUb.template ReinterpretCast<int32_t>();
        auto n = (rowCount + reduceSize - 1) / reduceSize;
        // 反 brcb
        WholeReduceMax(tmpUb, countUb, BYTE_BLOCK_PFA / sizeof(T), n, 1, 1, 1, ReduceOrder::ORDER_ONLY_VALUE);
        pipe_barrier(PIPE_V);
        Cast(ansUb, tmpUb, RoundMode::CAST_CEIL, n);
        auto eidv3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eidv3);
        WaitFlag<HardEvent::V_MTE3>(eidv3);
        DataCopyExtParams dce;
        dce.blockCount = 1;
        dce.blockLen = n * sizeof(int32_t);
        DataCopyPad(sparseCountTableGm[sparseCountTableGmOffset], ansUb, dce);
        uint32_t maskGmOffset = sparseCountTableGmOffset * blockNumKGlobal;
        DataCopy(maskGm[maskGmOffset],
            maskUb,
            {static_cast<uint16_t>(n),
                static_cast<uint16_t>(blockNumKGlobal * sizeof(uint8_t) / BYTE_BLOCK_PFA),
                static_cast<uint16_t>(
                    32 - blockNumKGlobal * sizeof(uint8_t) / BYTE_BLOCK_PFA),
                0});
    }

    __aicore__ inline void Res1VecCompute(SPFAEstParam *params)
    {
        uint32_t blockNumQGlobal = (seqLenQ + sparseSize - 1) / sparseSize;
        uint32_t blockNumKGlobal = (seqLenK + sparseSize - 1) / sparseSize;
        blockNumKGlobal = Align<uint32_t>(blockNumKGlobal, BYTE_BLOCK_PFA);

        auto colCount = params->mm1SingleCoreN;
        auto actColCount = params->singleProcessSInnerBmmTail;
        auto gmPingpong = params->gmPingpong;
        auto sInnerLoopIdx = params->sInnerLoopIdx;
        auto looptimes = params->sInnerLoopTimes;
        auto klen = params->actualSeqLengthKVPerBatch;

        uint32_t basicSInnerSize = singleCoreNBase;
        uint32_t reduceSize = sparseSize / stride;                                               // 16
        uint32_t reduceColCntAlign = basicSInnerSize / reduceSize + BYTE_BLOCK_PFA / sizeof(T);  // 8 for tailk
        uint32_t rowStart = params->sOuterOffset / stride;
        uint32_t rowCount = (params->singleProcessSOuterSize + stride - 1) / stride;
        uint32_t sparseCountTableGmOffset = params->bIdx * head_num_q * blockNumQGlobal +
                                            params->batchNOffset * blockNumQGlobal +
                                            params->sOuterLoopIdx * singleCoreMBase / reduceSize;
        auto QKTailUb = QKTailBuff.Get<T>();
        QKTailUb = QKTailUb[singleCoreMBase * (params->sOuterLoopIdx & 1)];
        auto tmpUb = tmpBuff1.Get<T>();
        int ubPingpong = 0;
        auto blockCountRow = (klen + sparseSize - 1) / sparseSize;
        bool hasktail = klen % stride > 0;

        if (this->softmaxSouterStepLen != this->softmaxFlashTilingData.srcM ||
            colCount != this->softmaxFlashTilingData.srcK) {
            SoftMaxShapeInfo softmaxShapeInfo{static_cast<uint32_t>(this->softmaxSouterStepLen),
                static_cast<uint32_t>(colCount),
                static_cast<uint32_t>(this->softmaxSouterStepLen),
                static_cast<uint32_t>(colCount)};
            // const bool isUpdate = false, const bool isBasicBlock = false, const bool isDataFormatNZ = false, const
            // bool isFlashOutputBrc = false
            this->softmaxFlashTilingDataNew = SoftMaxFlashV2TilingFunc(
                softmaxShapeInfo, sizeof(MM_OUT_T), sizeof(MM_OUT_T),
                BUFFER_SIZE_BYTE_16K * 3, true, false); // 3个BUFFER_SIZE_BYTE_16K大小
        }

        uint32_t reduceColCnt = (actColCount + reduceSize - 1) / reduceSize;
        uint32_t totreduceColCntAlign = (basicSInnerSize / reduceSize) * looptimes + BYTE_BLOCK_PFA / sizeof(T);

        Duplicate(firstReduceUb, (T)0, rowCount * reduceColCntAlign);
        auto ev2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(ev2);
        for (uint32_t souterOffset = 0, mm1ResGmOffset = 0, nextSouterOffset = 0; souterOffset < rowCount;
             souterOffset = nextSouterOffset) {  // Pending rectification
            uint32_t remainSouterSize = rowCount - souterOffset;
            uint32_t souterSize =
                (remainSouterSize >= this->softmaxSouterStepLen) ? this->softmaxSouterStepLen : remainSouterSize;
            nextSouterOffset = souterOffset + this->softmaxSouterStepLen;
            uint32_t nextLeftSouterSize = rowCount - nextSouterOffset;
            uint32_t nextSouterSize =
                (nextLeftSouterSize >= this->softmaxSouterStepLen) ? this->softmaxSouterStepLen : nextLeftSouterSize;
            uint32_t nextMm1ResGmOffset = mm1ResGmOffset + souterSize * colCount;
            uint32_t computeSize = souterSize * colCount;

            WaitFlag<HardEvent::V_MTE2>(ev2);
            if (nextSouterOffset < rowCount) {  // noLastSoftmaxLoop
                auto &srcGm = this->bmm1ResGmDb[gmPingpong];
                DataCopy(this->mmResUb[ubPingpong ^ 1].template ReinterpretCast<MM_OUT_T>(),
                    srcGm[nextMm1ResGmOffset],
                    nextSouterSize * colCount);
                SetFlag<HardEvent::MTE2_V>(this->bmm1ResCopyInEvent[ubPingpong ^ 1]);
            }
            WaitFlag<HardEvent::MTE2_V>(this->bmm1ResCopyInEvent[ubPingpong]);

            auto softmaxUb = this->mmResUb[ubPingpong].template ReinterpretCast<MM_OUT_T>();
            Muls(softmaxUb, softmaxUb, static_cast<MM_OUT_T>(this->scaleValue), computeSize);
            pipe_barrier(PIPE_V);
            auto brcbOffset = souterOffset * BYTE_BLOCK_PFA / sizeof(MM_OUT_T);
            if (sInnerLoopIdx == 0) {
                SoftmaxCompute<false>(softmaxUb,
                    softmaxMaxUb[brcbOffset],
                    softmaxSumUb[brcbOffset],
                    softmaxExpUb[brcbOffset],
                    this->tmpBuff1.template Get<uint8_t>(),
                    souterSize,
                    colCount,
                    actColCount);
            } else {
                SoftmaxCompute<true>(softmaxUb,
                    softmaxMaxUb[brcbOffset],
                    softmaxSumUb[brcbOffset],
                    softmaxExpUb[brcbOffset],
                    this->tmpBuff1.template Get<uint8_t>(),
                    souterSize,
                    colCount,
                    actColCount);
            }
            pipe_barrier(PIPE_V);
            if constexpr (sizeof(MM_OUT_T) > sizeof(T)) {
                Cast(this->mmResUb[ubPingpong], softmaxUb, RoundMode::CAST_ROUND, computeSize);
                pipe_barrier(PIPE_V);
            }
            constexpr auto alignNum = BYTE_BLOCK_PFA / sizeof(T);
            auto clearStart = actColCount / alignNum * alignNum;
            auto mulUb = this->tmpBuff1.template Get<int16_t>();
            if (colCount > clearStart) {
                Duplicate(mulUb, (int16_t)0, colCount - clearStart);
                pipe_barrier(PIPE_V);
            }
            if (actColCount > clearStart) {
                Duplicate(mulUb, (int16_t)1, actColCount - clearStart);
                pipe_barrier(PIPE_V);
            }

            for (auto i = 0; i < souterSize; i++) {  // souterSize = 16， i + souterOffset =0~M
                if (colCount > clearStart) {
                    Mul(this->mmResUb[ubPingpong][i * colCount + clearStart].template ReinterpretCast<int16_t>(),
                        this->mmResUb[ubPingpong][i * colCount + clearStart].template ReinterpretCast<int16_t>(),
                        mulUb,
                        colCount - clearStart);
                    pipe_barrier(PIPE_V);
                }
                RowSum(firstReduceUb[reduceColCntAlign * (i + souterOffset)],
                    this->mmResUb[ubPingpong][i * colCount],
                    reduceColCnt,
                    reduceSize,
                    reduceSize);
            }
            SetFlag<HardEvent::V_MTE2>(ev2);
            mm1ResGmOffset = nextMm1ResGmOffset;
            ubPingpong ^= 1;
        }
        WaitFlag<HardEvent::V_MTE2>(ev2);
        // reverse brcb
        WholeReduceMax(maxLocal[singleCoreMBase * sInnerLoopIdx],
            softmaxMaxUb,
            BYTE_BLOCK_PFA / sizeof(MM_OUT_T),
            singleCoreMBase,
            1,
            1,
            1,
            ReduceOrder::ORDER_ONLY_VALUE);
        pipe_barrier(PIPE_V);

        // firstReduceUb = [128, 72] firstReduceGm = [looptimes-1, 128, 72]
        if (sInnerLoopIdx < looptimes - 1) {
            auto eidv3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eidv3);
            WaitFlag<HardEvent::V_MTE3>(eidv3);
            DataCopy(firstReduceGm[sInnerLoopIdx * reduceColCntAlign * rowCount],
                firstReduceUb,
                reduceColCntAlign * rowCount);
        } else {
            auto &secReduceResUb = this->mmResUb[1];
            SecondReduce(secReduceResUb, rowCount, looptimes, gmPingpong);
            pipe_barrier(PIPE_V);
            for (uint32_t i = 0; i < (rowCount + reduceSize - 1) / reduceSize; i++) {
                auto actBlockCount = CAUSAL ? rowStart / reduceSize + i + 1 : blockCountRow;
                ScoreCompute(secReduceResUb[i * totreduceColCntAlign],
                    (rowCount + reduceSize - 1) / reduceSize,
                    actBlockCount,
                    i);
            }  // 4 rows
            MaskCopyOut(rowCount, blockNumKGlobal, sparseCountTableGmOffset);
        }
        event_t eventIDMTE3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
        WaitFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
        auto bmm1ResCopyOutEvent = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(bmm1ResCopyOutEvent);
        WaitFlag<HardEvent::MTE3_MTE2>(bmm1ResCopyOutEvent);
    }

    __aicore__ inline void SecondReduce(
        const LocalTensor<T> &dst, uint32_t srcRowCount, uint32_t innerLoopTimes, uint32_t gmPingpong)
    {
        uint32_t basicSInnerSize = singleCoreNBase;
        uint32_t reduceSize = sparseSize / stride;  // 16
        uint32_t reduceColCnt = basicSInnerSize / reduceSize;
        uint32_t reduceColCntAlign = reduceColCnt + BYTE_BLOCK_PFA / sizeof(T);  // for tailk
        uint32_t totreduceColCntAlign = reduceColCnt * innerLoopTimes + BYTE_BLOCK_PFA / sizeof(T);

        event_t eids[2];
        eids[0] = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        eids[1] = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        auto eidv2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        int ubPingpong = 0;
        auto tmpUb = tmpBuff1.Get<MM_OUT_T>();
        auto maxGlobal = maxLocal[(innerLoopTimes - 1) * singleCoreMBase];

        auto copysize = srcRowCount * reduceColCntAlign;
        if (innerLoopTimes > 1) {
            auto src = this->mmResUb[0][ubPingpong * copysize];
            DataCopy(src, firstReduceGm, copysize);
        }
        SetFlag<HardEvent::MTE2_V>(eids[ubPingpong]);
        SetFlag<HardEvent::V_MTE2>(eidv2);
        for (auto i = 0; i < innerLoopTimes; i++) {
            WaitFlag<HardEvent::V_MTE2>(eidv2);
            if (i + 1 < innerLoopTimes) {
                if (i + 1 < innerLoopTimes - 1) {
                    DataCopy(
                        this->mmResUb[0][(ubPingpong ^ 1) * copysize], firstReduceGm[(i + 1) * copysize], copysize);
                }
                SetFlag<HardEvent::MTE2_V>(eids[ubPingpong ^ 1]);
            }
            // update firstreduce
            auto midx = maxLocal[i * singleCoreMBase];  // midx =  [M, 1], softmaxSumUb = [M, 16]
            Sub(midx, midx, maxGlobal, singleCoreMBase);
            pipe_barrier(PIPE_V);
            Exp(midx, midx, singleCoreMBase);
            pipe_barrier(PIPE_V);  // exp(m_loopidx - M)
            constexpr uint32_t BRCB_ONCE = 8;
            Brcb(tmpUb, midx, singleCoreMBase / BRCB_ONCE, {1, 8});
            pipe_barrier(PIPE_V);  // !FLOAT
            Div(tmpUb, tmpUb, softmaxSumUb, srcRowCount * BYTE_BLOCK_PFA / sizeof(MM_OUT_T));
            pipe_barrier(PIPE_V);  //* exp(m_loopidx - M) / sum( exp(x-M))  FLOAT
            if constexpr (sizeof(MM_OUT_T) > sizeof(T)) {
                // ! row * 8 -> row *16 for RowMuls<T>
                auto tmpUbCopy = tmpUb[srcRowCount * BYTE_BLOCK_PFA / sizeof(MM_OUT_T)];
                DataCopy(tmpUbCopy, tmpUb, {(uint16_t)srcRowCount, 1, 0, 1});
                DataCopy(tmpUbCopy[BYTE_BLOCK_PFA / sizeof(MM_OUT_T)], tmpUb, {(uint16_t)srcRowCount, 1, 0, 1});
                pipe_barrier(PIPE_V);
                Cast(tmpUb.template ReinterpretCast<T>(),
                    tmpUbCopy,
                    RoundMode::CAST_ROUND,
                    srcRowCount * BYTE_BLOCK_PFA / sizeof(T));
                pipe_barrier(PIPE_V);
            }
            WaitFlag<HardEvent::MTE2_V>(eids[ubPingpong]);
            auto src = i < innerLoopTimes - 1 ? this->mmResUb[0][ubPingpong * copysize] : firstReduceUb;
            RowMuls(src, src, tmpUb.template ReinterpretCast<T>(), srcRowCount, reduceColCntAlign, reduceColCntAlign);
            pipe_barrier(PIPE_V);
            auto colCount = i < innerLoopTimes - 1 ? reduceColCnt : reduceColCntAlign;
            auto times = (srcRowCount + reduceSize - 1) / reduceSize;
            for (auto _t = 0; _t < times; _t++) {
                auto secReduceRows = _t == times - 1 ? srcRowCount - _t * reduceSize : reduceSize;
                ColumnSum(src[_t * reduceSize * reduceColCntAlign], secReduceRows, reduceColCntAlign);
                DataCopy(dst[(_t)*totreduceColCntAlign + reduceColCnt * i],
                    src[_t * reduceSize * reduceColCntAlign],
                    colCount);
            }
            SetFlag<HardEvent::V_MTE2>(eidv2);
            ubPingpong ^= 1;
        }
        WaitFlag<HardEvent::V_MTE2>(eidv2);
    }

    __aicore__ inline void ScoreCompute(
        LocalTensor<T> srcLocal, uint32_t rowCount, uint32_t actColCount, uint32_t rowIdx)  // 64 * 128K ==> 4 * 1K
    {
        uint32_t colCount = (actColCount + 31) / 32 * 32;
        auto maskLocal = maskUb[rowIdx * MAX_QK_LEN / sparseSize];
        auto maskHalf = tmpBuff1.Get<half>();
        Duplicate(countUb[rowIdx * BYTE_BLOCK_PFA / sizeof(half)], (half)0.0f, BYTE_BLOCK_PFA / sizeof(half));
        Duplicate(maskLocal.template ReinterpretCast<half>(), (half)0.0, MAX_QK_LEN / sparseSize / sizeof(half));
        pipe_barrier(PIPE_V);

        if (actColCount <= 2 && CAUSAL) { // 2 is activate col count
            Duplicate(maskHalf, (half)1.0f, actColCount);
            pipe_barrier(PIPE_V);
            Cast(maskLocal, maskHalf, RoundMode::CAST_CEIL, actColCount);
            pipe_barrier(PIPE_V);
        } else {
            auto eid_sv = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            auto eid_vs = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            const uint32_t sortApiByteSize = 8;
            auto tmpUb = tmpBuff1.Get<uint8_t>();
            auto sortedLocal = tmpBuff1.Get<half>();                                                             // 8k
            auto indexLocal = tmpUb[sortApiByteSize * colCount].template ReinterpretCast<uint32_t>();            // 4k
            auto dstIndexLocal = indexLocal[sizeof(int32_t) * colCount];                                         // 4k
            auto sortTmpLocal = dstIndexLocal.template ReinterpretCast<half>();                                  // 2k
            auto sortedValueLocal = dstIndexLocal[sizeof(int32_t) * colCount].template ReinterpretCast<half>();  // 2k
            // 不实际需要index
            // clear srcLocal
            Duplicate(sortedLocal.template ReinterpretCast<int16_t>(), (int16_t)0, colCount);
            pipe_barrier(PIPE_V);
            Duplicate(sortedLocal.template ReinterpretCast<int16_t>(),
                (int16_t)1,
                setDiag || (!CAUSAL) ? actColCount - 1 : actColCount);
            pipe_barrier(PIPE_V);  // diag 不参与排序
            if (setFirst) {
                Duplicate(sortedLocal.template ReinterpretCast<int16_t>(), (int16_t)0, 1);
                pipe_barrier(PIPE_V);
            }
            Mul(srcLocal.template ReinterpretCast<int16_t>(),
                srcLocal.template ReinterpretCast<int16_t>(),
                sortedLocal.template ReinterpretCast<int16_t>(),
                colCount);
            pipe_barrier(PIPE_V);
            DataCopy(sortedLocal, srcLocal, colCount);
            pipe_barrier(PIPE_V);
            RowSum(sortedLocal[colCount], sortedLocal, 1, colCount, setDiag ? actColCount - 1 : actColCount);
            pipe_barrier(PIPE_V);
            SetFlag<HardEvent::V_S>(eid_vs);
            WaitFlag<HardEvent::V_S>(eid_vs);
            // causal score求和归一化， 不包括 diag
            auto rs = static_cast<float>(sortedLocal.GetValue(colCount));
            SetFlag<HardEvent::S_V>(eid_sv);
            WaitFlag<HardEvent::S_V>(eid_sv);
            AscendC::Sort<half, true>(sortedLocal, srcLocal, indexLocal, sortTmpLocal, colCount / 32); // sort，一次32个
            pipe_barrier(PIPE_V);
            AscendC::Extract<half>(sortedValueLocal, dstIndexLocal, sortedLocal, colCount / 32); // 一次32个
            pipe_barrier(PIPE_V);
            constexpr uint32_t piece = 16;
            float s = 0;
            int32_t blk = 0;
            int32_t cnt = 0;
            float scoreGuard = 0.0f;

            if (threshold < 1.0f) {
                if (colCount > piece) {
                    auto tmpSrc = tmpUb.template ReinterpretCast<half>();
                    auto pieceSum = tmpSrc[colCount];
                    DataCopy(tmpSrc, sortedValueLocal, colCount);
                    pipe_barrier(PIPE_V);
                    RowSum(pieceSum, tmpSrc, colCount / piece, piece, piece);
                    SetFlag<HardEvent::V_S>(eid_vs);
                    WaitFlag<HardEvent::V_S>(eid_vs);
                    for (auto i = 0; i < colCount / piece; i++) {
                        float vi = static_cast<float>(pieceSum.GetValue(i));
                        if (s + vi > threshold * rs) {
                            blk = i * piece;
                            break;
                        }
                        s += vi;
                    }
                }
                SetFlag<HardEvent::V_S>(eid_vs);
                WaitFlag<HardEvent::V_S>(eid_vs);

                for (auto i = 0; i < piece; i++) {
                    auto vi = static_cast<float>(sortedValueLocal.GetValue(i + blk));
                    scoreGuard = vi;
                    s += vi;
                    cnt = blk + i;
                    if (s >= threshold * rs) {
                        break;
                    }
                }
            } else {
                cnt = actColCount;
                scoreGuard = 0.0f;
            }
            if ((cnt > actColCount * this->rowSparse) && actColCount >= 10) { // 10: 有效列数
                auto kk = static_cast<int32_t>(actColCount * this->rowSparse);
                scoreGuard = static_cast<float>(sortedValueLocal.GetValue(kk - 1));
            }
            scoreGuard -= EPS;
            if (setFirst) {
                srcLocal.SetValue(0, static_cast<half>(scoreGuard + 1.0f));
            }
            if (setDiag || !CAUSAL) {
                srcLocal.SetValue(actColCount - 1, static_cast<half>(scoreGuard + 1.0f));
            }
            SetFlag<HardEvent::S_V>(eid_sv);
            WaitFlag<HardEvent::S_V>(eid_sv);
            auto toRelu = tmpUb.template ReinterpretCast<float>();
            Cast(toRelu, srcLocal, RoundMode::CAST_NONE, colCount);
            pipe_barrier(PIPE_V);
            Adds(toRelu, toRelu, -scoreGuard, colCount);
            pipe_barrier(PIPE_V);
            Relu(toRelu, toRelu, colCount);
            pipe_barrier(PIPE_V);
            Adds(toRelu[colCount], toRelu, 0.001f, colCount);
            pipe_barrier(PIPE_V);
            Div(toRelu, toRelu, toRelu[colCount], colCount);
            pipe_barrier(PIPE_V);
            Cast(maskHalf, toRelu, RoundMode::CAST_NONE, actColCount);
            pipe_barrier(PIPE_V);
            Duplicate(maskHalf, (half)1.0, 1);
            pipe_barrier(PIPE_V);
            Cast(maskLocal, maskHalf, RoundMode::CAST_CEIL, actColCount);
            pipe_barrier(PIPE_V);
            Cast(maskHalf, maskLocal, RoundMode::CAST_NONE, actColCount);
            pipe_barrier(PIPE_V);
        }
        RowSum(countUb[rowIdx * BYTE_BLOCK_PFA / sizeof(half)], maskHalf, 1, colCount, actColCount);
        pipe_barrier(PIPE_V);
    }

    __aicore__ inline void ReorderQKTail(SPFAEstParam *params)
    {
        auto tensorAOffset = params->tensorAOffset;
        auto tensorBOffset = params->tensorBOffset;
        auto qLength = params->singleProcessSOuterSize;  // with stride but til not align
        auto kLength = params->actualSeqLengthKVPerBatch;
        auto gmPingpong = params->sOuterLoopIdx & 1;

        auto tailsize = kLength % stride;
        auto keyUb = maskBuff.Get<KV_T>();
        auto QKTailUb = QKTailBuff.Get<T>();

        uint32_t M = 4 * BUFFER_SIZE_BYTE_32K / sizeof(Q_T) / dim;
        auto times = (qLength + M - 1) / M;
        auto queryUb = mmResBuff.Get<Q_T>();

        for (auto t = 0; t < times; t++) {
            auto len = t == times - 1 ? qLength - M * t : M;
            if constexpr (LAYOUT_T == INPUT_LAYOUT::TND || LAYOUT_T == INPUT_LAYOUT::BSH) {
                DataCopyParams queryCopyParams;
                queryCopyParams.blockLen = dim * sizeof(Q_T) / 32; // 32 is BYTES_PER_BLOCK
                ;
                queryCopyParams.blockCount = len;
                queryCopyParams.srcStride = (head_num_q - 1) * dim / 32; // 32 is BYTES_PER_BLOCK
                queryCopyParams.dstStride = 0;  // 连续
                DataCopy(queryUb,
                    queryGm[tensorAOffset + t * M * head_num_q * dim],
                    queryCopyParams);  // TND layout, offset re compute
            } else if constexpr (LAYOUT_T == INPUT_LAYOUT::BNSD) {
                DataCopy(queryUb, queryGm[tensorAOffset + t * M * dim], len * dim);
            }

            auto eid = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eid);
            WaitFlag<HardEvent::MTE2_V>(eid);
            auto tl = len % stride;
            if (tl > 0) {
                Duplicate(queryUb[len * dim], (Q_T)0, (stride - tl) * dim);
            }

            auto eidv3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eidv3);
            WaitFlag<HardEvent::V_MTE3>(eidv3);
            // Stride = 4 k 只有三个 k0 k1 k2
            // q0 q1 q2 q3
            // 0  k2 k1 k0
            DataCopyParams queryCopyParams;
            auto blkLen = dim * sizeof(Q_T) / BYTE_BLOCK_PFA;
            auto ql = (len + stride - 1) / stride;
            queryCopyParams.blockLen = blkLen;
            queryCopyParams.blockCount = ql;
            queryCopyParams.srcStride = (stride - 1) * blkLen;
            queryCopyParams.dstStride = (stride - 1) * blkLen;
            for (auto j = 0; j < stride; j++) {
                DataCopy(
                    queryGmTmp[gmPingpong][j * dim + t * M * dim], queryUb[(stride - j - 1) * dim], queryCopyParams);
            }
            eidv3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
            SetFlag<HardEvent::MTE3_MTE2>(eidv3);
            WaitFlag<HardEvent::MTE3_MTE2>(eidv3);
            pipe_barrier(PIPE_V);
        }
    }

    __aicore__ inline void CopyParamsAttrOutOfInnerLoop(SPFAEstParam *dst, SPFAEstParam *src)
    {
        dst->bIdx = src->bIdx;
        dst->batchNOffset = src->batchNOffset;
        dst->sOuterLoopIdx = src->sOuterLoopIdx;
        dst->sInnerLoopIdx = src->sInnerLoopIdx;
        dst->singleProcessSOuterSize = src->singleProcessSOuterSize;
        dst->sOuterOffset = src->sOuterOffset;
        dst->mm1SingleCoreN = src->mm1SingleCoreN;
        dst->actualSeqLengthPerBatch = src->actualSeqLengthPerBatch;
        dst->actualSeqLengthKVPerBatch = src->actualSeqLengthKVPerBatch;
        dst->singleProcessSInnerBmmTail = src->singleProcessSInnerBmmTail;
        dst->tensorAOffset = src->tensorAOffset;
        dst->tensorBOffset = src->tensorBOffset;
    }
};
#endif